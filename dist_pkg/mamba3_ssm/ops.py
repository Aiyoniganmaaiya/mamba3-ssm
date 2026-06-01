"""
Core operations for Mamba-3:
  - RMSNorm
  - RoPE (Rotary Position Embeddings) utilities
  - SSM sequential scan (SISO and MIMO variants)

All operations use plain PyTorch — no custom CUDA/Triton kernels.
The sequential scans are O(L) in sequence length but easy to follow;
production code uses parallel chunk scans for throughput.

Notation (matching the paper):
    B = batch, L = seq_len, H = nheads, P = headdim, D = d_state, R = mimo_rank
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm does not center the input (no mean subtraction).
    Formula: y = (x / rms(x)) * weight,   where rms(x) = sqrt(mean(x²) + eps)
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight).to(x.dtype)


# =============================================================================
# RoPE (Rotary Position Embeddings)
# =============================================================================

def build_rope_freqs(num_angles: int, device: torch.device) -> torch.Tensor:
    """Build the standard RoPE inverse-frequency vector.

    freqs[i] = 1 / 10000^(2i / num_angles)

    Returns:
        freqs: (num_angles,) tensor of rotation frequencies
    """
    i = torch.arange(num_angles, device=device, dtype=torch.float32)
    return 1.0 / (10000.0 ** (i / num_angles))


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of dimensions of x by the given angles (in radians).

    For each pair (a, b) with angle θ:
        (a·cosθ - b·sinθ,  a·sinθ + b·cosθ)

    Args:
        x:       (..., 2 * num_angles)
        angles:  (..., num_angles)

    Returns:
        Rotated tensor of shape (..., 2 * num_angles)
    """
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # Split into even/odd pairs
    x1 = x[..., 0::2]   # (..., num_angles)
    x2 = x[..., 1::2]   # (..., num_angles)

    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    # Interleave back
    out = torch.stack([x1_rot, x2_rot], dim=-1)
    return out.flatten(-2)


# =============================================================================
# SSM Scan — SISO variant
# =============================================================================

def ssm_scan_siso(
    x: torch.Tensor,       # (B, L, H, P)
    B_proj: torch.Tensor,  # (B, L, H, D)
    C_proj: torch.Tensor,  # (B, L, H, D)
    ADT: torch.Tensor,     # (B, L, H)  — A·dt (negative)
    DT: torch.Tensor,      # (B, L, H)  — dt (positive)
    trap: torch.Tensor,    # (B, L, H)  — sigmoid gate
    D: torch.Tensor,       # (H,)       — skip connection weight
) -> torch.Tensor:
    """Sequential SSM scan in SISO mode.

    Recurrence:
        h_t = exp(A·dt_t) · h_{t-1}
            + dt_t · [ (1-σ(trap_t)) · (B_t ⊗ x_t)
                       + σ(trap_t) · (B_t ⊗ x_t + B_{t-1} ⊗ x_{t-1}) / 2 ]

        y_t = C_t · h_t + D · x_t

    State h has shape (B, H, P, D).  The outer product x_t ⊗ B_t means each
    of the P feature dimensions independently contributes to the D-dimensional
    state via its own vector of input weights.

    Returns:
        y: (B, L, H, P)
    """
    B_sz, L, H, P = x.shape
    D_state = B_proj.shape[-1]
    device = x.device
    dtype = x.dtype

    h = torch.zeros(B_sz, H, P, D_state, device=device, dtype=torch.float32)
    Bx_prev = torch.zeros(B_sz, H, P, D_state, device=device, dtype=torch.float32)
    outputs = []

    for t in range(L):
        x_t = x[:, t]                         # (B, H, P)
        B_t = B_proj[:, t]                    # (B, H, D)
        C_t = C_proj[:, t]                    # (B, H, D)
        adt_t = ADT[:, t]                     # (B, H)
        dt_t = DT[:, t]                       # (B, H)
        tr_t = trap[:, t]                     # (B, H)

        decay = torch.exp(adt_t).unsqueeze(-1).unsqueeze(-1)          # (B, H, 1, 1)
        dt_e = dt_t.unsqueeze(-1).unsqueeze(-1)                       # (B, H, 1, 1)
        tr_e = tr_t.unsqueeze(-1).unsqueeze(-1)                        # (B, H, 1, 1)

        # Outer product: Bx[b,h,p,d] = x[b,h,p] * B[b,h,d]
        Bx_curr = torch.einsum("bhp,bhd->bhpd", x_t.float(), B_t.float())

        # Trapezoidal blend
        Bx_blend = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + Bx_prev)

        # State update
        h = decay * h + dt_e * Bx_blend

        # Output
        y_t = torch.einsum("bhd,bhpd->bhp", C_t.float(), h)
        y_t = y_t + D.unsqueeze(0).unsqueeze(-1).to(torch.float32) * x_t.float()

        outputs.append(y_t.to(dtype))
        Bx_prev = Bx_curr

    return torch.stack(outputs, dim=1)  # (B, L, H, P)


# =============================================================================
# SSM Scan — MIMO variant
# =============================================================================

def ssm_scan_mimo(
    x: torch.Tensor,        # (B, L, H, P)
    B_proj: torch.Tensor,   # (B, L, R, H, D)
    C_proj: torch.Tensor,   # (B, L, R, H, D)
    ADT: torch.Tensor,      # (B, L, H)
    DT: torch.Tensor,       # (B, L, H)
    trap: torch.Tensor,     # (B, L, H)
    D: torch.Tensor,        # (H,)
    mimo_x: torch.Tensor,   # (H, R, P)  — down-projection
    mimo_o: torch.Tensor,   # (H, R, P)  — up-projection
) -> torch.Tensor:
    """Sequential SSM scan in MIMO mode.

    MIMO replaces the P×D state with a shared D-dimensional state updated by
    R rank-1 contributions projected from x.  This multiplies compute/byte by R.

    Algorithm per timestep t:
        x_r     = x_t · mimo_x          → (B, H, R)  (P-direction collapsed to R scalars)
        Bx_t    = Σ_r x_r[:,:,r] · B_t[:,r,:,:]  → (B, H, D)
        h_t     = exp(A·dt) · h_{t-1} + dt · blend(Bx_t, Bx_{t-1})
        y_r     = C_t · h_t             → (B, R, H)   (per-rank output scalar)
        y_t     = Σ_r (y_r + D·x_r) · mimo_o  → (B, H, P)

    Returns:
        y: (B, L, H, P)
    """
    B_sz, L, H, P = x.shape
    _, _, R, _, D_state = B_proj.shape
    device = x.device
    dtype = x.dtype

    h = torch.zeros(B_sz, H, D_state, device=device, dtype=torch.float32)
    Bx_prev = torch.zeros(B_sz, H, D_state, device=device, dtype=torch.float32)
    outputs = []

    for t in range(L):
        x_t = x[:, t]                          # (B, H, P)
        B_t = B_proj[:, t]                     # (B, R, H, D)
        C_t = C_proj[:, t]                     # (B, R, H, D)
        adt_t = ADT[:, t]                      # (B, H)
        dt_t = DT[:, t]                        # (B, H)
        tr_t = trap[:, t]                      # (B, H)

        decay = torch.exp(adt_t)               # (B, H)
        dt_e = dt_t                            # (B, H)

        # Down-project x: P → R
        x_r = torch.einsum("bhp,hrp->bhr", x_t.float(), mimo_x.float())  # (B, H, R)

        # Accumulate R rank-1 contributions into D-dim state
        Bx_curr = torch.einsum("bhr,brhd->bhd", x_r, B_t.float())        # (B, H, D)

        # Trapezoidal blend
        tr_e3 = tr_t.unsqueeze(-1)                                          # (B, H, 1)
        Bx_blend = (1.0 - tr_e3) * Bx_curr + tr_e3 * 0.5 * (Bx_curr + Bx_prev)

        # State update (scalar × D-vector, no P dimension)
        h = decay.unsqueeze(-1) * h + dt_e.unsqueeze(-1) * Bx_blend       # (B, H, D)

        # Per-rank output scalars
        y_r = torch.einsum("brhd,bhd->brh", C_t.float(), h)              # (B, R, H)

        # Skip connection
        skip = D.unsqueeze(0).unsqueeze(0) * x_r.permute(0, 2, 1)        # (B, R, H)
        y_pre = y_r + skip                                                 # (B, R, H)

        # Up-project: R → P
        y_t = torch.einsum("brh,hrp->bhp", y_pre, mimo_o.float())       # (B, H, P)

        outputs.append(y_t.to(dtype))
        Bx_prev = Bx_curr

    return torch.stack(outputs, dim=1)  # (B, L, H, P)
