"""
Core operations for Mamba-3:
  - RMSNorm, RoPE
  - SSM scan (SISO and MIMO) — optimized sequential scan

The scan uses a simple sequential loop optimized for torch.compile.
Use torch.compile(model) or torch.compile(layer.mixer.forward) for
2-3x speedup on the scan.

Notation: B=batch, L=seq_len, H=nheads, P=headdim, D=d_state, R=mimo_rank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms * self.weight).to(x.dtype)


# =============================================================================
# RoPE
# =============================================================================

def build_rope_freqs(num_angles: int, device: torch.device) -> torch.Tensor:
    i = torch.arange(num_angles, device=device, dtype=torch.float32)
    return 1.0 / (10000.0 ** (i / num_angles))


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(angles), torch.sin(angles)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


# =============================================================================
# SSM Scan — SISO (torch.compile-friendly)
# =============================================================================

def ssm_scan_siso(
    x: torch.Tensor,       # (B, L, H, P)
    B_proj: torch.Tensor,  # (B, L, H, D)
    C_proj: torch.Tensor,  # (B, L, H, D)
    ADT: torch.Tensor,     # (B, L, H)  negative
    DT: torch.Tensor,      # (B, L, H)  positive
    trap: torch.Tensor,    # (B, L, H)
    D: torch.Tensor,       # (H,)
) -> torch.Tensor:
    """SSM scan for SISO mode. Optimized for torch.compile."""
    B, L, H, P = x.shape
    Ds = B_proj.shape[-1]
    dtype = x.dtype

    # Pre-compute: cast once, compute all factors
    decay = torch.exp(ADT.float())                     # (B, L, H)
    dt = DT.float()                                     # (B, L, H)
    tr = torch.sigmoid(trap.float())                    # (B, L, H)

    h = torch.zeros(B, H, P, Ds, device=x.device)
    Bx_prev = torch.zeros(B, H, P, Ds, device=x.device)
    y_out = torch.empty(B, L, H, P, dtype=dtype, device=x.device)

    D_w = D.unsqueeze(0).unsqueeze(-1)                  # (1, H, 1)

    for t in range(L):
        x_t = x[:, t].float()                           # (B, H, P)
        B_t = B_proj[:, t].float()                      # (B, H, D)
        C_t = C_proj[:, t].float()                      # (B, H, D)

        # Outer product
        Bx = torch.einsum("bhp,bhd->bhpd", x_t, B_t)    # (B, H, P, D)

        # Trapezoidal blend
        blend = (1.0 - tr[:, t, :, None, None]) * Bx + \
                tr[:, t, :, None, None] * 0.5 * (Bx + Bx_prev)

        # State update
        h = decay[:, t, :, None, None] * h + dt[:, t, :, None, None] * blend

        # Output
        y_out[:, t] = (torch.einsum("bhd,bhpd->bhp", C_t, h) + D_w * x_t).to(dtype)
        Bx_prev = Bx

    return y_out


# =============================================================================
# SSM Scan — MIMO (torch.compile-friendly)
# =============================================================================

def ssm_scan_mimo(
    x: torch.Tensor,        # (B, L, H, P)
    B_proj: torch.Tensor,   # (B, L, R, H, D)
    C_proj: torch.Tensor,   # (B, L, R, H, D)
    ADT: torch.Tensor,      # (B, L, H)
    DT: torch.Tensor,       # (B, L, H)
    trap: torch.Tensor,     # (B, L, H)
    D: torch.Tensor,        # (H,)
    mimo_x: torch.Tensor,   # (H, R, P)
    mimo_o: torch.Tensor,   # (H, R, P)
) -> torch.Tensor:
    """SSM scan for MIMO mode. Optimized for torch.compile."""
    B, L, H, P = x.shape
    R = B_proj.shape[2]
    Ds = B_proj.shape[-1]
    dtype = x.dtype

    decay = torch.exp(ADT.float())
    dt = DT.float()
    tr = torch.sigmoid(trap.float())

    h = torch.zeros(B, H, Ds, device=x.device)
    Bx_prev = torch.zeros(B, H, Ds, device=x.device)
    y_out = torch.empty(B, L, H, P, dtype=dtype, device=x.device)

    for t in range(L):
        x_t = x[:, t].float()
        B_t = B_proj[:, t].float()
        C_t = C_proj[:, t].float()

        x_r = torch.einsum("bhp,hrp->bhr", x_t, mimo_x.float())
        Bx = torch.einsum("bhr,brhd->bhd", x_r, B_t)

        blend = (1.0 - tr[:, t, :, None]) * Bx + tr[:, t, :, None] * 0.5 * (Bx + Bx_prev)
        h = decay[:, t, :, None] * h + dt[:, t, :, None] * blend

        y_r = torch.einsum("brhd,bhd->brh", C_t, h)
        skip = D[None, None, :] * x_r.permute(0, 2, 1)
        y_pre = y_r + skip
        y_out[:, t] = torch.einsum("brh,hrp->bhp", y_pre, mimo_o.float()).to(dtype)

        Bx_prev = Bx

    return y_out
