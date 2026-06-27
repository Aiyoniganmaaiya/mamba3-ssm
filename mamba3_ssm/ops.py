"""
Core operations for Mamba-3:
  - RMSNorm, RoPE
  - SSM scan (SISO and MIMO) — CUDA/JIT accelerated with Python fallback

Notation: B=batch, L=seq_len, H=nheads, P=headdim, D=d_state, R=mimo_rank
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import cuda_backend

# =============================================================================
# JIT-compiled scan kernels (2-3x faster than Python loop)
# =============================================================================

@torch.jit.script
def _siso_scan_jit(
    x: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    decay: torch.Tensor,
    dt: torch.Tensor,
    tr: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    B, L, H, P = x.shape
    Ds = B_proj.shape[-1]
    d_dtype = x.dtype

    h = torch.zeros(B, H, P, Ds, device=x.device)
    Bx_prev = torch.zeros(B, H, P, Ds, device=x.device)
    y_out = torch.empty(B, L, H, P, dtype=d_dtype, device=x.device)
    D_w = D.unsqueeze(0).unsqueeze(-1)

    for t in range(L):
        x_t = x[:, t]
        B_t = B_proj[:, t]
        C_t = C_proj[:, t]

        Bx = torch.einsum("bhp,bhd->bhpd", x_t, B_t)
        blend = (1.0 - tr[:, t, :, None, None]) * Bx + \
                tr[:, t, :, None, None] * 0.5 * (Bx + Bx_prev)
        h = decay[:, t, :, None, None] * h + dt[:, t, :, None, None] * blend
        y_out[:, t] = (torch.einsum("bhd,bhpd->bhp", C_t, h) + D_w * x_t).to(d_dtype)
        Bx_prev = Bx

    return y_out


@torch.jit.script
def _mimo_scan_jit(
    x: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    decay: torch.Tensor,
    dt: torch.Tensor,
    tr: torch.Tensor,
    D: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
) -> torch.Tensor:
    B, L, H, P = x.shape
    R = B_proj.shape[2]
    Ds = B_proj.shape[-1]
    d_dtype = x.dtype

    h = torch.zeros(B, H, Ds, device=x.device)
    Bx_prev = torch.zeros(B, H, Ds, device=x.device)
    y_out = torch.empty(B, L, H, P, dtype=d_dtype, device=x.device)

    for t in range(L):
        x_t = x[:, t]
        B_t = B_proj[:, t]
        C_t = C_proj[:, t]

        x_r = torch.einsum("bhp,hrp->bhr", x_t, mimo_x)
        Bx = torch.einsum("bhr,brhd->bhd", x_r, B_t)
        blend = (1.0 - tr[:, t, :, None]) * Bx + tr[:, t, :, None] * 0.5 * (Bx + Bx_prev)
        h = decay[:, t, :, None] * h + dt[:, t, :, None] * blend

        y_r = torch.einsum("brhd,bhd->brh", C_t, h)
        skip = D[None, None, :] * x_r.permute(0, 2, 1)
        y_pre = y_r + skip
        y_out[:, t] = torch.einsum("brh,hrp->bhp", y_pre, mimo_o).to(d_dtype)
        Bx_prev = Bx

    return y_out


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
# SSM Scan — SISO
# =============================================================================

def ssm_scan_siso(
    x: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    trap: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    """SSM scan for SISO mode.

    Acceleration path:
      1. CUDA kernel (requires MSVC to compile)
      2. JIT-compiled (2-3x vs Python loop)
      3. Pure Python (always works)
    """
    B, L, H, P = x.shape
    Ds = B_proj.shape[-1]
    dtype = x.dtype

    decay = torch.exp(ADT.float())
    dt = DT.float()
    tr = torch.sigmoid(trap.float())

    # Try CUDA
    if x.is_cuda and Ds in (16, 32, 64, 128):
        y_cuda = cuda_backend.siso_scan_cuda(x, B_proj, C_proj, decay, dt, tr, D)
        if y_cuda is not None:
            return y_cuda.to(dtype)

    # Try JIT (CUDA only)
    if x.is_cuda:
        return _siso_scan_jit(x, B_proj, C_proj, decay, dt, tr, D).to(dtype)

    # Pure Python fallback
    h = torch.zeros(B, H, P, Ds, device=x.device)
    Bx_prev = torch.zeros(B, H, P, Ds, device=x.device)
    y_out = torch.empty(B, L, H, P, dtype=dtype, device=x.device)
    D_w = D.unsqueeze(0).unsqueeze(-1)

    for t in range(L):
        x_t = x[:, t].float()
        B_t = B_proj[:, t].float()
        C_t = C_proj[:, t].float()
        Bx = torch.einsum("bhp,bhd->bhpd", x_t, B_t)
        blend = (1.0 - tr[:, t, :, None, None]) * Bx + \
                tr[:, t, :, None, None] * 0.5 * (Bx + Bx_prev)
        h = decay[:, t, :, None, None] * h + dt[:, t, :, None, None] * blend
        y_out[:, t] = (torch.einsum("bhd,bhpd->bhp", C_t, h) + D_w * x_t).to(dtype)
        Bx_prev = Bx

    return y_out


# =============================================================================
# SSM Scan — MIMO
# =============================================================================

def ssm_scan_mimo(
    x: torch.Tensor,
    B_proj: torch.Tensor,
    C_proj: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    trap: torch.Tensor,
    D: torch.Tensor,
    mimo_x: torch.Tensor,
    mimo_o: torch.Tensor,
) -> torch.Tensor:
    """SSM scan for MIMO mode.

    Acceleration path:
      1. CUDA kernel (requires MSVC to compile)
      2. JIT-compiled (2-3x vs Python loop)
      3. Pure Python (always works)
    """
    B, L, H, P = x.shape
    R = B_proj.shape[2]
    Ds = B_proj.shape[-1]
    dtype = x.dtype

    decay = torch.exp(ADT.float())
    dt = DT.float()
    tr = torch.sigmoid(trap.float())

    # Try CUDA
    if x.is_cuda and Ds in (16, 32, 64, 128) and R in (1, 2, 4, 8):
        y_cuda = cuda_backend.mimo_state_cuda(x, B_proj, C_proj, decay, dt, tr, D, mimo_x, mimo_o, R)
        if y_cuda is not None:
            return y_cuda.to(dtype)

    # Try JIT (CUDA only)
    if x.is_cuda:
        return _mimo_scan_jit(x, B_proj, C_proj, decay, dt, tr, D, mimo_x, mimo_o).to(dtype)

    # Pure Python fallback
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
