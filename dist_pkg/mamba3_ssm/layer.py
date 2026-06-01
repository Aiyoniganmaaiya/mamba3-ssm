"""
Mamba-3 layer: the core sequence-mixing module.

Combines:
  1. Exponential-trapezoidal discretization (Mamba-3 vs Mamba-2 improvement)
  2. Complex-valued / RoPE state space (rotation on B and C projections)
  3. MIMO formulation (optional, for better decode efficiency)

Both SISO and MIMO modes share the same interface:
    Input:  (batch, seq_len, d_model)
    Output: (batch, seq_len, d_model)

The module also supports single-step autoregressive decoding via .step().
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .ops import RMSNorm, apply_rope, ssm_scan_siso, ssm_scan_mimo


class Mamba3(nn.Module):
    """Mamba-3 selective state space sequence mixer.

    This is a drop-in replacement for a Transformer attention layer.

    Architecture
    ------------
    Input u ──→ Linear ──→ split into {z, x, B, C, dt, A, trap, angles}
                   │
                   ├── RMSNorm(B) + B_bias  ──→ RoPE(B)
                   ├── RMSNorm(C) + C_bias  ──→ RoPE(C)
                   ├── softplus(dt_raw) + dt_bias → DT
                   ├── softplus(A_raw) → A (negated)
                   └── sigmoid(trap_raw) → trap_gate
                   │
                   ▼
             SSM scan (SISO or MIMO)
                   │
                   ▼
             y = scan_out ⊙ silu(z)
                   │
                   ▼
             Linear ──→ Output

    Key dimensions
    --------------
    d_model:  token embedding dimension (input/output)
    d_inner:  expand × d_model  (expanded hidden dim)
    nheads:   d_inner / headdim  (number of SSM heads, H)
    headdim:  per-head feature dim (P)
    d_state:  per-head state dim (D)
    mimo_rank: R parallel MIMO streams (R=1 for SISO)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # ── Dimension bookkeeping ─────────────────────────────────────────
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.A_floor = A_floor
        self.is_mimo = is_mimo
        self.mimo_rank = mimo_rank if is_mimo else 1
        self.num_bc_heads = ngroups

        self.d_inner = int(expand * d_model)
        assert self.d_inner % headdim == 0, \
            f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"
        self.nheads = self.d_inner // headdim  # H

        # ── RoPE / angle configuration ────────────────────────────────────
        # rope_fraction controls what fraction of d_state uses rotation.
        # 0.5 → first d_state/2 dims are complex pairs → d_state/4 angles
        # 1.0 → all d_state dims rotate → d_state/2 angles
        assert rope_fraction in (0.5, 1.0), "rope_fraction must be 0.5 or 1.0"
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1  # must be even (pairs)
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # ── Single fused input projection ─────────────────────────────────
        # Produces all SSM parameters from one linear to minimize memory traffic.
        # Layout: [z | x | B | C | dt_raw | A_raw | trap_raw | angle_raw]
        d_in_proj = (
            2 * self.d_inner                                 # z (gate) + x (values)
            + 2 * d_state * ngroups * self.mimo_rank         # B (K) + C (Q)
            + 3 * self.nheads                                # dt + A + trap
            + self.num_rope_angles                            # rotation rate
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)

        # ── dt bias (initialized from log-uniform in [dt_min, dt_max]) ────
        _dt = torch.exp(
            torch.rand(self.nheads, dtype=torch.float32)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse softplus so that softplus(dt_bias) ≈ dt_init
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias)
        self.dt_bias._no_weight_decay = True

        # ── B/C biases (initialized to 1 so they don't suppress updates) ──
        self.B_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32)
        )
        self.C_bias = nn.Parameter(
            torch.ones(self.nheads, self.mimo_rank, d_state, dtype=torch.float32)
        )
        self.B_bias._no_weight_decay = True
        self.C_bias._no_weight_decay = True

        # ── RMS norms for B and C projections ─────────────────────────────
        self.B_norm = RMSNorm(d_state)
        self.C_norm = RMSNorm(d_state)

        # ── MIMO projection matrices ──────────────────────────────────────
        if self.is_mimo:
            # mimo_x: (H, R, P) — down-project x from P to R scalars per head
            # mimo_o: (H, R, P) — up-project R scalars back to P dimensions
            # Initialized to 1/R so the sum ≈ 1x
            self.mimo_x = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs)
                / self.mimo_rank
            )
            self.mimo_o = nn.Parameter(
                torch.ones(self.nheads, self.mimo_rank, self.headdim, **factory_kwargs)
                / self.mimo_rank
            )

        # ── D skip connection ─────────────────────────────────────────────
        self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))
        self.D._no_weight_decay = True

        # ── Output projection ─────────────────────────────────────────────
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

    # =========================================================================
    # Forward pass (full sequence)
    # =========================================================================

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply Mamba-3 to an entire sequence.

        Args:
            u: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, L, _ = u.shape

        # ── Step 1: Fused projection ──────────────────────────────────────
        all_params = self.in_proj(u)  # (B, L, d_in_proj)

        splits = [
            self.d_inner,
            self.d_inner,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.nheads, self.nheads, self.nheads,
            self.num_rope_angles,
        ]
        z, x, B_raw, C_raw, dd_dt, dd_A, trap_raw, angle_raw = \
            torch.split(all_params, splits, dim=-1)

        # ── Step 2: Reshape to head-based tensors ─────────────────────────
        z = rearrange(z, "b l (h p) -> b l h p", p=self.headdim)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)

        B_raw = rearrange(
            B_raw, "b l (r g d) -> b l r g d",
            r=self.mimo_rank, g=self.num_bc_heads
        )
        C_raw = rearrange(
            C_raw, "b l (r g d) -> b l r g d",
            r=self.mimo_rank, g=self.num_bc_heads
        )

        # ── Step 3: Compute A, dt, and A·dt ──────────────────────────────
        A = -F.softplus(dd_A.float()).clamp(max=-self.A_floor)  # (B, L, H), negative
        DT = F.softplus(dd_dt.float() + self.dt_bias)             # (B, L, H), positive
        ADT = A * DT                                               # (B, L, H)

        # ── Step 4: Trapezoidal gate ──────────────────────────────────────
        trap = torch.sigmoid(trap_raw.float())  # (B, L, H), in [0, 1]

        # ── Step 5: B/C → RMSNorm → expand groups → add bias ─────────────
        B_normed = self.B_norm(B_raw.float())   # (B, L, R, G, D)
        C_normed = self.C_norm(C_raw.float())

        # Expand groups to heads (G can be 1 or nheads)
        B_exp = B_normed.expand(-1, -1, -1, self.nheads, -1)  # (B, L, R, H, D)
        C_exp = C_normed.expand(-1, -1, -1, self.nheads, -1)

        B_bias_t = rearrange(self.B_bias, "h r d -> r h d")  # (R, H, D)
        C_bias_t = rearrange(self.C_bias, "h r d -> r h d")
        B_exp = B_exp + B_bias_t  # broadcasts over (B, L)
        C_exp = C_exp + C_bias_t

        # ── Step 6: Apply RoPE to B and C ────────────────────────────────
        # Cumulative angle = cumsum(dt * angle_rate), per head
        angle_increments = (
            angle_raw.float().unsqueeze(2)    # (B, L, 1, S)
            * DT.float().unsqueeze(-1)        # (B, L, H, 1)
        )                                     # → (B, L, H, S)
        cum_angles = torch.cumsum(angle_increments, dim=1)  # (B, L, H, S)

        # Expand to all R ranks
        angles_rot = cum_angles.unsqueeze(2).expand(
            batch, L, self.mimo_rank, self.nheads, self.num_rope_angles
        )

        # Rotate the first split_tensor_size dims; leave the rest real-valued
        B_rot = apply_rope(B_exp[..., :self.split_tensor_size], angles_rot)
        C_rot = apply_rope(C_exp[..., :self.split_tensor_size], angles_rot)
        B_proj = torch.cat([B_rot, B_exp[..., self.split_tensor_size:]], dim=-1)
        C_proj = torch.cat([C_rot, C_exp[..., self.split_tensor_size:]], dim=-1)

        # ── Step 7: SSM scan ─────────────────────────────────────────────
        if self.is_mimo:
            y = ssm_scan_mimo(
                x=x, B_proj=B_proj, C_proj=C_proj,
                ADT=ADT, DT=DT, trap=trap, D=self.D,
                mimo_x=self.mimo_x, mimo_o=self.mimo_o,
            )
        else:
            y = ssm_scan_siso(
                x=x,
                B_proj=B_proj[:, :, 0],   # squeeze R=1
                C_proj=C_proj[:, :, 0],
                ADT=ADT, DT=DT, trap=trap, D=self.D,
            )

        # Gated output
        y = y * F.silu(z.float())

        # ── Step 8: Output projection ────────────────────────────────────
        y = rearrange(y, "b l h p -> b l (h p)")
        return self.out_proj(y.to(x.dtype))

    # =========================================================================
    # Single-step decode (autoregressive inference)
    # =========================================================================

    def step(
        self,
        u: torch.Tensor,
        angle_state: torch.Tensor,
        ssm_state: torch.Tensor,
        bx_prev_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single token for autoregressive decoding.

        Args:
            u:              (batch, d_model)       — current input token
            angle_state:    (batch, H, S)          — accumulated RoPE angles
            ssm_state:      (batch, H, P, D) or (batch, H, D) — SSM hidden state
            bx_prev_state:  same as ssm_state       — previous B*x for trapezoid

        Returns:
            out:            (batch, d_model)
            angle_state:    updated
            ssm_state:      updated
            bx_prev_state:  updated
        """
        batch = u.shape[0]

        # Fused projection (no sequence dimension)
        all_params = self.in_proj(u)  # (B, d_in_proj)

        splits = [
            self.d_inner,
            self.d_inner,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.d_state * self.num_bc_heads * self.mimo_rank,
            self.nheads, self.nheads, self.nheads,
            self.num_rope_angles,
        ]
        z, x, B_raw, C_raw, dd_dt, dd_A, trap_raw, angle_raw = \
            torch.split(all_params, splits, dim=-1)

        z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)

        B_raw = rearrange(B_raw, "b (r g d) -> b r g d",
                          r=self.mimo_rank, g=self.num_bc_heads)
        C_raw = rearrange(C_raw, "b (r g d) -> b r g d",
                          r=self.mimo_rank, g=self.num_bc_heads)

        A = -F.softplus(dd_A.float()).clamp(max=-self.A_floor)
        DT = F.softplus(dd_dt.float() + self.dt_bias)
        ADT = A * DT
        trap = torch.sigmoid(trap_raw.float())

        # RMSNorm + expand + bias
        B_normed = self.B_norm(B_raw.float())
        C_normed = self.C_norm(C_raw.float())
        B_exp = B_normed.expand(-1, -1, self.nheads, -1)
        C_exp = C_normed.expand(-1, -1, self.nheads, -1)
        B_exp = B_exp + rearrange(self.B_bias, "h r d -> r h d")
        C_exp = C_exp + rearrange(self.C_bias, "h r d -> r h d")

        # RoPE: update cumulative angle state
        delta_angle = angle_raw.float().unsqueeze(1) * DT.float().unsqueeze(-1)
        angle_state = angle_state + delta_angle
        angles_rot = angle_state.unsqueeze(1).expand(
            -1, self.mimo_rank, -1, -1
        )
        B_rot = apply_rope(B_exp[..., :self.split_tensor_size], angles_rot)
        C_rot = apply_rope(C_exp[..., :self.split_tensor_size], angles_rot)
        B_proj = torch.cat([B_rot, B_exp[..., self.split_tensor_size:]], dim=-1)
        C_proj = torch.cat([C_rot, C_exp[..., self.split_tensor_size:]], dim=-1)

        # State update (single timestep)
        decay = torch.exp(ADT)

        if self.is_mimo:
            # MIMO: state is (B, H, D)
            x_r = torch.einsum("bhp,hrp->bhr", x.float(), self.mimo_x.float())
            Bx_curr = torch.einsum("bhr,brhd->bhd", x_r, B_proj.float())
            tr_e = trap.unsqueeze(-1)
            Bx_blend = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + bx_prev_state)
            ssm_state = decay.unsqueeze(-1) * ssm_state + DT.unsqueeze(-1) * Bx_blend

            y_r = torch.einsum("brhd,bhd->brh", C_proj.float(), ssm_state)
            skip = self.D.unsqueeze(0).unsqueeze(0) * x_r.permute(0, 2, 1)
            y_pre = y_r + skip
            y = torch.einsum("brh,hrp->bhp", y_pre, self.mimo_o.float())
            y = y * F.silu(z.float())
            bx_prev_state = Bx_curr
        else:
            # SISO: state is (B, H, P, D)
            Bx_curr = torch.einsum("bhp,bhd->bhpd", x.float(), B_proj[:, 0].float())
            tr_e = trap.unsqueeze(-1).unsqueeze(-1)
            Bx_blend = (1.0 - tr_e) * Bx_curr + tr_e * 0.5 * (Bx_curr + bx_prev_state)
            ssm_state = (
                decay.unsqueeze(-1).unsqueeze(-1) * ssm_state
                + DT.unsqueeze(-1).unsqueeze(-1) * Bx_blend
            )
            y = torch.einsum("bhd,bhpd->bhp", C_proj[:, 0].float(), ssm_state)
            y = y + self.D.unsqueeze(0).unsqueeze(-1) * x.float()
            y = y * F.silu(z.float())
            bx_prev_state = Bx_curr

        y = rearrange(y, "b h p -> b (h p)")
        out = self.out_proj(y.to(u.dtype))
        return out, angle_state, ssm_state, bx_prev_state

    # =========================================================================
    # Inference cache helpers
    # =========================================================================

    def allocate_inference_cache(
        self, batch_size: int, device=None, dtype=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Allocate zero-initialized states for autoregressive decoding.

        Returns:
            angle_state:    (batch, H, num_rope_angles)
            ssm_state:      (batch, H, P, D) for SISO or (batch, H, D) for MIMO
            bx_prev:        same shape as ssm_state
        """
        device = device or self.in_proj.weight.device

        angle_state = torch.zeros(
            batch_size, self.nheads, self.num_rope_angles,
            device=device, dtype=torch.float32,
        )

        if self.is_mimo:
            def make_zeros(*shape):
                return torch.zeros(batch_size, *shape, device=device, dtype=torch.float32)
            ssm_state = make_zeros(self.nheads, self.d_state)
            bx_prev = make_zeros(self.nheads, self.d_state)
        else:
            ssm_state = torch.zeros(
                batch_size, self.nheads, self.headdim, self.d_state,
                device=device, dtype=torch.float32,
            )
            bx_prev = torch.zeros(
                batch_size, self.nheads, self.headdim, self.d_state,
                device=device, dtype=torch.float32,
            )

        return angle_state, ssm_state, bx_prev

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"d_inner={self.d_inner}, nheads={self.nheads}, "
            f"headdim={self.headdim}, is_mimo={self.is_mimo}, "
            f"mimo_rank={self.mimo_rank}"
        )
