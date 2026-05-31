# API Reference

## `Mamba3`

The core sequence-mixer. Processes sequences via state space recurrence instead of attention.

### Constructor

```python
Mamba3(
    d_model: int,              # Token embedding dimension
    d_state: int = 128,        # D: SSM state size per head
    expand: int = 2,           # Inner dim multiplier; d_inner = expand * d_model
    headdim: int = 64,         # P: per-head feature dimension
    ngroups: int = 1,          # G: groups for B/C projection sharing
    rope_fraction: float = 0.5,  # Fraction of d_state dims that rotate (0.5 or 1.0)
    dt_min: float = 0.001,     # Minimum initial time step
    dt_max: float = 0.1,       # Maximum initial time step
    dt_init_floor: float = 1e-4,
    A_floor: float = 1e-4,     # Minimum state decay magnitude
    is_mimo: bool = False,     # MIMO mode
    mimo_rank: int = 4,        # R: MIMO rank, only used when is_mimo=True
)
```

### `forward(u: Tensor) -> Tensor`

Full-sequence forward pass.

- **Input:** `u` — `(batch, seq_len, d_model)`
- **Output:** `(batch, seq_len, d_model)`

### `step(u, angle_state, ssm_state, bx_prev) -> (out, angle_state, ssm_state, bx_prev)`

Single autoregressive decode step.

- **Input:** `u` — `(batch, d_model)` — one token
- **Returns:**
  - `out` — `(batch, d_model)`
  - `angle_state` — `(batch, H, num_rope_angles)` — updated RoPE angles
  - `ssm_state` — `(batch, H, P, D)` (SISO) or `(batch, H, D)` (MIMO) — updated state
  - `bx_prev` — same shape as `ssm_state` — trapezoidal memory

### `allocate_inference_cache(batch_size) -> (angle_state, ssm_state, bx_prev)`

Allocate zero-initialized states for `step()`. Call once before decoding.

---

## `MambaLMHeadModel`

Stacked Mamba-3 language model.

### Constructor

```python
MambaLMHeadModel(
    config: MambaConfig,
)
```

### `forward(input_ids: Tensor) -> Tensor`

- **Input:** `input_ids` — `(batch, seq_len)` — integer token IDs
- **Output:** `(batch, seq_len, vocab_size)` — logits

---

## `MambaConfig`

```python
@dataclass
class MambaConfig:
    d_model: int = 2560
    d_intermediate: int = 0       # >0 adds SwiGLU MLP after each block
    n_layer: int = 64
    vocab_size: int = 50277
    ssm_cfg: dict = {}            # Passed to Mamba3()
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
```

---

## `SSMConfig`

```python
@dataclass
class SSMConfig:
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    is_mimo: bool = False
    mimo_rank: int = 4
```

---

## Low-level operations

```python
from mamba3_ssm import RMSNorm, apply_rope, ssm_scan_siso, ssm_scan_mimo
```

### `RMSNorm(d: int, eps: float = 1e-5)`

Root Mean Square Layer Normalization.

### `apply_rope(x: Tensor, angles: Tensor) -> Tensor`

Apply RoPE rotation to pairs of dimensions.

- `x`: `(..., 2 * num_angles)`
- `angles`: `(..., num_angles)`

### `ssm_scan_siso(x, B_proj, C_proj, ADT, DT, trap, D) -> y`

Sequential SSM scan (SISO). Returns `(B, L, H, P)`.

### `ssm_scan_mimo(x, B_proj, C_proj, ADT, DT, trap, D, mimo_x, mimo_o) -> y`

Sequential SSM scan (MIMO). Returns `(B, L, H, P)`.
