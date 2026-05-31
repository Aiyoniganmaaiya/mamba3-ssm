# Mamba-3: Improved Sequence Modeling using State Space Principles

A clean, readable, from-scratch PyTorch implementation of **Mamba-3** — a selective state space model that addresses three core limitations of Mamba-2. No Triton/CUDA kernels; designed for understanding and reproducing the algorithm.

**Paper:** [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)
**Authors:** Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu

## Installation

```bash
pip install mamba3-ssm
```

Or install from source:

```bash
pip install git+https://github.com/Aiyoniganmaaiya/mamba3-ssm.git
```

## Quick Start

```python
import torch
from mamba3_ssm import Mamba3, MambaLMHeadModel, MambaConfig

# ── SISO mode (standard) ──────────────────────────────
model = Mamba3(
    d_model=256,
    d_state=64,
    expand=2,
    headdim=32,
    is_mimo=False,
)
x = torch.randn(2, 128, 256)
y = model(x)  # (2, 128, 256)

# ── MIMO mode (better decode efficiency) ──────────────
model_mimo = Mamba3(
    d_model=256,
    d_state=64,
    expand=2,
    headdim=32,
    is_mimo=True,
    mimo_rank=4,
)
y = model_mimo(x)  # same I/O shape

# ── Autoregressive decode (one token at a time) ───────
angle_state, ssm_state, bx_prev = model.allocate_inference_cache(batch_size=2)
u = torch.randn(2, 256)
out, angle_state, ssm_state, bx_prev = model.step(
    u, angle_state, ssm_state, bx_prev
)

# ── Full language model ───────────────────────────────
cfg = MambaConfig(
    d_model=2048,
    n_layer=24,
    vocab_size=50277,
    ssm_cfg={"is_mimo": True, "mimo_rank": 4},
)
lm = MambaLMHeadModel(cfg)
logits = lm(torch.randint(0, 50277, (1, 512)))  # (1, 512, vocab_size)
```

## Core Ideas

### 1. Exponential-Trapezoidal Discretization

Mamba-2 used Zero-Order Hold (exponential-Euler), a first-order approximation. Mamba-3 adopts the **trapezoidal rule**, averaging the `B*x` contribution at times `t-1` and `t`:

```
h_t = exp(A·dt_t) · h_{t-1} + dt_t · σ(trap_t) · (B_t·x_t + B_{t-1}·x_{t-1}) / 2
```

`trap` is a learned sigmoid gate blending between Euler (`trap≈0`) and full trapezoidal (`trap≈1`).

### 2. Complex-Valued (Rotary) State Space

Real-valued SSM hidden states cannot easily represent oscillatory patterns. Mamba-3 applies **RoPE** to B and C projections, giving the state an effective complex-valued structure that tracks phase-dependent dependencies.

### 3. Multi-Input Multi-Output (MIMO) Formulation

Mamba-2 is SISO with state `(H, P, D)` — during decode the GPU is memory-bandwidth bound. **MIMO** reuses a shared `(H, D)` state for `R` rank streams, multiplying FLOPs/byte by `R`:

| | SISO | MIMO |
|---|---|---|
| State shape | `(H, P, D)` | `(H, D)` |
| Update | outer product `x ⊗ B` | sum of R rank-1 terms |

## Project Structure

```
mamba3_ssm/
├── __init__.py   # Public API
├── config.py     # MambaConfig / SSMConfig dataclasses
├── ops.py        # RMSNorm, RoPE, SSM scan (SISO + MIMO)
├── layer.py      # Mamba3 module (forward + step + inference cache)
├── block.py      # MambaBlock, MLPBlock, MambaLMHeadModel
├── tests.py      # 10 sanity checks
└── utils.py      # Parameter counting
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | — | Token embedding dimension |
| `d_state` | 128 | SSM state size per head (D) |
| `expand` | 2 | Inner dim multiplier; `d_inner = expand * d_model` |
| `headdim` | 64 | Features per SSM head (P) |
| `is_mimo` | False | Enable MIMO formulation |
| `mimo_rank` | 4 | Number of parallel MIMO streams (R) |
| `rope_fraction` | 0.5 | Fraction of state dims that rotate |

## Testing

```bash
python -m mamba3_ssm.tests
```

10/10 sanity checks pass, including shape tests, numerical consistency (step-by-step decode matches forward), gradient flow, and edge cases.

## Dependencies

- `torch>=2.0`
- `einops>=0.7`

## License

MIT

## References

- Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
- Official implementation: [state-spaces/mamba](https://github.com/state-spaces/mamba)
