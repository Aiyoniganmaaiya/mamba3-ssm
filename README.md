# Mamba-3: Improved Sequence Modeling using State Space Principles

A clean, readable, from-scratch PyTorch implementation of **Mamba-3** — a selective state space model that addresses three core limitations of Mamba-2. No Triton/CUDA kernels; designed for understanding and reproducing the algorithm.

**Paper:** [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)
**Authors:** Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu

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
| Output | `C @ h → P` | R scalars up-projected |

## Project Structure

```
mamba3-reproduce/
├── README.md
├── requirements.txt
├── setup.py
├── mamba3_ssm/
│   ├── __init__.py
│   ├── config.py        # Model configuration dataclass
│   ├── ops.py           # Core SSM operations (scan, RoPE, norms)
│   ├── layer.py         # Mamba3 layer (SISO + MIMO)
│   ├── block.py         # Residual block + optional MLP
│   ├── model.py         # Full MambaLMHeadModel
│   └── utils.py         # Parameter counting, initialization
├── tests/
│   ├── __init__.py
│   ├── test_ops.py       # Unit tests for SSM operations
│   ├── test_shapes.py    # Shape consistency tests
│   └── test_numerical.py # Numerical checks (trapezoidal, RoPE)
└── benchmarks/
    └── benchmark_scan.py  # Speed comparison SISO vs MIMO
```

## Quick Start

```python
import torch
from mamba3_ssm.layer import Mamba3
from mamba3_ssm.model import MambaLMHeadModel, MambaConfig

# SISO mode
model = Mamba3(d_model=256, d_state=64, expand=2, headdim=32)
x = torch.randn(2, 128, 256)
y = model(x)  # (2, 128, 256)

# MIMO mode
model_mimo = Mamba3(d_model=256, d_state=64, expand=2, headdim=32,
                     is_mimo=True, mimo_rank=4)
y = model_mimo(x)  # same I/O shape

# Autoregressive decode
angle_state, ssm_state, bx_prev = model.allocate_inference_cache(batch_size=2)
u = torch.randn(2, 256)
out, angle_state, ssm_state, bx_prev = model.step(
    u, angle_state, ssm_state, bx_prev
)

# Full language model
cfg = MambaConfig(d_model=2048, n_layer=24, vocab_size=50277,
                  ssm_cfg={"is_mimo": True, "mimo_rank": 4})
lm = MambaLMHeadModel(cfg)
logits = lm(torch.randint(0, 50277, (1, 512)))  # (1, 512, vocab_size)
```

## Dependencies

```
torch>=2.0
einops
```

## Notation

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| L | Sequence length |
| H | Number of SSM heads (`d_inner / headdim`) |
| P | Headdim — per-head feature dimension |
| D | `d_state` — SSM state size per head |
| R | `mimo_rank` — number of MIMO streams |
| G | `ngroups` — B/C projection sharing groups |

## References

- Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
- Official implementation: [state-spaces/mamba](https://github.com/state-spaces/mamba)
