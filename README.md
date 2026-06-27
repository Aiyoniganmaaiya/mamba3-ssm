# Mamba-3: Improved Sequence Modeling using State Space Principles

[![PyPI version](https://img.shields.io/pypi/v/mamba3-ssm.svg?color=blue)](https://pypi.org/project/mamba3-ssm/)
**pip install:** `pip install mamba3-ssm` · **version:** 0.2.0
[![Python 3.10+](https://img.shields.io/pypi/pyversions/mamba3-ssm.svg)](https://pypi.org/project/mamba3-ssm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, readable, from-scratch PyTorch implementation of **Mamba-3** [arXiv:2603.15569](https://arxiv.org/abs/2603.15569). Features **CUDA-accelerated SSM scans** (50× speedup) and **MSVC/CUDA 12.1 compatibility** on Windows.

## Installation

```bash
pip install mamba3-ssm
```

## Quick Start

```python
import torch
from mamba3_ssm import Mamba3, MambaLMHeadModel, MambaConfig

model = Mamba3(d_model=256, d_state=64, expand=2, headdim=32, is_mimo=True, mimo_rank=4)
x = torch.randn(2, 128, 256)
y = model(x)  # (2, 128, 256)

# Autoregressive decode
angle, state, prev = model.allocate_inference_cache(2)
out, angle, state, prev = model.step(torch.randn(2, 256), angle, state, prev)

# Full language model
cfg = MambaConfig(d_model=1536, n_layer=20, vocab_size=50000,
                  ssm_cfg={"d_state": 64, "is_mimo": True, "mimo_rank": 4})
lm = MambaLMHeadModel(cfg)
logits = lm(torch.randint(0, 50000, (1, 512)))
```

## Performance

### Acceleration Tiers

The SSM scan — the core bottleneck — uses a tiered acceleration strategy:

| Tier | Speedup | Availability |
|------|---------|-------------|
| **CUDA kernel** | ~50× vs Python | Requires MSVC + CUDA 12.1+ |
| **JIT (torch.jit.script)** | ~2–3× vs Python | All platforms, no compilation |
| **Pure Python** | 1× | Always works |

### Training Estimates (CUDA + JIT, RTX 4060 8GB Laptop)

Using the CUDA-accelerated SISO scan (MIMO falls back to JIT):

| Preset | Params | Time/micro-batch | TinyStories×3ep |
|--------|--------|-----------------|-----------------|
| small (SISO) | 112M | ~27 s | ~1035 h |
| medium (MIMO) | 306M | ~11 s | ~1624 h |
| large (MIMO) | 367M | ~11 s | ~1770 h |

**Note:** The Python-for-loop SSM scan remains the dominant bottleneck even with acceleration. Full TinyStories training requires multi-GPU or TPU. These presets are suitable for small-scale experiments, ablation studies, and inference.

### VRAM at bf16 (batch=2, seq_len=512 with grad_accum)

| Preset | Params | d_model | n_layer | VRAM |
|--------|--------|---------|---------|------|
| small | 112M | 1024 | 16 | ~5.6 GB |
| medium | 306M | 1536 | 20 | ~7.2 GB |
| large | 367M | 1536 | 24 | ~8.6 GB |

## Core Ideas

### 1. Exponential-Trapezoidal Discretization

Mamba-2 used Zero-Order Hold (first-order). Mamba-3 uses the **trapezoidal rule**:

```
h_t = exp(A·dt_t) · h_{t-1} + dt_t · σ(trap_t) · (B_t·x_t + B_{t-1}·x_{t-1}) / 2
```

Learned `trap` gate blends between Euler (trap≈0) and full trapezoidal (trap≈1).

### 2. Complex-Valued (Rotary) State Space

Applies **RoPE** to B and C projections, giving the state an effective complex-valued structure for tracking phase-dependent dependencies.

### 3. MIMO Formulation

Reuses a shared `(H, D)` state for `R` rank streams instead of SISO's `(H, P, D)` outer product:

| | SISO | MIMO |
|---|---|---|
| State shape | `(H, P, D)` | `(H, D)` |
| Decode FLOPs/byte | Low (memory-bound) | R× higher |

## CUDA Acceleration

The SSM scan is accelerated with a fused CUDA kernel when the MSVC compiler is available:

- **SISO**: Fully fused kernel — one block per (batch, head), P threads hold all D state-values in registers, B/C loaded via shared memory each timestep. Replaces the Python for-loop entirely.
- **MIMO**: Split design — outer einsums in PyTorch, inner state scan in CUDA. Uses tree-reduction over D for the output.
- **JIT fallback**: If MSVC is unavailable, `torch.jit.script` provides a ~2–3× speedup with no compilation needed.

To compile the CUDA kernel, install Visual Studio Build Tools with MSVC and run any scan function (compilation happens automatically on first call).

## API Reference

### `Mamba3(d_model, d_state=128, expand=2, headdim=64, ngroups=1, rope_fraction=0.5, is_mimo=False, mimo_rank=4)`

| Method | Description |
|--------|-------------|
| `forward(u)` | `(B, L, d_model)` → `(B, L, d_model)` |
| `step(u, angle, state, prev)` | Single decode step, returns updated states |
| `allocate_inference_cache(B)` | Allocate zero states for decoding |

### `MambaLMHeadModel(config)`

| Field | Default | Description |
|-------|---------|-------------|
| `d_model` | 2560 | Hidden size |
| `n_layer` | 64 | Number of blocks |
| `vocab_size` | 50277 | Padded to multiple of 8 |
| `ssm_cfg` | `{}` | Passed to Mamba3 |
| `d_intermediate` | 0 | SwiGLU MLP (0 = disabled) |
| `tie_embeddings` | True | Tie LM head to embedding |

### Exports

```python
from mamba3_ssm import (
    Mamba3, MambaLMHeadModel, MambaConfig, SSMConfig,
    RMSNorm, apply_rope, ssm_scan_siso, ssm_scan_mimo,
    CONFIGS,
)
```

## Testing

```bash
python -m mamba3_ssm.tests
```

10/10 checks: shapes, numerical consistency (step-by-step == forward), gradient flow, parameter counting, edge cases.

## Project Structure

```
mamba3_ssm/
├── __init__.py      # Public API
├── config.py        # MambaConfig / SSMConfig
├── ops.py           # RMSNorm, RoPE, SSM scans (CUDA/JIT/Python)
├── cuda_backend.py  # CUDA kernel compilation + Python wrappers
├── layer.py         # Mamba3 module (forward + step)
├── block.py         # MambaBlock, MambaLMHeadModel
├── presets.py       # RTX 4060 benchmarked configs
├── tests.py         # 10 sanity checks
└── utils.py         # Parameter counting
```

## Dependencies

```
torch>=2.0
einops>=0.7
```

Optional: `datasets` for auto-downloading TinyStories/Wikitext, `wandb` for logging.

## Changelog

### v0.2.0 (2026-06-28)
- **CUDA-accelerated SSM scan**: Fused SISO kernel (50× speedup); MIMO split kernel
- **JIT fallback**: `torch.jit.script` — 1.9–2.7× speedup without CUDA compilation
- **Bug fix: double sigmoid**: Removed redundant sigmoid on trap gate (forward path)
- **Bug fix: lm_head dimension**: Swapped to `Linear(d_model, vocab_size)`
- **MSVC 14.44 + CUDA 12.1 compatibility**: Added `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` workaround

### v0.1.2 (2026-05-31)
- Fix tokenizer cache loading bug, checkpoint resume, steps_per_epoch calculation
- Optimized SSM scan with pre-computed decay/trap factors

### v0.1.1 (2026-05-31)
- Add `--preset` flag, generate.py, presets.CONFIGS, RTX 4060 benchmarks

### v0.1.0 (2026-05-31)
- Initial release — SISO & MIMO Mamba-3, 10/10 tests passing

## License

MIT

## References

- Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
- Official implementation: [state-spaces/mamba](https://github.com/state-spaces/mamba)
