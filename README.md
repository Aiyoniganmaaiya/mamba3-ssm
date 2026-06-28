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

## Training

```bash
# Train on TinyStories with preset config
python train.py --dataset tinystories --preset small --epochs 3

# Custom data
python train.py --dataset custom --data-path myfile.txt --preset medium --epochs 1

# Resume from checkpoint
python train.py --dataset tinystories --resume checkpoints/best.pt

# Generate text from trained model
python generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time"
```

Available presets: `small` (112M, seq=512), `medium` (306M, seq=256), `large` (367M, seq=256). See `mamba3_ssm/presets.py` for details.

## Performance

### Acceleration Tiers

The SSM scan — the core bottleneck — uses a tiered acceleration strategy:

| Tier | Speedup | Availability |
|------|---------|-------------|
| **CUDA kernel** | ~50× vs Python | Requires MSVC + CUDA 12.1+ |
| **JIT (torch.jit.script)** | ~2–3× vs Python | All platforms, no compilation |
| **Pure Python** | 1× | Always works |

### Training Estimates (CUDA + JIT, RTX 4060 8GB Laptop)

SSM scan accelerated with CUDA SISO kernel (50× speedup) and JIT MIMO fallback:

| Preset | Params | Seq | VRAM | Micro-batch | Tok/s | Steps/ep | TinyStories×3ep |
|--------|--------|-----|------|-------------|-------|----------|-----------------|
| small | 112M | 512 | ~5.6GB | 117 ms | 8,780 | 5,798 | **~4 h 30 m** |
| medium | 306M | 256 | ~7.2GB | 125 ms | 2,040 | 11,596 | **~19 h 24 m** |
| large | 367M | 256 | ~8.6GB | 144 ms | 1,777 | 11,596 | **~22 h 16 m** |

**Note:** All three presets fit on an 8GB laptop GPU. Training `small` on TinyStories for 3 epochs completes in under 5 hours. Medium/large use JIT MIMO fallback — a fixed CUDA MIMO kernel would further improve throughput.

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
