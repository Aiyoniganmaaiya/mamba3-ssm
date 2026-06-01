# Mamba-3: Improved Sequence Modeling using State Space Principles

[![PyPI version](https://img.shields.io/pypi/v/mamba3-ssm.svg)](https://pypi.org/project/mamba3-ssm/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/mamba3-ssm.svg)](https://pypi.org/project/mamba3-ssm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, readable, from-scratch PyTorch implementation of **Mamba-3** [arXiv:2603.15569](https://arxiv.org/abs/2603.15569). No Triton/CUDA kernels. Train a 380M parameter model on a laptop GPU.

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

### Presets (benchmarked on RTX 4060 Laptop 8GB)

Based on actual VRAM measurements (bf16 + AdamW):

| Preset | Params | d_model | n_layer | d_state | batch | seq_len | VRAM | Status |
|--------|--------|---------|---------|---------|-------|---------|------|--------|
| `small` | 112M | 1024 | 16 | 64 | 2 | 512 | ~5.6GB | ✅ 舒适 |
| `medium` | 306M | 1536 | 20 | 64 | 1 | 256 | ~7.2GB | ✅ 推荐 |
| `large` | 367M | 1536 | 24 | 64 | 1 | 256 | ~8.6GB | ⚠️ 极限 |

Effective batch size = batch × grad_accum (default grad_accum=16 for all presets).

```bash
# Train 306M model on TinyStories (auto-downloads)
python train.py --dataset tinystories --preset medium --epochs 3

# Quick experiment with 112M on custom text
python train.py --dataset custom --data-path myfile.txt --preset small --epochs 5

# Wikitext-103 benchmark
python train.py --dataset wikitext --preset medium --epochs 5

# Resume training
python train.py --dataset tinystories --preset medium --resume checkpoints/best.pt

# Full custom config
python train.py --dataset tinystories --d-model 1024 --n-layer 16 --d-state 64 \
    --batch-size 2 --seq-len 512 --grad-accum 8 --learning-rate 3e-4 --epochs 3

# With W&B logging
python train.py --dataset tinystories --preset medium --wandb --wandb-project my-mamba3
```

### Text Generation

```bash
python generate.py --checkpoint checkpoints/best.pt \
    --prompt "Once upon a time" --max-tokens 200 --temperature 0.8
```

### Custom Training Code

```python
import torch
from mamba3_ssm import MambaLMHeadModel, MambaConfig, CONFIGS

# Use a preset or define your own config
cfg = CONFIGS["medium"]  # dict with d_model, n_layer, etc.
model = MambaLMHeadModel(MambaConfig(
    d_model=cfg["d_model"],
    n_layer=cfg["n_layer"],
    vocab_size=10000,
    ssm_cfg={"d_state": cfg["d_state"], "expand": 2, "headdim": 64,
             "is_mimo": True, "mimo_rank": 4},
)).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
```

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
    CONFIGS,  # RTX 4060 benchmarked presets
)
```

## Testing

```bash
python -m mamba3_ssm.tests
```

10/10 checks: shapes, numerical consistency (step-by-step == forward), gradient flow, parameter counting, edge cases.

## Project Structure

```
mamba3_ssm/          # pip installable package
├── __init__.py      # Public API
├── config.py        # MambaConfig / SSMConfig
├── ops.py           # RMSNorm, RoPE, SSM scans
├── layer.py         # Mamba3 module (forward + step)
├── block.py         # MambaBlock, MambaLMHeadModel
├── presets.py       # RTX 4060 benchmarked configs
├── tests.py         # 10 sanity checks
└── utils.py         # Parameter counting

train.py             # Training script
generate.py          # Text generation
docs/
├── API.md           # Full API reference
└── TRAINING.md      # Training guide with tips
```

## RTX 4060 Laptop Tips

- **bf16** is enabled automatically on RTX 40-series — no config needed
- **MIMO** gives ~20% speedup in decode over SISO
- **d_state=64** is the sweet spot for 8GB; go to 128 only if you reduce d_model
- **grad_accum** lets you simulate large batches without extra VRAM
- If OOM: reduce `seq_len` first (512→256→128), then `d_model`

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 4 GB | 8 GB |
| RAM | 8 GB | 16 GB |
| Disk | 1 GB | 5 GB (with datasets) |

Tested on RTX 4060 Laptop (8GB), PyTorch 2.6+cu124.

## Dependencies

```
torch>=2.0
einops>=0.7
```

Optional: `datasets` for auto-downloading TinyStories/Wikitext, `wandb` for logging.

## Changelog

### v0.1.1 (2026-05-31)
- Add `--preset` flag to train.py (small/medium/large) benchmarked on RTX 4060 8GB
- Fix default config to fit 8GB VRAM (d_state=64, bs=1, seq_len=256)
- Add `generate.py` for text generation from checkpoints
- Add `mamba3_ssm.presets.CONFIGS` with VRAM-benchmarked configurations
- Update README with training guide and benchmark table

### v0.1.0 (2026-05-31)
- Initial release — SISO & MIMO Mamba-3, 10/10 tests passing

## License

MIT

## References

- Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
- Official implementation: [state-spaces/mamba](https://github.com/state-spaces/mamba)
