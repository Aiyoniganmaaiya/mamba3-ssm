# Mamba-3: Improved Sequence Modeling using State Space Principles

[![PyPI version](https://img.shields.io/pypi/v/mamba3-ssm.svg)](https://pypi.org/project/mamba3-ssm/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/mamba3-ssm.svg)](https://pypi.org/project/mamba3-ssm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-10%2F%20passing-brightgreen.svg)]()

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
| Output | `C @ h → P` | R scalars up-projected |

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

## API Reference

### `Mamba3`

The core sequence-mixing module. Drop-in replacement for a Transformer attention layer.

```python
Mamba3(
    d_model: int,          # Token embedding dimension
    d_state: int = 128,    # SSM state size per head (D)
    expand: int = 2,       # Inner dim multiplier; d_inner = expand * d_model
    headdim: int = 64,     # Features per SSM head (P)
    ngroups: int = 1,      # Groups for B/C projection sharing (G)
    rope_fraction: float = 0.5,  # Fraction of state dims that rotate
    dt_min: float = 0.001,       # Minimum time step
    dt_max: float = 0.1,         # Maximum time step
    is_mimo: bool = False,       # Enable MIMO formulation
    mimo_rank: int = 4,          # Number of MIMO streams (R)
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `forward(u)` | Full-sequence forward pass. Input: `(B, L, d_model)` → Output: `(B, L, d_model)` |
| `step(u, angle_state, ssm_state, bx_prev)` | Single autoregressive decode step. Input: `(B, d_model)` → Output: `(B, d_model)` + updated states |
| `allocate_inference_cache(batch_size)` | Allocate zero-initialized states for decoding |

### `MambaLMHeadModel`

Full stacked language model.

```python
MambaLMHeadModel(
    config: MambaConfig,   # Model configuration
)
```

**`MambaConfig` fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `d_model` | 2560 | Hidden size |
| `n_layer` | 64 | Number of MambaBlocks |
| `vocab_size` | 50277 | Vocabulary size (padded to multiple of 8) |
| `ssm_cfg` | `{}` | Kwargs passed to `Mamba3` |
| `d_intermediate` | 0 | If >0, adds SwiGLU MLP after each block |
| `tie_embeddings` | True | Tie LM head weight to embedding weight |

### Low-level operations

```python
from mamba3_ssm import RMSNorm, apply_rope, ssm_scan_siso, ssm_scan_mimo
```

| Function | Description |
|----------|-------------|
| `RMSNorm(d, eps)` | Root Mean Square Layer Normalization |
| `apply_rope(x, angles)` | Rotate pairs of dimensions (RoPE) |
| `ssm_scan_siso(x, B, C, ADT, DT, trap, D)` | Sequential SSM scan (SISO mode) |
| `ssm_scan_mimo(x, B, C, ADT, DT, trap, D, mimo_x, mimo_o)` | Sequential SSM scan (MIMO mode) |

## Training Example

```python
import torch
import torch.nn.functional as F
from mamba3_ssm import MambaLMHeadModel, MambaConfig

# ── Small model for demo ──────────────────────────────
cfg = MambaConfig(
    d_model=256,
    n_layer=4,
    vocab_size=10000,
    ssm_cfg={"d_state": 64, "expand": 2, "headdim": 32, "is_mimo": True, "mimo_rank": 2},
)
model = MambaLMHeadModel(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Fake training loop ────────────────────────────────
for step in range(100):
    # Random data (replace with real data)
    input_ids = torch.randint(0, cfg.vocab_size, (4, 128))
    labels = torch.randint(0, cfg.vocab_size, (4, 128))

    logits = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
```

## Autoregressive Text Generation Example

```python
model.eval()
input_ids = torch.randint(0, cfg.vocab_size, (1, 10))  # seed tokens

# Allocate cache
states = [layer.mixer.allocate_inference_cache(1) for layer in model.layers]

# Generate 50 tokens
with torch.no_grad():
    # Process seed tokens
    x = model.embedding(input_ids)
    for i, block in enumerate(model.layers):
        x = block(x)  # forward handles the full sequence

    # Autoregressive decode
    for _ in range(50):
        x = model.embedding(logits.argmax(-1)[:, -1:])
        for i, block in enumerate(model.layers):
            x = x + block.mixer(block.norm(x))
            # In practice you'd use .step() with cached states
        logits = model.norm_f(x)
        next_token = logits.argmax(-1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

print("Generated sequence:", input_ids)
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
| `rope_fraction` | 0.5 | Fraction of state dims that rotate (0.5 or 1.0) |

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

## Training

Train a 380M parameter model on your RTX 4060 Laptop:

```bash
# Quick test with custom text
python train.py --dataset custom --data-path myfile.txt --epochs 1

# Train on TinyStories (auto-downloads ~30MB)
python train.py --dataset tinystories --epochs 3

# Train on Wikitext-103 (English LM benchmark)
python train.py --dataset wikitext --epochs 3

# Resume from checkpoint
python train.py --dataset tinystories --resume checkpoints/best.pt
```

Generate text from a trained model:

```bash
python generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time" --max-tokens 200
```

## Testing

```bash
python -m mamba3_ssm.tests
```

10/10 sanity checks:
- SISO/MIMO forward and step shape consistency
- Step-by-step decode matches full forward (numerical equality)
- Full language model integration
- Parameter counting
- Gradient flow
- Edge cases (rope_fraction=1.0)

## Dependencies

- `torch>=2.0`
- `einops>=0.7`

## Changelog

### v0.1.0 (2026-05-31)

- Initial release
- SISO and MIMO Mamba-3 implementations
- Exponential-trapezoidal discretization
- RoPE complex-valued state space
- Full MambaLMHeadModel
- 10/10 tests passing

## License

MIT

## References

- Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
- Official implementation: [state-spaces/mamba](https://github.com/state-spaces/mamba)
