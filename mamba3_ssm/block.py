"""
Building blocks for stacking Mamba-3 layers into a full model:
  - MambaBlock:    RMSNorm → Mamba3 → residual add
  - MLPBlock:      SwiGLU feed-forward (optional)
  - Embed / LMHead: Token embedding and language-model head
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field

from .ops import RMSNorm
from .layer import Mamba3
from .config import MambaConfig


class MambaBlock(nn.Module):
    """Single Mamba-3 residual block: Norm → Mixer → residual."""

    def __init__(self, d_model: int, ssm_cfg: dict, device=None, dtype=None):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba3(d_model=d_model, **ssm_cfg, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class MLPBlock(nn.Module):
    """SwiGLU-style feed-forward layer. Used when d_intermediate > 0."""

    def __init__(self, d_model: int, d_intermediate: int, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=False, **factory)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=False, **factory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * val)


class MambaLMHeadModel(nn.Module):
    """Stacked Mamba-3 language model.

    Architecture:
        Embedding → n_layer × (MambaBlock [+ MLPBlock]) → RMSNorm → LMHead

    The LM head weight can be tied to the embedding weight.
    Vocab size is padded up to the nearest multiple of pad_vocab_size_multiple.
    """

    def __init__(self, config: MambaConfig, device=None, dtype=None):
        factory = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config

        # Pad vocab size
        vocab_size = config.vocab_size
        r = vocab_size % config.pad_vocab_size_multiple
        if r != 0:
            vocab_size += config.pad_vocab_size_multiple - r
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, config.d_model, **factory)

        self.layers = nn.ModuleList([
            MambaBlock(config.d_model, config.ssm_cfg, **factory)
            for _ in range(config.n_layer)
        ])

        if config.d_intermediate > 0:
            self.mlp_norms = nn.ModuleList(
                [RMSNorm(config.d_model) for _ in range(config.n_layer)]
            )
            self.mlp_layers = nn.ModuleList([
                MLPBlock(config.d_model, config.d_intermediate, **factory)
                for _ in range(config.n_layer)
            ])
        else:
            self.mlp_norms = None
            self.mlp_layers = None

        self.norm_f = RMSNorm(config.d_model)
        # LM head: projects from d_model to vocab_size (tied with embedding)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False, **factory)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Args: input_ids (batch, seq_len) → logits (batch, seq_len, vocab_size)"""
        x = self.embedding(input_ids)
        for i, block in enumerate(self.layers):
            x = block(x)
            if self.mlp_layers is not None:
                x = x + self.mlp_layers[i](self.mlp_norms[i](x))
        x = self.norm_f(x)
        return self.lm_head(x)
