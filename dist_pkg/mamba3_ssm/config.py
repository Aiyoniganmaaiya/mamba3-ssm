"""
Configuration dataclasses for Mamba-3 models.

Paper reference: arXiv.2603.15569
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class SSMConfig:
    """Configuration for the SSM core inside Mamba-3."""

    # State dimensions
    d_state: int = 128        # D: SSM state size per head
    expand: int = 2            # Inner dimension multiplier; d_inner = expand * d_model
    headdim: int = 64          # P: per-head feature dimension; nheads = d_inner / headdim
    ngroups: int = 1           # G: number of groups for B/C projection sharing

    # RoPE
    rope_fraction: float = 0.5  # Fraction of state dims that rotate (0.5 or 1.0)

    # Time step bounds
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4

    # MIMO
    is_mimo: bool = False
    mimo_rank: int = 4         # R: number of MIMO streams

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items()
        }


@dataclass
class MambaConfig:
    """Full model configuration for Mamba-3 language models."""

    d_model: int = 2560
    d_intermediate: int = 0    # >0 adds an MLP sub-layer after each SSM block
    n_layer: int = 64
    vocab_size: int = 50277

    # SSM config (passed to each Mamba3 layer)
    ssm_cfg: Dict[str, Any] = field(default_factory=dict)

    # Norm / residual options
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True

    # Layer configuration
    attn_layer_idx: List[int] = field(default_factory=list)
    attn_cfg: Dict[str, Any] = field(default_factory=dict)
