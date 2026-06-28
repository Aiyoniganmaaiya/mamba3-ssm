"""
Mamba-3 v0.2.0: Improved Sequence Modeling using State Space Principles

CUDA-accelerated SSM scans with Python fallback.
Fixes double-sigmoid bug and lm_head dimension swap.
Paper: https://arxiv.org/abs/2603.15569
"""

from .config import MambaConfig, SSMConfig
from .layer import Mamba3
from .block import MambaLMHeadModel, MambaBlock
from .ops import RMSNorm, apply_rope, ssm_scan_siso, ssm_scan_mimo
from .presets import CONFIGS
from .tokenizer import CharTokenizer, BPETokenizer, load_tokenizer

__all__ = [
    "MambaConfig",
    "SSMConfig",
    "Mamba3",
    "MambaLMHeadModel",
    "MambaBlock",
    "RMSNorm",
    "apply_rope",
    "ssm_scan_siso",
    "ssm_scan_mimo",
    "CONFIGS",
    "CharTokenizer",
    "BPETokenizer",
    "load_tokenizer",
]
