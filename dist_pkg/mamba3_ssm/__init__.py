"""
Mambo-3: Improved Sequence Modeling using State Space Principles

Clean PyTorch implementation of the Mamba-3 SSM architecture.
Paper: https://arxiv.org/abs/2603.15569
"""

from .config import MambaConfig, SSMConfig
from .layer import Mamba3
from .block import MambaLMHeadModel, MambaBlock
from .ops import RMSNorm, apply_rope, ssm_scan_siso, ssm_scan_mimo
from .presets import CONFIGS

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
]
