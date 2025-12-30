"""
FlashOverlap Tuning Module

Provides offline tuning and runtime configuration loading for
FC2+AllToAll overlap optimization.
"""

from .flash_overlap_tuner import (
    load_tuning_config,
    get_config_key,
    MODEL_CONFIGS,
)

__all__ = [
    'load_tuning_config',
    'get_config_key',
    'MODEL_CONFIGS',
]
