"""
Layers Module - Complete Transformer layer implementations

Contains:
- transformer: TransformerLayer combining Attention + MoE
- megatron_integration: Megatron-LM integration utilities
"""

from .transformer import TransformerLayer
from .megatron_integration import (
    get_fluid_custom_layers,
    get_fluid_moe_layer_spec,
    is_fluid_enabled,
    print_fluid_layer_info,
)

__all__ = [
    'TransformerLayer',
    # Megatron integration
    'get_fluid_custom_layers',
    'get_fluid_moe_layer_spec',
    'is_fluid_enabled',
    'print_fluid_layer_info',
]
