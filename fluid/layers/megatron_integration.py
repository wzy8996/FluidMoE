"""
FluidMoE Megatron-LM Integration (Stub)

NOTE: This is a stub module. The full Megatron integration requires additional
modules that have been removed. To use Megatron integration, restore:
- fluid.attention_layers
- fluid.moe_layers
- fluid.attention_module
- fluid.moe_module
"""

from fluid.core.scheduler import get_backward_scheduler


def get_fluid_custom_layers():
    """
    Get Fluid custom layer implementation for Megatron.

    NOTE: This is currently a stub. Full implementation requires Megatron-specific
    integration modules.
    """
    raise NotImplementedError(
        "Megatron integration requires additional modules. "
        "Please restore fluid.attention_layers, fluid.moe_layers, "
        "fluid.attention_module, and fluid.moe_module."
    )


def get_fluid_moe_layer_spec():
    """Deprecated alias for get_fluid_custom_layers()"""
    return get_fluid_custom_layers()


def is_fluid_enabled():
    """Check if Fluid optimization is enabled"""
    scheduler = get_backward_scheduler()
    return scheduler.is_enabled()


def print_fluid_layer_info(model):
    """Print information about Fluid layers in the model"""
    print("Megatron integration not available")


__all__ = [
    'get_fluid_custom_layers',
    'get_fluid_moe_layer_spec',
    'is_fluid_enabled',
    'print_fluid_layer_info',
]
