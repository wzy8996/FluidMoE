"""
FluidMoE Complete Custom Layer Implementation for Megatron-LM

This module provides complete custom layer implementation (not just Layer Spec):
- Replaces entire SelfAttention and MoELayer modules
- Full control over forward and backward computation
- No global function patching required

NOTE: This is NOT traditional "Layer Spec" (which only replaces submodules).
We replace the entire attention and MoE layers for full control.

Design:
- FluidSelfAttention: Complete custom attention layer with Ulysses SP
- FluidMoELayer: Complete custom MoE layer with Expert Parallel (EP)
- Token dispatch/combine uses fluid_all_to_all_moe_dispatch/combine functions
- dW computation overlaps with AllToAll communication (backward)

Usage:
    from fluid.megatron_layers import get_fluid_custom_layers

    config = TransformerConfig(
        context_parallel_size=4,      # SP for attention
        expert_model_parallel_size=2, # EP for MoE
    )
    layer_spec = get_fluid_custom_layers()
    model = GPTModel(config=config, transformer_layer_spec=layer_spec)
"""

from typing import Optional
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer.identity_op import IdentityOp

from fluid.attention_layers import FluidColumnParallelLinear, FluidRowParallelLinear
from fluid.moe_layers import FluidGroupedMLP
from fluid.attention_module import FluidSelfAttention
from fluid.moe_module import FluidMoELayer
from fluid.scheduler import get_backward_scheduler


def get_fluid_custom_layers() -> ModuleSpec:
    """
    Get Fluid complete custom layer implementation for Megatron.

    NOTE: This is NOT traditional "Layer Spec" - we replace entire layers!
    - Layer Spec: Replace only submodules (linear_qkv, linear_proj, etc.)
    - This function: Replace entire SelfAttention and MoELayer

    This returns complete custom layers, allowing:
    1. Full control over forward computation logic
    2. Computation-communication overlap in both forward and backward
    3. Custom attention mechanisms (Ring Attention, etc.)
    4. Custom MoE routing and expert execution
    5. No global function patching required

    Architecture:
    - FluidSelfAttention: Custom attention layer
      - Calls fluid_all_to_all_sp2hp/hp2sp directly (no patch)
      - Uses FluidColumnParallelLinear/FluidRowParallelLinear for dW scheduling
      - Supports forward optimization (future)

    - FluidMoELayer: Custom MoE layer
      - Uses FluidTokenDispatcher for token routing
      - Calls fluid_all_to_all directly (no patch)
      - Uses FluidGroupedMLP for expert computation with dW scheduling
      - Supports forward optimization (future)

    Returns:
        ModuleSpec: Megatron layer specification using fully custom Fluid modules

    Maintenance:
    - Review Megatron updates for SelfAttention and MoELayer API changes
    - Update FluidSelfAttention/FluidMoELayer to match Megatron API
    - Test compatibility after Megatron updates
    - See MAINTENANCE_GUIDE.md for details

    Example:
        >>> from megatron.core import GPTModel
        >>> from megatron.core.transformer import TransformerConfig
        >>> from fluid.megatron_layers import get_fluid_custom_layers
        >>>
        >>> config = TransformerConfig(
        ...     num_layers=32,
        ...     hidden_size=4096,
        ...     num_moe_experts=8,
        ...     moe_router_topk=2,
        ...     context_parallel_size=4,  # SP
        ...     expert_model_parallel_size=2,  # EP
        ... )
        >>> custom_layers = get_fluid_custom_layers()
        >>> model = GPTModel(config=config, transformer_layer_spec=custom_layers)
    """
    # 1. Enable Fluid scheduler
    scheduler = get_backward_scheduler()
    scheduler.enable()
    print("[FluidMoE] Scheduler enabled")

    # 2. Determine layer norm implementation
    try:
        from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
        layer_norm_impl = FusedLayerNorm
    except ImportError:
        from megatron.core.transformer.torch_norm import WrappedTorchNorm
        layer_norm_impl = WrappedTorchNorm

    # 3. Build layer spec with FULLY CUSTOM components
    # Use TransformerLayerSubmodules (dataclass) instead of dict
    spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            # Layer normalization before attention
            input_layernorm=layer_norm_impl,

            # 完全自定义的注意力层 (内部自己创建所有子模块)
            # Use causal attention mask for GPT models
            self_attention=ModuleSpec(
                module=FluidSelfAttention,
                params={"attn_mask_type": AttnMaskType.causal}
            ),

            # Bias-dropout-add after attention
            self_attn_bda=get_bias_dropout_add,

            # Layer normalization before MoE (required for MoE layers)
            pre_mlp_layernorm=layer_norm_impl,

            # 完全自定义的 MoE 层 (内部自己创建所有子模块)
            mlp=ModuleSpec(module=FluidMoELayer),

            # Bias-dropout-add after MoE
            mlp_bda=get_bias_dropout_add,
        )
    )

    print("[FluidMoE] ✅ Custom layers created")
    print("[FluidMoE]    - FluidSelfAttention: Complete custom attention layer")
    print("[FluidMoE]    - FluidMoELayer: Complete custom MoE layer")
    print("[FluidMoE]    - No global patching required")
    return spec


# Backward compatibility alias
def get_fluid_moe_layer_spec() -> ModuleSpec:
    """
    Deprecated: Use get_fluid_custom_layers() instead.

    This is kept for backward compatibility.
    Note: The name "layer_spec" is misleading - we actually replace entire layers,
    not just submodules.
    """
    import warnings
    warnings.warn(
        "get_fluid_moe_layer_spec() is deprecated. Use get_fluid_custom_layers() instead. "
        "Note: This function replaces entire layers, not just submodules.",
        DeprecationWarning,
        stacklevel=2
    )
    return get_fluid_custom_layers()


# ============================================================
# Utility functions
# ============================================================

def is_fluid_enabled():
    """Check if Fluid optimization is enabled"""
    scheduler = get_backward_scheduler()
    return scheduler.is_enabled()


def print_fluid_layer_info(model):
    """
    Print information about Fluid layers in the model.

    Args:
        model: GPTModel or similar
    """
    print("\n" + "="*60)
    print("FluidMoE Layer Information")
    print("="*60)

    fluid_attention_layers = []
    fluid_moe_layers = []
    fluid_linear_layers = []

    for name, module in model.named_modules():
        if isinstance(module, FluidSelfAttention):
            fluid_attention_layers.append(name)
        elif isinstance(module, FluidMoELayer):
            fluid_moe_layers.append(name)
        elif isinstance(module, (FluidColumnParallelLinear, FluidRowParallelLinear)):
            fluid_linear_layers.append((name, type(module).__name__))
        elif isinstance(module, FluidGroupedMLP):
            pass  # Already counted in FluidMoELayer

    print(f"Attention layers: {len(fluid_attention_layers)} FluidSelfAttention")
    for name in fluid_attention_layers:
        print(f"  - {name}")

    print(f"\nMoE layers: {len(fluid_moe_layers)} FluidMoELayer")
    for name in fluid_moe_layers:
        print(f"  - {name}")

    print(f"\nLinear layers: {len(fluid_linear_layers)} Fluid linear layers")
    for name, layer_type in fluid_linear_layers[:5]:  # Show first 5
        print(f"  - {name}: {layer_type}")
    if len(fluid_linear_layers) > 5:
        print(f"  ... and {len(fluid_linear_layers) - 5} more")

    if not (fluid_attention_layers or fluid_moe_layers or fluid_linear_layers):
        print("⚠️  No Fluid layers found in model")

    scheduler = get_backward_scheduler()
    print(f"\nScheduler status: {'✅ Enabled' if scheduler.is_enabled() else '❌ Disabled'}")
    print("="*60 + "\n")
