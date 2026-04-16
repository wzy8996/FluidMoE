"""
FluidMoE: MoE/Attention with Communication-Computation Overlap

This package provides unified Transformer layers with optimized scheduling.

Key Scheduling Innovations:
1. Forward: P2P Round-Robin Tournament for dispatch/combine overlap
2. Backward: dW tasks deferred and executed during AllToAll communication
3. Unified autograd.Function: Single Function for complete Transformer layer

Modules:
- core: AllToAll primitives, P2P scheduling, BackwardScheduler
- moe: MoE forward/backward building blocks
- attention: Attention forward/backward building blocks
- layer: Unified Transformer layer with single autograd.Function
- distributed: Lightweight SP + EP parallel context

Quick Start:
    from fluid.distributed import init_parallel
    from fluid import TransformerLayer, TransformerModel

    ctx = init_parallel(ep_size=8)
    layer = TransformerLayer(hidden_size=2048, num_experts=8, parallel_ctx=ctx)
    model = TransformerModel(num_layers=24, ...)
"""

__version__ = "1.0.0"
__author__ = "FluidMoE Team"
__license__ = "Apache 2.0"

# =============================================================================
# Core module
# =============================================================================
from .core import (
    # AllToAll primitives
    _all_to_all,
    _all_to_all_sp2hp_forward,
    _all_to_all_hp2sp_forward,
    # P2P scheduling
    compute_round_robin_schedule,
    get_partner_for_round,
    # Overlap context
    MultiCardOverlapContext,
    # Scheduler
    BackwardScheduler,
    get_backward_scheduler,
    # P2P Backend
    get_p2p_backend,
)

# =============================================================================
# MoE module (forward/backward building blocks)
# =============================================================================
from .moe import (
    # Forward operations
    router_forward,
    merge_tokens_expert_major,
    precompute_backward_sort_indices,
    dispatch_fc1_p2p_forward,
    fc2_combine_p2p_forward,
    # Backward - Region 1 & 2
    combine_fc2_backward,
    fc1_dispatch_backward,
    register_moe_dw_tasks,
    router_backward,
    register_router_dw_task,
)

# =============================================================================
# Attention module (forward/backward building blocks)
# =============================================================================
from .attention import (
    # Forward operations
    qkv_projection_p2p_forward,
    scaled_dot_product_attention_forward,
    output_projection_p2p_forward,
    # Backward - Region 3 & 4
    outproj_sp2hp_backward,
    hp2sp_qkv_backward,
    output_projection_register_dw,
)

# =============================================================================
# Layer module (unified Transformer autograd.Function)
# =============================================================================
from .layer import (
    TransformerLayerFunction,
    TransformerLayer,
    TransformerModel,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Core - AllToAll primitives
    "_all_to_all",
    "_all_to_all_sp2hp_forward",
    "_all_to_all_hp2sp_forward",
    # Core - P2P scheduling
    "compute_round_robin_schedule",
    "get_partner_for_round",
    # Core - Overlap context
    "MultiCardOverlapContext",
    # Core - Scheduler
    "BackwardScheduler",
    # Core - P2P Backend
    "get_p2p_backend",
    "get_backward_scheduler",

    # MoE - Forward operations
    "router_forward",
    "merge_tokens_expert_major",
    "precompute_backward_sort_indices",
    "dispatch_fc1_p2p_forward",
    "fc2_combine_p2p_forward",
    # MoE - Backward (Region 1 & 2)
    "combine_fc2_backward",
    "fc1_dispatch_backward",
    "register_moe_dw_tasks",
    "router_backward",
    "register_router_dw_task",
    # Attention - Forward operations
    "qkv_projection_p2p_forward",
    "scaled_dot_product_attention_forward",
    "output_projection_p2p_forward",
    # Attention - Backward (Region 3 & 4)
    "outproj_sp2hp_backward",
    "hp2sp_qkv_backward",
    "output_projection_register_dw",

    # Layer - Unified Transformer (single autograd.Function)
    "TransformerLayerFunction",
    "TransformerLayer",
    "TransformerModel",

    # Setup (Megatron integration)
    "FluidDDP",
    "FluidOptimizerWrapper",
    "setup_model_and_optimizer",
]

from .setup import FluidDDP, FluidOptimizerWrapper, setup_model_and_optimizer
