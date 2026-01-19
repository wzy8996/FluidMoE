"""
FluidMoE: Standalone MoE/Attention Implementation with Communication-Computation Overlap

This package provides standalone implementations of:
- MoE (Mixture of Experts) with Expert Parallel
- Attention with Ulysses-style Context Parallel
- Scheduler for dW overlap during backward AllToAll
- P2P communication overlap for forward pass (Round-Robin Tournament)

Key Features:
- dW tasks registered during backward, executed during AllToAll communication
- Chunked dX computation pipelined with AllToAll
- Multi-card P2P overlap for both MoE dispatch/combine and Attention sp2hp/hp2sp
- No external dependencies (Megatron-free)

Modules:
- core: AllToAll primitives, P2P scheduling, overlap context, scheduler
- moe: MoE layer with P2P forward and chunked AllToAll backward
- attention: Attention layer with P2P forward and chunked AllToAll backward
- layers: Megatron-LM integration utilities
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
    _sort_chunks_by_idxs,
    # Utility functions
    _gelu_grad_exact,
    _compute_activation_derivative,
    _compute_activation_grad,
    get_optimal_num_chunks,
    # P2P scheduling
    compute_round_robin_schedule,
    get_partner_for_round,
    get_all_partners_ordered,
    get_num_rounds,
    # Overlap context
    MultiCardOverlapContext,
    AttentionMultiCardOverlapContext,
    # Scheduler
    BackwardScheduler,
    get_backward_scheduler,
)

# =============================================================================
# MoE module
# =============================================================================
from .moe import (
    # Forward operations
    router_forward,
    compute_fc1_act_per_source,
    compute_fc2_per_source,
    merge_tokens_expert_major,
    precompute_backward_sort_indices,
    dispatch_fc1_p2p_forward,
    fc2_combine_p2p_forward,
    # Backward operations
    recompute_fc1,
    register_moe_dw_tasks,
    combine_backward,
    expert_backward,
    dispatch_backward,
    router_backward,
    register_router_dw_task,
    # Complete MoE layer
    MoEP2PChunkedFunction,
    moe_p2p_chunked,
    MoELayer,
)

# =============================================================================
# Attention module
# =============================================================================
from .attention import (
    # Forward operations
    qkv_projection_p2p_forward,
    scaled_dot_product_attention_forward,
    output_projection_p2p_forward,
    # Backward operations
    output_projection_backward_chunked,
    attention_backward_chunked,
    qkv_projection_backward,
    output_projection_register_dw,
    # Complete attention layer
    AttentionP2PChunkedFunction,
    attention_p2p_chunked,
    AttentionLayer,
)

# =============================================================================
# Layers module (Megatron integration)
# =============================================================================
from .layers import (
    get_fluid_custom_layers,
    get_fluid_moe_layer_spec,  # Deprecated
    is_fluid_enabled,
    print_fluid_layer_info,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Core - AllToAll primitives
    "_all_to_all",
    "_all_to_all_sp2hp_forward",
    "_all_to_all_hp2sp_forward",
    "_sort_chunks_by_idxs",
    # Core - Utility functions
    "_gelu_grad_exact",
    "_compute_activation_derivative",
    "_compute_activation_grad",
    "get_optimal_num_chunks",
    # Core - P2P scheduling
    "compute_round_robin_schedule",
    "get_partner_for_round",
    "get_all_partners_ordered",
    "get_num_rounds",
    # Core - Overlap context
    "MultiCardOverlapContext",
    "AttentionMultiCardOverlapContext",
    # Core - Scheduler
    "BackwardScheduler",
    "get_backward_scheduler",

    # MoE - Forward operations
    "router_forward",
    "compute_fc1_act_per_source",
    "compute_fc2_per_source",
    "merge_tokens_expert_major",
    "precompute_backward_sort_indices",
    "dispatch_fc1_p2p_forward",
    "fc2_combine_p2p_forward",
    # MoE - Backward operations
    "recompute_fc1",
    "register_moe_dw_tasks",
    "combine_backward",
    "expert_backward",
    "dispatch_backward",
    "router_backward",
    "register_router_dw_task",
    # MoE - Complete layer
    "MoEP2PChunkedFunction",
    "moe_p2p_chunked",
    "MoELayer",

    # Attention - Forward operations
    "qkv_projection_p2p_forward",
    "scaled_dot_product_attention_forward",
    "output_projection_p2p_forward",
    # Attention - Backward operations
    "output_projection_backward_chunked",
    "attention_backward_chunked",
    "qkv_projection_backward",
    "output_projection_register_dw",
    # Attention - Complete layer
    "AttentionP2PChunkedFunction",
    "attention_p2p_chunked",
    "AttentionLayer",

    # Megatron integration
    "get_fluid_custom_layers",
    "get_fluid_moe_layer_spec",
    "is_fluid_enabled",
    "print_fluid_layer_info",
]


def print_status():
    """Print FluidMoE status and statistics"""
    scheduler = get_backward_scheduler()

    print("=" * 60)
    print("FluidMoE Status")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Scheduler enabled: {scheduler.is_enabled()}")

    if scheduler.is_enabled():
        stats = scheduler.get_stats()
        print("\nScheduler Statistics:")
        print(f"  Total dW tasks: {stats['total_dw_tasks']}")
        print(f"  Completed dW tasks: {stats['completed_dw_tasks']}")
        print(f"    - During overlap: {stats['overlap_completed_dw_tasks']}")
        print(f"    - In finish_batch: {stats['finish_batch_completed_dw_tasks']}")

        if stats['total_dw_tasks'] > 0:
            overlap_ratio = stats['overlap_completed_dw_tasks'] / stats['total_dw_tasks'] * 100
            print(f"  Overlap ratio: {overlap_ratio:.2f}%")

    print("=" * 60)
