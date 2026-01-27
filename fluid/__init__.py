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
    _sort_chunks_by_idxs,
    # P2P scheduling
    compute_round_robin_schedule,
    get_partner_for_round,
    get_all_partners_ordered,
    get_num_rounds,
    # Overlap context
    MultiCardOverlapContext,
    # Scheduler
    BackwardScheduler,
    get_backward_scheduler,
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
    # Backward operations
    recompute_fc1,
    register_moe_dw_tasks,
    combine_backward,
    expert_dispatch_backward,
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
    # Backward operations
    output_projection_backward_chunked,
    attention_backward_chunked,
    qkv_projection_backward,
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
# Distributed module (SP + EP parallel)
# =============================================================================
def _lazy_import_distributed():
    """Lazy import distributed module."""
    from . import distributed
    return distributed

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Core - AllToAll primitives
    "_all_to_all",
    "_all_to_all_sp2hp_forward",
    "_all_to_all_hp2sp_forward",
    "_sort_chunks_by_idxs",
    # Core - P2P scheduling
    "compute_round_robin_schedule",
    "get_partner_for_round",
    "get_all_partners_ordered",
    "get_num_rounds",
    # Core - Overlap context
    "MultiCardOverlapContext",
    # Core - Scheduler
    "BackwardScheduler",
    "get_backward_scheduler",

    # MoE - Forward operations
    "router_forward",
    "merge_tokens_expert_major",
    "precompute_backward_sort_indices",
    "dispatch_fc1_p2p_forward",
    "fc2_combine_p2p_forward",
    # MoE - Backward operations
    "recompute_fc1",
    "register_moe_dw_tasks",
    "combine_backward",
    "expert_dispatch_backward",
    "router_backward",
    "register_router_dw_task",

    # Attention - Forward operations
    "qkv_projection_p2p_forward",
    "scaled_dot_product_attention_forward",
    "output_projection_p2p_forward",
    # Attention - Backward operations
    "output_projection_backward_chunked",
    "attention_backward_chunked",
    "qkv_projection_backward",
    "output_projection_register_dw",

    # Layer - Unified Transformer (single autograd.Function)
    "TransformerLayerFunction",
    "TransformerLayer",
    "TransformerModel",

    # Optimizer wrapper
    "wrap_optimizer",
    "FluidOptimizerWrapper",
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


# =============================================================================
# Optimizer Wrapper
# =============================================================================

class FluidOptimizerWrapper:
    """
    Wrapper for optimizers that enables FluidMoE's dW-AllToAll overlap.

    This wrapper:
    1. Enables the BackwardScheduler before backward pass
    2. Calls finish_batch() before optimizer.step() to complete pending dW tasks
    3. Clears scheduler state after each step

    Usage:
        optimizer = get_megatron_optimizer(config, model)
        optimizer = FluidOptimizerWrapper(optimizer)

        # Training loop
        loss.backward()
        optimizer.step()  # Automatically handles dW completion
    """

    def __init__(self, optimizer):
        """
        Args:
            optimizer: The underlying optimizer to wrap
        """
        self._optimizer = optimizer
        self._scheduler = get_backward_scheduler()
        self._scheduler.enable()
        print(f"[FluidMoE] Optimizer wrapped, BackwardScheduler enabled")

    def step(self, *args, **kwargs):
        """
        Optimizer step with automatic dW completion.

        Calls finish_batch() to complete any pending dW tasks before
        the underlying optimizer updates weights.
        """
        # Complete pending dW tasks
        self._scheduler.finish_batch()

        # Call underlying optimizer step
        result = self._optimizer.step(*args, **kwargs)

        # Clear scheduler state for next iteration
        self._scheduler.clear_iteration()

        return result

    def zero_grad(self, *args, **kwargs):
        """Forward to underlying optimizer."""
        return self._optimizer.zero_grad(*args, **kwargs)

    def __getattr__(self, name):
        """Forward attribute access to underlying optimizer."""
        return getattr(self._optimizer, name)

    def __setattr__(self, name, value):
        """Handle attribute setting."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._optimizer, name, value)

    @property
    def scheduler(self):
        """Access to BackwardScheduler for advanced usage."""
        return self._scheduler


def wrap_optimizer(optimizer):
    """
    Wrap an optimizer to enable FluidMoE's dW-AllToAll overlap.

    This is the recommended way to enable FluidMoE scheduling with
    any training framework (Megatron, DeepSpeed, etc.)

    Args:
        optimizer: The optimizer to wrap

    Returns:
        FluidOptimizerWrapper: Wrapped optimizer with dW overlap enabled

    Example:
        from fluid import wrap_optimizer

        optimizer = get_megatron_optimizer(config, model)
        optimizer = wrap_optimizer(optimizer)

        # Training loop
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()  # dW tasks completed automatically
    """
    return FluidOptimizerWrapper(optimizer)
