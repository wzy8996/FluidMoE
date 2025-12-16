# Copyright (c) 2024, FluidMoE Team. All rights reserved.

"""
FluidMoE: Complete Custom Layer Implementation for Megatron-LM MoE

FluidMoE provides full custom layer implementation with computation-
communication overlap optimization:

- Complete custom FluidSelfAttention and FluidMoELayer
- No global function patching required
- Full control over forward and backward computation
- Support for forward optimization (future)

Architecture:
- FluidSelfAttention: Custom attention with Ulysses Sequence Parallel (SP)
- FluidMoELayer: Custom MoE with Expert Parallel (EP)
- FluidTokenDispatcher: Custom token routing with Fluid AllToAll
- dW computation overlaps with AllToAll communication

Quick Start:
    from megatron.core.transformer import TransformerConfig
    from megatron.core import GPTModel
    from fluid import get_fluid_custom_layers

    config = TransformerConfig(
        num_layers=32,
        hidden_size=4096,
        num_moe_experts=8,
        context_parallel_size=4,  # SP
        expert_model_parallel_size=2,  # EP
    )

    custom_layers = get_fluid_custom_layers()
    model = GPTModel(config, transformer_layer_spec=custom_layers)

Components:
- scheduler: Global backward scheduler with dW task queue
- communication: Fluid AllToAll primitives (no patching needed)
- attention_module: FluidSelfAttention with custom forward logic
- moe_module: FluidMoELayer and FluidTokenDispatcher
- attention_layers: Fluid linear layers for dW scheduling
- moe_layers: FluidGroupedMLP for expert computation
"""

__version__ = "0.5.0"  # Complete custom layer implementation
__author__ = "FluidMoE Team"
__license__ = "Apache 2.0"

# Core scheduler
from .scheduler import BackwardScheduler, get_backward_scheduler

# Optimizer component (for automatic finish_batch)
from .optimizer_wrapper import (
    FluidOptimizerWrapper,
    get_fluid_optimizer,
    wrap_optimizer,
)

# Communication primitives
from .communication import (
    _FluidAllToAll,
    fluid_all_to_all,
    fluid_all_to_all_sp2hp,
    fluid_all_to_all_hp2sp,
    fluid_all_to_all_moe_dispatch,
    fluid_all_to_all_moe_combine,
)

# MoE components
from .moe_layers import FluidGroupedMLP, FluidRouter

# Attention components
from .attention_layers import FluidColumnParallelLinear, FluidRowParallelLinear

# Custom layer modules
from .attention_module import FluidSelfAttention
from .moe_module import FluidMoELayer

# Primary API (complete custom layer implementation)
from .megatron_layers import (
    get_fluid_custom_layers,
    get_fluid_moe_layer_spec,  # Deprecated alias
    print_fluid_layer_info,
    is_fluid_enabled,
)

# Pretrain function (generic training entry point)
from .pretrain import pretrain

__all__ = [
    # Primary API
    "get_fluid_custom_layers",
    "get_fluid_optimizer",  # New: Fluid optimizer component
    "pretrain",  # FluidMoE pretrain function
    "get_fluid_moe_layer_spec",  # Deprecated, use get_fluid_custom_layers
    "print_fluid_layer_info",
    "is_fluid_enabled",

    # Custom components (for customization)
    "FluidSelfAttention",
    "FluidMoELayer",
    "FluidOptimizerWrapper",

    # Optimizer utilities
    "wrap_optimizer",

    # Scheduler (advanced users)
    "BackwardScheduler",
    "get_backward_scheduler",

    # Communication (advanced users)
    "_FluidAllToAll",
    "fluid_all_to_all",
    "fluid_all_to_all_sp2hp",
    "fluid_all_to_all_hp2sp",
    "fluid_all_to_all_moe_dispatch",
    "fluid_all_to_all_moe_combine",

    # Layer components (advanced users)
    "FluidColumnParallelLinear",
    "FluidRowParallelLinear",
    "FluidGroupedMLP",
]


def print_status():
    """Print FluidMoE status and statistics"""
    scheduler = get_backward_scheduler()

    print("="*60)
    print("FluidMoE Status")
    print("="*60)
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

    print("="*60)
