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

__version__ = "0.9.0"  # Multi-card P2P overlap with Round-Robin Tournament scheduling
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
    # Fused attention backward with chunked dX + AllToAll pipeline
    fluid_fused_hp2sp_linear_proj,
    fluid_fused_sp2hp_core_attention,
    # Pipelined Q/K/V sp2hp for forward compute-communication overlap
    fluid_pipelined_sp2hp_qkv,
    # TRUE overlap: V AllToAll with Q@K^T computation
    fluid_pipelined_sp2hp_with_qk_matmul,
)

# Fused forward kernels (v0.8)
# The new fused kernels (moe_alltoall_fc1_fused, moe_fc2_alltoall_fused)
# are exposed via fluid_kernels directly

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

# Forward compute-communication overlap (P2P based)
# DEPRECATED: 这些2卡专用函数已被统一的多卡Round-Robin实现替代
# 保留导出仅为向后兼容，请使用 multicard_p2p 中的函数
from .overlap_forward import (
    # QKV + sp2hp overlap (Heads Split方式) - Deprecated
    qkv_sp2hp_heads_split,
    prepare_qkv_split_weights,
    # hp2sp + output projection overlap - Deprecated
    hp2sp_output_proj_overlap,
    # MoE P2P overlap - Deprecated
    moe_p2p_overlap_forward,
    # Context manager - Deprecated
    OverlapContext,
)

# Multi-card P2P overlap (Round-Robin Tournament scheduling)
from .multicard_p2p import (
    # Round-Robin scheduling algorithm
    compute_round_robin_schedule,
    get_partner_for_round,
    get_all_partners_ordered,
    # MoE multi-card P2P overlap
    MultiCardOverlapContext,
    moe_multicard_p2p_overlap_forward,
    # Attention multi-card P2P overlap
    AttentionMultiCardOverlapContext,
    attention_multicard_qkv_sp2hp_with_grad,
    attention_multicard_hp2sp_proj,
)


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
    # Fused attention backward
    "fluid_fused_hp2sp_linear_proj",
    "fluid_fused_sp2hp_core_attention",
    # Pipelined Q/K/V sp2hp for forward overlap
    "fluid_pipelined_sp2hp_qkv",

    # Layer components (advanced users)
    "FluidColumnParallelLinear",
    "FluidRowParallelLinear",
    "FluidGroupedMLP",

    # Forward overlap (P2P based - Heads Split方式)
    # DEPRECATED: 已被统一的多卡实现替代，保留仅为向后兼容
    "qkv_sp2hp_heads_split",
    "prepare_qkv_split_weights",
    "hp2sp_output_proj_overlap",
    "moe_p2p_overlap_forward",
    "OverlapContext",

    # Multi-card P2P overlap (Round-Robin Tournament scheduling)
    # 统一实现：2卡场景自动退化为1轮通信
    "compute_round_robin_schedule",
    "get_partner_for_round",
    "get_all_partners_ordered",
    "MultiCardOverlapContext",
    "moe_multicard_p2p_overlap_forward",
    "AttentionMultiCardOverlapContext",
    "attention_multicard_qkv_sp2hp_with_grad",
    "attention_multicard_hp2sp_proj",
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
