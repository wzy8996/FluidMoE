"""
Core module - Basic infrastructure for FluidMoE

Contains:
- comm: Communication primitives (AllToAll, P2P scheduling, overlap context)
- scheduler: dW scheduler for backward pass
- nvtx: NVIDIA Tools Extension for profiling
"""

from .comm import (
    # AllToAll primitives
    _all_to_all,
    _all_to_all_sp2hp_forward,
    _all_to_all_hp2sp_forward,
    # P2P Scheduling
    compute_round_robin_schedule,
    get_partner_for_round,
    # Overlap Context
    MultiCardOverlapContext,
)

from .scheduler import (
    BackwardScheduler,
    get_backward_scheduler,
)

from .p2p_backend import (
    P2PBackend,
    NCCLBackend,
    get_p2p_backend,
    reset_p2p_backend,
)

from .nvtx import (
    nvtx_range,
    nvtx_range_push,
    nvtx_range_pop,
    NVTX_ENABLED,
)

__all__ = [
    # Communication
    '_all_to_all',
    '_all_to_all_sp2hp_forward',
    '_all_to_all_hp2sp_forward',
    # P2P Schedule
    'compute_round_robin_schedule',
    'get_partner_for_round',
    # Overlap Context
    'MultiCardOverlapContext',
    # Scheduler
    'BackwardScheduler',
    'get_backward_scheduler',
    # P2P Backend
    'P2PBackend',
    'NCCLBackend',
    'get_p2p_backend',
    'reset_p2p_backend',
    # NVTX Profiling
    'nvtx_range',
    'nvtx_range_push',
    'nvtx_range_pop',
    'NVTX_ENABLED',
]
