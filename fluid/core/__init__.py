"""
Core module - Basic infrastructure for FluidMoE

Contains:
- alltoall: Basic AllToAll communication primitives
- utils: Common utility functions
- forward_comm: Forward P2P scheduling and overlap context
- scheduler: dW scheduler for backward pass
"""

from .alltoall import (
    _all_to_all,
    _all_to_all_sp2hp_forward,
    _all_to_all_hp2sp_forward,
    _sort_chunks_by_idxs,
)

from .utils import (
    _gelu_grad_exact,
    _compute_activation_derivative,
    _compute_activation_grad,
    get_optimal_num_chunks,
)

from .forward_comm import (
    # P2P Scheduling
    compute_round_robin_schedule,
    get_partner_for_round,
    get_all_partners_ordered,
    get_num_rounds,
    # Overlap Context
    MultiCardOverlapContext,
    AttentionMultiCardOverlapContext,
)

from .scheduler import (
    BackwardScheduler,
    get_backward_scheduler,
)

__all__ = [
    # AllToAll
    '_all_to_all',
    '_all_to_all_sp2hp_forward',
    '_all_to_all_hp2sp_forward',
    '_sort_chunks_by_idxs',
    # Utils
    '_gelu_grad_exact',
    '_compute_activation_derivative',
    '_compute_activation_grad',
    'get_optimal_num_chunks',
    # P2P Schedule
    'compute_round_robin_schedule',
    'get_partner_for_round',
    'get_all_partners_ordered',
    'get_num_rounds',
    # Overlap Context
    'MultiCardOverlapContext',
    'AttentionMultiCardOverlapContext',
    # Scheduler
    'BackwardScheduler',
    'get_backward_scheduler',
]
