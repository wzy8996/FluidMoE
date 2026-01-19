"""
Core module - Basic infrastructure for FluidMoE

Contains:
- comm: Communication primitives (AllToAll, P2P scheduling, overlap context)
- scheduler: dW scheduler for backward pass
"""

from .comm import (
    # AllToAll primitives
    _all_to_all,
    _all_to_all_sp2hp_forward,
    _all_to_all_hp2sp_forward,
    _sort_chunks_by_idxs,
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
    # Communication
    '_all_to_all',
    '_all_to_all_sp2hp_forward',
    '_all_to_all_hp2sp_forward',
    '_sort_chunks_by_idxs',
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
