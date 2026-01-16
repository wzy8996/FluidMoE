"""
MoE Module - Mixture of Experts implementations

Contains:
- router: Router (gating network) with dW scheduling
- baseline: Baseline MoE with standard AllToAll
- chunked_backward: Chunked dX + AllToAll backward
- p2p_overlap: P2P communication overlap for forward pass
"""

from .router import _RouterFunction, compute_routing
from .baseline import MoEBaseline, _MoEBaselineFunction
from .chunked_backward import backward_dispatch_chunked
from .p2p_overlap import (
    moe_multicard_p2p_overlap_forward,
    _MoEMultiCardP2POverlapFunction,
    _compute_fc1_act_per_source,
    _compute_fc2_per_source,
    _compute_expert_forward_per_source,
    _merge_tokens_and_fc1_expert_major,
    _precompute_backward_sort_indices,
)

__all__ = [
    # Router
    '_RouterFunction',
    'compute_routing',
    # Baseline
    'MoEBaseline',
    '_MoEBaselineFunction',
    # Chunked backward
    'backward_dispatch_chunked',
    # P2P overlap
    'moe_multicard_p2p_overlap_forward',
    '_MoEMultiCardP2POverlapFunction',
    '_compute_fc1_act_per_source',
    '_compute_fc2_per_source',
    '_compute_expert_forward_per_source',
    '_merge_tokens_and_fc1_expert_major',
    '_precompute_backward_sort_indices',
]
