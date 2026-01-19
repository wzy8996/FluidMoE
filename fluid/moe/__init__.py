"""
MoE Module - Mixture of Experts implementations with Context Parallel

File structure:
- forward.py: All forward operations with P2P overlap
  - Router forward (logits, softmax, top-k, token permutation)
  - Dispatch + FC1 phase with P2P overlap
  - FC2 + Combine phase with P2P overlap
  - Helper functions for token merging and reordering

- backward.py: All backward operations with AllToAll + dX/dW scheduling
  - Combine AllToAll backward with dW overlap
  - Expert backward computation
  - Dispatch AllToAll backward with optional chunked dX overlap
  - dW task registration for weight1 and weight2
  - Router backward (grad through softmax, top-k, linear)
  - Router dW task registration
  - FC1 recomputation (Activation Recomputation optimization)

- layer.py: Complete MoE autograd functions with integrated routing
  - MoEP2PChunkedFunction: Full autograd with routing + P2P forward + chunked AllToAll backward
  - moe_p2p_chunked: Main API function with integrated routing
  - MoELayer: High-level nn.Module wrapper

Key design principles:
- Router integrated into layer.py with dW scheduling
- Forward uses P2P overlap for compute-communication overlap
- Backward uses AllToAll with optional chunked compute overlap
- FC1 Recomputation: FC1 is NOT saved in forward, recomputed during backward
  (saves ~2.5ms forward memcpy overhead, hidden by ~28ms Combine AllToAll)
- dW tasks are registered and executed during AllToAll communication
"""

from .forward import (
    router_forward,
    compute_fc1_act_per_source,
    compute_fc2_per_source,
    merge_tokens_expert_major,
    precompute_backward_sort_indices,
    dispatch_fc1_p2p_forward,
    fc2_combine_p2p_forward,
)
from .backward import (
    recompute_fc1,
    register_moe_dw_tasks,
    combine_backward,
    expert_backward,
    dispatch_backward,
    router_backward,
    register_router_dw_task,
)
from .layer import (
    MoEP2PChunkedFunction,
    moe_p2p_chunked,
    MoELayer,
)

__all__ = [
    # Router forward
    'router_forward',
    # Forward operations (P2P overlap)
    'compute_fc1_act_per_source',
    'compute_fc2_per_source',
    'merge_tokens_expert_major',
    'precompute_backward_sort_indices',
    'dispatch_fc1_p2p_forward',
    'fc2_combine_p2p_forward',
    # Backward operations (AllToAll + optional chunked overlap + dW scheduling)
    'recompute_fc1',
    'register_moe_dw_tasks',
    'combine_backward',
    'expert_backward',
    'dispatch_backward',
    # Router backward
    'router_backward',
    'register_router_dw_task',
    # Complete MoE layer (with integrated routing)
    'MoEP2PChunkedFunction',
    'moe_p2p_chunked',
    'MoELayer',
]
