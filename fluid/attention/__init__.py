"""
Attention Module - Self-Attention implementations with Context Parallel

File structure:
- forward.py: All forward operations with P2P overlap
  - QKV projection + sp2hp with P2P overlap
  - Scaled dot-product attention computation
  - hp2sp + output projection with P2P overlap

- backward.py: All backward operations with AllToAll + dX/dW scheduling
  - Output projection backward: chunked dX + sp2hp AllToAll + dW overlap
  - Attention backward: recompute + chunked grad_Q/K/V + hp2sp AllToAll + dW overlap
  - QKV projection backward: dX + dW

Note: Complete Attention layer (autograd.Function) has been moved to fluid.layer module
      for unified Transformer layer implementation.

Key design principles:
- Forward uses P2P overlap for compute-communication overlap
- Backward uses AllToAll with chunked compute overlap
- Memory-efficient: save Q, K, V instead of attention matrix
- dW tasks are registered and executed during AllToAll communication
"""

from .forward import (
    qkv_projection_p2p_forward,
    scaled_dot_product_attention_forward,
    output_projection_p2p_forward,
)
from .backward import (
    output_projection_backward_chunked,
    attention_backward_chunked,
    qkv_projection_backward,
    output_projection_register_dw,
)

__all__ = [
    # Forward operations (P2P overlap)
    'qkv_projection_p2p_forward',
    'scaled_dot_product_attention_forward',
    'output_projection_p2p_forward',
    # Backward operations (AllToAll + chunked overlap + dW scheduling)
    'output_projection_backward_chunked',
    'attention_backward_chunked',
    'qkv_projection_backward',
    'output_projection_register_dw',
]
