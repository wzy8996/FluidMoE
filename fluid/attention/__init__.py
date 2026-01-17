"""
Attention Module - Self-Attention implementations with Context Parallel

Contains:
- baseline: Baseline attention with Ulysses-style sp2hp/hp2sp AllToAll
- chunked_backward: Chunked dX + AllToAll backward
- p2p_overlap: P2P communication overlap for forward pass
"""

from .baseline import (
    AttentionBaseline,
    _QKVProjectionFunction,
    _OutputProjectionFunction,
    _SP2HPFunction,
    _HP2SPFunction,
    scaled_dot_product_attention,
)
from .chunked_backward import backward_output_proj_chunked
from .p2p_overlap import (
    AttentionMultiCardOverlapContext,
    qkv_sp2hp_multicard_overlap,
    hp2sp_output_proj_multicard_overlap,
    _QKVSp2HpMultiCardFunction,
    _HP2SpOutputProjMultiCardFunction,
)

__all__ = [
    # Baseline
    'AttentionBaseline',
    '_QKVProjectionFunction',
    '_OutputProjectionFunction',
    '_SP2HPFunction',
    '_HP2SPFunction',
    'scaled_dot_product_attention',
    # Chunked backward
    'backward_output_proj_chunked',
    # P2P overlap
    'AttentionMultiCardOverlapContext',
    'qkv_sp2hp_multicard_overlap',
    'hp2sp_output_proj_multicard_overlap',
    '_QKVSp2HpMultiCardFunction',
    '_HP2SpOutputProjMultiCardFunction',
]
