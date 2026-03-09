"""
Common Utility Functions for FluidMoE

Contains:
- Chunk size optimization
- Other common utilities
"""

import os


def get_optimal_num_chunks(
    hidden_size: int,
    total_tokens: int,
    ffn_hidden: int,
    estimated_a2a_time_us: float = 100.0,
) -> int:
    """
    Estimate optimal number of chunks for dX + AllToAll overlap.

    The goal is to balance chunk compute time with AllToAll time:
    - If chunk is too small: AllToAll launch overhead dominates
    - If chunk is too large: Less overlap opportunity

    Args:
        hidden_size: Model hidden dimension
        total_tokens: Total number of tokens
        ffn_hidden: FFN hidden dimension
        estimated_a2a_time_us: Estimated AllToAll time in microseconds

    Returns:
        Recommended number of chunks (1, 2, 4, or 8)
    """
    # Use environment variable if set
    default_chunks = int(os.environ.get('FLUID_DX_CHUNKS', '4'))

    # Ensure hidden_size is divisible by num_chunks
    for num_chunks in [default_chunks, 4, 2, 1]:
        if hidden_size % num_chunks == 0:
            return num_chunks

    return 1


__all__ = [
    'get_optimal_num_chunks',
]
