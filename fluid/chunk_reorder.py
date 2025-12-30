# Copyright (c) 2024, FluidMoE Team. All rights reserved.

"""
Chunk Reordering Utilities for dX + AllToAll Pipeline

This module provides functions to reorder tokens between different layouts:

1. Original layout (rank-sorted + expert-sorted):
   [Rank0: E0_tokens, E1_tokens, ... | Rank1: E0_tokens, E1_tokens, ... | ...]

2. Chunk-sorted layout:
   [Chunk0: R0_E0_c0, R0_E1_c0, R1_E0_c0, R1_E1_c0, ... | Chunk1: ... | ...]

Key insight: Each chunk is still internally sorted by rank, then by expert,
so grouped_gemm works directly without gather operations.

The reordering enables:
- Pre-reorder all data once at backward start
- Each chunk's data is contiguous, accessed by simple slicing
- grouped_gemm works on contiguous memory
- After computing dX for a chunk, the data is already in correct order for AllToAll
"""

import torch
from typing import List, Tuple, Optional


def compute_chunk_reorder_indices(
    tokens_per_expert: torch.Tensor,
    num_chunks: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Compute reordering indices for expert-sorted <-> chunk-sorted conversion.

    Args:
        tokens_per_expert: [num_experts] tensor with token counts per expert
        num_chunks: Number of chunks to divide the data into

    Returns:
        expert_to_chunk_indices: [total_tokens] indices to reorder from expert-sorted to chunk-sorted
        chunk_to_expert_indices: [total_tokens] indices to reorder from chunk-sorted to expert-sorted
        chunk_tokens_per_expert: List of [num_experts] tensors, one per chunk
                                  Each tensor contains the token count per expert for that chunk
    """
    num_experts = tokens_per_expert.shape[0]
    total_tokens = int(tokens_per_expert.sum().item())
    device = tokens_per_expert.device

    # Compute expert boundaries in expert-sorted layout
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)

    # For each expert, compute chunk boundaries
    # Each expert's tokens are divided into num_chunks portions
    chunk_tokens_per_expert = []

    # Build the mapping indices
    # expert_to_chunk_indices[i] = position of token i (in expert-sorted) in chunk-sorted layout
    # chunk_to_expert_indices[i] = position of token i (in chunk-sorted) in expert-sorted layout

    expert_to_chunk = torch.zeros(total_tokens, dtype=torch.int64, device=device)
    chunk_to_expert = torch.zeros(total_tokens, dtype=torch.int64, device=device)

    # Compute positions for each chunk
    chunk_start_positions = []  # chunk_start_positions[chunk_idx] = start position in chunk-sorted layout
    chunk_expert_counts = []    # chunk_expert_counts[chunk_idx][exp_idx] = tokens from expert exp_idx in chunk chunk_idx

    current_chunk_pos = 0
    for chunk_idx in range(num_chunks):
        chunk_expert_count = []
        chunk_start_positions.append(current_chunk_pos)

        for exp_idx in range(num_experts):
            exp_tokens = int(tokens_per_expert[exp_idx].item())

            # Divide tokens for this expert into chunks
            base_size = exp_tokens // num_chunks
            remainder = exp_tokens % num_chunks

            # Chunk chunk_idx gets base_size tokens, plus 1 if chunk_idx < remainder
            this_chunk_size = base_size + (1 if chunk_idx < remainder else 0)
            chunk_expert_count.append(this_chunk_size)

            current_chunk_pos += this_chunk_size

        chunk_expert_counts.append(chunk_expert_count)
        chunk_tokens_per_expert.append(
            torch.tensor(chunk_expert_count, dtype=torch.int32, device=device)
        )

    # Now build the actual index mappings
    # For each token in expert-sorted layout, compute its position in chunk-sorted layout
    for exp_idx in range(num_experts):
        exp_tokens = int(tokens_per_expert[exp_idx].item())
        exp_start = int(expert_offsets[exp_idx].item())

        base_size = exp_tokens // num_chunks
        remainder = exp_tokens % num_chunks

        for local_token_idx in range(exp_tokens):
            expert_sorted_pos = exp_start + local_token_idx

            # Determine which chunk this token belongs to
            # Tokens 0..remainder-1 go to chunks 0..remainder-1 (one extra each)
            # Then remaining tokens are distributed evenly
            if remainder == 0:
                chunk_idx = local_token_idx // base_size if base_size > 0 else 0
                pos_in_chunk_expert = local_token_idx % base_size if base_size > 0 else 0
            else:
                # First 'remainder' chunks each get (base_size + 1) tokens
                # Remaining chunks get base_size tokens
                threshold = remainder * (base_size + 1)
                if local_token_idx < threshold:
                    chunk_idx = local_token_idx // (base_size + 1)
                    pos_in_chunk_expert = local_token_idx % (base_size + 1)
                else:
                    adjusted_idx = local_token_idx - threshold
                    chunk_idx = remainder + (adjusted_idx // base_size if base_size > 0 else 0)
                    pos_in_chunk_expert = adjusted_idx % base_size if base_size > 0 else 0

            # Clamp chunk_idx (safety)
            chunk_idx = min(chunk_idx, num_chunks - 1)

            # Compute position in chunk-sorted layout
            # Position = chunk_start + sum of expert tokens in this chunk before exp_idx + pos_in_chunk_expert
            chunk_start = chunk_start_positions[chunk_idx]
            expert_offset_in_chunk = sum(chunk_expert_counts[chunk_idx][:exp_idx])
            chunk_sorted_pos = chunk_start + expert_offset_in_chunk + pos_in_chunk_expert

            expert_to_chunk[expert_sorted_pos] = chunk_sorted_pos
            chunk_to_expert[chunk_sorted_pos] = expert_sorted_pos

    return expert_to_chunk, chunk_to_expert, chunk_tokens_per_expert


def reorder_expert_to_chunk(
    data: torch.Tensor,
    expert_to_chunk_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Reorder data from expert-sorted to chunk-sorted layout.

    Args:
        data: [total_tokens, ...] tensor in expert-sorted order
        expert_to_chunk_indices: [total_tokens] index mapping

    Returns:
        Reordered tensor in chunk-sorted order
    """
    # Create output tensor
    out = torch.empty_like(data)
    # Scatter data to new positions
    out.scatter_(0, expert_to_chunk_indices.unsqueeze(-1).expand_as(data), data)
    return out


def reorder_chunk_to_expert(
    data: torch.Tensor,
    chunk_to_expert_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Reorder data from chunk-sorted to expert-sorted layout.

    Args:
        data: [total_tokens, ...] tensor in chunk-sorted order
        chunk_to_expert_indices: [total_tokens] index mapping

    Returns:
        Reordered tensor in expert-sorted order
    """
    # Create output tensor
    out = torch.empty_like(data)
    # Scatter data to new positions
    out.scatter_(0, chunk_to_expert_indices.unsqueeze(-1).expand_as(data), data)
    return out


def get_chunk_slice(
    chunk_idx: int,
    chunk_tokens_per_expert: List[torch.Tensor],
) -> Tuple[int, int]:
    """
    Get the start and end indices for a specific chunk in chunk-sorted layout.

    Args:
        chunk_idx: Index of the chunk
        chunk_tokens_per_expert: List of per-expert token counts for each chunk

    Returns:
        (start, end) indices in the chunk-sorted tensor
    """
    start = 0
    for i in range(chunk_idx):
        start += int(chunk_tokens_per_expert[i].sum().item())

    chunk_size = int(chunk_tokens_per_expert[chunk_idx].sum().item())
    return start, start + chunk_size


def get_expert_slice_in_chunk(
    chunk_idx: int,
    exp_idx: int,
    chunk_tokens_per_expert: List[torch.Tensor],
) -> Tuple[int, int, int]:
    """
    Get the slice for a specific expert within a chunk.

    Args:
        chunk_idx: Index of the chunk
        exp_idx: Index of the expert
        chunk_tokens_per_expert: List of per-expert token counts for each chunk

    Returns:
        (global_start, local_start, size) where:
        - global_start: position in the full chunk-sorted tensor
        - local_start: position within this chunk
        - size: number of tokens
    """
    chunk_start, _ = get_chunk_slice(chunk_idx, chunk_tokens_per_expert)

    local_start = 0
    for i in range(exp_idx):
        local_start += int(chunk_tokens_per_expert[chunk_idx][i].item())

    size = int(chunk_tokens_per_expert[chunk_idx][exp_idx].item())

    return chunk_start + local_start, local_start, size


class ChunkReorderContext:
    """
    Context manager for efficient chunk reordering during backward pass.

    Pre-computes and caches all reordering indices to avoid repeated computation.
    """

    def __init__(
        self,
        tokens_per_expert: torch.Tensor,
        num_chunks: int,
    ):
        """
        Initialize chunk reorder context.

        Args:
            tokens_per_expert: [num_experts] tensor with token counts
            num_chunks: Number of chunks for pipeline
        """
        self.tokens_per_expert = tokens_per_expert
        self.num_chunks = num_chunks
        self.num_experts = tokens_per_expert.shape[0]
        self.total_tokens = int(tokens_per_expert.sum().item())
        self.device = tokens_per_expert.device

        # Compute indices
        (
            self.expert_to_chunk_indices,
            self.chunk_to_expert_indices,
            self.chunk_tokens_per_expert,
        ) = compute_chunk_reorder_indices(tokens_per_expert, num_chunks)

        # Pre-compute chunk boundaries for fast slicing
        self._chunk_boundaries = []
        pos = 0
        for chunk_idx in range(num_chunks):
            chunk_size = int(self.chunk_tokens_per_expert[chunk_idx].sum().item())
            self._chunk_boundaries.append((pos, pos + chunk_size))
            pos += chunk_size

    def to_chunk_sorted(self, data: torch.Tensor) -> torch.Tensor:
        """Convert data from expert-sorted to chunk-sorted layout."""
        return reorder_expert_to_chunk(data, self.expert_to_chunk_indices)

    def to_expert_sorted(self, data: torch.Tensor) -> torch.Tensor:
        """Convert data from chunk-sorted to expert-sorted layout."""
        return reorder_chunk_to_expert(data, self.chunk_to_expert_indices)

    def get_chunk_data(self, chunk_sorted_data: torch.Tensor, chunk_idx: int) -> torch.Tensor:
        """Extract data for a specific chunk from chunk-sorted tensor."""
        start, end = self._chunk_boundaries[chunk_idx]
        return chunk_sorted_data[start:end]

    def get_chunk_tokens_per_expert(self, chunk_idx: int) -> torch.Tensor:
        """Get tokens_per_expert for a specific chunk."""
        return self.chunk_tokens_per_expert[chunk_idx]

    def get_chunk_size(self, chunk_idx: int) -> int:
        """Get total tokens in a specific chunk."""
        start, end = self._chunk_boundaries[chunk_idx]
        return end - start

    def get_chunk_boundaries(self, chunk_idx: int) -> Tuple[int, int]:
        """Get (start, end) boundaries for a chunk in chunk-sorted layout."""
        return self._chunk_boundaries[chunk_idx]

    def get_chunk_to_expert_indices_for_chunk(self, chunk_idx: int) -> torch.Tensor:
        """
        Get indices to scatter this chunk's results back to expert-sorted layout.

        Returns indices into the FULL expert-sorted tensor.
        """
        start, end = self._chunk_boundaries[chunk_idx]
        return self.chunk_to_expert_indices[start:end]


def compute_rank_chunk_reorder_indices(
    output_splits: List[int],
    num_chunks: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    """
    Compute reordering indices for rank-sorted <-> chunk-sorted conversion.

    This is for the case where data is organized by source rank:
    [rank0's tokens | rank1's tokens | ...]

    The chunk-sorted layout groups tokens by chunk across all ranks:
    [chunk0: rank0_c0 | rank1_c0 | ...][chunk1: rank0_c1 | rank1_c1 | ...]

    Args:
        output_splits: List of token counts per source rank
        num_chunks: Number of chunks to divide the data into
        device: Device to create tensors on

    Returns:
        rank_to_chunk_indices: [total_tokens] indices to reorder from rank-sorted to chunk-sorted
        chunk_to_rank_indices: [total_tokens] indices to reorder from chunk-sorted to rank-sorted
        chunk_splits: List[List[int]] - chunk_splits[chunk_idx][rank] = tokens from rank in chunk
    """
    ep_size = len(output_splits)
    total_tokens = sum(output_splits)

    if total_tokens == 0:
        return (
            torch.tensor([], dtype=torch.int64, device=device),
            torch.tensor([], dtype=torch.int64, device=device),
            [[0] * ep_size for _ in range(num_chunks)]
        )

    # Compute chunk sizes for each rank
    # chunk_splits[chunk_idx][rank] = number of tokens from rank in chunk chunk_idx
    chunk_splits = []
    for chunk_idx in range(num_chunks):
        chunk_rank_sizes = []
        for rank in range(ep_size):
            rank_tokens = output_splits[rank]
            if rank_tokens == 0:
                chunk_rank_sizes.append(0)
                continue

            base_size = rank_tokens // num_chunks
            remainder = rank_tokens % num_chunks

            # Chunk chunk_idx gets base_size tokens, plus 1 if chunk_idx < remainder
            this_chunk_size = base_size + (1 if chunk_idx < remainder else 0)
            chunk_rank_sizes.append(this_chunk_size)

        chunk_splits.append(chunk_rank_sizes)

    # Build index mappings
    rank_to_chunk = torch.zeros(total_tokens, dtype=torch.int64, device=device)
    chunk_to_rank = torch.zeros(total_tokens, dtype=torch.int64, device=device)

    # Compute starting positions for each chunk in chunk-sorted layout
    chunk_starts = [0]
    for chunk_idx in range(num_chunks):
        chunk_size = sum(chunk_splits[chunk_idx])
        chunk_starts.append(chunk_starts[-1] + chunk_size)

    # Compute starting positions for each rank within each chunk
    # chunk_rank_starts[chunk_idx][rank] = starting position of rank's data in chunk chunk_idx
    chunk_rank_starts = []
    for chunk_idx in range(num_chunks):
        rank_starts = [0]
        for rank in range(ep_size - 1):
            rank_starts.append(rank_starts[-1] + chunk_splits[chunk_idx][rank])
        chunk_rank_starts.append(rank_starts)

    # Compute starting positions for each rank in rank-sorted layout
    rank_starts_in_rank_sorted = [0]
    for rank in range(ep_size - 1):
        rank_starts_in_rank_sorted.append(rank_starts_in_rank_sorted[-1] + output_splits[rank])

    # Build mappings for each token
    for rank in range(ep_size):
        rank_tokens = output_splits[rank]
        if rank_tokens == 0:
            continue

        rank_start_in_rank_sorted = rank_starts_in_rank_sorted[rank]
        base_size = rank_tokens // num_chunks
        remainder = rank_tokens % num_chunks

        for local_idx in range(rank_tokens):
            rank_sorted_pos = rank_start_in_rank_sorted + local_idx

            # Determine which chunk this token belongs to
            if remainder == 0:
                chunk_idx = local_idx // base_size if base_size > 0 else 0
                pos_in_chunk_rank = local_idx % base_size if base_size > 0 else 0
            else:
                threshold = remainder * (base_size + 1)
                if local_idx < threshold:
                    chunk_idx = local_idx // (base_size + 1)
                    pos_in_chunk_rank = local_idx % (base_size + 1)
                else:
                    adjusted_idx = local_idx - threshold
                    chunk_idx = remainder + (adjusted_idx // base_size if base_size > 0 else 0)
                    pos_in_chunk_rank = adjusted_idx % base_size if base_size > 0 else 0

            # Clamp chunk_idx
            chunk_idx = min(chunk_idx, num_chunks - 1)

            # Compute position in chunk-sorted layout
            chunk_start = chunk_starts[chunk_idx]
            rank_offset_in_chunk = chunk_rank_starts[chunk_idx][rank]
            chunk_sorted_pos = chunk_start + rank_offset_in_chunk + pos_in_chunk_rank

            rank_to_chunk[rank_sorted_pos] = chunk_sorted_pos
            chunk_to_rank[chunk_sorted_pos] = rank_sorted_pos

    return rank_to_chunk, chunk_to_rank, chunk_splits


class RankChunkReorderContext:
    """
    Context manager for efficient rank-chunk reordering during backward pass.

    This handles the case where data is organized by source rank (EP AllToAll output).
    Pre-computes and caches all reordering indices to avoid repeated computation.
    """

    def __init__(
        self,
        output_splits: List[int],
        num_chunks: int,
        device: torch.device,
    ):
        """
        Initialize rank-chunk reorder context.

        Args:
            output_splits: List of token counts per source rank
            num_chunks: Number of chunks for pipeline
            device: Device for tensors
        """
        self.output_splits = output_splits
        self.num_chunks = num_chunks
        self.ep_size = len(output_splits)
        self.total_tokens = sum(output_splits)
        self.device = device

        # Compute indices
        (
            self.rank_to_chunk_indices,
            self.chunk_to_rank_indices,
            self.chunk_splits,
        ) = compute_rank_chunk_reorder_indices(output_splits, num_chunks, device)

        # Pre-compute chunk boundaries
        self._chunk_boundaries = []
        pos = 0
        for chunk_idx in range(num_chunks):
            chunk_size = sum(self.chunk_splits[chunk_idx])
            self._chunk_boundaries.append((pos, pos + chunk_size))
            pos += chunk_size

    def to_chunk_sorted(self, data: torch.Tensor) -> torch.Tensor:
        """Convert data from rank-sorted to chunk-sorted layout."""
        if self.total_tokens == 0:
            return data
        out = torch.empty_like(data)
        out.scatter_(0, self.rank_to_chunk_indices.unsqueeze(-1).expand_as(data), data)
        return out

    def to_rank_sorted(self, data: torch.Tensor) -> torch.Tensor:
        """Convert data from chunk-sorted to rank-sorted layout."""
        if self.total_tokens == 0:
            return data
        out = torch.empty_like(data)
        out.scatter_(0, self.chunk_to_rank_indices.unsqueeze(-1).expand_as(data), data)
        return out

    def get_chunk_data(self, chunk_sorted_data: torch.Tensor, chunk_idx: int) -> torch.Tensor:
        """Extract data for a specific chunk from chunk-sorted tensor."""
        start, end = self._chunk_boundaries[chunk_idx]
        return chunk_sorted_data[start:end]

    def get_chunk_splits(self, chunk_idx: int) -> List[int]:
        """Get output_splits for a specific chunk (tokens per rank in this chunk)."""
        return self.chunk_splits[chunk_idx]

    def get_chunk_size(self, chunk_idx: int) -> int:
        """Get total tokens in a specific chunk."""
        start, end = self._chunk_boundaries[chunk_idx]
        return end - start

    def get_chunk_boundaries(self, chunk_idx: int) -> Tuple[int, int]:
        """Get (start, end) boundaries for a chunk in chunk-sorted layout."""
        return self._chunk_boundaries[chunk_idx]

    def scatter_chunk_to_rank(
        self,
        chunk_data: torch.Tensor,
        chunk_idx: int,
        output: torch.Tensor,
    ) -> None:
        """
        Scatter a chunk's data back to its original positions in rank-sorted layout.

        This is more efficient than full reorder when only one chunk is ready.

        Args:
            chunk_data: [chunk_size, ...] tensor
            chunk_idx: Index of this chunk
            output: [total_tokens, ...] tensor to scatter into
        """
        start, end = self._chunk_boundaries[chunk_idx]
        indices = self.chunk_to_rank_indices[start:end]
        output.scatter_(0, indices.unsqueeze(-1).expand_as(chunk_data), chunk_data)


# ============================================================================
# Unified Chunk Reorder Context for rank + expert layout
# ============================================================================

def compute_full_chunk_reorder_indices(
    tokens_per_expert_per_rank: torch.Tensor,  # [ep_size, num_local_experts]
    num_chunks: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[List[int]], List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute reordering indices for full (rank + expert sorted) <-> chunk-sorted conversion.

    Original layout (rank-sorted + expert-sorted):
        [Rank0: E0_tokens, E1_tokens | Rank1: E0_tokens, E1_tokens | ...]

    Chunk-sorted layout (EXPERT-MAJOR within each chunk - for grouped_gemm):
        [Chunk0: E0_R0_c0, E0_R1_c0, E1_R0_c0, E1_R1_c0 | Chunk1: ...]

    This layout ensures:
    - grouped_gemm works correctly (tokens grouped by expert)
    - After dX computation, we need to reorder to rank-major for AllToAll

    Args:
        tokens_per_expert_per_rank: [ep_size, num_local_experts] token counts
        num_chunks: Number of chunks to divide the data into
        device: Device to create tensors on

    Returns:
        orig_to_chunk_indices: [total_tokens] indices for original -> chunk-sorted
        chunk_to_orig_indices: [total_tokens] indices for chunk-sorted -> original
        chunk_tokens_per_expert: List of [num_local_experts] tensors (merged across ranks)
        chunk_splits: List[List[int]] - chunk_splits[chunk_idx][rank] = tokens from rank in chunk
        chunk_expert_to_rank_indices: List of [chunk_size] tensors - within-chunk reorder to rank-major
        chunk_rank_to_expert_indices: List of [chunk_size] tensors - within-chunk reorder to expert-major
    """
    ep_size = tokens_per_expert_per_rank.shape[0]
    num_local_experts = tokens_per_expert_per_rank.shape[1]

    # Get tokens per expert per rank on CPU
    tpe_cpu = tokens_per_expert_per_rank.cpu().tolist()

    # Compute total tokens
    total_tokens = sum(sum(tpe_cpu[r]) for r in range(ep_size))

    if total_tokens == 0:
        return (
            torch.tensor([], dtype=torch.int64, device=device),
            torch.tensor([], dtype=torch.int64, device=device),
            [torch.zeros(num_local_experts, dtype=torch.int32, device=device) for _ in range(num_chunks)],
            [[0] * ep_size for _ in range(num_chunks)],
            [torch.tensor([], dtype=torch.int64, device=device) for _ in range(num_chunks)],
            [torch.tensor([], dtype=torch.int64, device=device) for _ in range(num_chunks)],
        )

    # Compute the chunk assignment for each token in each (rank, expert) group
    # For each (rank, expert) pair, divide its tokens into num_chunks parts

    # chunk_token_counts[chunk_idx][rank][exp] = tokens from (rank, exp) in chunk
    chunk_token_counts = [[[0] * num_local_experts for _ in range(ep_size)] for _ in range(num_chunks)]

    for rank in range(ep_size):
        for exp in range(num_local_experts):
            n_tokens = int(tpe_cpu[rank][exp])
            if n_tokens == 0:
                continue

            base_size = n_tokens // num_chunks
            remainder = n_tokens % num_chunks

            for chunk_idx in range(num_chunks):
                chunk_size = base_size + (1 if chunk_idx < remainder else 0)
                chunk_token_counts[chunk_idx][rank][exp] = chunk_size

    # Compute chunk_splits: total tokens from each rank in each chunk
    chunk_splits = []
    for chunk_idx in range(num_chunks):
        rank_totals = []
        for rank in range(ep_size):
            rank_totals.append(sum(chunk_token_counts[chunk_idx][rank]))
        chunk_splits.append(rank_totals)

    # Compute merged tokens_per_expert for each chunk (summed across ranks)
    chunk_tokens_per_expert = []
    for chunk_idx in range(num_chunks):
        merged = [0] * num_local_experts
        for rank in range(ep_size):
            for exp in range(num_local_experts):
                merged[exp] += chunk_token_counts[chunk_idx][rank][exp]
        chunk_tokens_per_expert.append(
            torch.tensor(merged, dtype=torch.int32, device=device)
        )

    # Build index mappings
    orig_to_chunk = torch.zeros(total_tokens, dtype=torch.int64, device=device)
    chunk_to_orig = torch.zeros(total_tokens, dtype=torch.int64, device=device)

    # Compute starting positions for each chunk in chunk-sorted layout
    chunk_starts = [0]
    for chunk_idx in range(num_chunks):
        chunk_size = sum(chunk_splits[chunk_idx])
        chunk_starts.append(chunk_starts[-1] + chunk_size)

    # ============================================================
    # EXPERT-MAJOR layout within each chunk (for grouped_gemm)
    # [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...]
    # ============================================================
    chunk_exp_rank_starts = []
    for chunk_idx in range(num_chunks):
        pos = chunk_starts[chunk_idx]
        starts = {}
        for exp in range(num_local_experts):  # Expert first!
            for rank in range(ep_size):
                starts[(rank, exp)] = pos
                pos += chunk_token_counts[chunk_idx][rank][exp]
        chunk_exp_rank_starts.append(starts)

    # Compute starting positions in original layout
    # orig_rank_exp_starts[(rank, exp)] = start position of (rank, exp) tokens
    orig_rank_exp_starts = {}
    pos = 0
    for rank in range(ep_size):
        for exp in range(num_local_experts):
            orig_rank_exp_starts[(rank, exp)] = pos
            pos += int(tpe_cpu[rank][exp])

    # Build mappings for each token
    for rank in range(ep_size):
        for exp in range(num_local_experts):
            n_tokens = int(tpe_cpu[rank][exp])
            if n_tokens == 0:
                continue

            orig_start = orig_rank_exp_starts[(rank, exp)]
            base_size = n_tokens // num_chunks
            remainder = n_tokens % num_chunks

            for local_idx in range(n_tokens):
                orig_pos = orig_start + local_idx

                # Determine which chunk this token belongs to
                if remainder == 0:
                    chunk_idx = local_idx // base_size if base_size > 0 else 0
                    pos_in_chunk = local_idx % base_size if base_size > 0 else 0
                else:
                    threshold = remainder * (base_size + 1)
                    if local_idx < threshold:
                        chunk_idx = local_idx // (base_size + 1)
                        pos_in_chunk = local_idx % (base_size + 1)
                    else:
                        adjusted = local_idx - threshold
                        chunk_idx = remainder + (adjusted // base_size if base_size > 0 else 0)
                        pos_in_chunk = adjusted % base_size if base_size > 0 else 0

                chunk_idx = min(chunk_idx, num_chunks - 1)

                # Compute position in chunk-sorted layout (EXPERT-MAJOR)
                chunk_sorted_pos = chunk_exp_rank_starts[chunk_idx][(rank, exp)] + pos_in_chunk

                orig_to_chunk[orig_pos] = chunk_sorted_pos
                chunk_to_orig[chunk_sorted_pos] = orig_pos

    # ============================================================
    # Compute within-chunk reorder indices: expert-major <-> rank-major
    # ============================================================
    # After dX computation (expert-major), we need rank-major for AllToAll
    # expert_to_rank_indices[i] = where token i (in expert-major) goes in rank-major
    # rank_to_expert_indices[i] = where token i (in rank-major) goes in expert-major

    chunk_expert_to_rank_indices = []
    chunk_rank_to_expert_indices = []

    for chunk_idx in range(num_chunks):
        chunk_size = sum(chunk_splits[chunk_idx])
        if chunk_size == 0:
            chunk_expert_to_rank_indices.append(torch.tensor([], dtype=torch.int64, device=device))
            chunk_rank_to_expert_indices.append(torch.tensor([], dtype=torch.int64, device=device))
            continue

        exp_to_rank = torch.zeros(chunk_size, dtype=torch.int64, device=device)
        rank_to_exp = torch.zeros(chunk_size, dtype=torch.int64, device=device)

        # Expert-major positions
        exp_major_pos = 0
        for exp in range(num_local_experts):
            for rank in range(ep_size):
                n_tok = chunk_token_counts[chunk_idx][rank][exp]
                for i in range(n_tok):
                    exp_major_pos += 1

        # Rank-major layout positions
        rank_major_starts = {}
        pos = 0
        for rank in range(ep_size):
            for exp in range(num_local_experts):
                rank_major_starts[(rank, exp)] = pos
                pos += chunk_token_counts[chunk_idx][rank][exp]

        # Build mappings
        exp_major_pos = 0
        for exp in range(num_local_experts):
            for rank in range(ep_size):
                n_tok = chunk_token_counts[chunk_idx][rank][exp]
                rank_major_start = rank_major_starts[(rank, exp)]
                for i in range(n_tok):
                    rank_major_pos = rank_major_start + i
                    exp_to_rank[exp_major_pos] = rank_major_pos
                    rank_to_exp[rank_major_pos] = exp_major_pos
                    exp_major_pos += 1

        chunk_expert_to_rank_indices.append(exp_to_rank)
        chunk_rank_to_expert_indices.append(rank_to_exp)

    return orig_to_chunk, chunk_to_orig, chunk_tokens_per_expert, chunk_splits, chunk_expert_to_rank_indices, chunk_rank_to_expert_indices


class FullChunkReorderContext:
    """
    Unified chunk reordering context for (rank + expert sorted) <-> chunk-sorted.

    Handles the full MoE layout where data is organized by source rank, with each
    rank's data internally sorted by expert.

    Original layout:
        [Rank0: E0_tokens, E1_tokens, ... | Rank1: E0_tokens, E1_tokens, ... | ...]

    Chunk-sorted layout (EXPERT-MAJOR within each chunk for grouped_gemm):
        [Chunk0: E0_R0_c0, E0_R1_c0, E1_R0_c0, E1_R1_c0, ...
         Chunk1: E0_R0_c1, E0_R1_c1, E1_R0_c1, E1_R1_c1, ...]

    Key features:
    - Expert-major layout within each chunk for grouped_gemm
    - After dX computation, use chunk_expert_to_rank() for rank-major AllToAll layout
    """

    def __init__(
        self,
        tokens_per_expert_per_rank: torch.Tensor,  # [ep_size, num_local_experts]
        num_chunks: int,
        device: torch.device,
    ):
        """
        Initialize chunk reorder context.

        Args:
            tokens_per_expert_per_rank: [ep_size, num_local_experts] token counts
            num_chunks: Number of chunks for pipeline
            device: Device for tensors
        """
        self.tokens_per_expert_per_rank = tokens_per_expert_per_rank
        self.num_chunks = num_chunks
        self.ep_size = tokens_per_expert_per_rank.shape[0]
        self.num_local_experts = tokens_per_expert_per_rank.shape[1]
        self.device = device

        # Compute output_splits (total tokens per rank)
        self.output_splits = tokens_per_expert_per_rank.sum(dim=1).tolist()
        self.total_tokens = sum(self.output_splits)

        # Compute indices
        (
            self.orig_to_chunk_indices,
            self.chunk_to_orig_indices,
            self.chunk_tokens_per_expert,
            self.chunk_splits,
            self._chunk_expert_to_rank_indices,
            self._chunk_rank_to_expert_indices,
        ) = compute_full_chunk_reorder_indices(
            tokens_per_expert_per_rank, num_chunks, device
        )

        # Pre-compute chunk boundaries
        self._chunk_boundaries = []
        pos = 0
        for chunk_idx in range(num_chunks):
            chunk_size = sum(self.chunk_splits[chunk_idx])
            self._chunk_boundaries.append((pos, pos + chunk_size))
            pos += chunk_size

    def to_chunk_sorted(self, data: torch.Tensor) -> torch.Tensor:
        """Convert data from original layout to chunk-sorted layout (expert-major within chunk)."""
        if self.total_tokens == 0:
            return data
        out = torch.empty_like(data)
        out.scatter_(0, self.orig_to_chunk_indices.unsqueeze(-1).expand_as(data), data)
        return out

    def to_original(self, data: torch.Tensor) -> torch.Tensor:
        """Convert data from chunk-sorted layout to original layout."""
        if self.total_tokens == 0:
            return data
        out = torch.empty_like(data)
        out.scatter_(0, self.chunk_to_orig_indices.unsqueeze(-1).expand_as(data), data)
        return out

    def get_chunk_data(self, chunk_sorted_data: torch.Tensor, chunk_idx: int) -> torch.Tensor:
        """Extract data for a specific chunk from chunk-sorted tensor."""
        start, end = self._chunk_boundaries[chunk_idx]
        return chunk_sorted_data[start:end]

    def get_chunk_tokens_per_expert(self, chunk_idx: int) -> torch.Tensor:
        """
        Get merged tokens_per_expert for a specific chunk.

        This is the sum of tokens across all ranks for each expert.
        Can be used directly with grouped_gemm.
        """
        return self.chunk_tokens_per_expert[chunk_idx]

    def get_chunk_splits(self, chunk_idx: int) -> List[int]:
        """Get output_splits for a specific chunk (tokens per rank in this chunk)."""
        return self.chunk_splits[chunk_idx]

    def get_chunk_size(self, chunk_idx: int) -> int:
        """Get total tokens in a specific chunk."""
        start, end = self._chunk_boundaries[chunk_idx]
        return end - start

    def get_chunk_boundaries(self, chunk_idx: int) -> Tuple[int, int]:
        """Get (start, end) boundaries for a chunk in chunk-sorted layout."""
        return self._chunk_boundaries[chunk_idx]

    def chunk_expert_to_rank(self, chunk_data: torch.Tensor, chunk_idx: int) -> torch.Tensor:
        """
        Reorder chunk data from expert-major to rank-major layout.

        Use this AFTER computing dX to prepare for AllToAll.
        Expert-major: [E0_R0, E0_R1, E1_R0, E1_R1, ...]
        Rank-major:   [R0_E0, R0_E1, R1_E0, R1_E1, ...]
        """
        indices = self._chunk_expert_to_rank_indices[chunk_idx]
        if len(indices) == 0:
            return chunk_data
        out = torch.empty_like(chunk_data)
        out.scatter_(0, indices.unsqueeze(-1).expand_as(chunk_data), chunk_data)
        return out

    def chunk_rank_to_expert(self, chunk_data: torch.Tensor, chunk_idx: int) -> torch.Tensor:
        """
        Reorder chunk data from rank-major to expert-major layout.

        Use this if you have rank-major data and need expert-major for grouped_gemm.
        Rank-major:   [R0_E0, R0_E1, R1_E0, R1_E1, ...]
        Expert-major: [E0_R0, E0_R1, E1_R0, E1_R1, ...]
        """
        indices = self._chunk_rank_to_expert_indices[chunk_idx]
        if len(indices) == 0:
            return chunk_data
        out = torch.empty_like(chunk_data)
        out.scatter_(0, indices.unsqueeze(-1).expand_as(chunk_data), chunk_data)
        return out


# Unit test
if __name__ == "__main__":
    print("Testing chunk reorder utilities...")

    # Test case: 2 experts, 4 tokens each, 2 chunks
    tokens_per_expert = torch.tensor([4, 4], dtype=torch.int32, device='cuda')
    num_chunks = 2

    # Create test data in expert-sorted order
    # E0: [0, 1, 2, 3], E1: [4, 5, 6, 7]
    data = torch.arange(8, dtype=torch.float32, device='cuda').unsqueeze(-1)
    print(f"Original (expert-sorted): {data.squeeze().tolist()}")

    # Compute indices
    e2c, c2e, chunk_tpe = compute_chunk_reorder_indices(tokens_per_expert, num_chunks)

    print(f"expert_to_chunk_indices: {e2c.tolist()}")
    print(f"chunk_to_expert_indices: {c2e.tolist()}")
    print(f"chunk_tokens_per_expert: {[t.tolist() for t in chunk_tpe]}")

    # Reorder to chunk-sorted
    chunk_sorted = reorder_expert_to_chunk(data, e2c)
    print(f"Chunk-sorted: {chunk_sorted.squeeze().tolist()}")
    # Expected: [0, 1, 4, 5, 2, 3, 6, 7]
    #           Chunk0: E0[0,1], E1[0,1] | Chunk1: E0[2,3], E1[2,3]

    # Reorder back to expert-sorted
    back_to_expert = reorder_chunk_to_expert(chunk_sorted, c2e)
    print(f"Back to expert-sorted: {back_to_expert.squeeze().tolist()}")

    # Verify round-trip
    assert torch.allclose(data, back_to_expert), "Round-trip failed!"
    print("Round-trip test passed!")

    # Test with uneven distribution
    print("\n--- Uneven distribution test ---")
    tokens_per_expert = torch.tensor([5, 3], dtype=torch.int32, device='cuda')
    num_chunks = 2

    data = torch.arange(8, dtype=torch.float32, device='cuda').unsqueeze(-1)
    print(f"Original: {data.squeeze().tolist()}")
    print(f"tokens_per_expert: {tokens_per_expert.tolist()}")

    e2c, c2e, chunk_tpe = compute_chunk_reorder_indices(tokens_per_expert, num_chunks)

    print(f"expert_to_chunk_indices: {e2c.tolist()}")
    print(f"chunk_to_expert_indices: {c2e.tolist()}")
    print(f"chunk_tokens_per_expert: {[t.tolist() for t in chunk_tpe]}")

    chunk_sorted = reorder_expert_to_chunk(data, e2c)
    print(f"Chunk-sorted: {chunk_sorted.squeeze().tolist()}")

    back_to_expert = reorder_chunk_to_expert(chunk_sorted, c2e)
    print(f"Back to expert-sorted: {back_to_expert.squeeze().tolist()}")

    assert torch.allclose(data, back_to_expert), "Round-trip failed for uneven case!"
    print("Uneven distribution test passed!")

    # Test ChunkReorderContext
    print("\n--- ChunkReorderContext test ---")
    ctx = ChunkReorderContext(tokens_per_expert, num_chunks)

    chunk_data = ctx.to_chunk_sorted(data)
    print(f"Via context - chunk-sorted: {chunk_data.squeeze().tolist()}")

    for i in range(num_chunks):
        chunk_slice = ctx.get_chunk_data(chunk_data, i)
        print(f"Chunk {i}: {chunk_slice.squeeze().tolist()}, tpe: {ctx.get_chunk_tokens_per_expert(i).tolist()}")

    print("\nAll tests passed!")

    # Test RankChunkReorderContext
    print("\n" + "="*60)
    print("Testing RankChunkReorderContext...")
    print("="*60)

    # Simulate EP=2, rank0 sends 4 tokens, rank1 sends 6 tokens
    output_splits = [4, 6]
    num_chunks = 2

    # Data in rank-sorted order: [R0_t0, R0_t1, R0_t2, R0_t3, R1_t0, R1_t1, R1_t2, R1_t3, R1_t4, R1_t5]
    data = torch.arange(10, dtype=torch.float32, device='cuda').unsqueeze(-1)
    print(f"Original (rank-sorted): {data.squeeze().tolist()}")
    print(f"output_splits: {output_splits}")

    ctx = RankChunkReorderContext(output_splits, num_chunks, device='cuda')

    print(f"\nChunk splits:")
    for i in range(num_chunks):
        print(f"  Chunk {i}: {ctx.get_chunk_splits(i)} (total: {ctx.get_chunk_size(i)})")

    chunk_sorted = ctx.to_chunk_sorted(data)
    print(f"\nChunk-sorted: {chunk_sorted.squeeze().tolist()}")
    # Expected:
    # Chunk 0: R0[0,1] + R1[0,1,2] = [0, 1, 4, 5, 6]
    # Chunk 1: R0[2,3] + R1[3,4,5] = [2, 3, 7, 8, 9]

    for i in range(num_chunks):
        chunk_data = ctx.get_chunk_data(chunk_sorted, i)
        print(f"  Chunk {i}: {chunk_data.squeeze().tolist()}")

    # Round-trip test
    back_to_rank = ctx.to_rank_sorted(chunk_sorted)
    print(f"\nBack to rank-sorted: {back_to_rank.squeeze().tolist()}")

    assert torch.allclose(data, back_to_rank), "Round-trip failed!"
    print("Round-trip test passed!")

    # Test scatter_chunk_to_rank
    print("\n--- Testing scatter_chunk_to_rank ---")
    output = torch.zeros_like(data)
    for i in range(num_chunks):
        chunk_data = ctx.get_chunk_data(chunk_sorted, i)
        ctx.scatter_chunk_to_rank(chunk_data, i, output)
        print(f"After scattering chunk {i}: {output.squeeze().tolist()}")

    assert torch.allclose(data, output), "Scatter test failed!"
    print("Scatter test passed!")

    print("\n" + "="*60)
    print("All RankChunkReorderContext tests passed!")
    print("="*60)

    # Test FullChunkReorderContext (multi-expert case)
    print("\n" + "="*60)
    print("Testing FullChunkReorderContext (multi-expert)...")
    print("="*60)

    # Simulate EP=2, 2 local experts per device
    # tokens_per_expert_per_rank[rank][expert]
    # Rank0: E0=3, E1=2
    # Rank1: E0=4, E1=3
    tokens_per_expert_per_rank = torch.tensor(
        [[3, 2], [4, 3]],
        dtype=torch.int32,
        device='cuda'
    )
    num_chunks = 2

    # Data layout in original order:
    # [R0_E0: 0,1,2 | R0_E1: 3,4 | R1_E0: 5,6,7,8 | R1_E1: 9,10,11]
    total_tokens = tokens_per_expert_per_rank.sum().item()
    data = torch.arange(total_tokens, dtype=torch.float32, device='cuda').unsqueeze(-1)
    print(f"Original layout: {data.squeeze().tolist()}")
    print(f"tokens_per_expert_per_rank:\n{tokens_per_expert_per_rank}")

    ctx = FullChunkReorderContext(tokens_per_expert_per_rank, num_chunks, device='cuda')

    print(f"\nOutput splits (tokens per rank): {ctx.output_splits}")
    print(f"\nChunk info:")
    for i in range(num_chunks):
        print(f"  Chunk {i}:")
        print(f"    - splits (tokens per rank): {ctx.get_chunk_splits(i)}")
        print(f"    - merged tokens_per_expert: {ctx.get_chunk_tokens_per_expert(i).tolist()}")
        print(f"    - total size: {ctx.get_chunk_size(i)}")

    # Reorder to chunk-sorted
    chunk_sorted = ctx.to_chunk_sorted(data)
    print(f"\nChunk-sorted: {chunk_sorted.squeeze().tolist()}")

    for i in range(num_chunks):
        chunk_data = ctx.get_chunk_data(chunk_sorted, i)
        print(f"  Chunk {i}: {chunk_data.squeeze().tolist()}")

    # Round-trip test
    back_to_orig = ctx.to_original(chunk_sorted)
    print(f"\nBack to original: {back_to_orig.squeeze().tolist()}")

    diff = torch.abs(data - back_to_orig).max().item()
    print(f"Max diff: {diff}")
    assert diff < 1e-6, "Round-trip failed!"
    print("Round-trip test passed!")

    # Verify that merged tokens_per_expert is correct for grouped_gemm
    print("\n--- Verifying merged tokens_per_expert ---")
    for chunk_idx in range(num_chunks):
        chunk_tpe = ctx.get_chunk_tokens_per_expert(chunk_idx)
        chunk_size = ctx.get_chunk_size(chunk_idx)
        expected_size = chunk_tpe.sum().item()
        print(f"Chunk {chunk_idx}: tokens_per_expert sum = {expected_size}, chunk_size = {chunk_size}")
        assert chunk_size == expected_size, f"Chunk {chunk_idx} size mismatch!"

    print("\n" + "="*60)
    print("All FullChunkReorderContext tests passed!")
    print("="*60)
