"""
Communication Primitives and P2P Overlap Management

This module provides:
1. AllToAll communication primitives (for MoE and Attention)
2. Round-Robin Tournament scheduling for P2P communication
3. CUDA resources (streams, events) for communication overlap

Used by both MoE and Attention layers for forward/backward communication.

Round-Robin Tournament ensures:
- Each rank communicates with exactly one peer per round
- All pairs communicate exactly once over all rounds
- No communication conflicts within a round

For P ranks (even): P-1 rounds
For P ranks (odd): P rounds (with dummy, some ranks idle)

Example for 4 ranks:
    Round 0: (0,3), (1,2)
    Round 1: (0,2), (1,3)
    Round 2: (0,1), (2,3)
"""

import os
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional


# =============================================================================
# AllToAll Communication Primitives
# =============================================================================

def _all_to_all(
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]],
    input_split_sizes: Optional[List[int]],
    group,
    debug_tag: str = "",
) -> torch.Tensor:
    """
    Direct call to PyTorch all_to_all_single (bypass Megatron wrapper)

    Args:
        input: Input tensor
        output_split_sizes: Size of each output chunk (None for equal split)
        input_split_sizes: Size of each input chunk (None for equal split)
        group: Process group

    Returns:
        Output tensor after all-to-all
    """
    world_size = group.size()
    if world_size == 1:
        return input.clone()

    nccl_stream = torch.cuda.current_stream()
    # The source tensor may be repacked on the current comm stream before NCCL
    # consumes it. Keep its storage alive until this stream reaches that point.
    input.record_stream(nccl_stream)
    input = input.contiguous()

    if output_split_sizes is None:
        # Equal split
        output = torch.empty_like(input)
    else:
        # Unequal split (all2all-v)
        output = input.new_empty(
            size=[sum(output_split_sizes)] + list(input.size()[1:]),
            dtype=input.dtype,
            device=input.device,
        )

    _debug_check_alltoallv_splits(
        input_rows=int(input.size(0)),
        output_rows=int(output.size(0)),
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=group,
        tag=debug_tag,
    )

    # Record NCCL stream usage to prevent the caching allocator from
    # reusing input/output memory before the NCCL kernel finishes.
    input.record_stream(nccl_stream)
    output.record_stream(nccl_stream)

    dist.all_to_all_single(
        output, input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def _debug_check_alltoallv_splits(
    input_rows: int,
    output_rows: int,
    input_split_sizes: Optional[List[int]],
    output_split_sizes: Optional[List[int]],
    group,
    tag: str,
) -> None:
    """Debug-only consistency checks for AllToAll-v split matrices.

    Enabled when FLUID_DEBUG_A2A_SPLITS=1.
    """
    if os.environ.get("FLUID_DEBUG_A2A_SPLITS", "0") != "1":
        return
    if input_split_sizes is None or output_split_sizes is None:
        return

    rank = dist.get_rank(group)
    world = group.size()

    in_splits = [int(x) for x in input_split_sizes]
    out_splits = [int(x) for x in output_split_sizes]

    if len(in_splits) != world or len(out_splits) != world:
        raise RuntimeError(
            f"[A2A:{tag}] rank{rank}: split len mismatch "
            f"in={len(in_splits)} out={len(out_splits)} world={world}"
        )
    if sum(in_splits) != input_rows:
        raise RuntimeError(
            f"[A2A:{tag}] rank{rank}: sum(input_split_sizes)={sum(in_splits)} "
            f"!= input_rows={input_rows}"
        )
    if sum(out_splits) != output_rows:
        raise RuntimeError(
            f"[A2A:{tag}] rank{rank}: sum(output_split_sizes)={sum(out_splits)} "
            f"!= output_rows={output_rows}"
        )

    in_mat = [None for _ in range(world)]
    out_mat = [None for _ in range(world)]
    dist.all_gather_object(in_mat, in_splits, group=group)
    dist.all_gather_object(out_mat, out_splits, group=group)

    mismatch = None
    for src in range(world):
        for dst in range(world):
            send = int(in_mat[src][dst])
            recv = int(out_mat[dst][src])
            if send != recv:
                mismatch = (src, dst, send, recv)
                break
        if mismatch is not None:
            break

    if mismatch is not None:
        src, dst, send, recv = mismatch
        raise RuntimeError(
            f"[A2A:{tag}] split matrix mismatch: "
            f"send[{src}->{dst}]={send} != recv[{dst}<-{src}]={recv}; "
            f"rank={rank}, in={in_splits}, out={out_splits}"
        )


def _all_to_all_sp2hp_forward(input_: torch.Tensor, group, output: torch.Tensor = None) -> torch.Tensor:
    """
    Forward-only sp2hp AllToAll (no autograd).
    Used in Attention for sequence parallel to head parallel conversion.

    Shape: [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim]

    Args:
        input_: Input tensor [seq_local, batch, heads, dim]
        group: Context parallel process group
        output: Optional pre-allocated output [seq_full, batch, heads_local, dim].
                If provided, AllToAll writes directly into it (no extra copy).

    Returns:
        Output tensor [seq_full, batch, heads_local, dim]
    """
    cp = group.size()
    seq_local, batch, heads, dim = input_.shape
    nccl_stream = torch.cuda.current_stream()
    # sp2hp does a permute+contiguous pack on the current comm stream before
    # launching NCCL. Protect the source storage across that cross-stream read.
    input_.record_stream(nccl_stream)

    # Rearrange: split heads, move CP to front, flatten (ensure contiguous)
    x = input_.contiguous().view(seq_local, batch, cp, heads // cp, dim)
    x = x.permute(2, 0, 1, 3, 4).contiguous()
    x = x.view(seq_local * cp, -1)

    # AllToAll communication (no grad)
    # sp2hp: AllToAll output is [seq_full, batch * heads_local * dim] which
    # reshapes directly to [seq_full, batch, heads_local, dim] — no permute needed,
    # so we can write directly into pre-allocated output.
    if output is not None:
        out_buf = output.view(seq_local * cp, -1)
    else:
        out_buf = torch.empty_like(x)
    x.record_stream(nccl_stream)
    out_buf.record_stream(nccl_stream)
    dist.all_to_all_single(
        out_buf, x,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        group=group,
    )

    if output is not None:
        return output
    return out_buf.view(seq_local * cp, batch, heads // cp, dim)


def _all_to_all_hp2sp_forward(input_: torch.Tensor, group, output: torch.Tensor = None) -> torch.Tensor:
    """
    Forward-only hp2sp AllToAll (no autograd).
    Used in Attention for head parallel to sequence parallel conversion.

    Shape: [seq, batch, heads/CP, dim] -> [seq/CP, batch, heads, dim]

    Args:
        input_: Input tensor [seq_full, batch, heads_local, dim]
        group: Context parallel process group
        output: Optional pre-allocated output [seq_local, batch, heads, dim].
                If provided, permute-copies directly into it (no extra allocation).

    Returns:
        Output tensor [seq_local, batch, heads, dim]
    """
    cp = group.size()
    seq, batch, heads_local, dim = input_.shape
    seq_local = seq // cp
    nccl_stream = torch.cuda.current_stream()
    # hp2sp may materialize a contiguous input view on the current comm stream
    # before NCCL runs. Protect the source storage across that handoff.
    input_.record_stream(nccl_stream)

    # Flatten to 2D (ensure contiguous for view)
    x = input_.contiguous().view(seq, batch * heads_local * dim)

    # AllToAll communication (no grad)
    raw_out = torch.empty_like(x)
    x.record_stream(nccl_stream)
    raw_out.record_stream(nccl_stream)
    dist.all_to_all_single(
        raw_out, x,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        group=group,
    )

    # Rearrange: unflatten, permute, merge heads
    # hp2sp requires a permute after AllToAll, so we permute-copy into output
    raw_5d = raw_out.view(cp, seq_local, batch, heads_local, dim)
    if output is not None:
        output.view(seq_local, batch, cp, heads_local, dim).copy_(
            raw_5d.permute(1, 2, 0, 3, 4))
        return output
    result = raw_5d.permute(1, 2, 0, 3, 4).contiguous()
    return result.view(seq_local, batch, heads_local * cp, dim)


# =============================================================================
# Round-Robin Tournament Scheduling
# =============================================================================

def compute_round_robin_schedule(num_ranks: int) -> List[List[Tuple[int, int]]]:
    """
    Compute Round-Robin Tournament schedule.

    For P participants, needs P-1 rounds (P even) or P rounds (P odd, add dummy).
    Each round, each participant pairs with exactly one other.

    Args:
        num_ranks: Number of participants

    Returns:
        schedule: List[List[Tuple[int, int]]]
            schedule[round_idx] = [(rank_a, rank_b), ...] pairs for that round
            If a rank pairs with dummy, it idles that round
    """
    P = num_ranks
    is_odd = (P % 2 == 1)
    if is_odd:
        P += 1  # Add dummy

    num_rounds = P - 1
    schedule = []

    # Initialize participants
    # Using standard circle method: fix one position, rotate others
    participants = list(range(P))

    for round_idx in range(num_rounds):
        pairs = []
        # Pairing: participants[0] vs participants[P-1]
        #          participants[1] vs participants[P-2]
        #          ...
        for i in range(P // 2):
            a = participants[i]
            b = participants[P - 1 - i]
            # Skip dummy (idx = num_ranks)
            if is_odd and (a == num_ranks or b == num_ranks):
                continue
            # Ensure a < b for consistency
            if a > b:
                a, b = b, a
            pairs.append((a, b))
        schedule.append(pairs)

        # Rotate: fix participants[0], counter-clockwise rotate others
        # [0, 1, 2, 3, 4, 5] -> [0, 5, 1, 2, 3, 4]
        new_participants = [participants[0]]
        new_participants.append(participants[-1])
        new_participants.extend(participants[1:-1])
        participants = new_participants

    return schedule


# =============================================================================
# Multi-Card Overlap Context
# =============================================================================

class MultiCardOverlapContext:
    """CUDA resources for multi-card P2P communication overlap.

    Unified manager for MoE and Attention overlap resources:
    - MoE: multi-round dispatch/combine P2P
    - Attention: QKV sp2hp and hp2sp output projection P2P

    Uses a global StreamManager so forward and backward share the same stream.
    """

    def __init__(self, device: torch.device, ep_size: int, cp_size: int = None):
        """
        Args:
            device: CUDA device
            ep_size: Expert Parallel size (MoE communication)
            cp_size: Context Parallel size (Attention; defaults to ep_size)
        """
        from fluid.core.stream import get_stream_manager

        self.device = device
        self.ep_size = ep_size
        self.cp_size = cp_size if cp_size is not None else ep_size
        self.num_rounds = ep_size - 1 if ep_size % 2 == 0 else ep_size

        # Shared stream across forward and backward.
        self._stream_manager = get_stream_manager()
        self._stream_manager.initialize(device)

        # Precomputed round-robin schedule.
        self.schedule = compute_round_robin_schedule(ep_size)

        # Per-(rank, round) partner cache.
        self._partner_cache = {}

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        """Communication stream (from StreamManager)."""
        return self._stream_manager.comm_stream

    @property
    def data_ready_event(self) -> torch.cuda.Event:
        """Data-ready event (from StreamManager)."""
        return self._stream_manager.data_ready_event

    def get_partner(self, my_rank: int, round_idx: int) -> int:
        """Partner rank for the given round (reads cached self.schedule, memoized per (rank, round))."""
        cache_key = (my_rank, round_idx)
        cached = self._partner_cache.get(cache_key)
        if cached is not None:
            return cached
        if round_idx >= len(self.schedule):
            partner = -1
        else:
            partner = -1
            for a, b in self.schedule[round_idx]:
                if a == my_rank:
                    partner = b
                    break
                if b == my_rank:
                    partner = a
                    break
        self._partner_cache[cache_key] = partner
        return partner

    def get_stream(self) -> torch.cuda.Stream:
        return self.comm_stream

    def get_round_event(self, tag: str, round_idx: int) -> torch.cuda.Event:
        """Get a reusable synchronization event for a specific overlap round."""
        return self._stream_manager.get_sync_event(("overlap", tag, round_idx))


__all__ = [
    # AllToAll primitives
    '_all_to_all',
    '_all_to_all_sp2hp_forward',
    '_all_to_all_hp2sp_forward',
    # Overlap Context
    'MultiCardOverlapContext',
]
