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
    group
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

    dist.all_to_all_single(
        output, input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def _all_to_all_sp2hp_forward(input_: torch.Tensor, group) -> torch.Tensor:
    """
    Forward-only sp2hp AllToAll (no autograd).
    Used in Attention for sequence parallel to head parallel conversion.

    Shape: [seq/CP, batch, heads, dim] -> [seq, batch, heads/CP, dim]

    Args:
        input_: Input tensor [seq_local, batch, heads, dim]
        group: Context parallel process group

    Returns:
        Output tensor [seq_full, batch, heads_local, dim]
    """
    cp = group.size()
    seq_local, batch, heads, dim = input_.shape

    # Rearrange: split heads, move CP to front, flatten (ensure contiguous)
    x = input_.contiguous().view(seq_local, batch, cp, heads // cp, dim)
    x = x.permute(2, 0, 1, 3, 4).contiguous()
    x = x.view(seq_local * cp, -1)

    # AllToAll communication (no grad)
    output = torch.empty_like(x)
    dist.all_to_all_single(
        output, x,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        group=group,
    )

    # Reshape to output format
    return output.view(seq_local * cp, batch, heads // cp, dim)


def _all_to_all_hp2sp_forward(input_: torch.Tensor, group) -> torch.Tensor:
    """
    Forward-only hp2sp AllToAll (no autograd).
    Used in Attention for head parallel to sequence parallel conversion.

    Shape: [seq, batch, heads/CP, dim] -> [seq/CP, batch, heads, dim]

    Args:
        input_: Input tensor [seq_full, batch, heads_local, dim]
        group: Context parallel process group

    Returns:
        Output tensor [seq_local, batch, heads, dim]
    """
    cp = group.size()
    seq, batch, heads_local, dim = input_.shape
    seq_local = seq // cp

    # Flatten to 2D (ensure contiguous for view)
    x = input_.contiguous().view(seq, batch * heads_local * dim)

    # AllToAll communication (no grad)
    output = torch.empty_like(x)
    dist.all_to_all_single(
        output, x,
        output_split_sizes=[seq_local] * cp,
        input_split_sizes=[seq_local] * cp,
        group=group,
    )

    # Rearrange: unflatten, permute, merge heads
    output = output.view(cp, seq_local, batch, heads_local, dim)
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    return output.view(seq_local, batch, heads_local * cp, dim)


def _sort_chunks_by_idxs(input_tensor, split_sizes, sorted_idxs):
    """
    Sort chunks of input tensor by indices.
    Used in MoE for reordering tokens between rank-major and expert-major layouts.

    Args:
        input_tensor: [total_tokens, hidden] tensor
        split_sizes: list or tensor of chunk sizes (can contain zeros)
        sorted_idxs: list or tensor of new order indices

    Returns:
        Reordered tensor
    """
    if input_tensor.numel() == 0:
        return input_tensor

    # Convert to list (avoid GPU sync)
    if torch.is_tensor(split_sizes):
        split_sizes = split_sizes.tolist()
    if torch.is_tensor(sorted_idxs):
        sorted_idxs = sorted_idxs.tolist()

    # Direct split and cat (Megatron style, simple and efficient)
    chunks = torch.split(input_tensor, split_sizes, dim=0)
    return torch.cat([chunks[i] for i in sorted_idxs], dim=0)


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


def get_partner_for_round(my_rank: int, round_idx: int, num_ranks: int) -> int:
    """
    Get the partner rank for a specific round.

    Args:
        my_rank: Current rank
        round_idx: Round index
        num_ranks: Total number of ranks

    Returns:
        partner_rank: Partner rank, -1 if idling this round
    """
    schedule = compute_round_robin_schedule(num_ranks)
    if round_idx >= len(schedule):
        return -1

    for a, b in schedule[round_idx]:
        if a == my_rank:
            return b
        if b == my_rank:
            return a
    return -1  # Idle this round


def get_all_partners_ordered(my_rank: int, num_ranks: int) -> List[Tuple[int, int]]:
    """
    Get all partners in order of communication rounds.

    Args:
        my_rank: Current rank
        num_ranks: Total number of ranks

    Returns:
        partners: List[(round_idx, partner_rank)]
            Ordered by round, -1 partner means idle
    """
    schedule = compute_round_robin_schedule(num_ranks)
    partners = []

    for round_idx, pairs in enumerate(schedule):
        found = False
        for a, b in pairs:
            if a == my_rank:
                partners.append((round_idx, b))
                found = True
                break
            if b == my_rank:
                partners.append((round_idx, a))
                found = True
                break
        if not found:
            partners.append((round_idx, -1))  # Idle

    return partners


def get_num_rounds(num_ranks: int) -> int:
    """Get number of rounds needed for complete exchange."""
    if num_ranks <= 1:
        return 0
    if num_ranks % 2 == 0:
        return num_ranks - 1
    return num_ranks  # Odd case with dummy


# =============================================================================
# Multi-Card Overlap Context
# =============================================================================

class MultiCardOverlapContext:
    """管理多卡P2P通信重叠所需的CUDA资源

    统一管理 MoE 层和 Attention 层的通信重叠资源：
    - MoE: dispatch/combine 的多轮 P2P 通信
    - Attention: QKV sp2hp 和 hp2sp output projection 的 P2P 通信
    """

    def __init__(self, device: torch.device, ep_size: int, cp_size: int = None):
        """
        Args:
            device: CUDA 设备
            ep_size: Expert Parallel size (MoE 通信)
            cp_size: Context Parallel size (Attention 通信，默认等于 ep_size)
        """
        self.device = device
        self.ep_size = ep_size
        self.cp_size = cp_size if cp_size is not None else ep_size
        self.num_rounds = ep_size - 1 if ep_size % 2 == 0 else ep_size

        # 通信流（MoE 和 Attention 共用）
        self.comm_stream = torch.cuda.Stream(device=device)

        # 复用单个Event，用于计算-通信同步
        self.data_ready_event = torch.cuda.Event()

        # 预计算调度表
        self.schedule = compute_round_robin_schedule(ep_size)

        # 缓存：my_rank的每轮partner
        self._partner_cache = {}

    def get_partner(self, my_rank: int, round_idx: int) -> int:
        """获取指定轮次的partner（带缓存）"""
        cache_key = (my_rank, round_idx)
        if cache_key not in self._partner_cache:
            self._partner_cache[cache_key] = get_partner_for_round(my_rank, round_idx, self.ep_size)
        return self._partner_cache[cache_key]

    def get_stream(self) -> torch.cuda.Stream:
        return self.comm_stream


class AttentionMultiCardOverlapContext:
    """注意力层多卡P2P通信的上下文管理（轻量级，仅用于Attention）"""

    def __init__(self, device: torch.device, cp_size: int):
        self.device = device
        self.cp_size = cp_size
        self.num_rounds = cp_size - 1 if cp_size % 2 == 0 else cp_size

        # 通信流
        self.comm_stream = torch.cuda.Stream(device=device)

        # 复用单个Event，用于计算-通信同步
        self.data_ready_event = torch.cuda.Event()

        # 预计算调度表
        self.schedule = compute_round_robin_schedule(cp_size)
        self._partner_cache = {}

    def get_partner(self, my_rank: int, round_idx: int) -> int:
        cache_key = (my_rank, round_idx)
        if cache_key not in self._partner_cache:
            self._partner_cache[cache_key] = get_partner_for_round(my_rank, round_idx, self.cp_size)
        return self._partner_cache[cache_key]

    def get_stream(self) -> torch.cuda.Stream:
        return self.comm_stream


__all__ = [
    # AllToAll primitives
    '_all_to_all',
    '_all_to_all_sp2hp_forward',
    '_all_to_all_hp2sp_forward',
    '_sort_chunks_by_idxs',
    # P2P Scheduling
    'compute_round_robin_schedule',
    'get_partner_for_round',
    'get_all_partners_ordered',
    'get_num_rounds',
    # Overlap Context
    'MultiCardOverlapContext',
    'AttentionMultiCardOverlapContext',
]
