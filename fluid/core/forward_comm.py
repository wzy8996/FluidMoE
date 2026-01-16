"""
Forward Communication Management - P2P Scheduling and Overlap Context

This module manages forward pass P2P communication for multi-card overlap:
1. Round-Robin Tournament scheduling for P2P communication
2. CUDA resources (streams, events) for communication overlap

Used by both MoE and Attention layers for forward P2P overlap.

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
from typing import List, Tuple


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

        # =================================================================
        # MoE 相关 events
        # =================================================================
        # 每轮的同步events
        # round_events[r] 表示第r轮通信完成的event
        self.round_events = [torch.cuda.Event() for _ in range(self.num_rounds)]

        # 数据准备好的event（用于触发通信）
        self.data_ready_event = torch.cuda.Event()

        # MoE dispatch/combine events（兼容旧接口）
        self.dispatch_event = torch.cuda.Event()
        self.combine_event = torch.cuda.Event()

        # =================================================================
        # Attention 相关 events (Ulysses SP / Context Parallel)
        # =================================================================
        # QKV sp2hp 相关 events
        self.qkv_ready_event = torch.cuda.Event()
        self.qkv_comm_done_event = torch.cuda.Event()

        # Output projection hp2sp 相关 events
        self.proj_ready_event = torch.cuda.Event()
        self.proj_comm_done_event = torch.cuda.Event()

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

    def get_round_event(self, round_idx: int) -> torch.cuda.Event:
        return self.round_events[round_idx]

    # =================================================================
    # 兼容 OverlapContext 的接口
    # =================================================================
    def get_events(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """获取 MoE 相关 events（兼容旧接口）"""
        return self.dispatch_event, self.combine_event

    def get_qkv_events(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """获取 QKV 相关 events"""
        return self.qkv_ready_event, self.qkv_comm_done_event

    def get_proj_events(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """获取 output projection 相关 events"""
        return self.proj_ready_event, self.proj_comm_done_event


class AttentionMultiCardOverlapContext:
    """注意力层多卡P2P通信的上下文管理（轻量级，仅用于Attention）"""

    def __init__(self, device: torch.device, cp_size: int):
        self.device = device
        self.cp_size = cp_size
        self.num_rounds = cp_size - 1 if cp_size % 2 == 0 else cp_size

        # 通信流
        self.comm_stream = torch.cuda.Stream(device=device)

        # 每轮的同步events
        self.round_events = [torch.cuda.Event() for _ in range(self.num_rounds)]

        # QKV相关events
        self.qkv_ready_event = torch.cuda.Event()
        self.qkv_comm_done_event = torch.cuda.Event()

        # 输出投影相关events
        self.proj_ready_event = torch.cuda.Event()
        self.proj_comm_done_event = torch.cuda.Event()

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

    def get_round_event(self, round_idx: int) -> torch.cuda.Event:
        return self.round_events[round_idx]

    def get_qkv_events(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        return self.qkv_ready_event, self.qkv_comm_done_event

    def get_proj_events(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        return self.proj_ready_event, self.proj_comm_done_event


__all__ = [
    # Scheduling
    'compute_round_robin_schedule',
    'get_partner_for_round',
    'get_all_partners_ordered',
    'get_num_rounds',
    # Context
    'MultiCardOverlapContext',
    'AttentionMultiCardOverlapContext',
]
