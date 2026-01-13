"""
多卡P2P通信重叠实现

设计原则：
1. 使用Round-Robin Tournament调度，将"每个rank和所有其他rank交换数据"拆成多轮
2. 每轮每张卡只和一个peer通信，避免冲突
3. 通信流在跑第r轮的同时，计算流在吃第r-1轮的数据
4. 注意力层通信规则（Ulysses SP），专家层通信不规则（EP+路由）

多轮调度：
- 若卡数P为偶数：总轮数R = P-1
- 若P为奇数：加一个dummy变偶数，轮数R = P，有的轮某些rank轮空
- 每轮结束就是"这个peer的数据ready"时刻，可以立刻触发计算
"""

import os
import torch
import torch.distributed as dist
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass


# =============================================================================
# Round-Robin Tournament 调度算法
# =============================================================================

def compute_round_robin_schedule(num_ranks: int) -> List[List[Tuple[int, int]]]:
    """
    计算Round-Robin Tournament调度表

    对于P个参与者，需要P-1轮（P为偶数）或P轮（P为奇数，添加dummy）
    每轮中，每个参与者恰好与一个其他参与者配对

    Args:
        num_ranks: 参与者数量

    Returns:
        schedule: List[List[Tuple[int, int]]]
            schedule[round_idx] = [(rank_a, rank_b), ...] 表示第round_idx轮的配对
            如果某个rank与dummy配对，则该轮它轮空
    """
    # 如果是奇数，添加一个dummy（-1表示）
    P = num_ranks
    is_odd = (P % 2 == 1)
    if is_odd:
        P += 1  # 添加dummy

    num_rounds = P - 1
    schedule = []

    # 初始化参与者列表
    # 使用标准的circle method：固定一个位置，其他位置轮转
    participants = list(range(P))  # 0, 1, 2, ..., P-1

    for round_idx in range(num_rounds):
        pairs = []
        # 配对：participants[0] vs participants[P-1]
        #       participants[1] vs participants[P-2]
        #       ...
        for i in range(P // 2):
            a = participants[i]
            b = participants[P - 1 - i]
            # 如果是dummy（idx = num_ranks），跳过
            if is_odd and (a == num_ranks or b == num_ranks):
                continue  # 这轮这个rank轮空
            # 确保 a < b 以便统一（可选）
            if a > b:
                a, b = b, a
            pairs.append((a, b))
        schedule.append(pairs)

        # 轮转：固定participants[0]，其他位置逆时针轮转
        # [0, 1, 2, 3, 4, 5] -> [0, 5, 1, 2, 3, 4]
        new_participants = [participants[0]]
        new_participants.append(participants[-1])
        new_participants.extend(participants[1:-1])
        participants = new_participants

    return schedule


def get_partner_for_round(my_rank: int, round_idx: int, num_ranks: int) -> int:
    """
    获取指定轮次中my_rank的配对partner

    Args:
        my_rank: 当前rank
        round_idx: 轮次索引
        num_ranks: 总rank数

    Returns:
        partner_rank: 配对的rank，-1表示轮空
    """
    schedule = compute_round_robin_schedule(num_ranks)
    if round_idx >= len(schedule):
        return -1

    for a, b in schedule[round_idx]:
        if a == my_rank:
            return b
        if b == my_rank:
            return a

    return -1  # 轮空


def get_all_partners_ordered(my_rank: int, num_ranks: int) -> List[Tuple[int, int]]:
    """
    获取所有轮次中my_rank的partner列表（按轮次顺序）

    Args:
        my_rank: 当前rank
        num_ranks: 总rank数

    Returns:
        partners: List[(round_idx, partner_rank)]，-1表示轮空
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
            partners.append((round_idx, -1))  # 轮空

    return partners


# =============================================================================
# 多卡P2P通信上下文
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


# =============================================================================
# MoE层多卡P2P前向重叠（不规则通信）
# =============================================================================

@dataclass
class MoEP2PRoundData:
    """每轮P2P通信的数据"""
    send_tokens: torch.Tensor  # 发送给partner的tokens
    recv_tokens: torch.Tensor  # 从partner接收的tokens
    send_count: int
    recv_count: int
    partner_rank: int


def moe_multicard_p2p_dispatch(
    permuted_tokens: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    ep_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    compute_fn_per_round: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, List[MoEP2PRoundData]]:
    """
    MoE多卡P2P Dispatch（前向）

    将tokens分发到各个expert所在的rank
    使用多轮P2P通信，每轮与一个partner交换数据

    Args:
        permuted_tokens: [num_tokens, hidden] 已按expert排序的tokens
        input_splits: [ep_size] 每个rank发送的token数
        output_splits: [ep_size] 从每个rank接收的token数
        ep_group: Expert Parallel进程组
        overlap_ctx: 多卡overlap上下文
        compute_fn_per_round: 可选的每轮计算函数，用于在等待通信时执行计算
            compute_fn_per_round(round_idx, received_data) -> processed_data

    Returns:
        all_received_tokens: [total_recv, hidden] 接收到的所有远程tokens
        round_data_list: 每轮的数据信息（用于combine阶段）
    """
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    device = permuted_tokens.device
    hidden_size = permuted_tokens.shape[-1]
    dtype = permuted_tokens.dtype

    input_splits_list = input_splits.tolist() if torch.is_tensor(input_splits) else list(input_splits)
    output_splits_list = output_splits.tolist() if torch.is_tensor(output_splits) else list(output_splits)

    # 计算input_offsets（permuted_tokens中各rank数据的起始位置）
    input_offsets = [0]
    for s in input_splits_list:
        input_offsets.append(input_offsets[-1] + s)

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()
    num_rounds = overlap_ctx.num_rounds

    # 结果存储
    round_data_list = []
    all_received = []

    # 记录数据准备好
    overlap_ctx.data_ready_event.record(default_stream)

    # 多轮P2P通信
    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)

        if partner == -1:
            # 轮空，跳过
            round_data_list.append(MoEP2PRoundData(
                send_tokens=None,
                recv_tokens=None,
                send_count=0,
                recv_count=0,
                partner_rank=-1,
            ))
            continue

        # 准备发送和接收数据
        send_count = input_splits_list[partner]
        recv_count = output_splits_list[partner]

        # 发送数据：permuted_tokens中发往partner的部分
        if send_count > 0:
            send_start = input_offsets[partner]
            send_tokens = permuted_tokens[send_start:send_start + send_count].contiguous()
        else:
            send_tokens = torch.empty(0, hidden_size, dtype=dtype, device=device)

        # 接收缓冲区
        recv_tokens = torch.empty(recv_count, hidden_size, dtype=dtype, device=device) if recv_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)

        # 启动P2P通信
        comm_stream.wait_event(overlap_ctx.data_ready_event)

        with torch.cuda.stream(comm_stream):
            p2p_ops = []
            if recv_count > 0:
                p2p_ops.append(dist.P2POp(dist.irecv, recv_tokens, partner, group=ep_group))
            if send_count > 0:
                p2p_ops.append(dist.P2POp(dist.isend, send_tokens, partner, group=ep_group))

            if p2p_ops:
                reqs = dist.batch_isend_irecv(p2p_ops)
                for req in reqs:
                    req.wait()

        # 记录本轮通信完成
        round_event = overlap_ctx.get_round_event(round_idx)
        round_event.record(comm_stream)

        # 如果有上一轮的数据，可以在等待本轮通信时处理
        if round_idx > 0 and compute_fn_per_round is not None:
            prev_round_data = round_data_list[round_idx - 1]
            if prev_round_data.recv_tokens is not None and prev_round_data.recv_count > 0:
                # 等待上一轮通信完成
                default_stream.wait_event(overlap_ctx.get_round_event(round_idx - 1))
                # 执行计算
                compute_fn_per_round(round_idx - 1, prev_round_data.recv_tokens)

        round_data_list.append(MoEP2PRoundData(
            send_tokens=send_tokens,
            recv_tokens=recv_tokens,
            send_count=send_count,
            recv_count=recv_count,
            partner_rank=partner,
        ))

        if recv_count > 0:
            all_received.append(recv_tokens)

    # 等待最后一轮通信完成
    if num_rounds > 0:
        default_stream.wait_event(overlap_ctx.get_round_event(num_rounds - 1))

    # 处理最后一轮的数据
    if compute_fn_per_round is not None and len(round_data_list) > 0:
        last_round_data = round_data_list[-1]
        if last_round_data.recv_tokens is not None and last_round_data.recv_count > 0:
            compute_fn_per_round(num_rounds - 1, last_round_data.recv_tokens)

    # 合并所有接收到的数据
    if all_received:
        all_received_tokens = torch.cat(all_received, dim=0)
    else:
        all_received_tokens = torch.empty(0, hidden_size, dtype=dtype, device=device)

    return all_received_tokens, round_data_list


def moe_multicard_p2p_combine(
    expert_output: torch.Tensor,
    round_data_list: List[MoEP2PRoundData],
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    ep_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
) -> torch.Tensor:
    """
    MoE多卡P2P Combine（前向）

    将expert计算结果发送回原始rank
    使用多轮P2P通信，顺序与dispatch相反

    Args:
        expert_output: [total_tokens, hidden] 所有expert的输出
        round_data_list: dispatch阶段的轮次数据（用于确定发送目标）
        input_splits: [ep_size] dispatch时每个rank发送的token数（现在变成接收）
        output_splits: [ep_size] dispatch时从每个rank接收的token数（现在变成发送）
        ep_group: Expert Parallel进程组
        overlap_ctx: 多卡overlap上下文

    Returns:
        combined_output: [original_tokens, hidden] 合并后的输出
    """
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    device = expert_output.device
    hidden_size = expert_output.shape[-1]
    dtype = expert_output.dtype

    input_splits_list = input_splits.tolist() if torch.is_tensor(input_splits) else list(input_splits)
    output_splits_list = output_splits.tolist() if torch.is_tensor(output_splits) else list(output_splits)

    # 计算输出offsets（combine的接收位置）
    input_offsets = [0]
    for s in input_splits_list:
        input_offsets.append(input_offsets[-1] + s)

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()
    num_rounds = overlap_ctx.num_rounds

    # 输出缓冲区
    total_output = sum(input_splits_list)
    combined_output = torch.empty(total_output, hidden_size, dtype=dtype, device=device)

    # 本地数据直接复制
    local_count = input_splits_list[my_rank]
    if local_count > 0:
        local_start = input_offsets[my_rank]
        # expert_output中本地部分的位置需要根据dispatch时的顺序确定
        # 这里假设本地数据在expert_output的开头
        combined_output[local_start:local_start + local_count] = expert_output[:local_count]

    # 记录数据准备好
    overlap_ctx.data_ready_event.record(default_stream)

    # 计算远程数据在expert_output中的起始位置
    remote_start = local_count

    # 多轮P2P通信（与dispatch顺序相反发送）
    for round_idx in range(num_rounds):
        round_data = round_data_list[round_idx]
        partner = round_data.partner_rank

        if partner == -1:
            continue

        # Combine时：发送给partner的是它发给我们的token的计算结果
        # 接收的是我们发给partner的token的计算结果
        send_count = round_data.recv_count  # dispatch时接收的，现在发送回去
        recv_count = round_data.send_count  # dispatch时发送的，现在接收回来

        # 发送数据
        if send_count > 0:
            send_tokens = expert_output[remote_start:remote_start + send_count].contiguous()
            remote_start += send_count
        else:
            send_tokens = torch.empty(0, hidden_size, dtype=dtype, device=device)

        # 接收缓冲区
        recv_start = input_offsets[partner]
        recv_tokens = combined_output[recv_start:recv_start + recv_count] if recv_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)

        # 启动P2P通信
        comm_stream.wait_event(overlap_ctx.data_ready_event)

        with torch.cuda.stream(comm_stream):
            p2p_ops = []
            if recv_count > 0:
                p2p_ops.append(dist.P2POp(dist.irecv, recv_tokens, partner, group=ep_group))
            if send_count > 0:
                p2p_ops.append(dist.P2POp(dist.isend, send_tokens, partner, group=ep_group))

            if p2p_ops:
                reqs = dist.batch_isend_irecv(p2p_ops)
                for req in reqs:
                    req.wait()

        # 记录本轮通信完成
        round_event = overlap_ctx.get_round_event(round_idx)
        round_event.record(comm_stream)

    # 等待所有通信完成
    if num_rounds > 0:
        default_stream.wait_event(overlap_ctx.get_round_event(num_rounds - 1))

    return combined_output


# =============================================================================
# MoE层多卡P2P前向重叠（完整实现）
# =============================================================================

class _MoEMultiCardP2POverlapFunction(torch.autograd.Function):
    """
    MoE多卡P2P重叠的自动求导包装

    前向：使用多轮P2P重叠（dispatch和combine）
    反向：使用标准AllToAll（保持反向调度不变）

    关键设计（两阶段）：

    阶段1: Dispatch + FC1 重叠
    ─────────────────────────────────────────────────────────────
    - 本地FC1+Act与第一轮P2P通信重叠
    - Round r的P2P通信与Round r-1的FC1+Act计算重叠
    - 只计算FC1和激活函数，不计算FC2

    阶段2: FC2 + Combine 重叠 (所有Dispatch和FC1完成后)
    ─────────────────────────────────────────────────────────────
    - 先计算远程数据的FC2（与Combine P2P重叠）
    - 最后计算本地数据的FC2
    - 本地FC2完成后保存重排激活值用于反向

    数据拼接：
    - 每个设备根据Round-Robin调度的partner顺序处理数据
    - combined_output按原始token顺序排列
    """

    @staticmethod
    def forward(ctx, tokens, input_splits, output_splits, weight1, weight2,
                ep_group, activation_func, overlap_ctx, layer_id,
                num_local_experts=1, tokens_per_expert=None,
                num_global_tokens_per_local_expert=None):
        """
        多卡流水线重叠实现：

        Dispatch阶段（通信 → 计算）流水线：
        ─────────────────────────────────────────────────────────────
        Round 0: 启动 P2P_0，同时计算 local FC1 + Act
        Round i: req.wait(P2P_{i-1})，启动 P2P_i，同时计算 recv_{i-1} 的 FC1 + Act
        最后:    req.wait(最后一轮)，计算最后的 FC1 + Act

        Combine阶段（计算 → 通信）流水线：
        ─────────────────────────────────────────────────────────────
        Round -1: 计算第一个 peer 的 FC2
        Round i:  event.synchronize()，启动 P2P_i，同时计算下一个 peer 的 FC2
        最后:     计算 local FC2（与最后一轮 P2P 并行）
        """
        ctx.ep_group = ep_group
        ctx.activation_func = activation_func
        ctx.layer_id = layer_id
        ctx.num_local_experts = num_local_experts

        # 保存原始权重引用（用于 backward 梯度计算）
        orig_weight1 = weight1
        orig_weight2 = weight2

        my_rank = ep_group.rank()
        ep_size = ep_group.size()
        device = tokens.device
        hidden_size = tokens.shape[-1]
        dtype = tokens.dtype

        # 权重维度
        total_ffn_hidden = weight1.shape[-1]
        ffn_hidden = total_ffn_hidden // num_local_experts

        w1 = weight1.view(num_local_experts, hidden_size, ffn_hidden)
        w2 = weight2.view(num_local_experts, ffn_hidden, hidden_size)

        input_splits_list = input_splits.tolist() if torch.is_tensor(input_splits) else list(input_splits)
        output_splits_list = output_splits.tolist() if torch.is_tensor(output_splits) else list(output_splits)

        ctx.input_splits_list = input_splits_list
        ctx.output_splits_list = output_splits_list
        ctx.my_rank = my_rank
        ctx.ffn_hidden = ffn_hidden

        # 计算offsets
        input_offsets = [0]
        for s in input_splits_list:
            input_offsets.append(input_offsets[-1] + s)

        output_offsets = [0]
        for s in output_splits_list:
            output_offsets.append(output_offsets[-1] + s)

        default_stream = torch.cuda.current_stream(device)
        comm_stream = overlap_ctx.get_stream()

        local_count = input_splits_list[my_rank]
        local_start = input_offsets[my_rank]
        ctx.local_count = local_count
        ctx.local_start = local_start

        # 计算每个expert处理的本地token数（CPU操作，提前准备）
        local_tokens_per_expert = None
        if num_global_tokens_per_local_expert is not None:
            local_tokens_per_expert = [
                num_global_tokens_per_local_expert[0, my_rank, exp_idx].item()
                for exp_idx in range(num_local_experts)
            ]

        # 获取 Round-Robin 调度的 partners
        partners = []
        for round_idx in range(overlap_ctx.num_rounds):
            partner = overlap_ctx.get_partner(my_rank, round_idx)
            if partner != -1:
                partners.append(partner)

        # 提取本地tokens
        local_tokens = tokens[local_start:local_start + local_count].clone() if local_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)

        # 准备发送数据（tokens 的 slice，几乎瞬时）
        send_chunks = {}
        for partner in partners:
            if input_splits_list[partner] > 0:
                send_chunks[partner] = tokens[input_offsets[partner]:input_offsets[partner+1]].contiguous()

        # 准备接收缓冲区（按 partner 顺序）
        recv_buffers = {}
        for partner in partners:
            recv_size = output_splits_list[partner]
            if recv_size > 0:
                recv_buffers[partner] = torch.empty(recv_size, hidden_size, dtype=dtype, device=device)

        # =========================================================================
        # Dispatch阶段流水线（通信 → 计算）
        # =========================================================================
        # Round 0: 启动 P2P_0，同时计算 local FC1 + Act
        # Round i: req.wait(P2P_{i-1})，启动 P2P_i，同时计算 recv_{i-1} 的 FC1 + Act
        # 注意：act_deriv 在保存激活值阶段统一计算，不在此处计算

        prev_reqs = None
        prev_partner = None
        recv_act_results = {}  # 存储每个 partner 的 Act 结果
        recv_fc1_results = {}  # 存储每个 partner 的 FC1 结果（用于 backward）
        local_fc1_saved = None  # 保存 local 的 FC1 结果

        for round_idx, partner in enumerate(partners):
            # 启动当前轮 P2P
            with torch.cuda.stream(comm_stream):
                p2p_ops = []
                if partner in recv_buffers:
                    p2p_ops.append(dist.P2POp(dist.irecv, recv_buffers[partner], partner, group=ep_group))
                if partner in send_chunks:
                    p2p_ops.append(dist.P2POp(dist.isend, send_chunks[partner], partner, group=ep_group))
                curr_reqs = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []

            # 与 P2P 并行：计算 FC1 + Act，同时保存 FC1 用于 backward
            if round_idx == 0:
                # 第一轮：计算 local FC1 + Act（与 P2P_0 并行）
                if local_count > 0 and local_tokens_per_expert is not None:
                    local_act = torch.zeros(local_count, ffn_hidden, dtype=dtype, device=device)
                    local_fc1_saved = torch.zeros(local_count, ffn_hidden, dtype=dtype, device=device)
                    start = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = local_tokens_per_expert[exp_idx]
                        if n_tok > 0:
                            exp_tokens = local_tokens[start:start + n_tok]
                            exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                            local_fc1_saved[start:start + n_tok] = exp_fc1
                            local_act[start:start + n_tok] = activation_func(exp_fc1)
                            start += n_tok
                elif local_count > 0:
                    local_fc1_saved = torch.matmul(local_tokens, w1[0])
                    local_act = activation_func(local_fc1_saved)
                else:
                    local_act = None
                    local_fc1_saved = None
            else:
                # 等待上一轮 P2P 完成
                for req in prev_reqs:
                    req.wait()
                # 计算上一轮接收数据的 FC1 + Act（与当前轮 P2P 并行）
                if prev_partner in recv_buffers:
                    recv_data = recv_buffers[prev_partner]
                    if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                        recv_act = torch.zeros(recv_data.shape[0], ffn_hidden, dtype=dtype, device=device)
                        recv_fc1 = torch.zeros(recv_data.shape[0], ffn_hidden, dtype=dtype, device=device)
                        offset = 0
                        for exp_idx in range(num_local_experts):
                            n_tok = num_global_tokens_per_local_expert[0, prev_partner, exp_idx].item()
                            if n_tok > 0:
                                exp_tokens = recv_data[offset:offset + n_tok]
                                exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                                recv_fc1[offset:offset + n_tok] = exp_fc1
                                recv_act[offset:offset + n_tok] = activation_func(exp_fc1)
                                offset += n_tok
                    else:
                        recv_fc1 = torch.matmul(recv_data, w1[0])
                        recv_act = activation_func(recv_fc1)
                    recv_act_results[prev_partner] = recv_act
                    recv_fc1_results[prev_partner] = recv_fc1

            prev_reqs = curr_reqs
            prev_partner = partner

        # 等待最后一轮 P2P 完成，计算最后的 FC1 + Act
        if prev_reqs is not None:
            for req in prev_reqs:
                req.wait()
            if prev_partner in recv_buffers:
                recv_data = recv_buffers[prev_partner]
                if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                    recv_act = torch.zeros(recv_data.shape[0], ffn_hidden, dtype=dtype, device=device)
                    recv_fc1 = torch.zeros(recv_data.shape[0], ffn_hidden, dtype=dtype, device=device)
                    offset = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = num_global_tokens_per_local_expert[0, prev_partner, exp_idx].item()
                        if n_tok > 0:
                            exp_tokens = recv_data[offset:offset + n_tok]
                            exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                            recv_fc1[offset:offset + n_tok] = exp_fc1
                            recv_act[offset:offset + n_tok] = activation_func(exp_fc1)
                            offset += n_tok
                else:
                    recv_fc1 = torch.matmul(recv_data, w1[0])
                    recv_act = activation_func(recv_fc1)
                recv_act_results[prev_partner] = recv_act
                recv_fc1_results[prev_partner] = recv_fc1

        # =========================================================================
        # Combine阶段流水线（计算 → 通信）
        # =========================================================================
        # Round -1: 计算第一个 peer 的 FC2
        # Round i:  event.synchronize()，启动 P2P_i，同时计算下一个 peer 的 FC2
        # 最后:     计算 local FC2（与最后一轮 P2P 并行）

        total_output = sum(input_splits_list)
        combined_output = torch.empty(total_output, hidden_size, dtype=dtype, device=device)

        peer_fc2_results = {}
        fc2_events = {}

        # Round -1：预计算第一个 peer 的 FC2
        if len(partners) > 0:
            first_partner = partners[0]
            if first_partner in recv_act_results:
                recv_act = recv_act_results[first_partner]
                if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                    peer_fc2 = torch.zeros(recv_act.shape[0], hidden_size, dtype=dtype, device=device)
                    offset = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = num_global_tokens_per_local_expert[0, first_partner, exp_idx].item()
                        if n_tok > 0:
                            exp_act = recv_act[offset:offset + n_tok]
                            peer_fc2[offset:offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                            offset += n_tok
                else:
                    peer_fc2 = torch.matmul(recv_act, w2[0])
                peer_fc2_results[first_partner] = peer_fc2
                fc2_events[first_partner] = torch.cuda.Event()
                fc2_events[first_partner].record(default_stream)

        # 流水线循环
        all_combine_reqs = []
        for round_idx, partner in enumerate(partners):
            # CPU 端等待 FC2 计算完成
            if partner in fc2_events:
                fc2_events[partner].synchronize()

            # 启动 P2P（发送 FC2 结果给 partner，接收 partner 发来的结果）
            with torch.cuda.stream(comm_stream):
                p2p_ops = []
                # 接收：partner 发给我的 FC2 结果
                recv_size = input_splits_list[partner]
                if recv_size > 0:
                    recv_chunk = combined_output[input_offsets[partner]:input_offsets[partner+1]]
                    p2p_ops.append(dist.P2POp(dist.irecv, recv_chunk, partner, group=ep_group))
                # 发送：我计算的 FC2 结果给 partner
                if partner in peer_fc2_results:
                    p2p_ops.append(dist.P2POp(dist.isend, peer_fc2_results[partner], partner, group=ep_group))
                reqs = dist.batch_isend_irecv(p2p_ops) if p2p_ops else []
                all_combine_reqs.extend(reqs)

            # 与当前轮 P2P 并行：计算下一轮的 FC2 或 local FC2
            if round_idx + 1 < len(partners):
                next_partner = partners[round_idx + 1]
                if next_partner in recv_act_results:
                    recv_act = recv_act_results[next_partner]
                    if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                        peer_fc2 = torch.zeros(recv_act.shape[0], hidden_size, dtype=dtype, device=device)
                        offset = 0
                        for exp_idx in range(num_local_experts):
                            n_tok = num_global_tokens_per_local_expert[0, next_partner, exp_idx].item()
                            if n_tok > 0:
                                exp_act = recv_act[offset:offset + n_tok]
                                peer_fc2[offset:offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                                offset += n_tok
                    else:
                        peer_fc2 = torch.matmul(recv_act, w2[0])
                    peer_fc2_results[next_partner] = peer_fc2
                    fc2_events[next_partner] = torch.cuda.Event()
                    fc2_events[next_partner].record(default_stream)
            else:
                # 最后一轮：计算 local FC2（与最后一轮 P2P 并行）
                if local_act is not None and local_tokens_per_expert is not None:
                    local_fc2 = torch.zeros(local_count, hidden_size, dtype=dtype, device=device)
                    start = 0
                    for exp_idx in range(num_local_experts):
                        n_tok = local_tokens_per_expert[exp_idx]
                        if n_tok > 0:
                            exp_act = local_act[start:start + n_tok]
                            local_fc2[start:start + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                            start += n_tok
                elif local_act is not None:
                    local_fc2 = torch.matmul(local_act, w2[0])
                else:
                    local_fc2 = None

        # =========================================================================
        # 保存用于backward：合并并重排激活值（与 Combine P2P 并行）
        # =========================================================================

        # 合并所有 recv_buffers 为单个 tensor（按 rank 顺序）
        total_recv = sum(output_splits_list[p] for p in partners if p in recv_buffers)
        if total_recv > 0:
            # 按 rank 顺序排列（不是按 partner 顺序）
            all_peer_tokens_list = []
            for i in range(ep_size):
                if i == my_rank:
                    continue
                if i in recv_buffers:
                    all_peer_tokens_list.append(recv_buffers[i])
            all_peer_tokens = torch.cat(all_peer_tokens_list, dim=0) if all_peer_tokens_list else torch.empty(0, hidden_size, dtype=dtype, device=device)
        else:
            all_peer_tokens = torch.empty(0, hidden_size, dtype=dtype, device=device)

        # 合并local和peer的fc1结果（按 rank 顺序）
        all_peer_fc1_list = []
        for i in range(ep_size):
            if i == my_rank:
                continue
            if i in recv_fc1_results:
                all_peer_fc1_list.append(recv_fc1_results[i])
        if all_peer_fc1_list:
            all_peer_fc1 = torch.cat(all_peer_fc1_list, dim=0)
        else:
            all_peer_fc1 = torch.empty(0, ffn_hidden, dtype=dtype, device=device)

        # 合并local和peer tokens/fc1用于backward（expert-major顺序）
        if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
            # 使用 _merge_tokens_and_fc1_expert_major 同时重排 tokens 和 fc1
            all_expert_tokens, all_fc1, all_tokens_per_expert = _merge_tokens_and_fc1_expert_major(
                local_tokens, all_peer_tokens,
                local_fc1_saved, all_peer_fc1,
                num_local_experts, num_global_tokens_per_local_expert,
                my_rank, ep_size, device
            )
        else:
            all_expert_tokens = torch.cat([local_tokens, all_peer_tokens], dim=0) if all_peer_tokens.numel() > 0 else local_tokens
            all_tokens_per_expert = [all_expert_tokens.shape[0]]
            # 合并 fc1 结果（单 expert 不需要重排）
            if all_peer_fc1.numel() > 0:
                if local_fc1_saved is not None:
                    all_fc1 = torch.cat([local_fc1_saved, all_peer_fc1], dim=0)
                else:
                    all_fc1 = all_peer_fc1
            else:
                all_fc1 = local_fc1_saved if local_fc1_saved is not None else torch.empty(0, ffn_hidden, dtype=dtype, device=device)

        # 等待所有 Combine P2P 完成
        for req in all_combine_reqs:
            req.wait()

        # 处理没有 partner 的情况（ep_size=1）
        if len(partners) == 0:
            if local_act is not None and local_tokens_per_expert is not None:
                local_fc2 = torch.zeros(local_count, hidden_size, dtype=dtype, device=device)
                start = 0
                for exp_idx in range(num_local_experts):
                    n_tok = local_tokens_per_expert[exp_idx]
                    if n_tok > 0:
                        exp_act = local_act[start:start + n_tok]
                        local_fc2[start:start + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                        start += n_tok
            elif local_act is not None:
                local_fc2 = torch.matmul(local_act, w2[0])
            else:
                local_fc2 = None

        # 本地结果写入 combined_output
        if local_fc2 is not None:
            combined_output[local_start:local_start + local_count] = local_fc2

        # 使用 ctx 属性代替 save_for_backward，避免 PyTorch 的额外检查开销
        # 所有张量都 detach，因为我们的自定义 backward 手动计算所有梯度
        ctx._all_expert_tokens = all_expert_tokens
        ctx._weight1 = orig_weight1.detach()
        ctx._weight2 = orig_weight2.detach()
        ctx._all_fc1 = all_fc1  # 保存 fc1，backward 从 fc1 计算 act 和 act_deriv
        ctx.all_tokens_per_expert = all_tokens_per_expert
        ctx.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert

        # 预计算backward需要的sort索引
        if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
            _precompute_backward_sort_indices(ctx, num_local_experts, ep_size,
                                              num_global_tokens_per_local_expert, device)

        return combined_output

    @staticmethod
    def backward(ctx, grad_output):
        # 使用标准AllToAll（与baseline相同）
        from fluid.overlap_forward import _compute_activation_grad, _compute_activation_derivative
        from fluid.communication import _all_to_all
        from fluid.scheduler import get_backward_scheduler
        from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs

        all_expert_tokens = ctx._all_expert_tokens
        weight1 = ctx._weight1
        weight2 = ctx._weight2
        all_fc1 = ctx._all_fc1  # 使用保存的 fc1

        ep_group = ctx.ep_group
        activation_func = ctx.activation_func
        input_splits_list = ctx.input_splits_list
        output_splits_list = ctx.output_splits_list
        num_local_experts = ctx.num_local_experts
        ffn_hidden = ctx.ffn_hidden
        all_tokens_per_expert = ctx.all_tokens_per_expert

        device = grad_output.device
        hidden_size = grad_output.shape[-1]

        w1 = weight1.view(num_local_experts, hidden_size, ffn_hidden)
        w2 = weight2.view(num_local_experts, ffn_hidden, hidden_size)

        total_all_tokens = all_expert_tokens.shape[0]

        # 从保存的 fc1 计算 act 和 act_deriv（无需重新计算 fc1 矩阵乘法）
        act_output = activation_func(all_fc1)
        act_deriv = _compute_activation_derivative(all_fc1, activation_func, gated_linear_unit=False)

        # Combine AllToAll（支持与之前层的 dW 重叠）
        scheduler = get_backward_scheduler()
        if scheduler.is_enabled():
            comm_stream = scheduler.comm_stream
            default_stream = scheduler.default_stream

            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_combined = _all_to_all(
                    grad_output.contiguous(),
                    output_split_sizes=output_splits_list,
                    input_split_sizes=input_splits_list,
                    group=ep_group
                )

            # 尝试执行之前层注册的 dW 任务
            scheduler._execute_all_dw_tasks_sync()
            default_stream.wait_stream(comm_stream)
        else:
            grad_combined = _all_to_all(
                grad_output.contiguous(),
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
                group=ep_group
            )

        # 转换布局：rank-major -> expert-major
        if hasattr(ctx, 'split_sizes_rank_major'):
            grad_all_fc2, _ = sort_chunks_by_idxs(
                grad_combined,
                ctx.split_sizes_rank_major,
                ctx.sorted_idxs_rank_to_exp
            )
        else:
            grad_all_fc2 = grad_combined

        # 计算grad_tokens
        grad_all_tokens = torch.zeros(total_all_tokens, hidden_size, dtype=grad_output.dtype, device=device)
        grad_all_fc1 = torch.zeros(total_all_tokens, ffn_hidden, dtype=grad_output.dtype, device=device)

        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_exp_act = torch.matmul(grad_all_fc2[start:start+n_tok], w2[exp_idx].t())
                grad_exp_fc1 = _compute_activation_grad(
                    grad_exp_act, act_deriv[start:start+n_tok], gated_linear_unit=False
                )
                grad_all_tokens[start:start+n_tok] = torch.matmul(grad_exp_fc1, w1[exp_idx].t())
                grad_all_fc1[start:start+n_tok] = grad_exp_fc1
                start += n_tok

        # 注册dW任务
        scheduler = get_backward_scheduler()

        num_local_experts_saved = num_local_experts
        all_tokens_per_expert_saved = all_tokens_per_expert
        ffn_hidden_saved = ffn_hidden
        hidden_size_saved = hidden_size
        grad_all_fc2_saved = grad_all_fc2.detach()
        grad_all_fc1_saved = grad_all_fc1.detach()
        act_output_saved = act_output.detach()
        all_expert_tokens_saved = all_expert_tokens.detach()

        def compute_dw_weight2():
            grad_w2 = torch.zeros_like(weight2)
            grad_w2_view = grad_w2.view(num_local_experts_saved, ffn_hidden_saved, hidden_size_saved)
            start = 0
            for exp_idx in range(num_local_experts_saved):
                n_tok = all_tokens_per_expert_saved[exp_idx]
                if n_tok > 0:
                    grad_w2_view[exp_idx] = torch.matmul(
                        act_output_saved[start:start+n_tok].t(),
                        grad_all_fc2_saved[start:start+n_tok]
                    )
                    start += n_tok
            return grad_w2

        def compute_dw_weight1():
            grad_w1 = torch.zeros_like(weight1)
            grad_w1_view = grad_w1.view(num_local_experts_saved, hidden_size_saved, ffn_hidden_saved)
            start = 0
            for exp_idx in range(num_local_experts_saved):
                n_tok = all_tokens_per_expert_saved[exp_idx]
                if n_tok > 0:
                    grad_w1_view[exp_idx] = torch.matmul(
                        all_expert_tokens_saved[start:start+n_tok].t(),
                        grad_all_fc1_saved[start:start+n_tok]
                    )
                    start += n_tok
            return grad_w1

        layer_id = ctx.layer_id
        scheduler.register_dw_task(
            layer_name=f"moe_multicard_weight2_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight2,
            priority=100,
            weight_param=weight2,
        )

        scheduler.register_dw_task(
            layer_name=f"moe_multicard_weight1_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight1,
            priority=99,
            weight_param=weight1,
        )

        # 转换布局：expert-major -> rank-major
        if hasattr(ctx, 'split_sizes_exp_major'):
            grad_dispatched, _ = sort_chunks_by_idxs(
                grad_all_tokens,
                ctx.split_sizes_exp_major,
                ctx.sorted_idxs_exp_to_rank
            )
        else:
            grad_dispatched = grad_all_tokens

        # dW-AllToAll重叠
        if scheduler.is_enabled():
            comm_stream = scheduler.comm_stream
            default_stream = scheduler.default_stream

            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_tokens = _all_to_all(
                    grad_dispatched.contiguous(),
                    output_split_sizes=input_splits_list,
                    input_split_sizes=output_splits_list,
                    group=ep_group
                )

            scheduler._execute_all_dw_tasks_sync()
            default_stream.wait_stream(comm_stream)
        else:
            grad_tokens = _all_to_all(
                grad_dispatched.contiguous(),
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )

        return (grad_tokens, None, None, None, None, None, None, None, None,
                None, None, None)


def _compute_fc1_act_per_source(
    tokens: torch.Tensor,
    w1: torch.Tensor,
    activation_func,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    source_rank: int,
) -> torch.Tensor:
    """
    计算来自单个source rank的tokens的FC1+激活（不包含FC2）

    Args:
        tokens: [num_tokens, hidden] 来自source_rank的tokens
        w1: [num_local_experts, hidden, ffn_hidden]
        activation_func: 激活函数
        num_local_experts: 本地expert数量
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        source_rank: 数据来源rank

    Returns:
        act_output: [num_tokens, ffn_hidden]
    """
    device = tokens.device
    ffn_hidden = w1.shape[-1]

    if tokens.numel() == 0:
        return torch.empty(0, ffn_hidden, dtype=tokens.dtype, device=device)

    if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
        act_output = torch.zeros(tokens.shape[0], ffn_hidden, dtype=tokens.dtype, device=device)

        offset = 0
        for exp_idx in range(num_local_experts):
            n_tok = num_global_tokens_per_local_expert[0, source_rank, exp_idx].item()
            if n_tok > 0:
                exp_tokens = tokens[offset:offset + n_tok]
                exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                act_output[offset:offset + n_tok] = activation_func(exp_fc1)
                offset += n_tok

        return act_output
    else:
        fc1 = torch.matmul(tokens, w1[0])
        return activation_func(fc1)


def _compute_fc2_per_source(
    act: torch.Tensor,
    w2: torch.Tensor,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    source_rank: int,
) -> torch.Tensor:
    """
    计算来自单个source rank的激活值的FC2

    Args:
        act: [num_tokens, ffn_hidden] 激活值
        w2: [num_local_experts, ffn_hidden, hidden]
        num_local_experts: 本地expert数量
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        source_rank: 数据来源rank

    Returns:
        fc2_output: [num_tokens, hidden]
    """
    device = act.device
    hidden_size = w2.shape[-1]

    if act.numel() == 0:
        return torch.empty(0, hidden_size, dtype=act.dtype, device=device)

    if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
        fc2_output = torch.zeros(act.shape[0], hidden_size, dtype=act.dtype, device=device)

        offset = 0
        for exp_idx in range(num_local_experts):
            n_tok = num_global_tokens_per_local_expert[0, source_rank, exp_idx].item()
            if n_tok > 0:
                exp_act = act[offset:offset + n_tok]
                fc2_output[offset:offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                offset += n_tok

        return fc2_output
    else:
        return torch.matmul(act, w2[0])


def _compute_expert_forward_per_source(
    tokens: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    activation_func,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    source_rank: int,
    my_rank: int,
) -> torch.Tensor:
    """
    计算来自单个source rank的tokens的expert前向（完整FC1+Act+FC2）

    注意：这个函数仅用于兼容旧代码，新的多卡实现应该分别调用
    _compute_fc1_act_per_source 和 _compute_fc2_per_source

    Args:
        tokens: [num_tokens, hidden] 来自source_rank的tokens
        w1: [num_local_experts, hidden, ffn_hidden]
        w2: [num_local_experts, ffn_hidden, hidden]
        activation_func: 激活函数
        num_local_experts: 本地expert数量
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]
        source_rank: 数据来源rank
        my_rank: 当前rank

    Returns:
        fc2_output: [num_tokens, hidden]
    """
    act = _compute_fc1_act_per_source(
        tokens, w1, activation_func, num_local_experts,
        num_global_tokens_per_local_expert, source_rank
    )
    return _compute_fc2_per_source(
        act, w2, num_local_experts,
        num_global_tokens_per_local_expert, source_rank
    )


def _merge_tokens_expert_major(
    local_tokens: torch.Tensor,
    all_peer_tokens: torch.Tensor,
    round_recv_data: Dict[int, Tuple[torch.Tensor, int, int]],
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    my_rank: int,
    ep_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """
    将local和peer tokens合并为expert-major顺序

    Returns:
        all_expert_tokens: [total, hidden] expert-major顺序
        all_tokens_per_expert: [num_local_experts] 每个expert的token数
    """
    hidden_size = local_tokens.shape[-1] if local_tokens.numel() > 0 else all_peer_tokens.shape[-1]
    dtype = local_tokens.dtype if local_tokens.numel() > 0 else all_peer_tokens.dtype

    # 计算每个expert的总token数
    all_tokens_per_expert = []
    for exp_idx in range(num_local_experts):
        total = 0
        for rank in range(ep_size):
            total += num_global_tokens_per_local_expert[0, rank, exp_idx].item()
        all_tokens_per_expert.append(total)

    total_tokens = sum(all_tokens_per_expert)

    if total_tokens == 0:
        return torch.empty(0, hidden_size, dtype=dtype, device=device), all_tokens_per_expert

    all_expert_tokens = torch.zeros(total_tokens, hidden_size, dtype=dtype, device=device)

    # 按expert-major顺序填充
    # expert-major: [E0_R0, E0_R1, ..., E0_Rn, E1_R0, E1_R1, ..., E1_Rn, ...]
    write_offset = 0
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            n_tok = num_global_tokens_per_local_expert[0, rank, exp_idx].item()
            if n_tok == 0:
                continue

            if rank == my_rank:
                # 从local_tokens提取
                # local_tokens按expert顺序排列
                local_exp_offset = sum(
                    num_global_tokens_per_local_expert[0, my_rank, e].item()
                    for e in range(exp_idx)
                )
                all_expert_tokens[write_offset:write_offset + n_tok] = \
                    local_tokens[local_exp_offset:local_exp_offset + n_tok]
            else:
                # 从对应round的recv_data提取
                # 需要找到哪个round接收了来自rank的数据
                for round_idx, (recv_tokens, partner, recv_count) in round_recv_data.items():
                    if partner == rank and recv_count > 0:
                        # recv_tokens按expert顺序排列
                        peer_exp_offset = sum(
                            num_global_tokens_per_local_expert[0, rank, e].item()
                            for e in range(exp_idx)
                        )
                        all_expert_tokens[write_offset:write_offset + n_tok] = \
                            recv_tokens[peer_exp_offset:peer_exp_offset + n_tok]
                        break

            write_offset += n_tok

    return all_expert_tokens, all_tokens_per_expert


def _merge_tokens_and_fc1_expert_major(
    local_tokens: torch.Tensor,
    all_peer_tokens: torch.Tensor,
    local_fc1: torch.Tensor,
    all_peer_fc1: torch.Tensor,
    num_local_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    my_rank: int,
    ep_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    将local和peer的tokens和fc1同时合并为expert-major顺序
    （只计算一次偏移量，同时重排两个张量）

    Args:
        local_tokens: [local_count, hidden_size] 本地tokens（按expert顺序）
        all_peer_tokens: [peer_count, hidden_size] 所有peer tokens（按rank顺序cat）
        local_fc1: [local_count, ffn_hidden] 本地fc1结果
        all_peer_fc1: [peer_count, ffn_hidden] 所有peer fc1结果（按rank顺序cat）

    Returns:
        all_expert_tokens: [total, hidden_size] expert-major顺序
        all_fc1: [total, ffn_hidden] expert-major顺序
        all_tokens_per_expert: [num_local_experts] 每个expert的token数
    """
    hidden_size = local_tokens.shape[-1] if local_tokens.numel() > 0 else all_peer_tokens.shape[-1]
    ffn_hidden = local_fc1.shape[-1] if local_fc1 is not None and local_fc1.numel() > 0 else all_peer_fc1.shape[-1]
    dtype = local_tokens.dtype if local_tokens.numel() > 0 else all_peer_tokens.dtype

    # 计算每个expert的总token数
    all_tokens_per_expert = []
    for exp_idx in range(num_local_experts):
        total = 0
        for rank in range(ep_size):
            total += num_global_tokens_per_local_expert[0, rank, exp_idx].item()
        all_tokens_per_expert.append(total)

    total_tokens = sum(all_tokens_per_expert)

    if total_tokens == 0:
        return (torch.empty(0, hidden_size, dtype=dtype, device=device),
                torch.empty(0, ffn_hidden, dtype=dtype, device=device),
                all_tokens_per_expert)

    all_expert_tokens = torch.zeros(total_tokens, hidden_size, dtype=dtype, device=device)
    all_fc1 = torch.zeros(total_tokens, ffn_hidden, dtype=dtype, device=device)

    # 预计算每个 rank 在 all_peer_tokens/all_peer_fc1 中的起始偏移
    peer_rank_offsets = {}
    offset = 0
    for rank in range(ep_size):
        if rank == my_rank:
            continue
        peer_rank_offsets[rank] = offset
        for exp_idx in range(num_local_experts):
            offset += num_global_tokens_per_local_expert[0, rank, exp_idx].item()

    # 按expert-major顺序填充（同时处理tokens和fc1）
    write_offset = 0
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            n_tok = num_global_tokens_per_local_expert[0, rank, exp_idx].item()
            if n_tok == 0:
                continue

            if rank == my_rank:
                # 从local提取（local按expert顺序排列）
                local_exp_offset = sum(
                    num_global_tokens_per_local_expert[0, my_rank, e].item()
                    for e in range(exp_idx)
                )
                all_expert_tokens[write_offset:write_offset + n_tok] = \
                    local_tokens[local_exp_offset:local_exp_offset + n_tok]
                if local_fc1 is not None and local_fc1.numel() > 0:
                    all_fc1[write_offset:write_offset + n_tok] = \
                        local_fc1[local_exp_offset:local_exp_offset + n_tok]
            else:
                # 从peer提取（peer按rank顺序cat，每个rank内按expert顺序）
                peer_base = peer_rank_offsets[rank]
                peer_exp_offset = sum(
                    num_global_tokens_per_local_expert[0, rank, e].item()
                    for e in range(exp_idx)
                )
                src_offset = peer_base + peer_exp_offset
                all_expert_tokens[write_offset:write_offset + n_tok] = \
                    all_peer_tokens[src_offset:src_offset + n_tok]
                if all_peer_fc1.numel() > 0:
                    all_fc1[write_offset:write_offset + n_tok] = \
                        all_peer_fc1[src_offset:src_offset + n_tok]

            write_offset += n_tok

    return all_expert_tokens, all_fc1, all_tokens_per_expert


def _precompute_backward_sort_indices(ctx, num_local_experts, ep_size,
                                       num_global_tokens_per_local_expert, device):
    """预计算backward需要的sort索引"""
    # rank-major chunk sizes: [R0_E0, R0_E1, R1_E0, R1_E1, ...]
    split_sizes_rank_major = []
    for rank in range(ep_size):
        for exp_idx in range(num_local_experts):
            split_sizes_rank_major.append(num_global_tokens_per_local_expert[0, rank, exp_idx].item())

    # rank-major -> expert-major的索引
    sorted_idxs_rank_to_exp = []
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            sorted_idxs_rank_to_exp.append(rank * num_local_experts + exp_idx)

    # expert-major chunk sizes: [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...]
    split_sizes_exp_major = []
    for exp_idx in range(num_local_experts):
        for rank in range(ep_size):
            split_sizes_exp_major.append(num_global_tokens_per_local_expert[0, rank, exp_idx].item())

    # expert-major -> rank-major的索引
    sorted_idxs_exp_to_rank = []
    for rank in range(ep_size):
        for exp_idx in range(num_local_experts):
            sorted_idxs_exp_to_rank.append(exp_idx * ep_size + rank)

    ctx.split_sizes_rank_major = torch.tensor(split_sizes_rank_major, dtype=torch.int64, device=device)
    ctx.sorted_idxs_rank_to_exp = torch.tensor(sorted_idxs_rank_to_exp, dtype=torch.int64, device=device)
    ctx.split_sizes_exp_major = torch.tensor(split_sizes_exp_major, dtype=torch.int64, device=device)
    ctx.sorted_idxs_exp_to_rank = torch.tensor(sorted_idxs_exp_to_rank, dtype=torch.int64, device=device)


# =============================================================================
# 公开API
# =============================================================================

def moe_multicard_p2p_overlap_forward(
    tokens: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    ep_group: dist.ProcessGroup,
    activation_func,
    overlap_ctx: MultiCardOverlapContext,
    layer_id: int = 0,
    num_local_experts: int = 1,
    tokens_per_expert: torch.Tensor = None,
    num_global_tokens_per_local_expert: torch.Tensor = None,
) -> torch.Tensor:
    """
    MoE多卡P2P重叠前向：本地token计算与远程token通信重叠

    使用多轮P2P通信调度，每轮每张卡只和一个peer通信
    通信流在跑第r轮的同时，计算流在吃第r-1轮的数据

    Args:
        tokens: [num_tokens, hidden] 输入token（已按expert排序）
        input_splits: [ep_size] 每个rank发送的token数
        output_splits: [ep_size] 每个rank接收的token数
        weight1: [hidden, ffn_hidden * num_local_experts] 第一层权重
        weight2: [ffn_hidden * num_local_experts, hidden] 第二层权重
        ep_group: Expert Parallel进程组
        activation_func: 激活函数
        overlap_ctx: 多卡overlap上下文
        layer_id: 层ID
        num_local_experts: 本地expert数量
        tokens_per_expert: [num_local_experts] 每个本地expert处理的token数
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts]

    Returns:
        output: [num_tokens, hidden]
    """
    if tokens.requires_grad:
        return _MoEMultiCardP2POverlapFunction.apply(
            tokens, input_splits, output_splits, weight1, weight2,
            ep_group, activation_func, overlap_ctx, layer_id,
            num_local_experts, tokens_per_expert, num_global_tokens_per_local_expert
        )
    else:
        # 推理模式：直接调用简化实现
        return _moe_multicard_p2p_impl(
            tokens, input_splits, output_splits, weight1, weight2,
            ep_group, activation_func, overlap_ctx, layer_id,
            num_local_experts, tokens_per_expert, num_global_tokens_per_local_expert
        )


# =============================================================================
# 注意力层多卡P2P前向重叠（规则通信）
# =============================================================================

class AttentionMultiCardOverlapContext:
    """注意力层多卡P2P通信的上下文管理"""

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


def attention_multicard_qkv_sp2hp(
    hidden_states: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    overlap_ctx: AttentionMultiCardOverlapContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    注意力层多卡QKV sp2hp P2P重叠实现

    流水线重叠核心思路（与 overlap_forward.py 一致）：
    - Round 0: 计算 partner_0 的 QKV → 启动 P2P_0
    - Round i (i > 0): 与 P2P_{i-1} 并行计算 partner_i 的 QKV → 启动 P2P_i
    - 最后: 与最后一轮 P2P 并行计算 local QKV → 等待所有 P2P

    这样每一轮的 QKV 计算都与上一轮的 P2P 通信重叠！

    Args:
        hidden_states: [seq_local, batch, hidden] 输入
        weight_qkv: [total_proj, hidden] QKV权重（interleaved布局）
        num_heads: Q heads总数
        num_kv_heads: K/V heads总数
        head_dim: 每个head的维度
        cp_group: Context Parallel进程组
        overlap_ctx: 多卡overlap上下文

    Returns:
        q, k, v: [seq_full, batch, heads_local, head_dim]
    """
    my_rank = cp_group.rank()
    cp_size = cp_group.size()
    device = hidden_states.device
    dtype = hidden_states.dtype
    seq_local, batch_size, hidden_size = hidden_states.shape

    # 计算各种维度
    q_per_group = num_heads // num_kv_heads
    groups_per_rank = num_kv_heads // cp_size
    group_size = (q_per_group + 2) * head_dim  # 每个group的QKV总维度
    proj_per_rank = groups_per_rank * group_size

    heads_local = groups_per_rank * q_per_group

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()
    num_rounds = overlap_ctx.num_rounds

    # 按group分割权重
    # 权重布局: [group0, group1, ...] 其中每个group = [Q, K, V]
    weight_grouped = weight_qkv.view(num_kv_heads, group_size, hidden_size)

    # 计算本地groups对应的权重
    local_group_start = my_rank * groups_per_rank
    local_group_end = local_group_start + groups_per_rank
    weight_local = weight_grouped[local_group_start:local_group_end].reshape(proj_per_rank, hidden_size)

    # 预先准备各个 partner 的权重
    weight_per_partner = {}
    partners = []
    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner == -1:
            continue
        partners.append(partner)
        partner_group_start = partner * groups_per_rank
        partner_group_end = partner_group_start + groups_per_rank
        weight_per_partner[partner] = weight_grouped[partner_group_start:partner_group_end].reshape(proj_per_rank, hidden_size)

    # =========================================================================
    # 流水线重叠核心：逐轮重叠！
    # - Round 0: 计算 QKV_0 → 启动 P2P_0
    # - Round i: P2P_{i-1} 运行中 + 计算 QKV_i → 启动 P2P_i
    # - 最后: 最后一轮 P2P 运行中 + 计算 local QKV → 等待所有 P2P
    # =========================================================================

    seq_full = seq_local * cp_size

    # 预分配输出缓冲区
    qkv_all = torch.empty(seq_full, batch_size, proj_per_rank, dtype=dtype, device=device)

    all_reqs = []
    send_tensors = []  # 保持发送tensor的引用

    for round_idx, partner in enumerate(partners):
        # 计算要发送给 partner 的 QKV（用 partner 的权重）
        # 这个计算与上一轮的 P2P 并行运行！
        qkv_to_send = torch.matmul(hidden_states, weight_per_partner[partner].t())
        send_tensors.append(qkv_to_send)  # 保持引用

        # 准备接收缓冲区
        partner_seq_start = partner * seq_local
        recv_buffer = qkv_all[partner_seq_start:partner_seq_start + seq_local]

        # 必须同步：确保 qkv_to_send 计算完成后才能发送
        overlap_ctx.qkv_ready_event.record(default_stream)
        comm_stream.wait_event(overlap_ctx.qkv_ready_event)

        with torch.cuda.stream(comm_stream):
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_buffer, partner, group=cp_group),
                dist.P2POp(dist.isend, qkv_to_send, partner, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(p2p_ops)
            all_reqs.extend(reqs)

    # 计算 local QKV（与最后一轮 P2P 并行！）
    qkv_local = torch.matmul(hidden_states, weight_local.t())  # [seq_local, batch, proj_per_rank]

    # 等待所有 P2P 完成
    for req in all_reqs:
        req.wait()

    # 记录通信完成并同步 stream
    overlap_ctx.qkv_comm_done_event.record(comm_stream)
    default_stream.wait_event(overlap_ctx.qkv_comm_done_event)

    # =========================================================================
    # 组装结果：本地数据写入对应位置
    # =========================================================================
    local_seq_start = my_rank * seq_local
    qkv_all[local_seq_start:local_seq_start + seq_local] = qkv_local

    # 分离Q, K, V
    qkv_all = qkv_all.view(seq_full, batch_size, groups_per_rank, group_size)
    q_size = q_per_group * head_dim
    q, k, v = torch.split(qkv_all, [q_size, head_dim, head_dim], dim=-1)

    # reshape
    q = q.reshape(seq_full, batch_size, heads_local, head_dim)
    # k, v 已经是 [seq_full, batch, kv_heads_local, head_dim]

    return q.contiguous(), k.contiguous(), v.contiguous()


def attention_multicard_hp2sp_proj(
    context: torch.Tensor,
    weight_proj: torch.Tensor,
    bias_proj: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    overlap_ctx: AttentionMultiCardOverlapContext,
) -> torch.Tensor:
    """
    注意力层多卡hp2sp + 输出投影P2P重叠实现

    每轮：
    1. 接收partner的部分结果
    2. 计算本地部分的输出投影

    Args:
        context: [seq_full, batch, heads_local, head_dim] 注意力输出
        weight_proj: [hidden, num_heads * head_dim] 输出投影权重
        bias_proj: [hidden] 偏置
        cp_group: Context Parallel进程组
        overlap_ctx: 多卡overlap上下文

    Returns:
        output: [seq_local, batch, hidden]
    """
    my_rank = cp_group.rank()
    cp_size = cp_group.size()
    device = context.device
    dtype = context.dtype

    seq_full, batch_size, heads_local, head_dim = context.shape
    seq_local = seq_full // cp_size
    hidden_size = weight_proj.shape[0]
    input_dim = heads_local * head_dim

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()
    num_rounds = overlap_ctx.num_rounds

    # 本地sequence位置
    local_seq_start = my_rank * seq_local
    local_seq_end = local_seq_start + seq_local

    # 提取本地sequence的attention输出
    context_local = context[local_seq_start:local_seq_end]  # [seq_local, batch, heads_local, head_dim]
    context_local_flat = context_local.view(seq_local, batch_size, -1)

    # 按heads分割权重
    # 每个rank负责heads_local个heads
    weight_per_rank = input_dim
    local_weight_start = my_rank * weight_per_rank
    weight_local = weight_proj[:, local_weight_start:local_weight_start + weight_per_rank]

    # 计算本地部分的输出投影
    output = torch.matmul(context_local_flat, weight_local.t())  # [seq_local, batch, hidden]

    # 记录数据准备好
    overlap_ctx.proj_ready_event.record(default_stream)

    # 多轮P2P接收其他rank的partial结果并累加
    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)

        if partner == -1:
            continue

        # 接收partner的数据并计算partial结果
        partner_seq_start = partner * seq_local
        context_partner = context[partner_seq_start:partner_seq_start + seq_local]
        context_partner_flat = context_partner.view(seq_local, batch_size, -1)

        # Partner的权重
        partner_weight_start = partner * weight_per_rank
        weight_partner = weight_proj[:, partner_weight_start:partner_weight_start + weight_per_rank]

        # 计算发送给partner的partial结果
        partial_to_send = torch.matmul(context_partner_flat, weight_partner.t())

        # 接收缓冲区
        recv_buffer = torch.empty(seq_local, batch_size, hidden_size, dtype=dtype, device=device)

        # 启动P2P通信
        comm_stream.wait_event(overlap_ctx.proj_ready_event)

        with torch.cuda.stream(comm_stream):
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_buffer, partner, group=cp_group),
                dist.P2POp(dist.isend, partial_to_send.contiguous(), partner, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        # 记录本轮通信完成
        round_event = overlap_ctx.get_round_event(round_idx)
        round_event.record(comm_stream)

        # 等待通信完成后累加
        default_stream.wait_event(round_event)
        output = output + recv_buffer

    # 添加bias
    if bias_proj is not None:
        output = output + bias_proj

    return output


class _AttentionMultiCardQKVFunction(torch.autograd.Function):
    """注意力层多卡QKV sp2hp的自动求导包装"""

    @staticmethod
    def forward(ctx, hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
                cp_group, overlap_ctx, layer_name, layer_id):
        ctx.cp_group = cp_group
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id

        ctx.save_for_backward(hidden_states, weight_qkv)

        # 不使用 torch.no_grad()，让 PyTorch 正确包装返回值
        q, k, v = attention_multicard_qkv_sp2hp(
            hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
            cp_group, overlap_ctx
        )

        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        hidden_states, weight_qkv = ctx.saved_tensors
        cp_group = ctx.cp_group
        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        head_dim = ctx.head_dim

        cp_size = cp_group.size()
        my_rank = cp_group.rank()
        seq_local, batch_size, hidden_size = hidden_states.shape

        q_per_group = num_heads // num_kv_heads
        groups_per_rank = num_kv_heads // cp_size
        group_size = (q_per_group + 2) * head_dim
        heads_local = groups_per_rank * q_per_group

        # 反向使用标准AllToAll hp2sp
        from fluid.communication import _all_to_all_hp2sp_forward

        # 合并grad_q, grad_k, grad_v为统一格式
        # grad_q: [seq_full, batch, heads_local, head_dim]
        # grad_k, grad_v: [seq_full, batch, kv_heads_local, head_dim]
        seq_full = grad_q.shape[0]

        # Reshape to interleaved format
        grad_q_grouped = grad_q.view(seq_full, batch_size, groups_per_rank, q_per_group * head_dim)
        grad_qkv = torch.cat([grad_q_grouped, grad_k, grad_v], dim=-1)
        # grad_qkv: [seq_full, batch, groups_per_rank, group_size]

        # hp2sp AllToAll
        grad_qkv_sp = _all_to_all_hp2sp_forward(
            grad_qkv.view(seq_full, batch_size, groups_per_rank * group_size, 1).squeeze(-1).unsqueeze(-1),
            cp_group
        ).squeeze(-1)
        # grad_qkv_sp: [seq_local, batch, groups_per_rank * group_size * cp_size]

        # 需要正确reshape
        grad_qkv_sp = grad_qkv_sp.view(seq_local, batch_size, -1)

        # 计算grad_hidden
        # 每个rank的权重对应不同的output位置
        weight_grouped = weight_qkv.view(num_kv_heads, group_size, hidden_size)

        grad_hidden = torch.zeros(seq_local, batch_size, hidden_size, dtype=hidden_states.dtype, device=hidden_states.device)

        for rank in range(cp_size):
            rank_group_start = rank * groups_per_rank
            rank_group_end = rank_group_start + groups_per_rank
            weight_rank = weight_grouped[rank_group_start:rank_group_end].reshape(groups_per_rank * group_size, hidden_size)

            # grad对应这个rank的部分
            grad_start = rank * groups_per_rank * group_size
            grad_end = grad_start + groups_per_rank * group_size
            grad_rank = grad_qkv_sp[:, :, grad_start:grad_end]

            grad_hidden += torch.matmul(grad_rank, weight_rank)

        # 注册dW任务
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        if scheduler.is_enabled():
            hidden_flat_saved = hidden_states.view(-1, hidden_size).detach()
            grad_qkv_sp_saved = grad_qkv_sp.view(-1, grad_qkv_sp.shape[-1]).detach()

            def compute_dw_qkv():
                # 计算完整的QKV权重梯度
                grad_weight = torch.zeros_like(weight_qkv)

                for rank in range(cp_size):
                    rank_group_start = rank * groups_per_rank
                    grad_start = rank * groups_per_rank * group_size
                    grad_end = grad_start + groups_per_rank * group_size
                    grad_rank = grad_qkv_sp_saved[:, grad_start:grad_end]

                    # dW = grad.T @ hidden
                    grad_weight_rank = torch.matmul(grad_rank.t(), hidden_flat_saved)
                    # 放到正确位置
                    weight_start = rank_group_start * group_size
                    weight_end = weight_start + groups_per_rank * group_size
                    grad_weight[weight_start:weight_end] = grad_weight_rank

                return grad_weight

            scheduler.register_dw_task(
                layer_name=f"{ctx.layer_name}_weight",
                layer_id=ctx.layer_id,
                compute_fn=compute_dw_qkv,
                priority=100,
                weight_param=weight_qkv,
            )

            grad_weight = None
        else:
            # 直接计算
            hidden_flat = hidden_states.view(-1, hidden_size)
            grad_qkv_sp_flat = grad_qkv_sp.view(-1, grad_qkv_sp.shape[-1])
            grad_weight = torch.matmul(grad_qkv_sp_flat.t(), hidden_flat)

        return (grad_hidden, grad_weight, None, None, None, None, None, None, None)


def attention_multicard_qkv_sp2hp_with_grad(
    hidden_states: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    overlap_ctx: AttentionMultiCardOverlapContext,
    layer_name: str = "qkv",
    layer_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    注意力层多卡QKV sp2hp（带梯度支持）

    Args:
        hidden_states: [seq_local, batch, hidden]
        weight_qkv: [total_proj, hidden] QKV权重
        num_heads: Q heads总数
        num_kv_heads: K/V heads总数
        head_dim: 每个head的维度
        cp_group: Context Parallel进程组
        overlap_ctx: 多卡overlap上下文
        layer_name: 层名称
        layer_id: 层ID

    Returns:
        q, k, v: [seq_full, batch, heads_local, head_dim]
    """
    if hidden_states.requires_grad:
        return _AttentionMultiCardQKVFunction.apply(
            hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
            cp_group, overlap_ctx, layer_name, layer_id
        )
    else:
        return attention_multicard_qkv_sp2hp(
            hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
            cp_group, overlap_ctx
        )


def _moe_multicard_p2p_impl(
    tokens: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    ep_group: dist.ProcessGroup,
    activation_func,
    overlap_ctx: MultiCardOverlapContext,
    layer_id: int = 0,
    num_local_experts: int = 1,
    tokens_per_expert: torch.Tensor = None,
    num_global_tokens_per_local_expert: torch.Tensor = None,
) -> torch.Tensor:
    """MoE多卡P2P重叠内部实现（无梯度追踪）

    关键：计算与通信重叠！
    - Dispatch P2P 与 本地 FC1+Act 并行
    - Combine P2P 与 本地 FC2 并行
    """
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    device = tokens.device
    hidden_size = tokens.shape[-1]
    dtype = tokens.dtype

    total_ffn_hidden = weight1.shape[-1]
    ffn_hidden = total_ffn_hidden // num_local_experts

    w1 = weight1.view(num_local_experts, hidden_size, ffn_hidden)
    w2 = weight2.view(num_local_experts, ffn_hidden, hidden_size)

    input_splits_list = input_splits.tolist() if torch.is_tensor(input_splits) else list(input_splits)
    output_splits_list = output_splits.tolist() if torch.is_tensor(output_splits) else list(output_splits)

    input_offsets = [0]
    for s in input_splits_list:
        input_offsets.append(input_offsets[-1] + s)

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()
    num_rounds = overlap_ctx.num_rounds

    local_count = input_splits_list[my_rank]
    local_start = input_offsets[my_rank]

    local_tokens_per_expert = None
    if num_global_tokens_per_local_expert is not None:
        local_tokens_per_expert = [
            num_global_tokens_per_local_expert[0, my_rank, exp_idx].item()
            for exp_idx in range(num_local_experts)
        ]

    # =========================================================================
    # Dispatch阶段：先启动通信，本地计算与通信并行！
    # =========================================================================

    # 记录数据准备好的事件
    overlap_ctx.data_ready_event.record(default_stream)
    comm_stream.wait_event(overlap_ctx.data_ready_event)

    # 准备所有dispatch的发送数据和接收buffer
    dispatch_send_data = {}
    dispatch_recv_data = {}

    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner == -1:
            continue

        send_count = input_splits_list[partner]
        recv_count = output_splits_list[partner]

        if send_count > 0:
            send_start = input_offsets[partner]
            send_tokens = tokens[send_start:send_start + send_count].contiguous()
        else:
            send_tokens = torch.empty(0, hidden_size, dtype=dtype, device=device)

        recv_tokens = torch.empty(recv_count, hidden_size, dtype=dtype, device=device) if recv_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)

        dispatch_send_data[round_idx] = (send_tokens, send_count, partner)
        dispatch_recv_data[round_idx] = (recv_tokens, recv_count, partner)

    # 启动所有dispatch P2P通信（在comm_stream上）
    dispatch_reqs = []
    with torch.cuda.stream(comm_stream):
        for round_idx in range(num_rounds):
            if round_idx not in dispatch_send_data:
                continue
            send_tokens, send_count, partner = dispatch_send_data[round_idx]
            recv_tokens, recv_count, _ = dispatch_recv_data[round_idx]

            p2p_ops = []
            if recv_count > 0:
                p2p_ops.append(dist.P2POp(dist.irecv, recv_tokens, partner, group=ep_group))
            if send_count > 0:
                p2p_ops.append(dist.P2POp(dist.isend, send_tokens, partner, group=ep_group))

            if p2p_ops:
                reqs = dist.batch_isend_irecv(p2p_ops)
                dispatch_reqs.extend(reqs)

    # 记录dispatch完成事件
    dispatch_done_event = torch.cuda.Event()
    dispatch_done_event.record(comm_stream)

    # 本地 FC1 + Act（与 dispatch 通信并行，在 default_stream 上执行！）
    local_tokens = tokens[local_start:local_start + local_count].clone() if local_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)

    if local_count > 0 and local_tokens_per_expert is not None:
        local_act = torch.zeros(local_count, ffn_hidden, dtype=dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = local_tokens_per_expert[exp_idx]
            if n_tok > 0:
                exp_tokens = local_tokens[start:start + n_tok]
                exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                local_act[start:start + n_tok] = activation_func(exp_fc1)
                start += n_tok
    elif local_count > 0:
        local_fc1 = torch.matmul(local_tokens, w1[0])
        local_act = activation_func(local_fc1)
    else:
        local_act = torch.empty(0, ffn_hidden, dtype=dtype, device=device)

    # 等待 dispatch P2P 完成
    default_stream.wait_event(dispatch_done_event)
    for req in dispatch_reqs:
        req.wait()

    # =========================================================================
    # 处理远端数据：FC1 + Act + FC2
    # =========================================================================
    round_fc2_results = {}
    for round_idx in range(num_rounds):
        if round_idx not in dispatch_recv_data:
            continue
        recv_tokens, recv_count, partner = dispatch_recv_data[round_idx]
        if recv_count > 0:
            peer_fc2 = _compute_expert_forward_per_source(
                recv_tokens, w1, w2, activation_func,
                num_local_experts, num_global_tokens_per_local_expert,
                partner, my_rank
            )
            round_fc2_results[round_idx] = (peer_fc2, partner, recv_count)

    # =========================================================================
    # Combine阶段：先启动通信，本地FC2与通信并行！
    # =========================================================================
    total_output = sum(input_splits_list)
    combined_output = torch.empty(total_output, hidden_size, dtype=dtype, device=device)

    # 记录combine准备好的事件
    overlap_ctx.data_ready_event.record(default_stream)
    comm_stream.wait_event(overlap_ctx.data_ready_event)

    # 准备所有combine的发送数据和接收buffer
    combine_recv_info = {}
    combine_reqs = []

    with torch.cuda.stream(comm_stream):
        for round_idx in range(num_rounds):
            partner = overlap_ctx.get_partner(my_rank, round_idx)
            if partner == -1:
                continue

            if round_idx in round_fc2_results:
                send_fc2, _, send_count = round_fc2_results[round_idx]
            else:
                send_fc2 = torch.empty(0, hidden_size, dtype=dtype, device=device)
                send_count = 0

            recv_count = input_splits_list[partner]
            recv_start = input_offsets[partner]
            recv_buffer = combined_output[recv_start:recv_start + recv_count] if recv_count > 0 else torch.empty(0, hidden_size, dtype=dtype, device=device)
            combine_recv_info[round_idx] = (recv_buffer, recv_count, recv_start)

            p2p_ops = []
            if recv_count > 0:
                p2p_ops.append(dist.P2POp(dist.irecv, recv_buffer, partner, group=ep_group))
            if send_count > 0 and send_fc2.numel() > 0:
                p2p_ops.append(dist.P2POp(dist.isend, send_fc2.contiguous(), partner, group=ep_group))

            if p2p_ops:
                reqs = dist.batch_isend_irecv(p2p_ops)
                combine_reqs.extend(reqs)

    # 记录combine完成事件
    combine_done_event = torch.cuda.Event()
    combine_done_event.record(comm_stream)

    # 本地 FC2（与 combine 通信并行，在 default_stream 上执行！）
    if local_count > 0 and local_tokens_per_expert is not None:
        local_fc2 = torch.zeros(local_count, hidden_size, dtype=dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = local_tokens_per_expert[exp_idx]
            if n_tok > 0:
                exp_act = local_act[start:start + n_tok]
                local_fc2[start:start + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                start += n_tok
    elif local_count > 0:
        local_fc2 = torch.matmul(local_act, w2[0])
    else:
        local_fc2 = None

    # 写入本地结果
    if local_fc2 is not None:
        combined_output[local_start:local_start + local_count] = local_fc2

    # 等待 combine P2P 完成
    default_stream.wait_event(combine_done_event)
    for req in combine_reqs:
        req.wait()

    return combined_output


# =============================================================================
# 注意力层多卡P2P通信重叠（Ulysses SP / Context Parallel）
# =============================================================================

def _qkv_sp2hp_multicard_impl(
    hidden_states: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV 计算 + sp2hp 多卡 P2P 重叠实现（无梯度追踪）

    多卡扩展：使用 Round-Robin 调度，流水线重叠计算和通信

    核心思路（流水线重叠，与 overlap_forward.py 一致）：
    - Round 0: 计算 peer_0 的 QKV，启动 P2P_0
    - Round i (i > 0): 与 P2P_{i-1} 并行计算 peer_i 的 QKV，启动 P2P_i
    - 最后: 与最后一轮 P2P 并行计算 local QKV，等待所有 P2P 完成

    这样每一轮的 QKV 计算都与上一轮的 P2P 通信重叠！

    Args:
        hidden_states: [seq_local, B, hidden]
        weight_qkv: [total_proj, hidden] 完整 QKV 权重
        num_heads: Q heads 总数
        num_kv_heads: K/V heads 总数 (groups)
        head_dim: 每个 head 的维度
        cp_group: Context Parallel 进程组
        overlap_ctx: 多卡 overlap 上下文

    Returns:
        q, k, v: [seq_full, B, heads_local, head_dim]
    """
    cp_size = cp_group.size()
    my_rank = cp_group.rank()
    device = hidden_states.device
    dtype = hidden_states.dtype
    seq_local, batch_size, hidden_size = hidden_states.shape

    if cp_size == 1:
        # 单卡：直接计算
        qkv = torch.matmul(hidden_states, weight_qkv.t())
        qkv = qkv.view(seq_local, batch_size, num_kv_heads, -1)
        q_per_group = num_heads // num_kv_heads
        q_size = q_per_group * head_dim
        q, k, v = torch.split(qkv, [q_size, head_dim, head_dim], dim=-1)
        q = q.reshape(seq_local, batch_size, num_heads, head_dim)
        return q, k, v

    # 多卡设置
    num_rounds = cp_size - 1
    seq_full = seq_local * cp_size
    q_per_group = num_heads // num_kv_heads
    groups_per_rank = num_kv_heads // cp_size
    heads_local = groups_per_rank * q_per_group
    group_size = (q_per_group + 2) * head_dim
    proj_per_rank = groups_per_rank * group_size

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    # 按 group 分割权重: [num_groups, group_size, hidden]
    weight_grouped = weight_qkv.view(num_kv_heads, group_size, hidden_size)

    # 本地 groups 的权重
    local_group_start = my_rank * groups_per_rank
    weight_local = weight_grouped[local_group_start:local_group_start + groups_per_rank]
    weight_local = weight_local.reshape(-1, hidden_size)  # [proj_per_rank, hidden]

    # 预先准备各个 peer 的权重
    weight_per_partner = {}
    partners = []
    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner == -1:
            continue
        partners.append(partner)
        r_group_start = partner * groups_per_rank
        weight_r = weight_grouped[r_group_start:r_group_start + groups_per_rank]
        weight_per_partner[partner] = weight_r.reshape(-1, hidden_size)

    # =========================================================================
    # 流水线重叠：每个 remote matmul 都与上一轮 P2P 并行
    # =========================================================================
    # 时序（假设有 partners = [p1, p2, p3]）:
    #   Round -1: 计算 matmul_p1，event_p1.synchronize()
    #   Round 0:  启动 P2P_0，同时计算 matmul_p2（与 P2P_0 并行）
    #   Round 1:  event_p2.synchronize()，启动 P2P_1，同时计算 matmul_p3
    #   Round 2:  event_p3.synchronize()，启动 P2P_2，同时计算 matmul_local
    #
    # 使用 event.synchronize()（CPU端等待）：
    # - CPU 等待 matmul 完成后，同时提交 P2P 和下一个 matmul
    # - 这样 GPU 可以识别出两者可以并行执行

    qkv_full = torch.empty(seq_full, batch_size, proj_per_rank, dtype=dtype, device=device)
    send_data_dict = {}
    send_data_events = {}

    # =========================================================================
    # Round -1：预计算第一轮的 send_data
    # =========================================================================
    if len(partners) > 0:
        first_partner = partners[0]
        send_data_dict[first_partner] = torch.matmul(
            hidden_states, weight_per_partner[first_partner].t()
        )
        send_data_events[first_partner] = torch.cuda.Event()
        send_data_events[first_partner].record(default_stream)

    # =========================================================================
    # 流水线循环：使用 CPU 端 event.synchronize() 确保数据准备好
    # =========================================================================
    all_reqs = []
    for round_idx, partner in enumerate(partners):
        # 接收缓冲区
        partner_seq_start = partner * seq_local
        recv_buffer = qkv_full[partner_seq_start:partner_seq_start + seq_local]

        # CPU 端等待 send_data 准备好
        send_data_events[partner].synchronize()

        # 启动 P2P（send_data 已准备好）
        with torch.cuda.stream(comm_stream):
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_buffer, partner, group=cp_group),
                dist.P2POp(dist.isend, send_data_dict[partner], partner, group=cp_group),
            ]
            reqs = dist.batch_isend_irecv(p2p_ops)
            all_reqs.extend(reqs)

        # 与当前轮 P2P 并行：在 default_stream 上计算下一轮数据
        if round_idx + 1 < len(partners):
            next_partner = partners[round_idx + 1]
            send_data_dict[next_partner] = torch.matmul(
                hidden_states, weight_per_partner[next_partner].t()
            )
            send_data_events[next_partner] = torch.cuda.Event()
            send_data_events[next_partner].record(default_stream)
        else:
            # 最后一轮：计算 local QKV（与最后一轮 P2P 并行）
            qkv_local = torch.matmul(hidden_states, weight_local.t())

    # 等待所有 P2P 完成
    for req in all_reqs:
        req.wait()

    # 处理没有 partner 的情况（cp_size=1）
    if len(partners) == 0:
        qkv_local = torch.matmul(hidden_states, weight_local.t())

    # =========================================================================
    # 组装结果：本地数据写入对应位置
    # =========================================================================
    local_seq_start = my_rank * seq_local
    qkv_full[local_seq_start:local_seq_start + seq_local] = qkv_local

    # 分离 Q, K, V
    qkv_full = qkv_full.view(seq_full, batch_size, groups_per_rank, group_size)
    q_size = q_per_group * head_dim
    q, k, v = torch.split(qkv_full, [q_size, head_dim, head_dim], dim=-1)
    q = q.reshape(seq_full, batch_size, heads_local, head_dim)
    # k, v: [seq_full, B, kv_heads_local, head_dim]

    return q, k, v


def _hp2sp_output_proj_multicard_impl(
    attn_output: torch.Tensor,
    weight_proj: torch.Tensor,
    bias_proj: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
) -> torch.Tensor:
    """hp2sp + 输出投影多卡 P2P 重叠实现（无梯度追踪）

    多卡扩展：使用 Round-Robin 调度，每轮与一个 peer 交换原始 attn 数据

    核心思路（流水线重叠）：
    - Round 0: 启动 P2P_0，同时计算 local partial
    - Round i (i > 0): 等待 P2P_{i-1}，启动 P2P_i，同时计算 recv_{i-1} 的 partial
    - 最后: 等待最后一轮 P2P，计算最后的 partial

    这样每一轮的 partial 计算都与下一轮的 P2P 通信重叠！

    hp2sp + projection 语义：
    - 输入: attn_output [seq_full, B, heads_local, head_dim] - 全序列，本地 heads
    - 输出: [seq_local, B, hidden] - 本地序列，完整 hidden

    Args:
        attn_output: [seq_full, B, heads_local, head_dim]
        weight_proj: [hidden, total_heads * head_dim]
        bias_proj: [hidden] 或 None
        cp_group: Context Parallel 进程组
        overlap_ctx: 多卡 overlap 上下文

    Returns:
        output: [seq_local, B, hidden]
    """
    cp_size = cp_group.size()
    my_rank = cp_group.rank()
    device = attn_output.device
    dtype = attn_output.dtype

    if cp_size == 1:
        attn_flat = attn_output.view(attn_output.shape[0], attn_output.shape[1], -1)
        output = torch.matmul(attn_flat, weight_proj.t())
        if bias_proj is not None:
            output = output + bias_proj
        return output

    seq_full, batch_size, heads_local, head_dim = attn_output.shape
    seq_local = seq_full // cp_size
    hidden_size = weight_proj.shape[0]
    input_dim_per_rank = heads_local * head_dim
    num_rounds = cp_size - 1

    default_stream = torch.cuda.current_stream(device)
    comm_stream = overlap_ctx.get_stream()

    # 本地序列的起始位置
    local_seq_start = my_rank * seq_local

    # 本地 heads 对应的权重
    weight_local_start = my_rank * input_dim_per_rank
    weight_local = weight_proj[:, weight_local_start:weight_local_start + input_dim_per_rank]

    # =========================================================================
    # 准备所有要发送的数据和接收缓冲区
    # =========================================================================
    send_data_dict = {}
    recv_buffers = {}
    weight_per_partner = {}
    partners = []

    for round_idx in range(num_rounds):
        partner = overlap_ctx.get_partner(my_rank, round_idx)
        if partner == -1:
            continue
        partners.append(partner)

        # 准备发送数据：peer 序列位置的 attn_output（我的 heads 数据）
        partner_seq_start = partner * seq_local
        send_data_dict[partner] = attn_output[partner_seq_start:partner_seq_start + seq_local].contiguous()

        # 准备接收缓冲区
        recv_buffers[partner] = torch.empty(seq_local, batch_size, heads_local, head_dim, dtype=dtype, device=device)

        # 缓存 partner 的权重
        partner_weight_start = partner * input_dim_per_rank
        weight_per_partner[partner] = weight_proj[:, partner_weight_start:partner_weight_start + input_dim_per_rank]

    # =========================================================================
    # 流水线重叠：每一轮 partial 计算与下一轮 P2P 通信重叠
    # =========================================================================
    # 初始化输出
    attn_local_seq = attn_output[local_seq_start:local_seq_start + seq_local]
    attn_local_flat = attn_local_seq.view(seq_local, batch_size, -1)

    prev_reqs = None
    prev_partner = None

    for round_idx, partner in enumerate(partners):
        # 启动当前轮的 P2P 通信
        with torch.cuda.stream(comm_stream):
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_buffers[partner], partner, group=cp_group),
                dist.P2POp(dist.isend, send_data_dict[partner], partner, group=cp_group),
            ]
            curr_reqs = dist.batch_isend_irecv(p2p_ops)

        # 与 P2P 并行：计算上一轮接收数据的 partial（或第一轮计算 local partial）
        if round_idx == 0:
            # 第一轮：计算 local partial（与 P2P_0 并行）
            output = torch.matmul(attn_local_flat, weight_local.t())
        else:
            # 等待上一轮 P2P 完成
            for req in prev_reqs:
                req.wait()
            # 计算上一轮接收数据的 partial（与当前轮 P2P 并行）
            recv_flat = recv_buffers[prev_partner].view(seq_local, batch_size, -1)
            output = output + torch.matmul(recv_flat, weight_per_partner[prev_partner].t())

        prev_reqs = curr_reqs
        prev_partner = partner

    # 等待最后一轮 P2P 完成，计算最后的 partial
    if prev_reqs is not None:
        for req in prev_reqs:
            req.wait()
        recv_flat = recv_buffers[prev_partner].view(seq_local, batch_size, -1)
        output = output + torch.matmul(recv_flat, weight_per_partner[prev_partner].t())

    if bias_proj is not None:
        output = output + bias_proj

    return output


# =============================================================================
# 注意力层多卡重叠的公共 API
# =============================================================================

def qkv_sp2hp_multicard_overlap(
    hidden_states: torch.Tensor,
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
    layer_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV 计算 + sp2hp 多卡 P2P 重叠

    多卡版本的 qkv_sp2hp_heads_split，支持任意数量的 GPU。

    Args:
        hidden_states: [seq_local, B, hidden]
        weight_qkv: [total_proj, hidden] 完整 QKV 权重
        num_heads: Q heads 总数
        num_kv_heads: K/V heads 总数 (groups)
        head_dim: 每个 head 的维度
        cp_group: Context Parallel 进程组
        overlap_ctx: 多卡 overlap 上下文
        layer_id: 层 ID（用于 dW 任务注册）

    Returns:
        q, k, v: [seq_full, B, heads_local, head_dim]
    """
    if hidden_states.requires_grad:
        return _QKVSp2HpMultiCardFunction.apply(
            hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
            cp_group, overlap_ctx, layer_id
        )
    else:
        return _qkv_sp2hp_multicard_impl(
            hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
            cp_group, overlap_ctx
        )


def hp2sp_output_proj_multicard_overlap(
    attn_output: torch.Tensor,
    weight_proj: torch.Tensor,
    bias_proj: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    overlap_ctx: MultiCardOverlapContext,
) -> torch.Tensor:
    """hp2sp + 输出投影多卡 P2P 重叠

    多卡版本的 hp2sp_output_proj_overlap，支持任意数量的 GPU。

    Args:
        attn_output: [seq_full, B, heads_local, head_dim]
        weight_proj: [hidden, total_heads * head_dim]
        bias_proj: [hidden] 或 None
        cp_group: Context Parallel 进程组
        overlap_ctx: 多卡 overlap 上下文

    Returns:
        output: [seq_local, B, hidden]
    """
    if attn_output.requires_grad:
        return _HP2SpOutputProjMultiCardFunction.apply(
            attn_output, weight_proj, bias_proj, cp_group, overlap_ctx
        )
    else:
        return _hp2sp_output_proj_multicard_impl(
            attn_output, weight_proj, bias_proj, cp_group, overlap_ctx
        )


# =============================================================================
# 注意力层多卡重叠的 autograd.Function 包装
# =============================================================================

class _QKVSp2HpMultiCardFunction(torch.autograd.Function):
    """QKV sp2hp 多卡 P2P 重叠的自动求导包装"""

    @staticmethod
    def forward(ctx, hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
                cp_group, overlap_ctx, layer_id):
        ctx.cp_group = cp_group
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim
        ctx.layer_id = layer_id

        ctx.save_for_backward(hidden_states, weight_qkv)

        # 不使用 torch.no_grad()，让 PyTorch 正确包装返回值
        q, k, v = _qkv_sp2hp_multicard_impl(
            hidden_states, weight_qkv, num_heads, num_kv_heads, head_dim,
            cp_group, overlap_ctx
        )

        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        """反向传播使用标准 AllToAll"""
        from fluid.communication import _all_to_all_hp2sp_forward

        hidden_states, weight_qkv = ctx.saved_tensors
        cp_group = ctx.cp_group
        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        head_dim = ctx.head_dim

        cp_size = cp_group.size()
        my_rank = cp_group.rank()
        device = hidden_states.device
        seq_local, batch_size, hidden_size = hidden_states.shape

        q_per_group = num_heads // num_kv_heads
        groups_per_rank = num_kv_heads // cp_size
        group_size = (q_per_group + 2) * head_dim
        heads_local = groups_per_rank * q_per_group
        seq_full = grad_q.shape[0]

        # 合并 grad_q, grad_k, grad_v 为 interleaved 格式
        # grad_q: [seq_full, batch, heads_local, head_dim]
        # grad_k, grad_v: [seq_full, batch, kv_heads_local, head_dim]
        grad_q_grouped = grad_q.view(seq_full, batch_size, groups_per_rank, q_per_group * head_dim)
        grad_qkv = torch.cat([grad_q_grouped, grad_k, grad_v], dim=-1)
        # grad_qkv: [seq_full, batch, groups_per_rank, group_size]

        # hp2sp AllToAll: 将 seq 维度从 full 变为 local
        # 需要先 reshape 成 AllToAll 期望的格式
        grad_qkv_flat = grad_qkv.view(seq_full, batch_size, -1)  # [seq_full, B, groups_local * group_size]

        # AllToAll: seq_full -> seq_local, 同时收集所有 rank 的 groups
        grad_qkv_parts = []
        for r in range(cp_size):
            r_seq_start = r * seq_local
            grad_qkv_parts.append(grad_qkv_flat[r_seq_start:r_seq_start + seq_local])

        # 使用 AllToAll 交换
        grad_qkv_send = torch.stack(grad_qkv_parts, dim=0)  # [cp_size, seq_local, B, groups_local * group_size]
        grad_qkv_recv = torch.empty_like(grad_qkv_send)

        dist.all_to_all_single(
            grad_qkv_recv.view(cp_size, -1),
            grad_qkv_send.view(cp_size, -1),
            group=cp_group
        )

        # 重组为 [seq_local, B, total_groups * group_size]
        grad_qkv_recv = grad_qkv_recv.permute(1, 2, 0, 3).contiguous()
        grad_qkv_sp = grad_qkv_recv.view(seq_local, batch_size, -1)

        # 计算 grad_hidden
        weight_grouped = weight_qkv.view(num_kv_heads, group_size, hidden_size)
        grad_hidden = torch.zeros(seq_local, batch_size, hidden_size, dtype=hidden_states.dtype, device=device)

        for rank in range(cp_size):
            rank_group_start = rank * groups_per_rank
            weight_rank = weight_grouped[rank_group_start:rank_group_start + groups_per_rank]
            weight_rank = weight_rank.reshape(-1, hidden_size)

            grad_start = rank * groups_per_rank * group_size
            grad_end = grad_start + groups_per_rank * group_size
            grad_rank = grad_qkv_sp[:, :, grad_start:grad_end]

            grad_hidden += torch.matmul(grad_rank, weight_rank)

        # 注册 dW 任务
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        if scheduler.is_enabled():
            hidden_flat_saved = hidden_states.view(-1, hidden_size).detach()
            grad_qkv_sp_saved = grad_qkv_sp.view(-1, grad_qkv_sp.shape[-1]).detach()
            weight_qkv_saved = weight_qkv
            num_kv_heads_saved = num_kv_heads
            groups_per_rank_saved = groups_per_rank
            group_size_saved = group_size
            cp_size_saved = cp_size
            layer_id_saved = ctx.layer_id

            def compute_dw_qkv():
                grad_weight = torch.zeros_like(weight_qkv_saved)
                for rank in range(cp_size_saved):
                    rank_group_start = rank * groups_per_rank_saved
                    grad_start = rank * groups_per_rank_saved * group_size_saved
                    grad_end = grad_start + groups_per_rank_saved * group_size_saved
                    grad_rank = grad_qkv_sp_saved[:, grad_start:grad_end]

                    grad_weight_rank = torch.matmul(grad_rank.t(), hidden_flat_saved)
                    weight_start = rank_group_start * group_size_saved
                    weight_end = weight_start + groups_per_rank_saved * group_size_saved
                    grad_weight[weight_start:weight_end] = grad_weight_rank

                return grad_weight

            scheduler.register_dw_task(
                layer_name=f"qkv_multicard_layer{layer_id_saved}",
                layer_id=layer_id_saved,
                compute_fn=compute_dw_qkv,
                priority=100,
                weight_param=weight_qkv_saved,
            )
            grad_weight = None
        else:
            hidden_flat = hidden_states.view(-1, hidden_size)
            grad_qkv_sp_flat = grad_qkv_sp.view(-1, grad_qkv_sp.shape[-1])

            grad_weight = torch.zeros_like(weight_qkv)
            for rank in range(cp_size):
                rank_group_start = rank * groups_per_rank
                grad_start = rank * groups_per_rank * group_size
                grad_end = grad_start + groups_per_rank * group_size
                grad_rank = grad_qkv_sp_flat[:, grad_start:grad_end]

                grad_weight_rank = torch.matmul(grad_rank.t(), hidden_flat)
                weight_start = rank_group_start * group_size
                weight_end = weight_start + groups_per_rank * group_size
                grad_weight[weight_start:weight_end] = grad_weight_rank

        return (grad_hidden, grad_weight, None, None, None, None, None, None)


class _HP2SpOutputProjMultiCardFunction(torch.autograd.Function):
    """hp2sp + 输出投影多卡 P2P 重叠的自动求导包装"""

    @staticmethod
    def forward(ctx, attn_output, weight_proj, bias_proj, cp_group, overlap_ctx):
        ctx.cp_group = cp_group
        ctx.has_bias = bias_proj is not None

        ctx.save_for_backward(attn_output, weight_proj)

        # 不使用 torch.no_grad()，让 PyTorch 正确包装返回值
        output = _hp2sp_output_proj_multicard_impl(
            attn_output, weight_proj, bias_proj, cp_group, overlap_ctx
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播使用标准 AllToAll"""
        attn_output, weight_proj = ctx.saved_tensors
        cp_group = ctx.cp_group
        has_bias = ctx.has_bias

        cp_size = cp_group.size()
        my_rank = cp_group.rank()
        device = grad_output.device

        seq_full, batch_size, heads_local, head_dim = attn_output.shape
        seq_local = seq_full // cp_size
        hidden_size = weight_proj.shape[0]
        input_dim_per_rank = heads_local * head_dim

        # grad_output: [seq_local, B, hidden]
        # 需要计算 grad_attn_output: [seq_full, B, heads_local, head_dim]

        # sp2hp AllToAll: seq_local -> seq_full
        grad_output_send = grad_output.contiguous()
        grad_output_recv = torch.empty(cp_size, seq_local, batch_size, hidden_size, dtype=grad_output.dtype, device=device)

        dist.all_gather_into_tensor(
            grad_output_recv.view(cp_size, -1),
            grad_output_send.view(-1),
            group=cp_group
        )

        # 重组为 [seq_full, B, hidden]
        grad_output_full = grad_output_recv.permute(1, 0, 2, 3).contiguous().view(seq_full, batch_size, hidden_size)

        # 计算 grad_attn_output
        # 每个 rank 只计算自己负责的 heads 的梯度
        weight_local_start = my_rank * input_dim_per_rank
        weight_local = weight_proj[:, weight_local_start:weight_local_start + input_dim_per_rank]

        grad_attn_local = torch.matmul(grad_output_full, weight_local)
        grad_attn_output = grad_attn_local.view(seq_full, batch_size, heads_local, head_dim)

        # 注册 dW 任务
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        if scheduler.is_enabled():
            # 提取本地序列用于计算 dW
            local_seq_start = my_rank * seq_local
            attn_local = attn_output[local_seq_start:local_seq_start + seq_local]
            attn_local_flat_saved = attn_local.view(seq_local * batch_size, -1).detach()
            grad_output_flat_saved = grad_output.view(seq_local * batch_size, hidden_size).detach()
            weight_proj_saved = weight_proj
            input_dim_per_rank_saved = input_dim_per_rank
            my_rank_saved = my_rank

            def compute_dw_proj():
                grad_weight = torch.zeros_like(weight_proj_saved)
                weight_start = my_rank_saved * input_dim_per_rank_saved
                weight_end = weight_start + input_dim_per_rank_saved

                # dW = grad_output.T @ attn_local_flat
                grad_weight[:, weight_start:weight_end] = torch.matmul(
                    grad_output_flat_saved.t(), attn_local_flat_saved
                )
                return grad_weight

            scheduler.register_dw_task(
                layer_name=f"proj_multicard_layer{my_rank_saved}",
                layer_id=my_rank_saved,
                compute_fn=compute_dw_proj,
                priority=99,
                weight_param=weight_proj_saved,
            )
            grad_weight = None
        else:
            local_seq_start = my_rank * seq_local
            attn_local = attn_output[local_seq_start:local_seq_start + seq_local]
            attn_local_flat = attn_local.view(seq_local * batch_size, -1)
            grad_output_flat = grad_output.view(seq_local * batch_size, hidden_size)

            grad_weight = torch.zeros_like(weight_proj)
            weight_start = my_rank * input_dim_per_rank
            weight_end = weight_start + input_dim_per_rank
            grad_weight[:, weight_start:weight_end] = torch.matmul(
                grad_output_flat.t(), attn_local_flat
            )

        # bias 梯度
        if has_bias:
            grad_bias = grad_output.sum(dim=(0, 1))
        else:
            grad_bias = None

        return (grad_attn_output, grad_weight, grad_bias, None, None)
