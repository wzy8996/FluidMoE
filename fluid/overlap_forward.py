"""
计算-通信重叠的前向传播实现

包含：
1. 注意力层 QKV + sp2hp：Heads Split 重叠
2. 注意力层 hp2sp + 输出投影：重叠优化
3. 专家层 MoE：P2P 重叠

设计原则：
- 前向使用 P2P 实现计算-通信重叠
- 反向使用标准 AllToAll，保持反向调度不变
- 使用 torch.autograd.Function 包装确保梯度正确流动
"""

import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Tuple

from fluid.communication import (
    _all_to_all_sp2hp_forward,
    _all_to_all_hp2sp_forward,
    fluid_all_to_all_moe_dispatch,  # 暂时保留，未使用
    fluid_all_to_all_moe_combine,    # 暂时保留，未使用
    _all_to_all,  # 底层AllToAll函数（用于backward）
)
from megatron.core.transformer.moe.moe_utils import sort_chunks_by_idxs
from fluid.scheduler import get_backward_scheduler  # 用于手动触发dW调度
from fluid.moe_layers import _gelu_grad_analytical  # 解析GELU梯度


def _compute_activation_derivative(fc1_output, activation_func, gated_linear_unit=False):
    """
    预计算激活函数导数（与 Baseline _FluidExpertComputation 一致）

    注意：使用 detach() 避免创建不必要的 autograd 计算图。
    act_deriv 在反向传播中仅作为常量使用，不需要对其求导。

    Returns:
        如果 gated_linear_unit=False: act_deriv
        如果 gated_linear_unit=True: (act_deriv, act_val, x_2)
    """
    # detach 避免为 act_deriv 计算创建 autograd 图（它只是反向传播中的常量）
    fc1_detached = fc1_output.detach()

    if gated_linear_unit:
        x_1, x_2 = torch.chunk(fc1_detached, 2, dim=-1)
        # 检测激活函数类型
        if activation_func == F.silu or (hasattr(activation_func, '__name__') and 'silu' in activation_func.__name__.lower()):
            sig = torch.sigmoid(x_1)
            act_deriv = sig * (1 + x_1 * (1 - sig))  # SiLU 导数
            act_val = x_1 * sig  # SiLU(x_1)
        else:
            act_deriv = _gelu_grad_analytical(x_1)  # GELU 导数
            act_val = F.gelu(x_1)  # GELU(x_1)
        return act_deriv, act_val, x_2
    else:
        # 非 GLU 激活
        if activation_func == F.silu or (hasattr(activation_func, '__name__') and 'silu' in activation_func.__name__.lower()):
            sig = torch.sigmoid(fc1_detached)
            act_deriv = sig * (1 + fc1_detached * (1 - sig))
        else:
            act_deriv = _gelu_grad_analytical(fc1_detached)
        return act_deriv


def _compute_activation_grad(grad_act, act_deriv, act_val=None, x_2=None, gated_linear_unit=False):
    """
    使用预计算的激活导数计算 grad_fc1（与 Baseline 一致）

    Args:
        grad_act: 激活输出的梯度
        act_deriv: 预计算的激活导数
        act_val: GLU 时的激活值 activation(x_1)
        x_2: GLU 时的 x_2 部分
        gated_linear_unit: 是否使用 GLU

    Returns:
        grad_fc1: FC1 输出的梯度
    """
    if gated_linear_unit:
        # GLU: output = activation(x_1) * x_2
        # grad_x_1 = grad_act * x_2 * activation'(x_1)
        # grad_x_2 = grad_act * activation(x_1)
        grad_x_1 = grad_act * x_2 * act_deriv
        grad_x_2 = grad_act * act_val
        grad_fc1 = torch.cat([grad_x_1, grad_x_2], dim=-1)
    else:
        # 非 GLU: output = activation(fc1)
        # grad_fc1 = grad_act * activation'(fc1)
        grad_fc1 = grad_act * act_deriv
    return grad_fc1


# =============================================================================
# 内部实现函数（无梯度）
# =============================================================================

def _all_to_all_qkv_hp2sp_batched(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将Q、K、V拼接后一次性hp2sp AllToAll（合并通信）

    支持 GQA：Q, K, V 可以有不同数量的 heads

    Args:
        q: [seq_full, B, q_heads_local, D]
        k: [seq_full, B, kv_heads_local, D]
        v: [seq_full, B, kv_heads_local, D]

    Returns:
        q_hp: [seq_local, B, q_heads, D]
        k_hp: [seq_local, B, kv_heads, D]
        v_hp: [seq_local, B, kv_heads, D]
    """
    cp_size = cp_group.size()
    if cp_size == 1:
        return q, k, v

    seq_full, batch, q_heads_local, head_dim = q.shape
    kv_heads_local = k.shape[2]
    q_heads = q_heads_local * cp_size
    kv_heads = kv_heads_local * cp_size

    # 拼接Q、K、V: [seq_full, B, q_heads_local + 2*kv_heads_local, D]
    qkv_batched = torch.cat([q, k, v], dim=2)

    # 一次性hp2sp AllToAll: [seq_full, B, total_local, D] -> [seq_local, B, total, D]
    qkv_hp = _all_to_all_hp2sp_forward(qkv_batched, cp_group)

    # 切分回Q、K、V
    # IMPORTANT: Make contiguous to avoid "view not compatible" errors
    q_hp = qkv_hp[:, :, :q_heads, :].contiguous()
    k_hp = qkv_hp[:, :, q_heads:q_heads+kv_heads, :].contiguous()
    v_hp = qkv_hp[:, :, q_heads+kv_heads:, :].contiguous()

    return q_hp, k_hp, v_hp


def _qkv_sp2hp_heads_split_impl(
    hidden_states: torch.Tensor,
    weight_local: torch.Tensor,
    weight_remote: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    comm_stream: torch.cuda.Stream,
    ev_ready: torch.cuda.Event,
    ev_comm_done: torch.cuda.Event,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """QKV 整体计算 + P2P 重叠（简化版，无梯度追踪）

    改进: 整体计算 QKV，不分离权重，只做 2 次 matmul

    Args:
        hidden_states: [seq_local, B, hidden]
        weight_local: [groups_local * group_size, hidden] 本地 groups 的权重
        weight_remote: [groups_local * group_size, hidden] 远程 groups 的权重
        num_heads: Q heads 总数
        num_kv_heads: K/V heads 总数 (groups)
        head_dim: 每个 head 的维度
        ev_ready: CUDA Event，用于同步 default stream 和 comm stream
        ev_comm_done: CUDA Event，用于标记通信完成
    """
    cp_size = cp_group.size()
    my_rank = cp_group.rank()
    device = hidden_states.device
    seq_local, batch_size, hidden_size = hidden_states.shape

    q_per_group = num_heads // num_kv_heads
    groups_local = num_kv_heads // cp_size
    heads_local = groups_local * q_per_group
    kv_heads_local = groups_local

    if cp_size == 1:
        # 直接用完整权重计算
        qkv = torch.matmul(hidden_states, weight_local.t())  # [seq, B, total_proj]
        # 分离 Q, K, V
        qkv = qkv.view(seq_local, batch_size, num_kv_heads, (q_per_group + 2) * head_dim)
        q_size = q_per_group * head_dim
        q, k, v = torch.split(qkv, [q_size, head_dim, head_dim], dim=-1)
        q = q.reshape(seq_local, batch_size, num_heads, head_dim)
        return q, k, v

    default_stream = torch.cuda.current_stream(device)
    peer_rank = 1 - my_rank

    # Step 1: 计算 remote groups (整体 matmul)
    qkv_remote = torch.matmul(hidden_states, weight_remote.t())  # [seq_local, B, groups_local * group_size]

    # Step 2: 启动 P2P 通信
    qkv_recv = torch.empty_like(qkv_remote)

    # 必须同步：确保 qkv_remote 计算完成后才能发送
    ev_ready.record(default_stream)
    comm_stream.wait_event(ev_ready)

    with torch.cuda.stream(comm_stream):
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.irecv, qkv_recv, peer_rank, group=cp_group),
            dist.P2POp(dist.isend, qkv_remote.contiguous(), peer_rank, group=cp_group),
        ])

    # Step 3: 计算 local groups（与通信重叠）
    qkv_local = torch.matmul(hidden_states, weight_local.t())  # [seq_local, B, groups_local * group_size]

    # Step 4: 等待 P2P 完成并同步 stream
    # 注意: batch_isend_irecv 在 comm_stream 上下文中启动
    # req.wait() 是 CPU 等待，但还需要 GPU stream 同步
    for req in reqs:
        req.wait()

    # NCCL 在 comm_stream 上完成，记录 event 并让 default_stream 等待
    ev_comm_done.record(comm_stream)
    default_stream.wait_event(ev_comm_done)

    # Step 5: 拼接序列，再分离 Q, K, V
    group_size = (q_per_group + 2) * head_dim
    q_size = q_per_group * head_dim

    # 拼接: [seq_local, B, ...] + [seq_local, B, ...] -> [seq_full, B, ...]
    if my_rank == 0:
        qkv_full = torch.cat([qkv_local, qkv_recv], dim=0)
    else:
        qkv_full = torch.cat([qkv_recv, qkv_local], dim=0)

    seq_full = seq_local * 2

    # 分离 Q, K, V
    qkv_full = qkv_full.view(seq_full, batch_size, groups_local, group_size)
    q, k, v = torch.split(qkv_full, [q_size, head_dim, head_dim], dim=-1)
    # q: [seq_full, B, groups_local, q_per_group * head_dim]
    # k, v: [seq_full, B, groups_local, head_dim]

    q = q.reshape(seq_full, batch_size, heads_local, head_dim)
    # k, v 已经是 [seq_full, B, kv_heads_local, head_dim]

    return q, k, v


def _hp2sp_output_proj_overlap_impl(
    attn_output: torch.Tensor,
    weight_proj: torch.Tensor,
    bias_proj: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    comm_stream: torch.cuda.Stream,
    ev_ready: torch.cuda.Event,
    ev_comm_done: torch.cuda.Event,
    weight_local: torch.Tensor = None,
    weight_peer: torch.Tensor = None,
) -> torch.Tensor:
    """hp2sp + 输出投影重叠内部实现（无梯度追踪）

    优化版本：使用 event-based 同步

    关键优化：Output Projection 的输入数据（attn_output）来自上一步 attention，
    已经在 default_stream 上准备好，不需要 ev_ready 同步。
    只需要 ev_comm_done 来确保 P2P 完成后才使用 recv_buffer。

    Args:
        ev_ready: CUDA Event（Output Proj 场景下不使用，保留参数以保持接口统一）
        ev_comm_done: CUDA Event，用于标记通信完成
        weight_local: 预先准备好的本地权重切片（可选，如果为 None 则在函数内切片）
        weight_peer: 预先准备好的对端权重切片（可选）
    """
    cp_size = cp_group.size()
    my_rank = cp_group.rank()
    device = attn_output.device

    if cp_size == 1:
        attn_flat = attn_output.view(attn_output.shape[0], attn_output.shape[1], -1)
        output = torch.matmul(attn_flat, weight_proj.t())
        if bias_proj is not None:
            output = output + bias_proj
        return output

    seq_full, batch_size, heads_local, head_dim = attn_output.shape
    seq_local = seq_full // cp_size
    input_dim_per_rank = heads_local * head_dim
    peer_rank = 1 - my_rank

    # Step 1: 准备数据
    local_seq_start = my_rank * seq_local
    peer_seq_start = peer_rank * seq_local

    attn_local_seq = attn_output[local_seq_start:local_seq_start + seq_local]
    send_data = attn_output[peer_seq_start:peer_seq_start + seq_local].contiguous()
    recv_buffer = torch.empty_like(send_data)

    # 使用预先准备的权重切片，或在函数内切片
    if weight_local is None or weight_peer is None:
        local_head_start = my_rank * input_dim_per_rank
        peer_head_start = peer_rank * input_dim_per_rank
        weight_local = weight_proj[:, local_head_start:local_head_start + input_dim_per_rank]
        weight_peer = weight_proj[:, peer_head_start:peer_head_start + input_dim_per_rank]

    attn_local_flat = attn_local_seq.view(seq_local, batch_size, -1)

    # Step 2: 启动 P2P 通信
    # 注意：Output Projection 不需要 ev_ready 同步！
    # 因为 attn_output 来自上一步 attention，已在 default_stream 上完成
    # 跳过 ev_ready 可以避免不必要的串行化，实现 1.20x 加速
    default_stream = torch.cuda.current_stream(device)

    with torch.cuda.stream(comm_stream):
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.irecv, recv_buffer, peer_rank, group=cp_group),
            dist.P2POp(dist.isend, send_data, peer_rank, group=cp_group),
        ])

    # Step 3: 计算本地输出投影（与 P2P 重叠）
    output_local_partial = torch.matmul(attn_local_flat, weight_local.t())

    # Step 4: 等待 P2P 完成并同步 stream
    # 注意: batch_isend_irecv 在 comm_stream 上下文中启动
    # req.wait() 是 CPU 等待，但还需要 GPU stream 同步
    for req in reqs:
        req.wait()

    # NCCL 在 comm_stream 上完成，记录 event 并让 default_stream 等待
    ev_comm_done.record(comm_stream)
    default_stream.wait_event(ev_comm_done)

    # Step 5: 计算远程输出投影并求和
    recv_flat = recv_buffer.view(seq_local, batch_size, -1)
    output_peer_partial = torch.matmul(recv_flat, weight_peer.t())

    # Step 6: 求和得到最终输出
    output = output_local_partial + output_peer_partial

    if bias_proj is not None:
        output = output + bias_proj

    return output


# =============================================================================
# 自动求导包装（前向 P2P，反向 AllToAll）
# =============================================================================

class _QKVSp2HpHeadsSplitFunction(torch.autograd.Function):
    """
    QKV Heads 分离重叠的自动求导包装（简化版：2个整体权重）

    前向：使用 P2P 重叠，2次 matmul
    反向：使用标准 AllToAll（保持反向调度不变）
    """

    @staticmethod
    def forward(ctx, hidden_states, weight_local, weight_remote,
                num_heads, num_kv_heads, head_dim, cp_group, comm_stream,
                weight_qkv_fused, layer_name, layer_id, ev_ready, ev_comm_done):
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim
        ctx.cp_group = cp_group
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id

        # 保存用于反向
        saved_tensors = [hidden_states, weight_local, weight_remote]
        if weight_qkv_fused is not None:
            saved_tensors.append(weight_qkv_fused)
            ctx.has_fused_weight = True
        else:
            ctx.has_fused_weight = False

        ctx.save_for_backward(*saved_tensors)

        # 前向使用 P2P 重叠（复用events）
        # 不使用 torch.no_grad()，让 PyTorch 正确包装返回值
        q, k, v = _qkv_sp2hp_heads_split_impl(
            hidden_states, weight_local, weight_remote,
            num_heads, num_kv_heads, head_dim,
            cp_group, comm_stream, ev_ready, ev_comm_done
        )

        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        # 提取saved tensors
        saved = ctx.saved_tensors
        hidden_states = saved[0]
        weight_local = saved[1]
        weight_remote = saved[2]

        if ctx.has_fused_weight:
            weight_qkv_fused = saved[3]
        else:
            weight_qkv_fused = None

        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        head_dim = ctx.head_dim
        cp_group = ctx.cp_group

        cp_size = cp_group.size()
        my_rank = cp_group.rank()
        peer_rank = 1 - my_rank
        seq_local, batch_size, hidden_size = hidden_states.shape

        # 计算关键维度
        q_per_group = num_heads // num_kv_heads
        groups_local = num_kv_heads // cp_size
        heads_local = groups_local * q_per_group
        kv_heads_local = groups_local
        group_size = (q_per_group + 2) * head_dim

        # 反向使用标准 AllToAll hp2sp（保持反向调度不变）
        # grad_q/k/v: [seq_full, B, heads_local, D] -> [seq_local, B, H, D]
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        if scheduler.is_enabled():
            comm_stream = scheduler.comm_stream
            default_stream = scheduler.default_stream

            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_q_sp, grad_k_sp, grad_v_sp = _all_to_all_qkv_hp2sp_batched(
                    grad_q, grad_k, grad_v, cp_group
                )

            scheduler._execute_all_dw_tasks_sync()
            default_stream.wait_stream(comm_stream)
        else:
            grad_q_sp, grad_k_sp, grad_v_sp = _all_to_all_qkv_hp2sp_batched(
                grad_q, grad_k, grad_v, cp_group
            )

        # grad_*_sp: [seq_local, B, H, D]
        # 分离 local 和 remote groups
        local_group_start = my_rank * groups_local
        remote_group_start = peer_rank * groups_local

        local_q_start = local_group_start * q_per_group
        remote_q_start = remote_group_start * q_per_group

        # Q 梯度按 heads 分离
        grad_q_local = grad_q_sp[:, :, local_q_start:local_q_start+heads_local, :]
        grad_q_remote = grad_q_sp[:, :, remote_q_start:remote_q_start+heads_local, :]

        # K/V 梯度按 groups 分离
        grad_k_local = grad_k_sp[:, :, local_group_start:local_group_start+kv_heads_local, :]
        grad_k_remote = grad_k_sp[:, :, remote_group_start:remote_group_start+kv_heads_local, :]
        grad_v_local = grad_v_sp[:, :, local_group_start:local_group_start+kv_heads_local, :]
        grad_v_remote = grad_v_sp[:, :, remote_group_start:remote_group_start+kv_heads_local, :]

        # 重组为 interleaved 布局（与权重布局匹配）
        # grad_q: [seq, B, groups_local, q_per_group, D] -> [seq, B, groups_local, q_per_group * D]
        grad_q_local = grad_q_local.view(seq_local, batch_size, groups_local, q_per_group * head_dim)
        grad_q_remote = grad_q_remote.view(seq_local, batch_size, groups_local, q_per_group * head_dim)

        # 拼接成 interleaved 布局: [Q, K, V] per group
        grad_qkv_local = torch.cat([grad_q_local, grad_k_local, grad_v_local], dim=-1)
        grad_qkv_remote = torch.cat([grad_q_remote, grad_k_remote, grad_v_remote], dim=-1)
        # grad_qkv_*: [seq_local, B, groups_local, group_size]

        # Flatten for matmul
        grad_qkv_local_flat = grad_qkv_local.view(seq_local, batch_size, -1)
        grad_qkv_remote_flat = grad_qkv_remote.view(seq_local, batch_size, -1)

        # 计算 grad_hidden = grad_qkv_local @ weight_local + grad_qkv_remote @ weight_remote
        grad_hidden = torch.matmul(grad_qkv_local_flat, weight_local)
        grad_hidden = grad_hidden + torch.matmul(grad_qkv_remote_flat, weight_remote)

        # ===== 注册dW任务（如果有融合QKV权重参数）=====
        if ctx.has_fused_weight and weight_qkv_fused is not None:
            # 保存中间变量用于dW计算
            hidden_flat_saved = hidden_states.view(-1, hidden_size).detach()

            # 需要将 grad 转换回 interleaved 布局来计算 dW
            # grad_qkv_local/remote: [seq_local, B, groups_local, group_size]
            grad_qkv_local_saved = grad_qkv_local.view(-1, groups_local * group_size).detach()
            grad_qkv_remote_saved = grad_qkv_remote.view(-1, groups_local * group_size).detach()

            q_proj_size = num_heads * head_dim
            kv_proj_size = num_kv_heads * head_dim

            def compute_dw_qkv_fused():
                # 计算 local 和 remote 的权重梯度
                # grad_weight_local = grad_qkv_local.T @ hidden
                grad_weight_local = torch.matmul(grad_qkv_local_saved.t(), hidden_flat_saved)
                grad_weight_remote = torch.matmul(grad_qkv_remote_saved.t(), hidden_flat_saved)

                # grad_weight_*: [groups_local * group_size, hidden]
                # 需要重组为完整的 interleaved 布局

                # 初始化完整权重梯度
                total_proj = q_proj_size + 2 * kv_proj_size
                grad_weight_qkv = torch.zeros(total_proj, hidden_size,
                                              device=weight_qkv_fused.device, dtype=weight_qkv_fused.dtype)

                # 将 local 和 remote 的梯度放到正确位置
                # 权重布局: [group0, group1, ...] 其中每个 group = [Q, K, V]
                local_size = groups_local * group_size
                grad_weight_local_grouped = grad_weight_local.view(groups_local, group_size, hidden_size)
                grad_weight_remote_grouped = grad_weight_remote.view(groups_local, group_size, hidden_size)

                # 将每个 group 的梯度放回正确位置
                for g in range(groups_local):
                    global_group_local = local_group_start + g
                    global_group_remote = remote_group_start + g

                    # Local group
                    start_local = global_group_local * group_size
                    grad_weight_qkv[start_local:start_local+group_size, :] = grad_weight_local_grouped[g]

                    # Remote group
                    start_remote = global_group_remote * group_size
                    grad_weight_qkv[start_remote:start_remote+group_size, :] = grad_weight_remote_grouped[g]

                return grad_weight_qkv

            scheduler.register_dw_task(
                layer_name=f"{ctx.layer_name}_weight",
                layer_id=ctx.layer_id,
                compute_fn=compute_dw_qkv_fused,
                priority=100,
                weight_param=weight_qkv_fused,
            )

            grad_weight_local = None
            grad_weight_remote = None
        else:
            # 如果没有融合权重参数，直接计算权重梯度
            hidden_flat = hidden_states.view(-1, hidden_size)
            grad_qkv_local_2d = grad_qkv_local.view(-1, groups_local * group_size)
            grad_qkv_remote_2d = grad_qkv_remote.view(-1, groups_local * group_size)

            grad_weight_local = torch.matmul(grad_qkv_local_2d.t(), hidden_flat)
            grad_weight_remote = torch.matmul(grad_qkv_remote_2d.t(), hidden_flat)

        return (grad_hidden, grad_weight_local, grad_weight_remote,
                None, None, None, None, None,  # num_heads, num_kv_heads, head_dim, cp_group, comm_stream
                None, None, None, None, None)  # weight_qkv_fused, layer_name, layer_id, ev_ready, ev_comm_done


def _sort_chunks_by_idxs(input_tensor, split_sizes, sorted_idxs):
    """
    Sort chunks of input tensor by indices (simplified version of Megatron's sort_chunks_by_idxs)

    Args:
        input_tensor: [total_tokens, hidden] input tensor
        split_sizes: [num_chunks] size of each chunk
        sorted_idxs: [num_chunks] new order of chunks

    Returns:
        sorted_tensor: [total_tokens, hidden] sorted tensor
    """
    if input_tensor.numel() == 0:
        return input_tensor

    # Split into chunks
    split_sizes_list = split_sizes.tolist() if torch.is_tensor(split_sizes) else list(split_sizes)
    chunks = torch.split(input_tensor, split_sizes_list, dim=0)

    # Reorder chunks
    sorted_idxs_list = sorted_idxs.tolist() if torch.is_tensor(sorted_idxs) else list(sorted_idxs)
    sorted_chunks = [chunks[i] for i in sorted_idxs_list if i < len(chunks) and chunks[i].numel() > 0]

    if not sorted_chunks:
        return input_tensor

    return torch.cat(sorted_chunks, dim=0)


class _MoEP2POverlapFunction(torch.autograd.Function):
    """
    MoE P2P 重叠的自动求导包装

    前向：使用 P2P 重叠（dispatch 和 combine）
    反向：使用标准 AllToAll（保持反向调度不变）

    支持 num_local_experts > 1 的多 expert 场景
    """

    @staticmethod
    def forward(ctx, tokens, input_splits, output_splits, weight1, weight2,
                ep_group, activation_func, comm_stream, dispatch_event, combine_event, layer_id,
                num_local_experts=1, tokens_per_expert=None, num_global_tokens_per_local_expert=None,
                sort_indices=None, restore_indices=None):
        ctx.ep_group = ep_group
        ctx.activation_func = activation_func
        ctx.layer_id = layer_id
        ctx.num_local_experts = num_local_experts

        my_rank = ep_group.rank()
        ep_size = ep_group.size()
        device = tokens.device
        hidden_size = tokens.shape[-1]

        # 处理多 expert 的权重 shape
        # weight1: [hidden_size, ffn_hidden * num_local_experts]
        # weight2: [ffn_hidden * num_local_experts, hidden_size]
        total_ffn_hidden = weight1.shape[-1]
        ffn_hidden = total_ffn_hidden // num_local_experts

        # Reshape weights to per-expert view
        # NOTE: 必须与 BASELINE (_FluidExpertComputation) 保持一致！
        # BASELINE 使用: w1 = weight1.view(num_local_experts, hidden_size, -1)
        # 这里使用相同的方式以确保计算结果一致
        w1 = weight1.view(num_local_experts, hidden_size, ffn_hidden)  # [num_local_experts, hidden, ffn]
        w2 = weight2.view(num_local_experts, ffn_hidden, hidden_size)  # [num_local_experts, ffn, hidden]

        input_splits_list = input_splits.tolist() if torch.is_tensor(input_splits) else list(input_splits)
        output_splits_list = output_splits.tolist() if torch.is_tensor(output_splits) else list(output_splits)

        ctx.input_splits_list = input_splits_list
        ctx.output_splits_list = output_splits_list
        ctx.my_rank = my_rank
        ctx.ffn_hidden = ffn_hidden

        default_stream = torch.cuda.current_stream(device)

        # 计算偏移量
        input_offsets = [0]
        for s in input_splits_list:
            input_offsets.append(input_offsets[-1] + s)

        output_offsets = [0]
        for s in output_splits_list:
            output_offsets.append(output_offsets[-1] + s)

        local_count = input_splits_list[my_rank]
        recv_count = sum(output_splits_list) - output_splits_list[my_rank]
        send_count = sum(input_splits_list) - input_splits_list[my_rank]

        ctx.local_count = local_count
        ctx.recv_count = recv_count
        ctx.send_count = send_count

        # ===== Dispatch 阶段 (P2P) =====
        recv_buffer = torch.empty(recv_count, hidden_size, dtype=tokens.dtype, device=device) if recv_count > 0 else torch.empty(0, hidden_size, dtype=tokens.dtype, device=device)

        dispatch_event.record(default_stream)
        comm_stream.wait_event(dispatch_event)

        local_start = input_offsets[my_rank]
        ctx.local_start = local_start
        local_tokens = tokens[local_start:local_start + local_count].clone() if local_count > 0 else torch.empty(0, hidden_size, dtype=tokens.dtype, device=device)

        # 准备发送数据
        send_chunks = {}
        for i in range(ep_size):
            if i == my_rank: continue
            if input_splits_list[i] > 0:
                send_chunks[i] = tokens[input_offsets[i] : input_offsets[i+1]].contiguous()

        # 启动 P2P 通信 (Dispatch)
        with torch.cuda.stream(comm_stream):
            dispatch_ops = []
            recv_ptr = 0
            for i in range(ep_size):
                if i == my_rank: continue

                recv_size = output_splits_list[i]
                if recv_size > 0:
                    recv_chunk = recv_buffer[recv_ptr : recv_ptr + recv_size]
                    dispatch_ops.append(dist.P2POp(dist.irecv, recv_chunk, i, group=ep_group))
                    recv_ptr += recv_size

                if i in send_chunks:
                    dispatch_ops.append(dist.P2POp(dist.isend, send_chunks[i], i, group=ep_group))

            dispatch_reqs = dist.batch_isend_irecv(dispatch_ops) if dispatch_ops else []

        combine_event.record(comm_stream)

        # 本地 FC1 + Act（与 dispatch 重叠）
        # 注意：local_tokens 只包含 my_rank 发给自己的 tokens
        # 需要用 num_global_tokens_per_local_expert[0, my_rank, :] 而不是全局的 tokens_per_expert
        # 预计算 local_tokens_per_expert_list（后续 merge 操作也会用到）
        local_tokens_per_expert_list = None
        if num_global_tokens_per_local_expert is not None:
            local_tokens_per_expert_list = [
                num_global_tokens_per_local_expert[0, my_rank, exp_idx].item()
                for exp_idx in range(num_local_experts)
            ]

        if local_count > 0 and local_tokens_per_expert_list is not None:
            local_fc1 = torch.zeros(local_count, ffn_hidden, dtype=tokens.dtype, device=device)
            local_act = torch.zeros(local_count, ffn_hidden, dtype=tokens.dtype, device=device)

            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = local_tokens_per_expert_list[exp_idx]
                if n_tok > 0:
                    exp_tokens = local_tokens[start:start + n_tok]
                    exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                    local_fc1[start:start + n_tok] = exp_fc1
                    local_act[start:start + n_tok] = activation_func(exp_fc1)
                    start += n_tok
        elif local_count > 0:
            # 单 expert 回退
            local_fc1 = torch.matmul(local_tokens, w1[0])
            local_act = activation_func(local_fc1)
        else:
            local_fc1 = torch.empty(0, ffn_hidden, dtype=tokens.dtype, device=device)
            local_act = torch.empty(0, ffn_hidden, dtype=tokens.dtype, device=device)

        # 等待 Dispatch 通信完成（CUDA stream 同步）
        default_stream.wait_event(combine_event)

        # 等待所有 Dispatch P2P 请求完成（确保数据已到达）
        for req in dispatch_reqs:
            req.wait()

        # 远程数据处理
        # 直接按 source-rank-major 处理（前向计算不需要排序）
        # 保存激活值的排序放在 local FC2 之后，与 combine 通信重叠
        peer_act_src_major = None  # 用于后续排序
        peer_recv_chunk_sizes = None
        peer_sort_indices = None

        if recv_count > 0:
            peer_data = recv_buffer

            # peer_data 布局: [peer0_all_experts, peer1_all_experts, ...]
            # 直接按 peer 处理，每个 peer 内部按 expert 计算
            peer_fc1 = torch.zeros(recv_count, ffn_hidden, dtype=tokens.dtype, device=device)
            peer_act_src_major = torch.zeros(recv_count, ffn_hidden, dtype=tokens.dtype, device=device)  # source-rank-major
            peer_fc2 = torch.zeros(recv_count, hidden_size, dtype=tokens.dtype, device=device)

            if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
                # 按 source-rank 顺序处理
                peer_offset = 0
                for src_rank in range(ep_size):
                    if src_rank == my_rank:
                        continue
                    # 该 peer 的 tokens 按 expert 分布
                    for exp_idx in range(num_local_experts):
                        n_tok = num_global_tokens_per_local_expert[0, src_rank, exp_idx].item()
                        if n_tok > 0:
                            exp_tokens = peer_data[peer_offset:peer_offset + n_tok]
                            exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                            exp_act = activation_func(exp_fc1)
                            exp_fc2 = torch.matmul(exp_act, w2[exp_idx])
                            peer_fc1[peer_offset:peer_offset + n_tok] = exp_fc1
                            peer_act_src_major[peer_offset:peer_offset + n_tok] = exp_act
                            peer_fc2[peer_offset:peer_offset + n_tok] = exp_fc2
                            peer_offset += n_tok

                # 预计算排序索引（实际排序放在 local FC2 之后与 combine 重叠）
                peer_recv_chunk_sizes = []
                for src_rank in range(ep_size):
                    if src_rank == my_rank:
                        continue
                    for exp_idx in range(num_local_experts):
                        chunk_size = num_global_tokens_per_local_expert[0, src_rank, exp_idx].item()
                        peer_recv_chunk_sizes.append(chunk_size)

                peer_sort_indices = []
                for exp_idx in range(num_local_experts):
                    for src_idx in range(ep_size - 1):
                        chunk_idx = src_idx * num_local_experts + exp_idx
                        peer_sort_indices.append(chunk_idx)
            else:
                # 单 expert 情况
                peer_fc1 = torch.matmul(peer_data, w1[0])
                peer_act_src_major = activation_func(peer_fc1)
                peer_fc2 = torch.matmul(peer_act_src_major, w2[0])
        else:
            peer_data = torch.empty(0, hidden_size, dtype=tokens.dtype, device=device)
            peer_fc1 = torch.empty(0, ffn_hidden, dtype=tokens.dtype, device=device)
            peer_fc2 = None

        # ===== Debug: Print peer FC2 info =====
        debug_overlap = os.environ.get('FLUID_DEBUG_OVERLAP', '0') == '1'
        if debug_overlap:
            print(f"[DEBUG Rank {my_rank}] OVERLAP forward entered, peer_fc2 exists: {peer_fc2 is not None}", flush=True)
            if peer_fc2 is not None:
                print(f"[DEBUG Rank {my_rank}] peer_fc2 shape: {peer_fc2.shape}, first 4 sums: {peer_fc2.sum(dim=-1).tolist()[:4]}", flush=True)

        # ===== Combine 阶段 =====
        dispatch_event.record(default_stream)
        comm_stream.wait_event(dispatch_event)

        total_tokens = tokens.shape[0]
        combine_recv = torch.empty(total_tokens, hidden_size, dtype=tokens.dtype, device=device)

        with torch.cuda.stream(comm_stream):
            combine_ops = []
            send_ptr = 0
            for i in range(ep_size):
                if i == my_rank: continue

                recv_size = input_splits_list[i]
                if recv_size > 0:
                    recv_chunk = combine_recv[input_offsets[i] : input_offsets[i+1]]
                    combine_ops.append(dist.P2POp(dist.irecv, recv_chunk, i, group=ep_group))

                send_size = output_splits_list[i]
                if send_size > 0 and peer_fc2 is not None:
                    send_chunk = peer_fc2[send_ptr : send_ptr + send_size].contiguous()
                    combine_ops.append(dist.P2POp(dist.isend, send_chunk, i, group=ep_group))
                    send_ptr += send_size

            combine_reqs = dist.batch_isend_irecv(combine_ops) if combine_ops else []

        combine_event.record(comm_stream)

        # 本地 FC2（与 combine 重叠）
        # 注意：使用 local_tokens_per_expert_list（FC1 阶段已计算）
        if local_count > 0 and local_tokens_per_expert_list is not None:
            local_fc2 = torch.zeros(local_count, hidden_size, dtype=tokens.dtype, device=device)

            start = 0
            for exp_idx in range(num_local_experts):
                n_tok = local_tokens_per_expert_list[exp_idx]
                if n_tok > 0:
                    exp_act = local_act[start:start + n_tok]
                    local_fc2[start:start + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                    start += n_tok
        elif local_count > 0:
            local_fc2 = torch.matmul(local_act, w2[0])
        else:
            local_fc2 = None

        # ===== 保存激活值排序（与 combine 重叠）=====
        # 将 peer_act 和 peer_data 从 source-rank-major 重排为 expert-major（用于反向传播）
        # 排序操作与 combine 通信重叠执行
        # 使用 Megatron 的 sort_chunks_by_idxs（带 autograd 支持）
        if peer_act_src_major is not None and peer_recv_chunk_sizes is not None and peer_sort_indices is not None:
            peer_chunk_sizes_tensor = torch.tensor(peer_recv_chunk_sizes, dtype=torch.int64, device=device)
            peer_sort_idxs_tensor = torch.tensor(peer_sort_indices, dtype=torch.int64, device=device)
            peer_act, _ = sort_chunks_by_idxs(peer_act_src_major, peer_chunk_sizes_tensor, peer_sort_idxs_tensor)
            peer_data, _ = sort_chunks_by_idxs(peer_data, peer_chunk_sizes_tensor, peer_sort_idxs_tensor)
        elif peer_act_src_major is not None:
            peer_act = peer_act_src_major
        else:
            peer_act = torch.empty(0, ffn_hidden, dtype=tokens.dtype, device=device)

        # ===== 合并 local_tokens/local_act 和 peer_data/peer_act =====
        # 合并后布局: 标准 expert-major [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...] (和 BASELINE 一致)
        # local_tokens 布局: [E0_local, E1_local, ...]
        # peer_data 布局: 已排序为 expert-major [E0_peer0, E0_peer1, ..., E1_peer0, E1_peer1, ...]
        if num_global_tokens_per_local_expert is not None and recv_count > 0 and peer_data.numel() > 0:
            # 拼接: [local_E0, local_E1, ..., peer_E0_sorted, peer_E1_sorted, ...]
            tokens_concat = torch.cat([local_tokens, peer_data], dim=0)
            act_concat = torch.cat([local_act, peer_act], dim=0)

            # 构建 split_sizes 和 sorted_idxs 用于合并
            # local chunks: [E0_local, E1_local, ...]
            # peer chunks (已排序): [E0_all_peers, E1_all_peers, ...]
            split_sizes = []
            for exp_idx in range(num_local_experts):
                split_sizes.append(num_global_tokens_per_local_expert[0, my_rank, exp_idx].item())
            for exp_idx in range(num_local_experts):
                peer_total = sum(num_global_tokens_per_local_expert[0, r, exp_idx].item()
                               for r in range(ep_size) if r != my_rank)
                split_sizes.append(peer_total)
            split_sizes_tensor = torch.tensor(split_sizes, dtype=torch.int64, device=device)

            # sorted_idxs: 交错合并 [E0_local, E0_peers, E1_local, E1_peers, ...]
            sorted_idxs = []
            for exp_idx in range(num_local_experts):
                sorted_idxs.append(exp_idx)  # local chunk for expert
                sorted_idxs.append(num_local_experts + exp_idx)  # peer chunk for expert
            sorted_idxs_tensor = torch.tensor(sorted_idxs, dtype=torch.int64, device=device)

            all_expert_tokens, _ = sort_chunks_by_idxs(tokens_concat, split_sizes_tensor, sorted_idxs_tensor)
            all_act, _ = sort_chunks_by_idxs(act_concat, split_sizes_tensor, sorted_idxs_tensor)
        elif recv_count > 0 and peer_data.numel() > 0:
            # 单 expert 情况，简单拼接（local 在前，peer 在后）
            all_expert_tokens = torch.cat([local_tokens, peer_data], dim=0)
            all_act = torch.cat([local_act, peer_act], dim=0)
        else:
            # 没有 peer tokens，直接使用 local_tokens/local_act
            all_expert_tokens = local_tokens
            all_act = local_act

        # 计算每个 expert 的总 token 数（用于 backward）
        if num_global_tokens_per_local_expert is not None:
            all_tokens_per_expert = []
            for exp_idx in range(num_local_experts):
                total = sum(num_global_tokens_per_local_expert[0, rank, exp_idx].item() for rank in range(ep_size))
                all_tokens_per_expert.append(total)
        else:
            # 单 expert 情况，总 token 数就是 all_expert_tokens 的第一维
            all_tokens_per_expert = [all_expert_tokens.shape[0]]

        # 等待 Combine 通信完成（CUDA stream 同步）
        default_stream.wait_event(combine_event)

        # 等待所有 Combine P2P 请求完成（确保数据已到达）
        for req in combine_reqs:
            req.wait()

        # 组装结果
        if local_fc2 is not None:
            combine_recv[local_start : local_start + local_count] = local_fc2

        combined_output = combine_recv

        # Debug: Print combined_output
        if debug_overlap:
            print(f"[Rank {my_rank}] combined_output shape: {combined_output.shape}, first 4 sums: {combined_output.sum(dim=-1).tolist()[:4]}", flush=True)

        # ===== 预计算 backward 需要的 sort 索引 =====
        # backward 需要将 grad_combined (rank-major) 转换为 expert-major
        # 使用与 baseline 相同的布局转换逻辑
        ctx.need_sort = num_global_tokens_per_local_expert is not None

        if num_global_tokens_per_local_expert is not None:
            # rank-major chunk sizes: [R0_E0, R0_E1, R1_E0, R1_E1, ...]
            split_sizes_rank_major = []
            for rank in range(ep_size):
                for exp_idx in range(num_local_experts):
                    split_sizes_rank_major.append(num_global_tokens_per_local_expert[0, rank, exp_idx].item())

            # rank-major -> expert-major 的索引
            sorted_idxs_rank_to_exp = []
            for exp_idx in range(num_local_experts):
                for rank in range(ep_size):
                    sorted_idxs_rank_to_exp.append(rank * num_local_experts + exp_idx)

            # expert-major chunk sizes: [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...]
            split_sizes_exp_major = []
            for exp_idx in range(num_local_experts):
                for rank in range(ep_size):
                    split_sizes_exp_major.append(num_global_tokens_per_local_expert[0, rank, exp_idx].item())

            # expert-major -> rank-major 的索引
            sorted_idxs_exp_to_rank = []
            for rank in range(ep_size):
                for exp_idx in range(num_local_experts):
                    sorted_idxs_exp_to_rank.append(exp_idx * ep_size + rank)

            ctx.split_sizes_rank_major = torch.tensor(split_sizes_rank_major, dtype=torch.int64, device=device)
            ctx.sorted_idxs_rank_to_exp = torch.tensor(sorted_idxs_rank_to_exp, dtype=torch.int64, device=device)
            ctx.split_sizes_exp_major = torch.tensor(split_sizes_exp_major, dtype=torch.int64, device=device)
            ctx.sorted_idxs_exp_to_rank = torch.tensor(sorted_idxs_exp_to_rank, dtype=torch.int64, device=device)
        else:
            ctx.split_sizes_rank_major = None
            ctx.sorted_idxs_rank_to_exp = None
            ctx.split_sizes_exp_major = None
            ctx.sorted_idxs_exp_to_rank = None

        # 保存用于 backward（all_act 已排序为 expert-major，与 combine 重叠执行）
        ctx.save_for_backward(all_expert_tokens, weight1, weight2, all_act)
        ctx.all_tokens_per_expert = all_tokens_per_expert

        return combined_output

    @staticmethod
    def backward(ctx, grad_output):
        # 提取合并后的 saved tensors（all_act 从 forward 保存，避免重计算）
        all_expert_tokens, weight1, weight2, all_act = ctx.saved_tensors

        ep_group = ctx.ep_group
        activation_func = ctx.activation_func
        input_splits_list = ctx.input_splits_list
        output_splits_list = ctx.output_splits_list
        local_count = ctx.local_count
        recv_count = ctx.recv_count
        send_count = ctx.send_count
        my_rank = ctx.my_rank
        local_start = ctx.local_start
        num_local_experts = ctx.num_local_experts
        ffn_hidden = ctx.ffn_hidden

        # 获取 per-expert token 数量
        all_tokens_per_expert = ctx.all_tokens_per_expert

        device = grad_output.device
        hidden_size = grad_output.shape[-1]

        # Reshape weights to per-expert view
        # NOTE: 必须与 forward 保持一致！Forward 使用 view(num_local_experts, hidden_size, ffn_hidden)
        w1 = weight1.view(num_local_experts, hidden_size, ffn_hidden)  # [num_local_experts, hidden, ffn]
        w2 = weight2.view(num_local_experts, ffn_hidden, hidden_size)  # [num_local_experts, ffn, hidden]

        total_all_tokens = all_expert_tokens.shape[0]

        # ===== 使用保存的 all_act =====
        act_output = all_act  # 直接使用保存的激活值

        # ===== 反向 Combine AllToAll（与 BASELINE 完全一致的调度）=====
        # BASELINE 使用 _FluidAllToAll.backward：
        # 1. 在 comm_stream 上启动 AllToAll
        # 2. 尝试执行 dW 任务（此时没有 dW 任务）
        # 3. 等待 AllToAll 完成
        scheduler = get_backward_scheduler()
        if scheduler.is_enabled():
            comm_stream = scheduler.comm_stream
            default_stream = scheduler.default_stream

            # Step 1: 在 comm_stream 上启动 AllToAll
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_combined = _all_to_all(
                    grad_output.contiguous(),
                    output_split_sizes=output_splits_list,
                    input_split_sizes=input_splits_list,
                    group=ep_group
                )

            # Step 2: 尝试执行 dW 任务（此时没有 dW 任务，与 BASELINE 一致）
            scheduler._execute_all_dw_tasks_sync()

            # Step 3: 等待 AllToAll 完成
            default_stream.wait_stream(comm_stream)
        else:
            # Scheduler 未启用，直接执行 AllToAll
            grad_combined = _all_to_all(
                grad_output.contiguous(),
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
                group=ep_group
            )

        # ===== 使用 sort_chunks_by_idxs 转换 grad_combined 布局（rank-major -> expert-major）=====
        # grad_combined 布局: rank-major [R0_E0, R0_E1, ..., R1_E0, R1_E1, ...]
        # 目标布局: 标准 expert-major [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...] (和 BASELINE 一致)
        # 使用 forward 中预计算的 sort 索引（避免重复计算）
        if ctx.need_sort:
            grad_all_fc2, _ = sort_chunks_by_idxs(
                grad_combined,
                ctx.split_sizes_rank_major,
                ctx.sorted_idxs_rank_to_exp
            )
        else:
            # 单 expert 无需 sort，直接使用 grad_combined
            grad_all_fc2 = grad_combined

        # ===== fc1 重计算和 act_deriv 计算（与 BASELINE 一致的调度：在 sort 之后）=====
        # BASELINE 在 expert backward 中重计算 fc1 和 activation
        fc1_output = torch.zeros(total_all_tokens, ffn_hidden, dtype=all_expert_tokens.dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                fc1_output[start:start + n_tok] = torch.matmul(
                    all_expert_tokens[start:start + n_tok], w1[exp_idx]
                )
                start += n_tok

        # 一次性计算 act_deriv（与 BASELINE 一致，在循环外）
        if activation_func == F.silu or (hasattr(activation_func, '__name__') and 'silu' in activation_func.__name__.lower()):
            sig = torch.sigmoid(fc1_output)
            act_deriv = sig * (1 + fc1_output * (1 - sig))
        else:
            act_deriv = _gelu_grad_analytical(fc1_output)

        # ===== 分步计算 grad_tokens（与 BASELINE 完全一致的三步结构）=====
        # Step 1: FC2 backward - 循环计算 grad_intermediate
        grad_intermediate = torch.zeros(total_all_tokens, ffn_hidden, dtype=grad_output.dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_intermediate[start:start+n_tok] = torch.matmul(
                    grad_all_fc2[start:start+n_tok], w2[exp_idx].t()
                )
                start += n_tok

        # Step 2: Activation backward - 一次性处理整个 tensor（与 BASELINE 一致，在循环外）
        grad_all_fc1 = grad_intermediate * act_deriv

        # Step 3: FC1 backward - 循环计算 grad_all_tokens
        grad_all_tokens = torch.zeros(total_all_tokens, hidden_size, dtype=grad_output.dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = all_tokens_per_expert[exp_idx]
            if n_tok > 0:
                grad_all_tokens[start:start+n_tok] = torch.matmul(
                    grad_all_fc1[start:start+n_tok], w1[exp_idx].t()
                )
                start += n_tok

        # ===== 注册 dW 任务（使用合并后的 tensor，和 BASELINE 一致）=====
        scheduler = get_backward_scheduler()

        # Capture variables for closure
        num_local_experts_saved = num_local_experts
        all_tokens_per_expert_saved = all_tokens_per_expert
        ffn_hidden_saved = ffn_hidden
        hidden_size_saved = hidden_size
        grad_all_fc2_saved = grad_all_fc2.detach()
        grad_all_fc1_saved = grad_all_fc1.detach()
        act_output_saved = act_output.detach()
        all_expert_tokens_saved = all_expert_tokens.detach()

        def compute_dw_weight2():
            # weight2: [ffn_hidden * num_local_experts, hidden_size]
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
            # weight1: [hidden_size, ffn_hidden * num_local_experts]
            # NOTE: 必须与 BASELINE 保持一致！BASELINE 使用 view(num_local_experts, hidden_size, -1)
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
            layer_name=f"moe_p2p_overlap_weight2_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight2,
            priority=100,
            weight_param=weight2,
        )

        scheduler.register_dw_task(
            layer_name=f"moe_p2p_overlap_weight1_layer{layer_id}",
            layer_id=layer_id,
            compute_fn=compute_dw_weight1,
            priority=99,
            weight_param=weight1,
        )

        grad_weight1 = None
        grad_weight2 = None

        # ===== 使用 sort_chunks_by_idxs 转换 grad_all_tokens 布局（expert-major -> rank-major）=====
        # expert-major: [E0_R0, E0_R1, ..., E1_R0, E1_R1, ...]
        # rank-major: [R0_E0, R0_E1, ..., R1_E0, R1_E1, ...]
        # 使用 forward 中预计算的 sort 索引（避免重复计算）
        if ctx.need_sort:
            grad_dispatched, _ = sort_chunks_by_idxs(
                grad_all_tokens,
                ctx.split_sizes_exp_major,
                ctx.sorted_idxs_exp_to_rank
            )
        else:
            # 单 expert 无需 sort，直接使用 grad_all_tokens
            grad_dispatched = grad_all_tokens

        # ===== dW-AllToAll重叠优化 =====
        # 与Baseline模式相同的策略：
        # 1. 在comm_stream上启动AllToAll（异步）
        # 2. 在default_stream上执行dW任务（与AllToAll并行）
        # 3. 等待AllToAll完成
        scheduler = get_backward_scheduler()
        if scheduler.is_enabled():
            comm_stream = scheduler.comm_stream
            default_stream = scheduler.default_stream

            # Step 1: 在comm_stream上启动AllToAll（非阻塞）
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_tokens = _all_to_all(
                    grad_dispatched.contiguous(),
                    output_split_sizes=input_splits_list,    # 前向的input_splits
                    input_split_sizes=output_splits_list,    # 前向的output_splits
                    group=ep_group
                )

            # Step 2: 在default_stream上执行dW任务（与AllToAll并行）
            scheduler._execute_all_dw_tasks_sync()

            # Step 3: 等待AllToAll完成
            default_stream.wait_stream(comm_stream)
        else:
            # Scheduler未启用，直接执行AllToAll
            grad_tokens = _all_to_all(
                grad_dispatched.contiguous(),
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group
            )

        # 返回梯度: tokens, input_splits, output_splits, weight1, weight2, ep_group, activation_func,
        # comm_stream, dispatch_event, combine_event, layer_id, num_local_experts, tokens_per_expert,
        # num_global_tokens_per_local_expert, sort_indices, restore_indices
        return (grad_tokens, None, None, grad_weight1, grad_weight2, None, None, None, None, None, None,
                None, None, None, None, None)


class _Hp2SpOutputProjOverlapFunction(torch.autograd.Function):
    """
    hp2sp + 输出投影重叠的自动求导包装

    前向：使用 P2P 重叠
    反向：使用标准 AllToAll（保持反向调度不变）
    """

    @staticmethod
    def forward(ctx, attn_output, weight_proj, bias_proj, cp_group, comm_stream,
                layer_name, layer_id, ev_ready, ev_comm_done,
                weight_local=None, weight_peer=None):
        ctx.cp_group = cp_group
        ctx.has_bias = bias_proj is not None
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id

        # 保存用于dW注册
        saved_tensors = [attn_output, weight_proj]
        if bias_proj is not None:
            saved_tensors.append(bias_proj)
        ctx.save_for_backward(*saved_tensors)

        # 前向使用 P2P 重叠
        with torch.no_grad():
            output = _hp2sp_output_proj_overlap_impl(
                attn_output, weight_proj, bias_proj, cp_group, comm_stream,
                ev_ready, ev_comm_done, weight_local, weight_peer
            )

        output = output.requires_grad_(attn_output.requires_grad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 提取saved tensors
        saved = ctx.saved_tensors
        attn_output = saved[0]
        weight_proj = saved[1]
        bias_proj = saved[2] if ctx.has_bias else None

        cp_group = ctx.cp_group
        has_bias = ctx.has_bias

        cp_size = cp_group.size()
        seq_full, batch_size, heads_local, head_dim = attn_output.shape
        seq_local = seq_full // cp_size
        hidden_size = weight_proj.shape[0]

        # 反向使用标准 AllToAll（保持反向调度不变）
        # grad_output: [seq_local, B, hidden]
        # 需要 sp2hp 变回 [seq_full, B, heads_local, D]

        # dL/d_attn = dL/d_output @ W_proj (部分)
        grad_attn_flat = torch.matmul(grad_output, weight_proj)
        # grad_attn_flat: [seq_local, B, H*D]
        grad_attn_sp = grad_attn_flat.view(seq_local, batch_size, -1, head_dim)

        # sp2hp 通信
        grad_attn_output = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)

        # ===== 注册dW任务 =====
        from fluid.scheduler import get_backward_scheduler
        scheduler = get_backward_scheduler()

        if scheduler.is_enabled():
            # 保存中间变量用于dW计算
            attn_hp_saved = _all_to_all_hp2sp_forward(attn_output, cp_group)
            attn_flat_saved = attn_hp_saved.view(seq_local * batch_size, -1).detach()
            grad_output_flat_saved = grad_output.view(seq_local * batch_size, hidden_size).detach()

            # 定义dW计算函数
            def compute_dw_proj():
                return torch.matmul(grad_output_flat_saved.t(), attn_flat_saved)

            # 注册dW任务
            scheduler.register_dw_task(
                layer_name=f"{ctx.layer_name}_weight",
                layer_id=ctx.layer_id,
                compute_fn=compute_dw_proj,
                priority=100,
                weight_param=weight_proj,
            )

            # 如果有bias，注册bias的dW任务
            if has_bias and bias_proj is not None:
                grad_output_for_bias = grad_output.detach()

                def compute_dbias():
                    return grad_output_for_bias.sum(dim=(0, 1))

                scheduler.register_dw_task(
                    layer_name=f"{ctx.layer_name}_bias",
                    layer_id=ctx.layer_id,
                    compute_fn=compute_dbias,
                    priority=99,
                    weight_param=bias_proj,
                )

            # 返回None作为权重梯度（调度器会计算）
            grad_weight_proj = None
            grad_bias = None
        else:
            # 如果调度器未启用，直接计算梯度（回退模式）
            attn_hp = _all_to_all_hp2sp_forward(attn_output, cp_group)
            attn_flat = attn_hp.view(seq_local * batch_size, -1)
            grad_output_flat = grad_output.view(seq_local * batch_size, hidden_size)
            grad_weight_proj = torch.matmul(grad_output_flat.t(), attn_flat)
            grad_bias = grad_output.sum(dim=(0, 1)) if has_bias else None

        # 返回顺序：attn_output, weight_proj, bias_proj, cp_group, comm_stream, layer_name, layer_id, ev_ready, ev_comm_done, weight_local, weight_peer
        return grad_attn_output, grad_weight_proj, grad_bias, None, None, None, None, None, None, None, None


# =============================================================================
# 公开 API
# =============================================================================

def qkv_sp2hp_heads_split(
    hidden_states: torch.Tensor,
    weight_local: torch.Tensor,
    weight_remote: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    cp_group: dist.ProcessGroup,
    overlap_ctx: 'OverlapContext',
    weight_qkv_fused: Optional[torch.Tensor] = None,
    layer_name: str = "qkv",
    layer_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    QKV 整体计算 + P2P 重叠（简化版）

    改进: 只做 2 次 matmul，计算后再分离 Q/K/V

    Args:
        hidden_states: [seq_local, B, hidden]
        weight_local: [groups_local * group_size, hidden] 本地 groups 权重
        weight_remote: [groups_local * group_size, hidden] 远程 groups 权重
        num_heads: Q heads 总数
        num_kv_heads: K/V heads 总数 (groups)
        head_dim: 每个头的维度
        cp_group: Context Parallel 进程组
        overlap_ctx: OverlapContext，管理 CUDA stream 和 events
        weight_qkv_fused: 完整 QKV 权重（用于训练时注册 dW 任务）
        layer_name: 层名称（用于 dW 调度）
        layer_id: 层 ID（用于 dW 调度）

    Returns:
        q, k, v: [seq_full, B, heads_local, head_dim]
    """
    comm_stream = overlap_ctx.get_stream()
    ev_ready, ev_comm_done = overlap_ctx.get_qkv_events()

    if hidden_states.requires_grad:
        # 训练模式：使用 autograd.Function 包装
        return _QKVSp2HpHeadsSplitFunction.apply(
            hidden_states, weight_local, weight_remote,
            num_heads, num_kv_heads, head_dim, cp_group, comm_stream,
            weight_qkv_fused, layer_name, layer_id, ev_ready, ev_comm_done
        )
    else:
        # 推理模式：直接调用实现
        return _qkv_sp2hp_heads_split_impl(
            hidden_states, weight_local, weight_remote,
            num_heads, num_kv_heads, head_dim,
            cp_group, comm_stream,
            ev_ready, ev_comm_done
        )


def prepare_qkv_split_weights(
    weight_qkv: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    my_rank: int,
    cp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    按 group 整体分割 QKV 权重为 local 和 remote

    Megatron 使用 interleaved 布局: [Q0,K0,V0, Q1,K1,V1, ...]
    每个 "group" 包含 (q_per_group + 2) 个 heads

    简化版: 不分离 Q/K/V，整体按 group 分割，计算后再分离

    Args:
        weight_qkv: [total_proj, hidden] 完整 QKV 权重
        num_heads: Q heads 总数
        num_kv_heads: K/V heads 总数 (groups)
        head_dim: 每个 head 的维度
        my_rank: 当前 rank
        cp_size: CP 大小

    Returns:
        (weight_local, weight_remote) 或 cp_size=1 时 (weight_qkv, None)
    """
    if cp_size == 1:
        return weight_qkv, None

    num_groups = num_kv_heads
    q_per_group = num_heads // num_kv_heads
    group_size = (q_per_group + 2) * head_dim
    hidden_size = weight_qkv.shape[1]
    groups_local = num_groups // cp_size
    peer_rank = 1 - my_rank

    # Reshape to [num_groups, group_size, hidden]
    weight_grouped = weight_qkv.view(num_groups, group_size, hidden_size)

    # 按 group 分割
    local_group_start = my_rank * groups_local
    local_group_end = local_group_start + groups_local
    remote_group_start = peer_rank * groups_local
    remote_group_end = remote_group_start + groups_local

    local_size = groups_local * group_size
    weight_local = weight_grouped[local_group_start:local_group_end].reshape(local_size, hidden_size).contiguous()
    weight_remote = weight_grouped[remote_group_start:remote_group_end].reshape(local_size, hidden_size).contiguous()

    return weight_local, weight_remote


def prepare_proj_split_weights(
    weight_proj: torch.Tensor,
    num_heads: int,
    head_dim: int,
    my_rank: int,
    cp_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预先准备 Output Projection 的权重切片

    Args:
        weight_proj: [hidden, num_heads * head_dim] 输出投影权重
        num_heads: 总 head 数
        head_dim: 每个 head 的维度
        my_rank: 当前 rank
        cp_size: CP 大小

    Returns:
        (weight_local, weight_peer): 预先切片好的权重，都是 contiguous
    """
    if cp_size == 1:
        return weight_proj, None

    heads_local = num_heads // cp_size
    input_dim_per_rank = heads_local * head_dim
    peer_rank = 1 - my_rank

    local_head_start = my_rank * input_dim_per_rank
    peer_head_start = peer_rank * input_dim_per_rank

    weight_local = weight_proj[:, local_head_start:local_head_start + input_dim_per_rank].contiguous()
    weight_peer = weight_proj[:, peer_head_start:peer_head_start + input_dim_per_rank].contiguous()

    return weight_local, weight_peer


def hp2sp_output_proj_overlap(
    attn_output: torch.Tensor,
    weight_proj: torch.Tensor,
    bias_proj: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    comm_stream: torch.cuda.Stream,
    layer_name: str = "output_proj",
    layer_id: int = 0,
    overlap_ctx: 'OverlapContext' = None,
    weight_local: torch.Tensor = None,
    weight_peer: torch.Tensor = None,
) -> torch.Tensor:
    """
    hp2sp + 输出投影重叠：本地投影与远程通信重叠

    前向：使用 P2P 重叠优化
    反向：使用标准 AllToAll，注册dW任务（保持反向调度不变）

    Args:
        attn_output: [seq_full, B, heads_local, head_dim] 注意力输出
        weight_proj: [hidden, num_heads * head_dim] 输出投影权重
        bias_proj: [hidden] 输出投影偏置
        cp_group: Context Parallel 进程组
        comm_stream: 通信用的 CUDA stream (已废弃，将使用overlap_ctx.get_stream())
        layer_name: 层名称（用于dW调度）
        layer_id: 层ID（用于dW调度）
        overlap_ctx: OverlapContext，管理CUDA stream和events（必需）
        weight_local: 预先准备好的本地权重切片（可选）
        weight_peer: 预先准备好的对端权重切片（可选）

    Returns:
        output: [seq_local, B, hidden]
    """
    # 从overlap_ctx提取stream和events
    comm_stream = overlap_ctx.get_stream()
    ev_ready, ev_comm_done = overlap_ctx.get_proj_events()

    if attn_output.requires_grad:
        return _Hp2SpOutputProjOverlapFunction.apply(
            attn_output, weight_proj, bias_proj, cp_group, comm_stream,
            layer_name, layer_id, ev_ready, ev_comm_done,
            weight_local, weight_peer
        )
    else:
        return _hp2sp_output_proj_overlap_impl(
            attn_output, weight_proj, bias_proj, cp_group, comm_stream,
            ev_ready, ev_comm_done, weight_local, weight_peer
        )


def _moe_p2p_overlap_impl(
    tokens: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    ep_group: dist.ProcessGroup,
    activation_func,
    comm_stream: torch.cuda.Stream,
    dispatch_event: torch.cuda.Event,
    combine_event: torch.cuda.Event,
    num_local_experts: int = 1,
    tokens_per_expert: torch.Tensor = None,
    num_global_tokens_per_local_expert: torch.Tensor = None,
    sort_indices: torch.Tensor = None,
    restore_indices: torch.Tensor = None,
) -> torch.Tensor:
    """MoE P2P 重叠内部实现（无梯度追踪，支持多 expert）"""
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    device = tokens.device
    hidden_size = tokens.shape[-1]

    # 处理多 expert 的权重 shape
    total_ffn_hidden = weight1.shape[-1]
    ffn_hidden = total_ffn_hidden // num_local_experts

    # Reshape weights to per-expert view
    # NOTE: 必须与 BASELINE (_FluidExpertComputation) 保持一致！
    # BASELINE 使用: w1 = weight1.view(num_local_experts, hidden_size, -1)
    w1 = weight1.view(num_local_experts, hidden_size, ffn_hidden)  # [num_local_experts, hidden, ffn]
    w2 = weight2.view(num_local_experts, ffn_hidden, hidden_size)  # [num_local_experts, ffn, hidden]

    input_splits_list = input_splits.tolist() if torch.is_tensor(input_splits) else list(input_splits)
    output_splits_list = output_splits.tolist() if torch.is_tensor(output_splits) else list(output_splits)

    default_stream = torch.cuda.current_stream(device)
    local_count = input_splits_list[my_rank]

    input_offsets = [0]
    for s in input_splits_list:
        input_offsets.append(input_offsets[-1] + s)

    output_offsets = [0]
    for s in output_splits_list:
        output_offsets.append(output_offsets[-1] + s)

    total_recv = sum(output_splits_list) - output_splits_list[my_rank]
    recv_buffer = torch.empty(total_recv, hidden_size, dtype=tokens.dtype, device=device)

    # ===== Dispatch 阶段 =====
    dispatch_event.record(default_stream)
    comm_stream.wait_event(dispatch_event)

    local_start = input_offsets[my_rank]
    local_tokens = tokens[local_start:local_start + local_count].clone() if local_count > 0 else torch.empty(0, hidden_size, dtype=tokens.dtype, device=device)

    send_chunks = {}
    for i in range(ep_size):
        if i == my_rank: continue
        if input_splits_list[i] > 0:
            send_chunks[i] = tokens[input_offsets[i] : input_offsets[i+1]].contiguous()

    with torch.cuda.stream(comm_stream):
        dispatch_ops = []
        recv_ptr = 0
        for i in range(ep_size):
            if i == my_rank: continue
            recv_size = output_splits_list[i]
            if recv_size > 0:
                recv_chunk = recv_buffer[recv_ptr : recv_ptr + recv_size]
                dispatch_ops.append(dist.P2POp(dist.irecv, recv_chunk, i, group=ep_group))
                recv_ptr += recv_size
            if i in send_chunks:
                dispatch_ops.append(dist.P2POp(dist.isend, send_chunks[i], i, group=ep_group))
        dispatch_reqs = dist.batch_isend_irecv(dispatch_ops) if dispatch_ops else []

    combine_event.record(comm_stream)

    # 本地 FC1 + Act（与 dispatch 重叠）- 多 expert 情况
    # NOTE: 需要用 num_global_tokens_per_local_expert[0, my_rank, :] 而不是 tokens_per_expert
    # 因为 local_tokens 只包含 my_rank 发给自己的 tokens
    if local_count > 0 and num_global_tokens_per_local_expert is not None:
        local_tokens_per_expert_list = [
            num_global_tokens_per_local_expert[0, my_rank, exp_idx].item()
            for exp_idx in range(num_local_experts)
        ]
        local_act = torch.zeros(local_count, ffn_hidden, dtype=tokens.dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = local_tokens_per_expert_list[exp_idx]
            if n_tok > 0:
                exp_tokens = local_tokens[start:start + n_tok]
                exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                local_act[start:start + n_tok] = activation_func(exp_fc1)
                start += n_tok
    elif local_count > 0:
        local_fc1 = torch.matmul(local_tokens, w1[0])
        local_act = activation_func(local_fc1)
    else:
        local_act = None

    default_stream.wait_event(combine_event)

    # 等待 Dispatch P2P 请求完成（确保 recv_buffer 数据已到达）
    for req in dispatch_reqs:
        req.wait()

    # 远程数据处理 - 直接按 source-rank-major 处理（无需排序，因为不需要保存激活值）
    if total_recv > 0:
        peer_data = recv_buffer
        peer_fc2 = torch.zeros(total_recv, hidden_size, dtype=tokens.dtype, device=device)

        if num_local_experts > 1 and num_global_tokens_per_local_expert is not None:
            # 直接按 source-rank 顺序处理，每个 peer 内部按 expert 计算
            peer_offset = 0
            for src_rank in range(ep_size):
                if src_rank == my_rank:
                    continue
                for exp_idx in range(num_local_experts):
                    n_tok = num_global_tokens_per_local_expert[0, src_rank, exp_idx].item()
                    if n_tok > 0:
                        exp_tokens = peer_data[peer_offset:peer_offset + n_tok]
                        exp_fc1 = torch.matmul(exp_tokens, w1[exp_idx])
                        exp_act = activation_func(exp_fc1)
                        peer_fc2[peer_offset:peer_offset + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                        peer_offset += n_tok
        else:
            # 单 expert 情况
            peer_fc1 = torch.matmul(peer_data, w1[0])
            peer_act = activation_func(peer_fc1)
            peer_fc2 = torch.matmul(peer_act, w2[0])
    else:
        peer_fc2 = None

    # ===== Combine 阶段 =====
    dispatch_event.record(default_stream)
    comm_stream.wait_event(dispatch_event)

    total_tokens_count = tokens.shape[0]
    combine_recv = torch.empty(total_tokens_count, hidden_size, dtype=tokens.dtype, device=device)

    with torch.cuda.stream(comm_stream):
        combine_ops = []
        send_ptr = 0
        for i in range(ep_size):
            if i == my_rank: continue
            recv_size = input_splits_list[i]
            if recv_size > 0:
                recv_chunk = combine_recv[input_offsets[i] : input_offsets[i+1]]
                combine_ops.append(dist.P2POp(dist.irecv, recv_chunk, i, group=ep_group))
            send_size = output_splits_list[i]
            if send_size > 0 and peer_fc2 is not None:
                send_chunk = peer_fc2[send_ptr : send_ptr + send_size].contiguous()
                combine_ops.append(dist.P2POp(dist.isend, send_chunk, i, group=ep_group))
                send_ptr += send_size
        combine_reqs = dist.batch_isend_irecv(combine_ops) if combine_ops else []

    combine_event.record(comm_stream)

    # 本地 FC2（与 combine 重叠）- 多 expert 情况
    # NOTE: 复用 local_tokens_per_expert_list（FC1 阶段已计算）
    if local_act is not None and num_global_tokens_per_local_expert is not None:
        local_fc2 = torch.zeros(local_count, hidden_size, dtype=tokens.dtype, device=device)
        start = 0
        for exp_idx in range(num_local_experts):
            n_tok = local_tokens_per_expert_list[exp_idx]
            if n_tok > 0:
                exp_act = local_act[start:start + n_tok]
                local_fc2[start:start + n_tok] = torch.matmul(exp_act, w2[exp_idx])
                start += n_tok
    elif local_act is not None:
        local_fc2 = torch.matmul(local_act, w2[0])
    else:
        local_fc2 = None

    default_stream.wait_event(combine_event)

    if local_fc2 is not None:
        combine_recv[local_start : local_start + local_count] = local_fc2

    # 等待 Combine P2P 请求完成（dispatch_reqs 已在使用 recv_buffer 前等待）
    for req in combine_reqs:
        req.wait()

    return combine_recv


def moe_p2p_overlap_forward(
    tokens: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    ep_group: dist.ProcessGroup,
    activation_func,
    comm_stream: torch.cuda.Stream,
    dispatch_event: torch.cuda.Event,
    combine_event: torch.cuda.Event,
    layer_id: int = 0,
    num_local_experts: int = 1,
    tokens_per_expert: torch.Tensor = None,
    num_global_tokens_per_local_expert: torch.Tensor = None,
    sort_indices: torch.Tensor = None,
    restore_indices: torch.Tensor = None,
) -> torch.Tensor:
    """
    MoE P2P 重叠前向：本地 token 计算与远程 token 通信重叠

    前向：使用 P2P 重叠优化
    反向：使用标准 AllToAll（保持反向调度不变）

    Args:
        tokens: [num_tokens, hidden] 输入 token（已按 expert 排序）
        input_splits: [ep_size] 每个 rank 发送的 token 数
        output_splits: [ep_size] 每个 rank 接收的 token 数
        weight1: [hidden, ffn_hidden * num_local_experts] 第一层权重
        weight2: [ffn_hidden * num_local_experts, hidden] 第二层权重
        ep_group: Expert Parallel 进程组
        activation_func: 激活函数
        comm_stream: 通信用的 CUDA stream
        dispatch_event: Dispatch 同步事件
        combine_event: Combine 同步事件
        layer_id: 层ID（用于区分不同层的dW任务）
        num_local_experts: 本地 expert 数量
        tokens_per_expert: [num_local_experts] 每个本地 expert 处理的 token 数
        num_global_tokens_per_local_expert: [tp_size, ep_size, num_local_experts] 每个 rank 发给每个 expert 的 token 数
        sort_indices: 用于将 source-rank-major 转为 expert-major 的索引
        restore_indices: 用于将 expert-major 转回 source-rank-major 的索引

    Returns:
        output: [num_tokens, hidden]
    """
    if tokens.requires_grad:
        return _MoEP2POverlapFunction.apply(
            tokens, input_splits, output_splits, weight1, weight2,
            ep_group, activation_func, comm_stream, dispatch_event, combine_event, layer_id,
            num_local_experts, tokens_per_expert, num_global_tokens_per_local_expert,
            sort_indices, restore_indices
        )
    else:
        return _moe_p2p_overlap_impl(
            tokens, input_splits, output_splits, weight1, weight2,
            ep_group, activation_func, comm_stream, dispatch_event, combine_event,
            num_local_experts, tokens_per_expert, num_global_tokens_per_local_expert,
            sort_indices, restore_indices
        )


# =============================================================================
# 辅助类：重叠上下文管理
# =============================================================================

class OverlapContext:
    """管理计算-通信重叠所需的 CUDA 资源"""

    def __init__(self, device: torch.device):
        self.device = device
        self.comm_stream = torch.cuda.Stream(device=device)

        # MoE 相关 events
        self.dispatch_event = torch.cuda.Event()
        self.combine_event = torch.cuda.Event()

        # Attention QKV 相关 events
        self.qkv_ready_event = torch.cuda.Event()
        self.qkv_comm_done_event = torch.cuda.Event()

        # Attention output projection 相关 events
        self.proj_ready_event = torch.cuda.Event()      # 数据准备好，可以开始通信
        self.proj_comm_done_event = torch.cuda.Event()  # 通信完成

    def get_stream(self) -> torch.cuda.Stream:
        return self.comm_stream

    def get_events(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """获取MoE相关events（保持向后兼容）"""
        return self.dispatch_event, self.combine_event

    def get_qkv_events(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """获取QKV相关events"""
        return self.qkv_ready_event, self.qkv_comm_done_event

    def get_proj_events(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        """获取output projection相关events"""
        return self.proj_ready_event, self.proj_comm_done_event
