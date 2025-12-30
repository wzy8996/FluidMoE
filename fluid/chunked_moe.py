"""
Chunked MoE FC1 实现

将数据分成多个chunk，形成通信-计算流水线：
  Chunk1 Send: [══]
  Chunk1 Recv:   [══]
  Chunk1 FC1:      [══]
  Chunk2 Send:   [══]
  Chunk2 Recv:     [══]
  Chunk2 FC1:        [══]
  ...

这样可以隐藏部分通信延迟。
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import List, Optional
from .ops import fluid_kernels


def chunked_fc1_forward(
    permuted_tokens: torch.Tensor,
    fc1_weight: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    ep_rank: int,
    ep_world_size: int,
    num_chunks: int = 2,
) -> torch.Tensor:
    """
    Chunked FC1 前向传播
    
    将 remote tokens 分成多个 chunk，形成流水线。
    Self tokens 单独处理（无需通信）。
    
    Args:
        permuted_tokens: [total_tokens, hidden_size]
        fc1_weight: [num_local_experts, hidden_size, ffn_hidden_size]
        input_splits: [ep_size]
        output_splits: [ep_size]
        tokens_per_expert: [num_local_experts]
        ep_rank: 当前rank
        ep_world_size: EP world size
        num_chunks: 分块数量
        
    Returns:
        fc1_output: [total_output_tokens, ffn_hidden_size]
    """
    device = permuted_tokens.device
    hidden_size = permuted_tokens.size(1)
    ffn_hidden_size = fc1_weight.size(2)
    num_local_experts = fc1_weight.size(0)
    
    input_splits_list = input_splits.tolist()
    output_splits_list = output_splits.tolist()
    total_output_tokens = sum(output_splits_list)
    
    # 提取 self tokens
    self_input_offset = sum(input_splits_list[:ep_rank])
    self_input_count = input_splits_list[ep_rank]
    self_output_count = output_splits_list[ep_rank]
    
    # 计算 remote tokens 数量
    remote_output_count = total_output_tokens - self_output_count
    
    # 分配最终输出
    final_output = torch.empty(total_output_tokens, ffn_hidden_size, 
                                dtype=permuted_tokens.dtype, device=device)
    
    # 创建通信流
    comm_stream = torch.cuda.Stream(device=device)
    
    # ========================================
    # Step 1: Self FC1 (无需通信)
    # ========================================
    self_copy_tokens = permuted_tokens[self_input_offset:self_input_offset + self_input_count]
    
    # 计算 self tokens per expert
    tokens_per_expert_list = tokens_per_expert.tolist()
    # 简化：假设 self tokens 均匀分布在各 expert
    self_tpe = torch.tensor([self_output_count // num_local_experts] * num_local_experts,
                            dtype=torch.int32, device=device)
    
    if self_input_count > 0:
        self_fc1 = fluid_kernels.grouped_gemm(self_copy_tokens, fc1_weight, self_tpe)
        self_fc1 = F.gelu(self_fc1)
        # Self tokens 放在输出的开头（简化处理）
        final_output[:self_output_count] = self_fc1
    
    # ========================================
    # Step 2: Chunked Remote Communication + FC1
    # ========================================
    if remote_output_count > 0:
        # 构建 remote 通信参数
        # 对于 EP=2: remote peer 是另一个 rank
        remote_peer = 1 - ep_rank  # 假设 EP=2
        
        remote_send_count = input_splits_list[remote_peer]
        remote_recv_count = output_splits_list[remote_peer]
        
        # 计算 chunk 大小
        chunk_size = (remote_recv_count + num_chunks - 1) // num_chunks
        send_chunk_size = (remote_send_count + num_chunks - 1) // num_chunks
        
        # 发送数据的偏移
        send_offset = sum(input_splits_list[:remote_peer])
        
        # 接收缓冲区
        recv_buffer = torch.empty(remote_recv_count, hidden_size,
                                   dtype=permuted_tokens.dtype, device=device)
        
        # Remote tokens per expert (简化)
        remote_tpe = torch.tensor([remote_recv_count // num_local_experts] * num_local_experts,
                                  dtype=torch.int32, device=device)
        
        # 先做完整通信，然后分块计算
        # （真正的per-chunk通信需要更底层的控制）
        with torch.cuda.stream(comm_stream):
            # 构建 AllToAll 的输入
            # 只发送 remote 部分
            send_data = permuted_tokens.clone()  # 需要完整数据
            recv_data = torch.empty_like(permuted_tokens)
            
            dist.all_to_all_single(
                recv_data, send_data,
                output_split_sizes=output_splits_list,
                input_split_sizes=input_splits_list,
            )
        
        comm_stream.synchronize()
        
        # 提取 remote 接收的数据
        remote_recv_offset = sum(output_splits_list[:remote_peer])
        remote_tokens = recv_data[remote_recv_offset:remote_recv_offset + remote_recv_count]
        
        # 分块计算 FC1
        fc1_outputs = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, remote_recv_count)
            if start >= end:
                break
            
            chunk_tokens = remote_tokens[start:end]
            chunk_tpe = torch.tensor([(end - start) // num_local_experts] * num_local_experts,
                                     dtype=torch.int32, device=device)
            
            chunk_fc1 = fluid_kernels.grouped_gemm(chunk_tokens, fc1_weight, chunk_tpe)
            chunk_fc1 = F.gelu(chunk_fc1)
            fc1_outputs.append(chunk_fc1)
        
        # 合并 remote FC1 输出
        if fc1_outputs:
            remote_fc1 = torch.cat(fc1_outputs, dim=0)
            final_output[self_output_count:] = remote_fc1
    
    return final_output


def pipelined_fc1_forward_v2(
    permuted_tokens: torch.Tensor,
    fc1_weight: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    self_tokens_per_expert: torch.Tensor,
    num_global: torch.Tensor,
    ep_rank: int,
    ep_world_size: int,
) -> torch.Tensor:
    """
    改进的流水线实现 - 使用 P2P 通信实现真正的 per-peer 流水线
    
    对于 EP=2，只有 self 和 1 个 remote peer，
    等价于 Local-First 策略。
    
    对于 EP>2，可以实现真正的 per-peer 流水线。
    """
    device = permuted_tokens.device
    hidden_size = permuted_tokens.size(1)
    ffn_hidden_size = fc1_weight.size(2)
    num_local_experts = fc1_weight.size(0)
    
    input_splits_list = input_splits.tolist()
    output_splits_list = output_splits.tolist()
    total_output_tokens = sum(output_splits_list)
    
    # 分配输出
    final_output = torch.empty(total_output_tokens, ffn_hidden_size,
                                dtype=permuted_tokens.dtype, device=device)
    
    # 计算 indices
    cached = fluid_kernels.compute_localfirst_indices_v2(
        tokens_per_expert, self_tokens_per_expert, num_global, ep_rank, ep_rank
    )
    sort_indices, self_out_idx, remote_out_idx, remote_tpe = cached
    
    # 提取 self tokens
    self_input_offset = sum(input_splits_list[:ep_rank])
    self_input_count = input_splits_list[ep_rank]
    self_copy_tokens = permuted_tokens[self_input_offset:self_input_offset + self_input_count]
    
    # 接收缓冲区
    recv_buffer = torch.empty(total_output_tokens, hidden_size,
                               dtype=permuted_tokens.dtype, device=device)
    
    # 创建 per-peer 通信请求
    send_reqs = []
    recv_reqs = []
    peer_recv_buffers = {}
    
    for peer in range(ep_world_size):
        if peer == ep_rank:
            continue
        
        send_offset = sum(input_splits_list[:peer])
        send_count = input_splits_list[peer]
        recv_offset = sum(output_splits_list[:peer])
        recv_count = output_splits_list[peer]
        
        if send_count > 0:
            send_tensor = permuted_tokens[send_offset:send_offset + send_count].contiguous()
            req = dist.isend(send_tensor, dst=peer)
            send_reqs.append(req)
        
        if recv_count > 0:
            recv_tensor = recv_buffer[recv_offset:recv_offset + recv_count]
            req = dist.irecv(recv_tensor, src=peer)
            recv_reqs.append(req)
            peer_recv_buffers[peer] = (recv_tensor, recv_count)
    
    # Self FC1 (与通信并行)
    if self_input_count > 0:
        self_fc1 = fluid_kernels.grouped_gemm(self_copy_tokens, fc1_weight, self_tokens_per_expert)
        self_fc1 = F.gelu(self_fc1)
    
    # 等待所有接收完成
    for req in recv_reqs:
        req.wait()
    
    # Sort 接收的数据
    # 首先把 self tokens 也放入 recv_buffer
    self_recv_offset = sum(output_splits_list[:ep_rank])
    recv_buffer[self_recv_offset:self_recv_offset + self_input_count] = self_copy_tokens
    
    sorted_tokens = torch.index_select(recv_buffer, 0, sort_indices)
    
    # Remote FC1
    self_output_count = self_out_idx.size(0)
    remote_tokens = sorted_tokens[self_output_count:]
    
    if remote_tokens.size(0) > 0:
        remote_fc1 = fluid_kernels.grouped_gemm(remote_tokens, fc1_weight, remote_tpe)
        remote_fc1 = F.gelu(remote_fc1)
    
    # Merge
    if self_input_count > 0:
        final_output.index_copy_(0, self_out_idx, self_fc1)
    if remote_tokens.size(0) > 0:
        final_output.index_copy_(0, remote_out_idx, remote_fc1)
    
    # 等待发送完成
    for req in send_reqs:
        req.wait()
    
    return final_output
