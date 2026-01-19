#!/usr/bin/env python3
"""
FluidMoE vs Megatron Baseline Comparison

对比 FluidMoE (P2P overlap + dW scheduling) 与 Megatron baseline (同步 AllToAll) 的性能。
两者使用相同的模型结构，确保公平对比。

Megatron Baseline 实现参考：
- megatron/core/transformer/moe/token_dispatcher.py (MoEAlltoAllTokenDispatcher)
- megatron/core/transformer/attention.py (context parallel via AllToAll)

核心区别：
- Megatron Baseline: 同步 AllToAll，dispatch_preprocess → token_dispatch → dispatch_postprocess
                     → expert_compute → combine_preprocess → token_combine → combine_postprocess
- FluidMoE: P2P Round-Robin overlap (forward), dW+AllToAll overlap (backward)

Usage:
    torchrun --nproc_per_node=4 tests/test_transformer_speedup.py
    torchrun --nproc_per_node=4 tests/test_transformer_speedup.py --mode training
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

from fluid.core.forward_comm import MultiCardOverlapContext
from fluid.core.scheduler import get_backward_scheduler
from fluid.core import _all_to_all
from fluid.moe.layer import moe_p2p_chunked
from fluid.attention.layer import attention_p2p_chunked


@dataclass
class TransformerConfig:
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 32
    ffn_hidden_size: int = 14336
    num_experts: int = 8
    top_k: int = 2
    seq_len: int = 4096
    batch_size: int = 4  # batch >= 4 for meaningful compute/comm overlap
    num_layers: int = 2
    num_chunks: int = 1  # Number of chunks for backward overlap (1 = no chunking)
    dtype: torch.dtype = torch.bfloat16


def setup_distributed():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    return rank, world_size, device


# =============================================================================
# Differentiable AllToAll for Baseline (autograd-compatible)
# =============================================================================

class DifferentiableAllToAll(torch.autograd.Function):
    """
    可微分的 AllToAll 通信。
    Backward 就是反方向的 AllToAll（input/output splits 互换）。
    """

    @staticmethod
    def forward(ctx, input_tensor, output_split_sizes, input_split_sizes, group):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = group.size()
        if world_size == 1:
            return input_tensor.clone()

        input_tensor = input_tensor.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input_tensor)
        else:
            output = input_tensor.new_empty(
                size=[sum(output_split_sizes)] + list(input_tensor.size()[1:]),
            )

        dist.all_to_all_single(
            output, input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        # Backward: swap input/output splits
        output_split_sizes = ctx.input_split_sizes
        input_split_sizes = ctx.output_split_sizes

        world_size = group.size()
        if world_size == 1:
            return grad_output.clone(), None, None, None

        grad_output = grad_output.contiguous()

        if output_split_sizes is None:
            grad_input = torch.empty_like(grad_output)
        else:
            grad_input = grad_output.new_empty(
                size=[sum(output_split_sizes)] + list(grad_output.size()[1:]),
            )

        dist.all_to_all_single(
            grad_input, grad_output,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return grad_input, None, None, None


def differentiable_all_to_all(input_tensor, output_split_sizes, input_split_sizes, group):
    """可微分 AllToAll 的便捷函数"""
    return DifferentiableAllToAll.apply(input_tensor, output_split_sizes, input_split_sizes, group)


def differentiable_all_to_all_equal(input_tensor, group):
    """Equal-split 可微分 AllToAll（用于 Attention）"""
    return DifferentiableAllToAll.apply(input_tensor, None, None, group)


# =============================================================================
# Baseline Attention Autograd Function (保存 Q/K/V，backward 重计算 attention)
# =============================================================================

class BaselineAttentionFunction(torch.autograd.Function):
    """
    Baseline attention autograd function.
    Forward: 保存 Q, K, V (不保存 attention scores 矩阵)
    Backward: 重新计算 attention scores，然后计算 grad_Q, grad_K, grad_V
    """

    @staticmethod
    def forward(ctx, q, k, v, scale, is_causal=True):
        """
        Args:
            q, k, v: [batch, heads, seq, head_dim]
            scale: attention scale factor
            is_causal: whether to apply causal mask
        """
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        if is_causal:
            seq_len = q.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Output
        output = torch.matmul(attn_probs, v)

        # Save for backward (不保存 attn_scores/attn_probs，节省 O(n^2) 内存)
        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.is_causal = is_causal

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: 重新计算 attention scores
        """
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal

        # Recompute attention (与 forward 相同)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            seq_len = q.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)

        # Compute gradients
        # grad_V = attn_probs.T @ grad_output
        grad_v = torch.matmul(attn_probs.transpose(-2, -1), grad_output)

        # grad_attn_probs = grad_output @ V.T
        grad_attn_probs = torch.matmul(grad_output, v.transpose(-2, -1))

        # Softmax backward: grad_scores = probs * (grad_probs - sum(grad_probs * probs))
        sum_grad = (grad_attn_probs * attn_probs).sum(dim=-1, keepdim=True)
        grad_attn_scores = attn_probs * (grad_attn_probs - sum_grad)

        # Apply causal mask to gradient (masked positions have zero gradient)
        if is_causal:
            grad_attn_scores = grad_attn_scores.masked_fill(causal_mask, 0.0)

        # grad_Q = grad_scores @ K * scale
        grad_q = torch.matmul(grad_attn_scores, k) * scale

        # grad_K = grad_scores.T @ Q * scale
        grad_k = torch.matmul(grad_attn_scores.transpose(-2, -1), q) * scale

        return grad_q, grad_k, grad_v, None, None


# =============================================================================
# Megatron Baseline: 同步 AllToAll (参考 MoEAlltoAllTokenDispatcher)
# =============================================================================

class MegatronBaselineMoE(nn.Module):
    """
    Megatron-style MoE baseline implementation.

    严格遵循 Megatron MoEAlltoAllTokenDispatcher 的 workflow:
    1. dispatch_preprocess: 计算 routing metadata, permute tokens by expert
    2. token_dispatch: AllToAll 通信
    3. dispatch_postprocess: sort tokens by local expert
    4. expert_compute: 专家计算
    5. combine_preprocess: unsort tokens
    6. token_combine: AllToAll 通信
    7. combine_postprocess: unpermute tokens

    参考: megatron/core/transformer/moe/token_dispatcher.py
    """

    def __init__(self, config, ep_group, device):
        super().__init__()
        self.ep_group = ep_group
        self.ep_size = ep_group.size()
        self.ep_rank = dist.get_rank(ep_group)

        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.experts_per_rank = self.num_experts // self.ep_size

        # Router weight (fp32 for stability, like Megatron)
        self.router_weight = nn.Parameter(torch.randn(
            config.hidden_size, config.num_experts,
            dtype=torch.float32, device=device) * 0.02)

        # Expert weights - GroupedMLP style: [num_local_experts, hidden, ffn] per expert
        # Megatron uses separate weights per expert in GroupedMLP
        self.w1 = nn.Parameter(torch.randn(
            self.experts_per_rank, config.hidden_size, config.ffn_hidden_size,
            dtype=config.dtype, device=device) * 0.02)
        self.w2 = nn.Parameter(torch.randn(
            self.experts_per_rank, config.ffn_hidden_size, config.hidden_size,
            dtype=config.dtype, device=device) * 0.02)

    def _permute_by_expert(self, hidden_states: torch.Tensor, routing_map: torch.Tensor,
                           probs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permute tokens according to routing_map (dispatch_preprocess in Megatron).

        Args:
            hidden_states: [num_tokens, hidden_size]
            routing_map: [num_tokens, num_experts] bool mask
            probs: [num_tokens, num_experts] routing probabilities

        Returns:
            permuted_tokens: [num_out_tokens, hidden_size]
            permuted_probs: [num_out_tokens]
            reverse_indices: for unpermuting later
        """
        num_tokens = hidden_states.shape[0]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Get indices where routing_map is True
        # [num_out_tokens, 2] where [:, 0] is token_idx and [:, 1] is expert_idx
        indices = routing_map.nonzero()

        if indices.shape[0] == 0:
            # No tokens routed
            return (hidden_states.new_zeros(0, self.hidden_size),
                    probs.new_zeros(0),
                    None)

        token_indices = indices[:, 0]
        expert_indices = indices[:, 1]

        # Sort by expert index (stable sort to preserve order within expert)
        sorted_order = expert_indices.argsort(stable=True)
        token_indices = token_indices[sorted_order]
        expert_indices = expert_indices[sorted_order]

        # Permute tokens and probs
        permuted_tokens = hidden_states[token_indices]
        permuted_probs = probs[token_indices, expert_indices]

        # Store for unpermute
        self._reverse_indices = (token_indices, expert_indices, num_tokens)

        return permuted_tokens, permuted_probs, sorted_order

    def _unpermute(self, output_tokens: torch.Tensor, permuted_probs: torch.Tensor) -> torch.Tensor:
        """
        Unpermute tokens back to original order (combine_postprocess in Megatron).
        """
        token_indices, expert_indices, num_tokens = self._reverse_indices

        # Weight by probs and accumulate
        output = output_tokens.new_zeros(num_tokens, self.hidden_size)
        weighted_output = output_tokens * permuted_probs.unsqueeze(-1).to(output_tokens.dtype)
        output.scatter_add_(0, token_indices.unsqueeze(-1).expand_as(weighted_output), weighted_output)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Megatron baseline forward: 同步 AllToAll + 顺序计算

        Args:
            x: [num_tokens, hidden_size]

        Returns:
            output: [num_tokens, hidden_size]
        """
        num_tokens = x.shape[0]
        dtype = x.dtype
        device = x.device

        # ============================================================
        # Step 1: Router (TopKRouter in Megatron)
        # ============================================================
        router_logits = x.float() @ self.router_weight  # [num_tokens, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # normalize

        # Create routing map (multi-hot encoding)
        routing_map = torch.zeros(num_tokens, self.num_experts, dtype=torch.bool, device=device)
        routing_map.scatter_(1, topk_indices, True)

        # Create probs tensor matching routing_map
        probs = torch.zeros(num_tokens, self.num_experts, dtype=topk_probs.dtype, device=device)
        probs.scatter_(1, topk_indices, topk_probs)

        # ============================================================
        # Step 2: dispatch_preprocess - Permute tokens by expert
        # ============================================================
        permuted_tokens, permuted_probs, _ = self._permute_by_expert(x, routing_map, probs)

        # Calculate tokens per expert for AllToAll splits
        tokens_per_expert = routing_map.sum(dim=0).long()  # [num_experts]

        # Calculate input_splits (tokens going to each EP rank)
        # Expert i goes to rank i // experts_per_rank
        input_splits = []
        for r in range(self.ep_size):
            start_e = r * self.experts_per_rank
            end_e = start_e + self.experts_per_rank
            count = tokens_per_expert[start_e:end_e].sum().item()
            input_splits.append(count)

        # ============================================================
        # Step 3: token_dispatch - AllToAll communication
        # ============================================================
        # Exchange split sizes
        input_splits_tensor = torch.tensor(input_splits, device=device, dtype=torch.long)
        output_splits_tensor = torch.empty_like(input_splits_tensor)
        dist.all_to_all_single(output_splits_tensor, input_splits_tensor, group=self.ep_group)
        output_splits = output_splits_tensor.tolist()

        # AllToAll for tokens (使用可微分版本)
        total_recv = sum(output_splits)
        if sum(input_splits) > 0 and total_recv > 0:
            recv_tokens = differentiable_all_to_all(
                permuted_tokens, output_splits, input_splits, self.ep_group
            )

            # AllToAll for probs (使用可微分版本)
            recv_probs = differentiable_all_to_all(
                permuted_probs.unsqueeze(-1), output_splits, input_splits, self.ep_group
            ).squeeze(-1)
        elif total_recv > 0:
            recv_tokens = differentiable_all_to_all(
                permuted_tokens.new_zeros(0, self.hidden_size),
                output_splits, input_splits, self.ep_group
            )
            recv_probs = differentiable_all_to_all(
                permuted_probs.new_zeros(0, 1),
                output_splits, input_splits, self.ep_group
            ).squeeze(-1)
        else:
            recv_tokens = permuted_tokens.new_zeros(0, self.hidden_size)
            recv_probs = permuted_probs.new_zeros(0)

        # ============================================================
        # Step 4: dispatch_postprocess - Sort by local expert
        # (In Megatron this uses sort_chunks_by_idxs when num_local_experts > 1)
        # ============================================================
        # Tokens are already grouped by expert from the sender side
        # We need to figure out tokens_per_local_expert

        # Gather global tokens_per_expert info
        global_tokens_per_expert = torch.zeros(self.ep_size, self.num_experts,
                                               dtype=torch.long, device=device)
        dist.all_gather_into_tensor(
            global_tokens_per_expert.view(-1),
            tokens_per_expert,
            group=self.ep_group
        )

        # Calculate tokens per local expert (from all ranks)
        local_expert_start = self.ep_rank * self.experts_per_rank
        tokens_per_local_expert = global_tokens_per_expert[:, local_expert_start:local_expert_start + self.experts_per_rank].sum(dim=0)

        # ============================================================
        # Step 5: Expert computation (GroupedMLP in Megatron)
        # ============================================================
        if total_recv > 0:
            expert_outputs = []
            offset = 0
            for local_e in range(self.experts_per_rank):
                count = tokens_per_local_expert[local_e].item()
                if count > 0:
                    expert_input = recv_tokens[offset:offset + count]
                    # FC1 + GeLU + FC2
                    h = expert_input @ self.w1[local_e]
                    h = F.gelu(h)
                    out = h @ self.w2[local_e]
                    expert_outputs.append(out)
                    offset += count

            if expert_outputs:
                expert_output = torch.cat(expert_outputs, dim=0)
            else:
                expert_output = recv_tokens.new_zeros(0, self.hidden_size)

            # Scale by probs (this happens after combine in some Megatron variants)
            expert_output = expert_output * recv_probs.unsqueeze(-1).to(dtype)
        else:
            expert_output = recv_tokens.new_zeros(0, self.hidden_size)

        # ============================================================
        # Step 6: combine_preprocess - Unsort tokens (reverse of dispatch_postprocess)
        # ============================================================
        # Already in correct order for AllToAll back

        # ============================================================
        # Step 7: token_combine - AllToAll communication (使用可微分版本)
        # ============================================================
        if sum(input_splits) > 0 and total_recv > 0:
            send_tokens = differentiable_all_to_all(
                expert_output, input_splits, output_splits, self.ep_group
            )
        elif sum(input_splits) > 0:
            send_tokens = differentiable_all_to_all(
                expert_output.new_zeros(0, self.hidden_size),
                input_splits, output_splits, self.ep_group
            )
        else:
            send_tokens = permuted_tokens.new_zeros(0, self.hidden_size)

        # ============================================================
        # Step 8: combine_postprocess - Unpermute tokens
        # ============================================================
        if send_tokens.shape[0] > 0:
            # Unpermute back to original token order
            # The output already has probs applied, so we use uniform probs for unpermute
            output = self._unpermute(send_tokens, torch.ones(send_tokens.shape[0], device=device))
        else:
            output = torch.zeros(num_tokens, self.hidden_size, dtype=dtype, device=device)

        return output


class MegatronBaselineAttention(nn.Module):
    """
    Megatron-style Ulysses Attention baseline.
    使用同步 AllToAll 实现 context parallel.

    Ulysses 流程:
    1. QKV projection
    2. SP → HP AllToAll (sequence parallel → head parallel)
    3. Scaled dot-product attention
    4. HP → SP AllToAll
    5. Output projection

    参考: megatron/core/transformer/attention.py + dot_product_attention.py
    """

    def __init__(self, config, cp_group, device):
        super().__init__()
        self.cp_group = cp_group
        self.cp_size = cp_group.size()
        self.cp_rank = dist.get_rank(cp_group)

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        # Heads per rank (after SP→HP AllToAll)
        self.q_heads_local = self.num_heads // self.cp_size
        self.kv_heads_local = self.num_kv_heads // self.cp_size

        # QKV weight (interleaved GQA layout like Megatron)
        # Layout: [q0, q1, ..., qn, k, v] per KV group
        q_per_kv_group = self.num_heads // self.num_kv_heads
        group_size = (q_per_kv_group + 2) * self.head_dim  # q_heads + k + v per group
        total_proj_size = self.num_kv_heads * group_size

        self.weight_qkv = nn.Parameter(
            torch.randn(total_proj_size, config.hidden_size, dtype=config.dtype, device=device) * 0.02)

        # Output projection
        self.weight_proj = nn.Parameter(
            torch.randn(config.hidden_size, self.num_heads * self.head_dim,
                       dtype=config.dtype, device=device) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Megatron baseline Ulysses attention: 同步 AllToAll

        Args:
            x: [seq_local, batch, hidden]

        Returns:
            output: [seq_local, batch, hidden]
        """
        seq_local, batch, hidden = x.shape
        dtype = x.dtype
        device = x.device
        seq_full = seq_local * self.cp_size

        # ============================================================
        # Step 1: QKV projection
        # ============================================================
        x_2d = x.view(-1, hidden)  # [seq_local * batch, hidden]
        qkv = x_2d @ self.weight_qkv.t()  # [seq_local * batch, total_proj_size]

        # Parse QKV from interleaved GQA layout (向量化，避免Python循环)
        # Layout: [Q_g0, K_g0, V_g0, Q_g1, K_g1, V_g1, ...]
        # 每个 group 包含 q_per_kv_group 个 Q heads + 1 K + 1 V
        q_per_kv_group = self.num_heads // self.num_kv_heads
        group_size = (q_per_kv_group + 2) * self.head_dim
        q_size = q_per_kv_group * self.head_dim

        # Reshape to [seq_local * batch, num_kv_heads, group_size]
        qkv = qkv.view(-1, self.num_kv_heads, group_size)

        # Split Q, K, V within each group
        q_grouped = qkv[:, :, :q_size]  # [N, num_kv_heads, q_per_kv_group * head_dim]
        k = qkv[:, :, q_size:q_size + self.head_dim]  # [N, num_kv_heads, head_dim]
        v = qkv[:, :, q_size + self.head_dim:]  # [N, num_kv_heads, head_dim]

        # Reshape Q: [N, num_kv_heads, q_per_kv_group * head_dim] -> [N, num_heads, head_dim]
        q = q_grouped.view(-1, self.num_kv_heads * q_per_kv_group, self.head_dim)

        # Reshape to [seq_local, batch, heads, head_dim]
        q = q.view(seq_local, batch, self.num_heads, self.head_dim)
        k = k.view(seq_local, batch, self.num_kv_heads, self.head_dim)
        v = v.view(seq_local, batch, self.num_kv_heads, self.head_dim)

        # ============================================================
        # Step 2: SP → HP AllToAll (同步)
        # [seq_local, batch, heads, head_dim] → [seq_full, batch, heads_local, head_dim]
        # ============================================================
        # Reshape: [seq_local, batch, cp_size, heads_local, head_dim]
        q = q.view(seq_local, batch, self.cp_size, self.q_heads_local, self.head_dim)
        k = k.view(seq_local, batch, self.cp_size, self.kv_heads_local, self.head_dim)
        v = v.view(seq_local, batch, self.cp_size, self.kv_heads_local, self.head_dim)

        # Permute for AllToAll: [cp_size, seq_local, batch, heads_local, head_dim]
        q_send = q.permute(2, 0, 1, 3, 4).contiguous()
        k_send = k.permute(2, 0, 1, 3, 4).contiguous()
        v_send = v.permute(2, 0, 1, 3, 4).contiguous()

        # 合并 Q, K, V 为一次 AllToAll (优化通信效率)
        # Concat along last dim: [cp_size, seq_local, batch, heads_local, head_dim * 3]
        qkv_send = torch.cat([
            q_send.reshape(self.cp_size, seq_local, batch, self.q_heads_local * self.head_dim),
            k_send.reshape(self.cp_size, seq_local, batch, self.kv_heads_local * self.head_dim),
            v_send.reshape(self.cp_size, seq_local, batch, self.kv_heads_local * self.head_dim),
        ], dim=-1)

        # Flatten for AllToAll
        qkv_flat = qkv_send.reshape(-1, qkv_send.shape[-1] * batch)
        qkv_hp_flat = differentiable_all_to_all_equal(qkv_flat, self.cp_group)

        # Split back to Q, K, V
        qkv_hp = qkv_hp_flat.view(self.cp_size, seq_local, batch, -1)
        q_size = self.q_heads_local * self.head_dim
        k_size = self.kv_heads_local * self.head_dim
        v_size = self.kv_heads_local * self.head_dim
        q_hp, k_hp, v_hp = torch.split(qkv_hp, [q_size, k_size, v_size], dim=-1)

        # Reshape to [seq_full, batch, heads_local, head_dim]
        q_hp = q_hp.reshape(seq_full, batch, self.q_heads_local, self.head_dim)
        k_hp = k_hp.reshape(seq_full, batch, self.kv_heads_local, self.head_dim)
        v_hp = v_hp.reshape(seq_full, batch, self.kv_heads_local, self.head_dim)

        # ============================================================
        # Step 3: GQA expansion + Scaled dot-product attention
        # ============================================================
        if self.num_heads > self.num_kv_heads:
            expand_ratio = self.num_heads // self.num_kv_heads
            k_hp = k_hp.repeat_interleave(expand_ratio, dim=2)
            v_hp = v_hp.repeat_interleave(expand_ratio, dim=2)

        # Transpose for SDPA: [batch, heads_local, seq_full, head_dim]
        q_hp = q_hp.permute(1, 2, 0, 3)
        k_hp = k_hp.permute(1, 2, 0, 3)
        v_hp = v_hp.permute(1, 2, 0, 3)

        scale = 1.0 / (self.head_dim ** 0.5)
        # 使用自定义 autograd Function，保存 Q/K/V，backward 重计算 attention
        attn_out = BaselineAttentionFunction.apply(q_hp, k_hp, v_hp, scale, True)

        # [batch, heads_local, seq_full, head_dim] → [seq_full, batch, heads_local, head_dim]
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous()

        # ============================================================
        # Step 4: HP → SP AllToAll (使用可微分版本)
        # [seq_full, batch, heads_local, head_dim] → [seq_local, batch, heads, head_dim]
        # ============================================================
        # Reshape: [cp_size, seq_local, batch, heads_local, head_dim]
        attn_out = attn_out.view(self.cp_size, seq_local, batch, self.q_heads_local, self.head_dim)
        attn_out_flat = attn_out.reshape(-1, batch * self.q_heads_local * self.head_dim)

        output_hp_flat = differentiable_all_to_all_equal(attn_out_flat, self.cp_group)
        output_hp = output_hp_flat.view(self.cp_size, seq_local, batch, self.q_heads_local, self.head_dim)

        # Permute back: [seq_local, batch, cp_size, heads_local, head_dim] → [seq_local, batch, heads, head_dim]
        output_hp = output_hp.permute(1, 2, 0, 3, 4).contiguous()
        output_hp = output_hp.view(seq_local, batch, self.num_heads, self.head_dim)

        # ============================================================
        # Step 5: Output projection
        # ============================================================
        output_hp = output_hp.view(seq_local * batch, -1)
        output = output_hp @ self.weight_proj.t()
        output = output.view(seq_local, batch, hidden)

        return output


# =============================================================================
# FluidMoE: P2P overlap + dW scheduling
# =============================================================================

class FluidMoELayer(nn.Module):
    """FluidMoE with P2P overlap for dispatch/combine"""

    def __init__(self, config, ep_group, device, overlap_ctx, layer_id=0):
        super().__init__()
        self.ep_group = ep_group
        self.ep_size = ep_group.size()
        self.overlap_ctx = overlap_ctx
        self.layer_id = layer_id

        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.num_chunks = config.num_chunks
        self.experts_per_rank = self.num_experts // self.ep_size

        # Router weight
        self.router_weight = nn.Parameter(torch.randn(
            config.hidden_size, config.num_experts,
            dtype=torch.float32, device=device) * 0.02)

        # Expert weights - FluidMoE 新 API 格式 (merged for all local experts)
        self.w1 = nn.Parameter(torch.randn(
            config.hidden_size, config.ffn_hidden_size * self.experts_per_rank,
            dtype=config.dtype, device=device) * 0.02)
        self.w2 = nn.Parameter(torch.randn(
            config.ffn_hidden_size * self.experts_per_rank, config.hidden_size,
            dtype=config.dtype, device=device) * 0.02)

    def forward(self, x):
        return moe_p2p_chunked(
            x,
            self.router_weight,
            self.w1,
            self.w2,
            self.ep_group,
            self.overlap_ctx,
            layer_id=self.layer_id,
            num_experts=self.num_experts,
            top_k=self.top_k,
            activation_func=F.gelu,
            num_chunks=self.num_chunks,
        )


class FluidAttentionLayer(nn.Module):
    """FluidMoE Attention with P2P overlap"""

    def __init__(self, config, cp_group, device, qkv_ctx, proj_ctx, layer_id=0):
        super().__init__()
        self.cp_group = cp_group
        self.cp_size = cp_group.size()
        self.qkv_ctx = qkv_ctx
        self.proj_ctx = proj_ctx
        self.layer_id = layer_id

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_chunks = config.num_chunks

        # QKV weight (interleaved layout)
        q_per_group = self.num_heads // self.num_kv_heads
        group_size = (q_per_group + 2) * self.head_dim
        total_proj = self.num_kv_heads * group_size
        self.weight_qkv = nn.Parameter(
            torch.randn(total_proj, config.hidden_size, dtype=config.dtype, device=device) * 0.02)

        # Output projection
        self.weight_proj = nn.Parameter(
            torch.randn(config.hidden_size, self.num_heads * self.head_dim,
                       dtype=config.dtype, device=device) * 0.02)

    def forward(self, x):
        return attention_p2p_chunked(
            x,
            self.weight_qkv,
            self.weight_proj,
            None,  # bias_proj
            self.cp_group,
            self.qkv_ctx,
            self.proj_ctx,
            layer_id=self.layer_id,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            num_chunks=self.num_chunks,
        )


# =============================================================================
# Transformer Models
# =============================================================================

class MegatronBaselineTransformerLayer(nn.Module):
    def __init__(self, config, cp_group, ep_group, device, layer_id=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=config.dtype, device=device)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=config.dtype, device=device)
        self.attn = MegatronBaselineAttention(config, cp_group, device)
        self.moe = MegatronBaselineMoE(config, ep_group, device)

    def forward(self, x):
        # x: [batch, seq_local, hidden]
        B, S, H = x.shape

        # Attention (with residual)
        x_norm = self.ln1(x)
        x_t = x_norm.transpose(0, 1).contiguous()  # [seq_local, batch, hidden]
        attn_out = self.attn(x_t)
        attn_out = attn_out.transpose(0, 1).contiguous()  # [batch, seq_local, hidden]
        x = x + attn_out

        # MoE (with residual)
        x_norm = self.ln2(x)
        x_flat = x_norm.view(-1, H)  # [batch * seq_local, hidden]
        moe_out = self.moe(x_flat)
        moe_out = moe_out.view(B, S, H)
        x = x + moe_out

        return x


class FluidTransformerLayer(nn.Module):
    def __init__(self, config, cp_group, ep_group, device,
                 qkv_ctx, proj_ctx, moe_ctx, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=config.dtype, device=device)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=config.dtype, device=device)
        self.attn = FluidAttentionLayer(config, cp_group, device, qkv_ctx, proj_ctx, layer_id)
        self.moe = FluidMoELayer(config, ep_group, device, moe_ctx, layer_id)

    def forward(self, x):
        B, S, H = x.shape

        # Attention
        x_norm = self.ln1(x)
        x_t = x_norm.transpose(0, 1).contiguous()
        attn_out = self.attn(x_t)
        attn_out = attn_out.transpose(0, 1).contiguous()
        x = x + attn_out

        # MoE
        x_norm = self.ln2(x)
        x_flat = x_norm.view(-1, H)
        moe_out = self.moe(x_flat)
        moe_out = moe_out.view(B, S, H)
        x = x + moe_out

        return x


class MegatronBaselineTransformer(nn.Module):
    def __init__(self, config, cp_group, ep_group, device):
        super().__init__()
        self.layers = nn.ModuleList([
            MegatronBaselineTransformerLayer(config, cp_group, ep_group, device, i)
            for i in range(config.num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FluidTransformer(nn.Module):
    def __init__(self, config, cp_group, ep_group, device,
                 qkv_ctx, proj_ctx, moe_ctx):
        super().__init__()
        # Create separate contexts per layer to avoid resource conflicts
        self.qkv_ctxs = [MultiCardOverlapContext(device, cp_group.size(), cp_group.size())
                        for _ in range(config.num_layers)]
        self.proj_ctxs = [MultiCardOverlapContext(device, cp_group.size(), cp_group.size())
                         for _ in range(config.num_layers)]
        self.moe_ctxs = [MultiCardOverlapContext(device, ep_group.size(), ep_group.size())
                        for _ in range(config.num_layers)]
        self.layers = nn.ModuleList([
            FluidTransformerLayer(config, cp_group, ep_group, device,
                                  self.qkv_ctxs[i], self.proj_ctxs[i], self.moe_ctxs[i], i)
            for i in range(config.num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(model, x_template, mode='training', warmup=10, iters=20,
              use_scheduler=False, scheduler=None):
    """Benchmark model performance"""

    def create_input():
        if mode == 'inference':
            return x_template.detach().clone()
        else:
            return x_template.detach().clone().requires_grad_(True)

    # Warmup
    for _ in range(warmup):
        x = create_input()
        if mode == 'inference':
            with torch.no_grad():
                _ = model(x)
        elif mode == 'forward':
            _ = model(x)
        else:  # training
            model.zero_grad()
            out = model(x)
            out.sum().backward()
            if use_scheduler and scheduler:
                scheduler.finish_batch()
                scheduler.clear_iteration()

    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    times = []
    for i in range(iters):
        x = create_input()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        if mode == 'inference':
            with torch.no_grad():
                _ = model(x)
        elif mode == 'forward':
            _ = model(x)
        else:  # training
            model.zero_grad()
            out = model(x)
            out.sum().backward()
            if use_scheduler and scheduler:
                scheduler.finish_batch()
                # Don't clear on last iteration so we can see stats
                if i < iters - 1:
                    scheduler.clear_iteration()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def verify_correctness(config, cp_group, ep_group, device, rank):
    """Verify that baseline models produce valid outputs"""
    if rank == 0:
        print("\nVerifying correctness...")

    scheduler = get_backward_scheduler()
    scheduler.disable()  # Disable scheduler for correctness check

    seq_per_rank = config.seq_len // dist.get_world_size()

    # Create identical input
    torch.manual_seed(42 + rank)
    x = torch.randn(config.batch_size, seq_per_rank, config.hidden_size,
                    dtype=config.dtype, device=device)

    # Create models
    baseline_moe = MegatronBaselineMoE(config, ep_group, device)
    baseline_attn = MegatronBaselineAttention(config, cp_group, device)

    # Test MoE forward
    x_flat = x.view(-1, config.hidden_size)

    with torch.no_grad():
        baseline_out = baseline_moe(x_flat.clone())

    # Check for NaN/Inf
    baseline_valid = not (torch.isnan(baseline_out).any() or torch.isinf(baseline_out).any())
    valid_tensor = torch.tensor([baseline_valid], device=device)
    dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)

    if rank == 0:
        if valid_tensor.item() == 1:
            print("  Baseline MoE: OK (no NaN/Inf)")
        else:
            print("  Baseline MoE: FAILED (contains NaN/Inf)")

    # Test Attention
    x_t = x.transpose(0, 1).contiguous()
    with torch.no_grad():
        attn_out = baseline_attn(x_t.clone())

    attn_valid = not (torch.isnan(attn_out).any() or torch.isinf(attn_out).any())
    valid_tensor = torch.tensor([attn_valid], device=device)
    dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)

    if rank == 0:
        if valid_tensor.item() == 1:
            print("  Baseline Attention: OK (no NaN/Inf)")
        else:
            print("  Baseline Attention: FAILED (contains NaN/Inf)")

    del baseline_moe, baseline_attn
    return True


def main(mode='training', batch_size=None, seq_len=None, num_chunks=None):
    rank, world_size, device = setup_distributed()

    # Configuration
    sp_size = ep_size = world_size

    # Ensure heads and experts are divisible by world_size
    num_kv_heads = max(world_size, 32 // world_size * world_size)
    num_experts = world_size * 2  # 2 experts per rank

    config = TransformerConfig(
        hidden_size=4096,
        num_attention_heads=num_kv_heads,
        num_kv_heads=num_kv_heads,
        ffn_hidden_size=14336,
        num_experts=num_experts,
        top_k=2,
        seq_len=seq_len or 4096,
        batch_size=batch_size or 4,  # batch >= 4 for meaningful compute/comm overlap
        num_layers=2,
        num_chunks=num_chunks or 1,  # 1 = no chunking, 4 = chunked overlap
        dtype=torch.bfloat16,
    )

    # Validate
    assert config.num_kv_heads % sp_size == 0
    assert config.num_experts % ep_size == 0
    assert config.seq_len % sp_size == 0

    cp_group = ep_group = dist.group.WORLD
    seq_per_rank = config.seq_len // world_size

    if rank == 0:
        print("\n" + "=" * 70)
        print("FluidMoE vs Megatron Baseline Comparison")
        print("=" * 70)
        print(f"World size: {world_size} (SP={sp_size}, EP={ep_size})")
        print(f"Hidden: {config.hidden_size}, Heads: {config.num_attention_heads}")
        print(f"FFN: {config.ffn_hidden_size}, Experts: {config.num_experts} ({config.num_experts // ep_size}/rank)")
        print(f"Seq: {config.seq_len} ({seq_per_rank}/rank), Batch: {config.batch_size}, Layers: {config.num_layers}")
        print(f"Test mode: {mode}")
        print("-" * 70)
        print("Megatron Baseline: Synchronous AllToAll (MoEAlltoAllTokenDispatcher style)")
        print("FluidMoE: P2P Round-Robin overlap + dW scheduling")
        print("=" * 70)

    dist.barrier()

    # Verify correctness first
    verify_correctness(config, cp_group, ep_group, device, rank)

    dist.barrier()

    # Create models
    scheduler = get_backward_scheduler()

    # FluidMoE contexts
    qkv_ctx = MultiCardOverlapContext(device, sp_size, sp_size)
    proj_ctx = MultiCardOverlapContext(device, sp_size, sp_size)
    moe_ctx = MultiCardOverlapContext(device, ep_size, ep_size)

    # Input
    x = torch.randn(config.batch_size, seq_per_rank, config.hidden_size,
                    dtype=config.dtype, device=device)

    # Models
    baseline = MegatronBaselineTransformer(config, cp_group, ep_group, device)
    fluid = FluidTransformer(config, cp_group, ep_group, device,
                             qkv_ctx, proj_ctx, moe_ctx)

    dist.barrier()

    if rank == 0:
        print(f"\nBenchmarking {mode}...")

    # Benchmark Megatron baseline
    scheduler.disable()
    baseline_time = benchmark(baseline, x, mode, use_scheduler=False)

    # Benchmark FluidMoE
    scheduler.enable()
    fluid_time = benchmark(fluid, x, mode, use_scheduler=True, scheduler=scheduler)

    dist.barrier()

    if rank == 0:
        speedup = baseline_time / fluid_time
        print("\n" + "=" * 70)
        print("Results")
        print("=" * 70)
        print(f"\n{mode.title()} Performance:")
        print(f"  Megatron Baseline: {baseline_time:.2f} ms")
        print(f"  FluidMoE:          {fluid_time:.2f} ms")
        print(f"  Speedup:           {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")

        if mode == 'training':
            stats = scheduler.get_stats()
            print(f"\ndW Scheduling Stats:")
            print(f"  Total dW tasks:    {stats['total_dw_tasks']}")
            print(f"  Overlap completed: {stats['overlap_completed_dw_tasks']}")
            print(f"  Finish completed:  {stats['finish_batch_completed_dw_tasks']}")
            if stats['total_dw_tasks'] > 0:
                ratio = stats['overlap_completed_dw_tasks'] / stats['total_dw_tasks'] * 100
                print(f"  Overlap ratio:     {ratio:.1f}%")

        if world_size == 2:
            print("\n" + "-" * 70)
            print("Note: 2-GPU testing shows overhead from P2P/stream management.")
            print("FluidMoE optimizations are designed for 4+ GPUs where P2P overlap")
            print("across multiple rounds provides significant communication hiding.")
            print("Consider testing with 4+ GPUs for realistic performance gains.")

        print("\n" + "=" * 70)

    # Cleanup
    del baseline, fluid, qkv_ctx, proj_ctx, moe_ctx
    torch.cuda.empty_cache()

    dist.destroy_process_group()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='FluidMoE vs Megatron Baseline')
    parser.add_argument('--mode', '-m', type=str, default='training',
                        choices=['inference', 'forward', 'training'],
                        help='Test mode (default: training)')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='Batch size (default: 2)')
    parser.add_argument('--seq-len', '-s', type=int, default=None,
                        help='Sequence length (default: 4096)')
    parser.add_argument('--num-chunks', '-c', type=int, default=None,
                        help='Number of chunks for backward overlap (default: 1, no chunking)')

    args = parser.parse_args()
    main(mode=args.mode, batch_size=args.batch_size, seq_len=args.seq_len, num_chunks=args.num_chunks)
