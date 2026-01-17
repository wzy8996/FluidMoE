#!/usr/bin/env python3
"""
FluidMoE Complete Transformer Speedup Test

测试完整 Transformer 结构（Attention + MoE）的加速效果。
对比 Baseline (AllGather/AllToAll) 和 Overlap (P2P重叠) 两种模式的前向性能。

Usage:
    torchrun --nproc_per_node=2 tests/test_transformer_speedup.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from dataclasses import dataclass

from fluid.attention.baseline import scaled_dot_product_attention, AttentionBaseline
# 使用 fluid 包中的 P2P 重叠实现 (根据 requires_grad 自动选择)
from fluid.core.forward_comm import MultiCardOverlapContext
from fluid.core.scheduler import get_backward_scheduler
from fluid.attention.p2p_overlap import (
    qkv_sp2hp_multicard_overlap,
    hp2sp_output_proj_multicard_overlap,
)
from fluid.moe.p2p_overlap import moe_multicard_p2p_overlap_forward
from fluid.moe.baseline import MoEBaseline
from fluid.moe.router import compute_routing


@dataclass
class TransformerConfig:
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_kv_heads: int = 32
    ffn_hidden_size: int = 14336
    num_experts: int = 4
    top_k: int = 2
    seq_len: int = 2048
    batch_size: int = 2
    num_layers: int = 2
    dtype: torch.dtype = torch.bfloat16


def setup_distributed():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    return rank, world_size, device


# =============================================================================
# Simple MoE Layer Wrapper (uses fluid/moe/baseline.py MoEBaseline)
# =============================================================================

class SimpleMoELayer(nn.Module):
    """MoE wrapper using MoEBaseline from fluid package (has proper autograd)"""

    def __init__(self, config, ep_group, device, layer_id=0):
        super().__init__()
        # Convert dataclass config to dict for MoEBaseline
        moe_config = {
            'hidden_size': config.hidden_size,
            'ffn_hidden_size': config.ffn_hidden_size,
            'num_experts': config.num_experts,
            'top_k': config.top_k,
        }
        self.moe = MoEBaseline(moe_config, ep_group, device, config.dtype, layer_id)
        self.moe.init_weights(requires_grad=True)

    def forward(self, x):
        # x: [batch*seq, hidden]
        return self.moe.forward(x, do_backward=False)


# =============================================================================
# Overlap MoE Layer (uses P2P overlap)
# =============================================================================

class OverlapMoELayer(nn.Module):
    """MoE using P2P overlap for dispatch/combine"""

    def __init__(self, config, ep_group, device, overlap_ctx):
        super().__init__()
        self.ep_group = ep_group
        self.ep_size = ep_group.size()
        self.my_rank = ep_group.rank()
        self.overlap_ctx = overlap_ctx

        self.hidden_size = config.hidden_size
        self.ffn_hidden_size = config.ffn_hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.experts_per_rank = self.num_experts // self.ep_size

        # Router weight (使用 compute_routing 函数，与 MoEBaseline 一致)
        self.router_weight = torch.randn(
            config.hidden_size, config.num_experts,
            dtype=torch.float32, device=device, requires_grad=True
        )

        # Expert weights - 格式适配 moe_multicard_p2p_overlap_forward
        # weight1: [hidden, ffn_hidden * num_local_experts]
        # weight2: [ffn_hidden * num_local_experts, hidden]
        self.w1 = nn.Parameter(torch.randn(
            config.hidden_size, config.ffn_hidden_size * self.experts_per_rank,
            dtype=config.dtype, device=device) * 0.02)
        self.w2 = nn.Parameter(torch.randn(
            config.ffn_hidden_size * self.experts_per_rank, config.hidden_size,
            dtype=config.dtype, device=device) * 0.02)

    def forward(self, x, layer_id=0):
        # x: [batch*seq, hidden]
        num_tokens = x.shape[0]

        # Routing (使用 compute_routing，与 MoEBaseline 一致，注册 router dW 任务)
        permuted_tokens, input_splits, output_splits, permuted_probs, \
            restore_indices, local_tokens_per_expert, global_tokens_per_expert, \
            tokens_per_expert_2d = compute_routing(
                x, self.router_weight, self.num_experts, self.top_k,
                self.ep_group, layer_id)

        # 构建 num_global_tokens_per_local_expert [tp_size=1, ep_size, num_local_experts]
        local_expert_start = self.my_rank * self.experts_per_rank
        num_global_tokens_per_local_expert = tokens_per_expert_2d[
            :, local_expert_start:local_expert_start + self.experts_per_rank
        ].unsqueeze(0)  # [1, ep_size, num_local_experts]

        # 本地专家的 token 数
        local_tokens = local_tokens_per_expert[
            local_expert_start:local_expert_start + self.experts_per_rank
        ]

        # 使用 P2P overlap 前向
        combined = moe_multicard_p2p_overlap_forward(
            permuted_tokens.contiguous(),
            input_splits,
            output_splits,
            self.w1,
            self.w2,
            self.ep_group,
            F.gelu,
            self.overlap_ctx,
            layer_id,
            self.experts_per_rank,
            local_tokens,
            num_global_tokens_per_local_expert,
        )

        # Apply probs and restore order
        combined = combined * permuted_probs.unsqueeze(-1).to(combined.dtype)
        restored = combined[restore_indices]
        output = restored.view(num_tokens, self.top_k, -1).sum(dim=1)

        return output


# =============================================================================
# Simple Attention Wrapper (uses fluid/attention/baseline.py AttentionBaseline)
# =============================================================================

class SimpleAttention(nn.Module):
    """Attention wrapper using AttentionBaseline from fluid package (Ulysses SP)"""

    def __init__(self, config, cp_group, device, layer_id=0):
        super().__init__()
        self.cp_group = cp_group
        self.cp_size = cp_group.size()
        self.layer_id = layer_id

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_kv_heads', config.num_attention_heads)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.heads_per_rank = self.num_heads // self.cp_size

        # GQA parameters
        self.q_per_group = self.num_heads // self.num_kv_heads
        self.group_size = (self.q_per_group + 2) * self.head_dim
        self.total_proj = self.num_kv_heads * self.group_size

        # QKV projection weight: [total_proj, hidden] interleaved layout (same as Overlap)
        self.weight_qkv = nn.Parameter(torch.randn(
            self.total_proj, config.hidden_size,
            dtype=config.dtype, device=device) * 0.02)

        # Output projection weight: [hidden, num_heads * head_dim]
        self.weight_proj = nn.Parameter(torch.randn(
            config.hidden_size, self.num_heads * self.head_dim,
            dtype=config.dtype, device=device) * 0.02)

        # 导入需要的函数
        from fluid.attention.baseline import (
            _QKVProjectionFunction, _OutputProjectionFunction,
            _SP2HPFunction, _HP2SPFunction
        )
        self._QKVProjectionFunction = _QKVProjectionFunction
        self._OutputProjectionFunction = _OutputProjectionFunction
        self._SP2HPFunction = _SP2HPFunction
        self._HP2SPFunction = _HP2SPFunction

    def forward(self, x):
        # x: [seq_local, batch, hidden]
        seq_local, batch, hidden = x.shape

        # 1. QKV Projection with interleaved layout
        # Detach + contiguous to avoid retaining computation graph across iterations
        # (we compute dW manually in backward, so no need for autograd to track weight)
        qkv = self._QKVProjectionFunction.apply(
            x, self.weight_qkv.t().detach().contiguous(),  # Transpose for matmul
            f"attn_qkv_L{self.layer_id}", self.layer_id,
            self.weight_qkv  # Pass original weight for gradient
        )

        # Extract Q, K, V from interleaved layout
        qkv = qkv.view(seq_local, batch, self.num_kv_heads, self.group_size)
        q_dim = self.q_per_group * self.head_dim

        q = qkv[:, :, :, :q_dim]
        k = qkv[:, :, :, q_dim:q_dim + self.head_dim]
        v = qkv[:, :, :, q_dim + self.head_dim:]

        # Reshape Q to [seq, batch, num_heads, head_dim]
        q = q.reshape(seq_local, batch, self.num_heads, self.head_dim)
        if self.q_per_group > 1:
            k = k.repeat_interleave(self.q_per_group, dim=2)
            v = v.repeat_interleave(self.q_per_group, dim=2)
        else:
            k = k.view(seq_local, batch, self.num_heads, self.head_dim)
            v = v.view(seq_local, batch, self.num_heads, self.head_dim)

        # 2. sp2hp AllToAll
        if self.cp_size > 1:
            q = self._SP2HPFunction.apply(q, self.cp_group, self.layer_id)
            k = self._SP2HPFunction.apply(k, self.cp_group, self.layer_id)
            v = self._SP2HPFunction.apply(v, self.cp_group, self.layer_id)

        # After sp2hp: [seq_full, batch, heads_local, head_dim]
        seq_full = seq_local * self.cp_size

        # 3. Core Attention
        q = q.permute(1, 2, 0, 3)  # [batch, heads_local, seq_full, head_dim]
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        attn_out = scaled_dot_product_attention(q, k, v)

        # Reshape back: [seq_full, batch, heads_local, head_dim]
        attn_out = attn_out.permute(2, 0, 1, 3)

        # 4. hp2sp AllToAll
        if self.cp_size > 1:
            attn_out = self._HP2SPFunction.apply(attn_out, self.cp_group, self.layer_id)

        # After hp2sp: [seq_local, batch, heads, head_dim]
        attn_out = attn_out.reshape(seq_local, batch, self.hidden_size)

        # 5. Output Projection
        # Detach weight to avoid retaining computation graph
        output = self._OutputProjectionFunction.apply(
            attn_out, self.weight_proj.detach(),
            f"attn_proj_L{self.layer_id}", self.layer_id,
            self.weight_proj  # Pass original weight for gradient
        )

        return output


# =============================================================================
# Overlap Attention (uses P2P overlap)
# =============================================================================

class OverlapAttention(nn.Module):
    """Attention with P2P overlap"""

    def __init__(self, config, cp_group, device, overlap_ctx):
        super().__init__()
        self.cp_group = cp_group
        self.cp_size = cp_group.size()
        self.overlap_ctx = overlap_ctx

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        # QKV weight (interleaved layout)
        q_per_group = self.num_heads // self.num_kv_heads
        group_size = (q_per_group + 2) * self.head_dim
        total_proj = self.num_kv_heads * group_size
        self.weight_qkv = nn.Parameter(
            torch.randn(total_proj, config.hidden_size, dtype=config.dtype, device=device) * 0.02)

        # Output projection
        self.weight_proj = nn.Parameter(
            torch.randn(config.hidden_size, config.num_attention_heads * self.head_dim,
                       dtype=config.dtype, device=device) * 0.02)

    def forward(self, x, layer_id=0):
        # QKV with P2P overlap (根据 requires_grad 自动选择实现)
        # - requires_grad=False: 使用纯前向实现
        # - requires_grad=True: 使用 autograd function
        q, k, v = qkv_sp2hp_multicard_overlap(
            x, self.weight_qkv,
            self.num_heads, self.num_kv_heads, self.head_dim,
            self.cp_group, self.overlap_ctx, layer_id)

        # Attention
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        attn_out = scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.permute(2, 0, 1, 3).contiguous()

        # hp2sp + output projection with P2P overlap
        output = hp2sp_output_proj_multicard_overlap(
            attn_out, self.weight_proj, None,
            self.cp_group, self.overlap_ctx)

        return output


# =============================================================================
# Transformers
# =============================================================================

class BaselineTransformerLayer(nn.Module):
    def __init__(self, config, cp_group, ep_group, device, layer_id=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=config.dtype, device=device)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=config.dtype, device=device)
        self.attn = SimpleAttention(config, cp_group, device, layer_id)
        self.moe = SimpleMoELayer(config, ep_group, device, layer_id)

    def forward(self, x):
        B, S, H = x.shape
        x_norm = self.ln1(x)
        x_t = x_norm.transpose(0, 1).contiguous()
        attn_out = self.attn(x_t)
        attn_out = attn_out.transpose(0, 1).contiguous()
        x = x + attn_out

        x_norm = self.ln2(x)
        x_flat = x_norm.view(-1, H)
        moe_out = self.moe(x_flat)
        moe_out = moe_out.view(B, S, H)
        x = x + moe_out
        return x


class OverlapTransformerLayer(nn.Module):
    def __init__(self, config, cp_group, ep_group, device, overlap_ctx, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(config.hidden_size, dtype=config.dtype, device=device)
        self.ln2 = nn.LayerNorm(config.hidden_size, dtype=config.dtype, device=device)
        self.attn = OverlapAttention(config, cp_group, device, overlap_ctx)
        self.moe = OverlapMoELayer(config, ep_group, device, overlap_ctx)  # MoE with P2P overlap

    def forward(self, x):
        B, S, H = x.shape
        x_norm = self.ln1(x)
        x_t = x_norm.transpose(0, 1).contiguous()
        attn_out = self.attn(x_t, self.layer_id)
        attn_out = attn_out.transpose(0, 1).contiguous()
        x = x + attn_out

        x_norm = self.ln2(x)
        x_flat = x_norm.view(-1, H)
        moe_out = self.moe(x_flat, self.layer_id)
        moe_out = moe_out.view(B, S, H)
        x = x + moe_out
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config, cp_group, ep_group, device):
        super().__init__()
        self.layers = nn.ModuleList([
            BaselineTransformerLayer(config, cp_group, ep_group, device, i)
            for i in range(config.num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OverlapTransformer(nn.Module):
    def __init__(self, config, cp_group, ep_group, device, overlap_ctx):
        super().__init__()
        self.layers = nn.ModuleList([
            OverlapTransformerLayer(config, cp_group, ep_group, device, overlap_ctx, i)
            for i in range(config.num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(model, x_template, mode='inference', warmup=20, iters=30, use_scheduler=False):
    scheduler = get_backward_scheduler()

    # Enable scheduler for training mode overlap model
    if mode == 'training' and use_scheduler:
        scheduler.enable()
    else:
        scheduler.disable()

    def create_input():
        # 每次都 clone，根据模式设置 requires_grad
        # inference: requires_grad=False，跳过保存中间结果
        # forward/training: requires_grad=True
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
        else:
            model.zero_grad()  # Clear gradients before forward
            out = model(x)
            out.sum().backward()
            if use_scheduler:
                scheduler.finish_batch()
                scheduler.clear_iteration()

    torch.cuda.synchronize()
    dist.barrier()

    # Benchmark
    times = []
    for _ in range(iters):
        x = create_input()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        if mode == 'inference':
            with torch.no_grad():
                _ = model(x)
        elif mode == 'forward':
            _ = model(x)
        else:
            model.zero_grad()  # Clear gradients before forward
            out = model(x)
            out.sum().backward()
            if use_scheduler:
                scheduler.finish_batch()
                scheduler.clear_iteration()
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return sum(times) / len(times)


def main():
    rank, world_size, device = setup_distributed()

    # SP = EP = world_size (单机多卡，所有卡参与 Context Parallel 和 Expert Parallel)
    sp_size = ep_size = world_size

    # 动态配置：确保 heads 和 experts 能被卡数整除
    # num_kv_heads 必须能被 sp_size 整除 (Context Parallel 要求)
    # num_experts 必须能被 ep_size 整除 (Expert Parallel 要求)
    base_kv_heads = 32  # 基础 KV heads 数
    base_experts_per_rank = 2  # 每个 rank 的专家数

    # 调整 kv_heads 使其能被 world_size 整除
    num_kv_heads = max(world_size, (base_kv_heads // world_size) * world_size)
    if num_kv_heads < world_size:
        num_kv_heads = world_size

    config = TransformerConfig(
        hidden_size=4096,
        num_attention_heads=num_kv_heads,  # Q heads = KV heads (MHA)
        num_kv_heads=num_kv_heads,
        ffn_hidden_size=14336,
        num_experts=ep_size * base_experts_per_rank,  # 每个 rank 2 个专家
        top_k=2,
        seq_len=4096,
        batch_size=2,
        num_layers=2,
        dtype=torch.bfloat16,
    )

    # 验证配置
    assert config.num_kv_heads % sp_size == 0, f"num_kv_heads ({config.num_kv_heads}) must be divisible by sp_size ({sp_size})"
    assert config.num_experts % ep_size == 0, f"num_experts ({config.num_experts}) must be divisible by ep_size ({ep_size})"
    assert config.seq_len % sp_size == 0, f"seq_len ({config.seq_len}) must be divisible by sp_size ({sp_size})"

    cp_group = ep_group = dist.group.WORLD
    seq_per_rank = config.seq_len // world_size

    if rank == 0:
        print("\n" + "=" * 60)
        print("FluidMoE Transformer Speedup Test")
        print("=" * 60)
        print(f"World size: {world_size} (SP={sp_size}, EP={ep_size})")
        print(f"Hidden: {config.hidden_size}, Heads: {config.num_attention_heads}, KV Heads: {config.num_kv_heads}")
        print(f"FFN: {config.ffn_hidden_size}, Experts: {config.num_experts} ({config.num_experts // ep_size} per rank)")
        print(f"Seq: {config.seq_len} ({seq_per_rank} per rank), Batch: {config.batch_size}, Layers: {config.num_layers}")
        print("=" * 60)

    dist.barrier()

    # =========================================================================
    # 全局预热：先运行所有模式一次，让 CUDA kernel 编译完成
    # =========================================================================
    if rank == 0:
        print("\nGlobal warmup (compiling CUDA kernels)...")

    overlap_ctx = MultiCardOverlapContext(device, ep_size, sp_size)
    x = torch.randn(config.batch_size, seq_per_rank, config.hidden_size,
                    dtype=config.dtype, device=device)
    baseline = BaselineTransformer(config, cp_group, ep_group, device)
    overlap = OverlapTransformer(config, cp_group, ep_group, device, overlap_ctx)

    scheduler = get_backward_scheduler()

    for warmup_mode in ['inference', 'forward', 'training']:
        for model in [baseline, overlap]:
            for _ in range(3):  # 每种模式预热 3 次
                if warmup_mode == 'inference':
                    with torch.no_grad():
                        _ = model(x.detach())
                elif warmup_mode == 'forward':
                    _ = model(x.detach().clone().requires_grad_(True))
                else:
                    scheduler.enable()
                    model.zero_grad()
                    out = model(x.detach().clone().requires_grad_(True))
                    out.sum().backward()
                    scheduler.finish_batch()
                    scheduler.clear_iteration()
                    scheduler.disable()

    del baseline, overlap, overlap_ctx, x
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print("Global warmup done.\n")

    # =========================================================================
    # 正式测试
    # =========================================================================
    results = {}

    for mode in ['inference', 'forward', 'training']:
        if rank == 0:
            print(f"Benchmarking {mode}...")

        # 每种模式重新创建模型
        torch.cuda.empty_cache()

        overlap_ctx = MultiCardOverlapContext(device, ep_size, sp_size)
        x = torch.randn(config.batch_size, seq_per_rank, config.hidden_size,
                        dtype=config.dtype, device=device)

        baseline = BaselineTransformer(config, cp_group, ep_group, device)
        overlap = OverlapTransformer(config, cp_group, ep_group, device, overlap_ctx)

        dist.barrier()

        # Baseline never uses scheduler (it doesn't do any overlap)
        # Overlap uses scheduler only in training mode
        base_time = benchmark(baseline, x, mode, use_scheduler=False)
        ovlp_time = benchmark(overlap, x, mode, use_scheduler=(mode == 'training'))
        results[mode] = (base_time, ovlp_time)

        # 清理模型释放显存
        del baseline, overlap, overlap_ctx, x
        torch.cuda.empty_cache()

    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)

        for mode, (base, ovlp) in results.items():
            speedup = base / ovlp
            print(f"\n{mode.title()}:")
            print(f"  Baseline: {base:.2f} ms")
            print(f"  Overlap:  {ovlp:.2f} ms")
            print(f"  Speedup:  {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")

        print("\n" + "=" * 60)

    dist.destroy_process_group()


def test_chunked_backward():
    """测试 dX 切块反向传播的加速效果"""
    rank, world_size, device = setup_distributed()

    # Import chunked backward functions
    from fluid.attention.chunked_backward import backward_output_proj_chunked
    from fluid.moe.chunked_backward import backward_dispatch_chunked
    from fluid.core import _all_to_all_sp2hp_forward, _all_to_all

    dtype = torch.bfloat16

    if rank == 0:
        print("\n" + "=" * 60)
        print("dX Chunked Backward Speedup Test")
        print("=" * 60)

    scheduler = get_backward_scheduler()
    scheduler.enable()

    # ========================================
    # Test 1: Attention Output Projection Chunked Backward
    # ========================================
    if rank == 0:
        print("\n[Test 1] Attention Output Projection Chunked Backward")
        print("-" * 50)

    # Config
    seq_local = 2048
    batch_size = 2
    hidden_size = 4096
    num_heads = 32
    head_dim = hidden_size // num_heads
    cp_group = dist.group.WORLD
    cp_size = world_size

    # Create test data
    grad_output = torch.randn(seq_local, batch_size, hidden_size, dtype=dtype, device=device)
    weight_proj = (torch.randn(hidden_size, num_heads * head_dim, dtype=dtype, device=device) * 0.02)

    warmup = 3
    iters = 10

    # Baseline: non-chunked (num_chunks=1)
    for _ in range(warmup):
        _ = backward_output_proj_chunked(grad_output, weight_proj, num_heads, head_dim, cp_group, num_chunks=1)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = backward_output_proj_chunked(grad_output, weight_proj, num_heads, head_dim, cp_group, num_chunks=1)
    end.record()
    torch.cuda.synchronize()
    base_time = start.elapsed_time(end) / iters

    # Chunked: num_chunks=4
    for _ in range(warmup):
        _ = backward_output_proj_chunked(grad_output, weight_proj, num_heads, head_dim, cp_group, num_chunks=4)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        _ = backward_output_proj_chunked(grad_output, weight_proj, num_heads, head_dim, cp_group, num_chunks=4)
    end.record()
    torch.cuda.synchronize()
    chunk4_time = start.elapsed_time(end) / iters

    if rank == 0:
        speedup = base_time / chunk4_time
        print(f"  Baseline (chunks=1): {base_time:.2f} ms")
        print(f"  Chunked (chunks=4):  {chunk4_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")

    # ========================================
    # Test 2: MoE Dispatch Chunked Backward
    # ========================================
    if rank == 0:
        print("\n[Test 2] MoE Dispatch Chunked Backward")
        print("-" * 50)

    # Config
    num_experts = world_size * 2
    experts_per_rank = num_experts // world_size
    ffn_hidden = 14336
    num_tokens = 4096
    ep_group = dist.group.WORLD
    ep_size = world_size

    # Simulate token distribution (uniform across experts and ranks)
    tokens_per_expert = num_tokens // experts_per_rank
    tokens_per_expert_list = [tokens_per_expert] * experts_per_rank
    total_recv = sum(tokens_per_expert_list)

    # Create test data
    grad_all_fc1 = torch.randn(total_recv, ffn_hidden, dtype=dtype, device=device)
    weight1 = (torch.randn(experts_per_rank, hidden_size, ffn_hidden, dtype=dtype, device=device) * 0.02)

    # Create split info for AllToAll (uniform distribution)
    # Each rank sends/receives the same amount
    tokens_per_rank = total_recv // ep_size
    input_splits_list = [tokens_per_rank] * ep_size
    output_splits_list = [tokens_per_rank] * ep_size

    # Create expert-major → rank-major reorder indices
    # For uniform distribution, we need to interleave experts across ranks
    # Expert 0 tokens go to rank 0, expert 1 tokens go to rank 1, etc (round-robin)
    # split_sizes_exp_major: sizes per expert in expert-major order
    split_sizes_exp_major = tokens_per_expert_list

    # sorted_idxs: maps chunk index in expert-major to chunk index in rank-major
    # For 2 experts, 2 ranks: expert0->rank0, expert1->rank1
    # So sorted_idxs = [0, 1] (identity for this simple case)
    sorted_idxs_exp_to_rank = list(range(experts_per_rank))

    # Baseline: non-chunked (num_chunks=1)
    for _ in range(warmup):
        _ = backward_dispatch_chunked(
            grad_all_fc1, weight1, split_sizes_exp_major, sorted_idxs_exp_to_rank,
            tokens_per_expert_list, input_splits_list, output_splits_list, ep_group, num_chunks=1
        )
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        _ = backward_dispatch_chunked(
            grad_all_fc1, weight1, split_sizes_exp_major, sorted_idxs_exp_to_rank,
            tokens_per_expert_list, input_splits_list, output_splits_list, ep_group, num_chunks=1
        )
    end.record()
    torch.cuda.synchronize()
    base_time = start.elapsed_time(end) / iters

    # Chunked: num_chunks=4
    for _ in range(warmup):
        _ = backward_dispatch_chunked(
            grad_all_fc1, weight1, split_sizes_exp_major, sorted_idxs_exp_to_rank,
            tokens_per_expert_list, input_splits_list, output_splits_list, ep_group, num_chunks=4
        )
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        _ = backward_dispatch_chunked(
            grad_all_fc1, weight1, split_sizes_exp_major, sorted_idxs_exp_to_rank,
            tokens_per_expert_list, input_splits_list, output_splits_list, ep_group, num_chunks=4
        )
    end.record()
    torch.cuda.synchronize()
    chunk4_time = start.elapsed_time(end) / iters

    if rank == 0:
        speedup = base_time / chunk4_time
        print(f"  Baseline (chunks=1): {base_time:.2f} ms")
        print(f"  Chunked (chunks=4):  {chunk4_time:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x ({(speedup-1)*100:+.1f}%)")
        print("\n" + "=" * 60)

    scheduler.clear_iteration()
    dist.destroy_process_group()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--chunked':
        test_chunked_backward()
    else:
        main()
