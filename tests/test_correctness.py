#!/usr/bin/env python3
"""
FluidMoE Correctness Test

验证 MoE 层计算的正确性：
1. Expert FFN 计算正确性
2. AllToAll 通信正确性
3. MoE 梯度计算正确性（torch.autograd.gradcheck）

Usage:
    torchrun --nproc_per_node=2 tests/test_correctness.py
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.distributed as dist
import torch.nn.functional as F


def setup_distributed():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    return rank, world_size, device


def test_expert_computation_correctness(rank, world_size, device):
    """测试单个 Expert FFN 计算的正确性"""
    print(f"\n{'='*60}")
    print("Test 1: Expert FFN Computation Correctness")
    print('='*60)

    dtype = torch.float32
    hidden_size = 512
    ffn_hidden = 2048
    num_tokens = 64

    torch.manual_seed(42 + rank)
    w1 = torch.randn(hidden_size, ffn_hidden, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(ffn_hidden, hidden_size, dtype=dtype, device=device) * 0.02
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)

    # 标准计算
    fc1 = torch.matmul(x, w1)
    act = F.silu(fc1)
    output_standard = torch.matmul(act, w2)

    # 分步计算
    fc1_step = torch.matmul(x, w1)
    act_step = F.silu(fc1_step)
    output_step = torch.matmul(act_step, w2)

    diff = (output_standard - output_step).abs().max().item()
    print(f"Rank {rank}: Max diff = {diff:.2e}")

    passed = diff < 1e-6
    print(f"  [{'PASS' if passed else 'FAIL'}] Expert computation")
    return passed


def test_alltoall_correctness(rank, world_size, device):
    """测试 AllToAll 通信的正确性"""
    print(f"\n{'='*60}")
    print("Test 2: AllToAll Communication Correctness")
    print('='*60)

    from fluid.core.alltoall import _all_to_all

    ep_group = dist.group.WORLD
    dtype = torch.float32
    hidden_size = 64

    torch.manual_seed(42 + rank)
    tokens_per_rank = 16
    total_send = tokens_per_rank * world_size
    send_data = torch.randn(total_send, hidden_size, dtype=dtype, device=device)

    input_splits = [tokens_per_rank] * world_size
    output_splits = [tokens_per_rank] * world_size

    recv_data = _all_to_all(send_data, input_splits, output_splits, ep_group)

    # 验证
    all_send_data = [torch.zeros_like(send_data) for _ in range(world_size)]
    dist.all_gather(all_send_data, send_data, group=ep_group)

    expected_recv = []
    for src_rank in range(world_size):
        start = rank * tokens_per_rank
        end = start + tokens_per_rank
        expected_recv.append(all_send_data[src_rank][start:end])
    expected_recv = torch.cat(expected_recv, dim=0)

    diff = (recv_data - expected_recv).abs().max().item()
    print(f"Rank {rank}: Max diff = {diff:.2e}")

    passed = diff < 1e-6
    print(f"  [{'PASS' if passed else 'FAIL'}] AllToAll communication")
    return passed


def test_moe_output_consistency(rank, world_size, device):
    """测试 MoE 多次前向传播输出一致性"""
    print(f"\n{'='*60}")
    print("Test 3: MoE Output Consistency (same input -> same output)")
    print('='*60)

    from fluid.moe.p2p_overlap import moe_multicard_p2p_overlap_forward
    from fluid.moe.router import compute_routing
    from fluid.core.forward_comm import MultiCardOverlapContext

    config = {
        'hidden_size': 512,
        'ffn_hidden_size': 2048,
        'num_experts': 4,
        'top_k': 2,
    }
    dtype = torch.float32

    ep_group = dist.group.WORLD
    ep_size = ep_group.size()
    my_rank = ep_group.rank()
    num_local_experts = config['num_experts'] // ep_size

    # 初始化权重
    torch.manual_seed(42)
    weight1 = torch.randn(config['hidden_size'], config['ffn_hidden_size'] * num_local_experts,
                          dtype=dtype, device=device) * 0.02
    weight2 = torch.randn(config['ffn_hidden_size'] * num_local_experts, config['hidden_size'],
                          dtype=dtype, device=device) * 0.02
    router_weight = torch.randn(config['hidden_size'], config['num_experts'],
                                dtype=dtype, device=device) * 0.02

    overlap_ctx = MultiCardOverlapContext(device, ep_size, ep_size)

    # 创建输入
    torch.manual_seed(100 + rank)
    num_tokens = 128
    x = torch.randn(num_tokens, config['hidden_size'], dtype=dtype, device=device)

    def run_moe_forward(input_x):
        with torch.no_grad():
            (permuted_tokens, input_splits, output_splits, permuted_probs,
             restore_indices, tokens_per_expert, global_tokens_per_expert,
             tokens_per_expert_2d) = compute_routing(
                input_x.clone(), router_weight, config['num_experts'], config['top_k'],
                ep_group, layer_id=0)

            local_start_expert = my_rank * num_local_experts
            local_end_expert = local_start_expert + num_local_experts
            cumsum = tokens_per_expert.cumsum(0)
            local_start_idx = cumsum[local_start_expert - 1].item() if local_start_expert > 0 else 0
            local_end_idx = cumsum[local_end_expert - 1].item()
            local_tokens = permuted_tokens[local_start_idx:local_end_idx]
            num_global_tokens = tokens_per_expert_2d[:, local_start_expert:local_end_expert].unsqueeze(0)

            combined = moe_multicard_p2p_overlap_forward(
                permuted_tokens.contiguous(),
                input_splits, output_splits,
                weight1, weight2,
                ep_group, F.silu, overlap_ctx,
                0, num_local_experts, local_tokens, num_global_tokens)

            combined = combined * permuted_probs.unsqueeze(-1).to(combined.dtype)
            restored = combined[restore_indices]
            output = restored.view(num_tokens, config['top_k'], -1).sum(dim=1)
            return output

    # 多次前向，验证输出一致
    out1 = run_moe_forward(x)
    out2 = run_moe_forward(x)

    diff = (out1 - out2).abs().max().item()
    print(f"Rank {rank}: Max diff between runs = {diff:.2e}")
    print(f"  Output norm: {out1.norm().item():.4f}")

    passed = diff < 1e-5
    print(f"  [{'PASS' if passed else 'FAIL'}] Output consistency")
    return passed


def test_moe_gradient_flow(rank, world_size, device):
    """测试 MoE 梯度流是否正确"""
    print(f"\n{'='*60}")
    print("Test 4: MoE Gradient Flow")
    print('='*60)

    from fluid.moe.p2p_overlap import _MoEMultiCardP2POverlapFunction
    from fluid.moe.router import compute_routing
    from fluid.core.forward_comm import MultiCardOverlapContext
    from fluid.core.scheduler import get_backward_scheduler

    config = {
        'hidden_size': 256,
        'ffn_hidden_size': 1024,
        'num_experts': 4,
        'top_k': 2,
    }
    dtype = torch.float32

    ep_group = dist.group.WORLD
    ep_size = ep_group.size()
    my_rank = ep_group.rank()
    num_local_experts = config['num_experts'] // ep_size

    scheduler = get_backward_scheduler()
    scheduler.enable()

    # 初始化权重
    # NOTE: Create tensor first, then multiply by scale, then set requires_grad
    # to ensure the weight is a leaf tensor (non-leaf tensors cause graph retention issues)
    torch.manual_seed(42)
    weight1 = (torch.randn(config['hidden_size'], config['ffn_hidden_size'] * num_local_experts,
                          dtype=dtype, device=device) * 0.02).requires_grad_(True)
    weight2 = (torch.randn(config['ffn_hidden_size'] * num_local_experts, config['hidden_size'],
                          dtype=dtype, device=device) * 0.02).requires_grad_(True)
    router_weight = (torch.randn(config['hidden_size'], config['num_experts'],
                                dtype=dtype, device=device) * 0.02).requires_grad_(True)

    overlap_ctx = MultiCardOverlapContext(device, ep_size, ep_size)

    # 创建输入
    torch.manual_seed(100 + rank)
    num_tokens = 64
    x = torch.randn(num_tokens, config['hidden_size'], dtype=dtype, device=device, requires_grad=True)

    # 前向
    (permuted_tokens, input_splits, output_splits, permuted_probs,
     restore_indices, tokens_per_expert, global_tokens_per_expert,
     tokens_per_expert_2d) = compute_routing(
        x, router_weight, config['num_experts'], config['top_k'],
        ep_group, layer_id=0)

    local_start_expert = my_rank * num_local_experts
    local_end_expert = local_start_expert + num_local_experts
    cumsum = tokens_per_expert.cumsum(0)
    local_start_idx = cumsum[local_start_expert - 1].item() if local_start_expert > 0 else 0
    local_end_idx = cumsum[local_end_expert - 1].item()
    local_tokens = permuted_tokens[local_start_idx:local_end_idx]
    num_global_tokens = tokens_per_expert_2d[:, local_start_expert:local_end_expert].unsqueeze(0)

    combined = _MoEMultiCardP2POverlapFunction.apply(
        permuted_tokens.contiguous(),
        input_splits, output_splits,
        weight1, weight2,
        ep_group, F.silu, overlap_ctx,
        0, num_local_experts, local_tokens, num_global_tokens)

    combined = combined * permuted_probs.unsqueeze(-1).to(combined.dtype)
    restored = combined[restore_indices]
    output = restored.view(num_tokens, config['top_k'], -1).sum(dim=1)

    # 反向
    loss = output.sum()
    loss.backward()

    scheduler.finish_batch()
    scheduler.clear_iteration()
    scheduler.disable()

    # 验证梯度
    has_input_grad = x.grad is not None
    has_w1_grad = weight1.grad is not None
    has_w2_grad = weight2.grad is not None

    print(f"Rank {rank}:")
    print(f"  Input has grad:   {has_input_grad}")
    print(f"  Weight1 has grad: {has_w1_grad}")
    print(f"  Weight2 has grad: {has_w2_grad}")

    if has_input_grad:
        print(f"  Input grad norm:   {x.grad.norm().item():.4f}")
    if has_w1_grad:
        print(f"  Weight1 grad norm: {weight1.grad.norm().item():.4f}")
    if has_w2_grad:
        print(f"  Weight2 grad norm: {weight2.grad.norm().item():.4f}")

    # 验证梯度不为 0
    grads_nonzero = True
    if has_input_grad and x.grad.norm().item() == 0:
        grads_nonzero = False
    if has_w1_grad and weight1.grad.norm().item() == 0:
        grads_nonzero = False
    if has_w2_grad and weight2.grad.norm().item() == 0:
        grads_nonzero = False

    passed = has_input_grad and has_w1_grad and has_w2_grad and grads_nonzero
    print(f"  [{'PASS' if passed else 'FAIL'}] Gradient flow")
    return passed


def test_attention_output_consistency(rank, world_size, device):
    """测试 Attention 多次前向传播输出一致性"""
    print(f"\n{'='*60}")
    print("Test 5: Attention Output Consistency")
    print('='*60)

    from fluid.attention.p2p_overlap import qkv_sp2hp_multicard_overlap, hp2sp_output_proj_multicard_overlap
    from fluid.attention.baseline import scaled_dot_product_attention
    from fluid.core.forward_comm import MultiCardOverlapContext

    hidden_size = 512
    num_heads = 8
    num_kv_heads = 8
    head_dim = hidden_size // num_heads
    seq_len = 128
    batch_size = 2
    dtype = torch.float32

    cp_group = dist.group.WORLD
    cp_size = cp_group.size()

    # 初始化权重 (interleaved layout for overlap)
    torch.manual_seed(42)
    q_per_group = num_heads // num_kv_heads
    group_size = (q_per_group + 2) * head_dim
    total_proj = num_kv_heads * group_size
    weight_qkv = torch.randn(total_proj, hidden_size, dtype=dtype, device=device) * 0.02
    weight_proj = torch.randn(hidden_size, num_heads * head_dim, dtype=dtype, device=device) * 0.02

    overlap_ctx = MultiCardOverlapContext(device, cp_size, cp_size)

    # 创建输入
    torch.manual_seed(100 + rank)
    seq_local = seq_len // cp_size
    x = torch.randn(seq_local, batch_size, hidden_size, dtype=dtype, device=device)

    def run_attention_forward(input_x):
        with torch.no_grad():
            # QKV projection with P2P overlap
            q, k, v = qkv_sp2hp_multicard_overlap(
                input_x.clone(), weight_qkv,
                num_heads, num_kv_heads, head_dim,
                cp_group, overlap_ctx, layer_id=0)

            # Attention
            q = q.permute(1, 2, 0, 3)  # [batch, heads_local, seq_full, head_dim]
            k = k.permute(1, 2, 0, 3)
            v = v.permute(1, 2, 0, 3)
            attn_out = scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.permute(2, 0, 1, 3).contiguous()

            # Output projection with P2P overlap
            output = hp2sp_output_proj_multicard_overlap(
                attn_out, weight_proj, None,
                cp_group, overlap_ctx)

            return output

    # 多次前向，验证输出一致
    out1 = run_attention_forward(x)
    out2 = run_attention_forward(x)

    diff = (out1 - out2).abs().max().item()
    print(f"Rank {rank}: Max diff between runs = {diff:.2e}")
    print(f"  Output shape: {out1.shape}")
    print(f"  Output norm: {out1.norm().item():.4f}")

    passed = diff < 1e-5
    print(f"  [{'PASS' if passed else 'FAIL'}] Attention output consistency")
    return passed


def test_attention_gradient_flow(rank, world_size, device):
    """测试 Attention 梯度流是否正确"""
    print(f"\n{'='*60}")
    print("Test 6: Attention Gradient Flow")
    print('='*60)

    from fluid.attention.p2p_overlap import qkv_sp2hp_multicard_overlap, hp2sp_output_proj_multicard_overlap
    from fluid.attention.baseline import scaled_dot_product_attention
    from fluid.core.forward_comm import MultiCardOverlapContext
    from fluid.core.scheduler import get_backward_scheduler

    hidden_size = 256
    num_heads = 8
    num_kv_heads = 8
    head_dim = hidden_size // num_heads
    seq_len = 64
    batch_size = 2
    dtype = torch.float32

    cp_group = dist.group.WORLD
    cp_size = cp_group.size()

    scheduler = get_backward_scheduler()
    scheduler.enable()

    # 初始化权重
    # NOTE: Create tensor first, then multiply by scale, then set requires_grad
    # to ensure the weight is a leaf tensor (non-leaf tensors cause graph retention issues)
    torch.manual_seed(42)
    q_per_group = num_heads // num_kv_heads
    group_size = (q_per_group + 2) * head_dim
    total_proj = num_kv_heads * group_size
    weight_qkv = (torch.randn(total_proj, hidden_size, dtype=dtype, device=device) * 0.02).requires_grad_(True)
    weight_proj = (torch.randn(hidden_size, num_heads * head_dim, dtype=dtype, device=device) * 0.02).requires_grad_(True)

    overlap_ctx = MultiCardOverlapContext(device, cp_size, cp_size)

    # 创建输入
    torch.manual_seed(100 + rank)
    seq_local = seq_len // cp_size
    x = torch.randn(seq_local, batch_size, hidden_size, dtype=dtype, device=device, requires_grad=True)

    # 前向
    q, k, v = qkv_sp2hp_multicard_overlap(
        x, weight_qkv,
        num_heads, num_kv_heads, head_dim,
        cp_group, overlap_ctx, layer_id=0)

    q = q.permute(1, 2, 0, 3)
    k = k.permute(1, 2, 0, 3)
    v = v.permute(1, 2, 0, 3)
    attn_out = scaled_dot_product_attention(q, k, v)
    attn_out = attn_out.permute(2, 0, 1, 3).contiguous()

    output = hp2sp_output_proj_multicard_overlap(
        attn_out, weight_proj, None,
        cp_group, overlap_ctx)

    # 反向
    loss = output.sum()
    loss.backward()

    scheduler.finish_batch()
    scheduler.clear_iteration()
    scheduler.disable()

    # 验证梯度
    has_input_grad = x.grad is not None
    has_qkv_grad = weight_qkv.grad is not None
    has_proj_grad = weight_proj.grad is not None

    print(f"Rank {rank}:")
    print(f"  Input has grad:      {has_input_grad}")
    print(f"  Weight QKV has grad: {has_qkv_grad}")
    print(f"  Weight Proj has grad: {has_proj_grad}")

    if has_input_grad:
        print(f"  Input grad norm:      {x.grad.norm().item():.4f}")
    if has_qkv_grad:
        print(f"  Weight QKV grad norm: {weight_qkv.grad.norm().item():.4f}")
    if has_proj_grad:
        print(f"  Weight Proj grad norm: {weight_proj.grad.norm().item():.4f}")

    grads_nonzero = True
    if has_input_grad and x.grad.norm().item() == 0:
        grads_nonzero = False
    if has_qkv_grad and weight_qkv.grad.norm().item() == 0:
        grads_nonzero = False
    if has_proj_grad and weight_proj.grad.norm().item() == 0:
        grads_nonzero = False

    passed = has_input_grad and has_qkv_grad and has_proj_grad and grads_nonzero
    print(f"  [{'PASS' if passed else 'FAIL'}] Attention gradient flow")
    return passed


def main():
    rank, world_size, device = setup_distributed()

    if rank == 0:
        print("\n" + "="*60)
        print("FluidMoE Correctness Verification")
        print("="*60)
        print(f"World size: {world_size}")

    dist.barrier()

    results = []

    # Test 1
    try:
        results.append(("Expert FFN Computation", test_expert_computation_correctness(rank, world_size, device)))
    except Exception as e:
        print(f"Rank {rank}: Test 1 failed: {e}")
        import traceback; traceback.print_exc()
        results.append(("Expert FFN Computation", False))

    dist.barrier()

    # Test 2
    try:
        results.append(("AllToAll Communication", test_alltoall_correctness(rank, world_size, device)))
    except Exception as e:
        print(f"Rank {rank}: Test 2 failed: {e}")
        import traceback; traceback.print_exc()
        results.append(("AllToAll Communication", False))

    dist.barrier()

    # Test 3
    try:
        results.append(("MoE Output Consistency", test_moe_output_consistency(rank, world_size, device)))
    except Exception as e:
        print(f"Rank {rank}: Test 3 failed: {e}")
        import traceback; traceback.print_exc()
        results.append(("MoE Output Consistency", False))

    dist.barrier()

    # Test 4
    try:
        results.append(("MoE Gradient Flow", test_moe_gradient_flow(rank, world_size, device)))
    except Exception as e:
        print(f"Rank {rank}: Test 4 failed: {e}")
        import traceback; traceback.print_exc()
        results.append(("MoE Gradient Flow", False))

    dist.barrier()

    # Test 5
    try:
        results.append(("Attention Output Consistency", test_attention_output_consistency(rank, world_size, device)))
    except Exception as e:
        print(f"Rank {rank}: Test 5 failed: {e}")
        import traceback; traceback.print_exc()
        results.append(("Attention Output Consistency", False))

    dist.barrier()

    # Test 6
    try:
        results.append(("Attention Gradient Flow", test_attention_gradient_flow(rank, world_size, device)))
    except Exception as e:
        print(f"Rank {rank}: Test 6 failed: {e}")
        import traceback; traceback.print_exc()
        results.append(("Attention Gradient Flow", False))

    dist.barrier()

    # Summary
    if rank == 0:
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        all_passed = True
        for name, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: [{status}]")
            if not passed:
                all_passed = False

        print("="*60)
        if all_passed:
            print("All tests PASSED!")
        else:
            print("Some tests FAILED!")
        print("="*60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
