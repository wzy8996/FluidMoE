"""
Test script for fused MoE operators (requires multiple GPUs)
"""

import torch
import torch.distributed as dist
import os
import sys

sys.path.insert(0, '/home/zju/wzy/FluidMoE/fluid/ops')


def get_nccl_comm_ptr():
    """
    Get the NCCL communicator pointer from PyTorch distributed.
    This is a workaround since PyTorch doesn't expose this directly.
    """
    # For now, we'll use a simpler approach: test the non-fused version first
    # The fused version requires direct access to NCCL comm which is complex
    return None


def test_grouped_gemm():
    """Test grouped GEMM on single GPU"""
    import fluid_kernels

    print("Testing grouped_gemm...")

    num_experts = 4
    hidden_size = 256
    ffn_size = 512
    tokens_per_expert_list = [32, 48, 16, 64]
    total_tokens = sum(tokens_per_expert_list)

    # Forward test
    input_act = torch.randn(total_tokens, hidden_size, dtype=torch.float16, device='cuda')
    weights = torch.randn(num_experts, hidden_size, ffn_size, dtype=torch.float16, device='cuda')
    tokens_per_expert = torch.tensor(tokens_per_expert_list, dtype=torch.int32, device='cuda')

    output = fluid_kernels.grouped_gemm(input_act, weights, tokens_per_expert)

    # Reference computation
    output_ref = torch.zeros(total_tokens, ffn_size, dtype=torch.float16, device='cuda')
    offset = 0
    for i, num_tokens in enumerate(tokens_per_expert_list):
        if num_tokens > 0:
            expert_input = input_act[offset:offset+num_tokens]
            expert_weight = weights[i]
            output_ref[offset:offset+num_tokens] = expert_input @ expert_weight
        offset += num_tokens

    diff = (output - output_ref).abs().max().item()
    print(f"  Forward max diff: {diff}")
    assert diff < 1e-2, f"Forward diff too large: {diff}"

    # Backward dX test (trans_b=True)
    # dX = grad_out @ W^T
    # grad_out: [total_tokens, ffn_size]
    # W: [num_experts, hidden, ffn_size]  -> W^T: [num_experts, ffn_size, hidden]
    # Output: [total_tokens, hidden]
    grad_out = torch.randn(total_tokens, ffn_size, dtype=torch.float16, device='cuda')

    # For trans_b=True, we pass weights as [num_experts, hidden, ffn] and the kernel does B^T
    # So output = A @ B^T = [tokens, ffn] @ [ffn, hidden] = [tokens, hidden]
    grad_input = fluid_kernels.grouped_gemm(grad_out, weights, tokens_per_expert, trans_b=True)

    # Reference
    grad_input_ref = torch.zeros(total_tokens, hidden_size, dtype=torch.float16, device='cuda')
    offset = 0
    for i, num_tokens in enumerate(tokens_per_expert_list):
        if num_tokens > 0:
            expert_grad = grad_out[offset:offset+num_tokens]  # [tokens, ffn]
            expert_weight = weights[i]  # [hidden, ffn]
            # dX = grad @ W^T = [tokens, ffn] @ [ffn, hidden]
            grad_input_ref[offset:offset+num_tokens] = expert_grad @ expert_weight.t()
        offset += num_tokens

    diff = (grad_input - grad_input_ref).abs().max().item()
    print(f"  Backward dX max diff: {diff}")
    assert diff < 1e-2, f"Backward dX diff too large: {diff}"

    # Backward dW test
    # dW = input^T @ grad_intermediate
    # For FFN1: input is [tokens, hidden], grad is [tokens, ffn]
    # dW1: [hidden, ffn] = input^T @ grad_intermediate
    grad_intermediate = torch.randn(total_tokens, ffn_size, dtype=torch.float16, device='cuda')
    dW = fluid_kernels.grouped_gemm_dw(input_act, grad_intermediate, tokens_per_expert, hidden_size, ffn_size)

    # Reference
    dW_ref = torch.zeros(num_experts, hidden_size, ffn_size, dtype=torch.float16, device='cuda')
    offset = 0
    for i, num_tokens in enumerate(tokens_per_expert_list):
        if num_tokens > 0:
            expert_input = input_act[offset:offset+num_tokens]  # [tokens, hidden]
            expert_grad = grad_intermediate[offset:offset+num_tokens]  # [tokens, ffn]
            # dW = input^T @ grad = [hidden, tokens] @ [tokens, ffn] = [hidden, ffn]
            dW_ref[i] = expert_input.t() @ expert_grad
        offset += num_tokens

    diff = (dW - dW_ref).abs().max().item()
    print(f"  Backward dW max diff: {diff}")
    assert diff < 1e-2, f"Backward dW diff too large: {diff}"

    print("  All grouped_gemm tests passed!")


def test_barrier():
    """Test barrier operations"""
    import fluid_kernels

    print("Testing barrier operations...")

    barrier = fluid_kernels.create_barrier(4, torch.device('cuda:0'))
    print(f"  Created barrier: {barrier}")

    fluid_kernels.signal_barrier(barrier, 0, 1)
    torch.cuda.synchronize()
    print(f"  After signal(0, 1): {barrier}")

    fluid_kernels.reset_barrier(barrier)
    torch.cuda.synchronize()
    print(f"  After reset: {barrier}")

    print("  Barrier tests passed!")


def test_moe_layer_integration():
    """Test integration with MoE layer (single GPU, no communication)"""
    import fluid_kernels

    print("Testing MoE layer integration...")

    # Simulate MoE forward pass
    batch_size = 32
    seq_len = 64
    hidden_size = 256
    ffn_hidden_size = 512
    num_experts = 4
    top_k = 2

    # Input tokens
    tokens = torch.randn(batch_size * seq_len, hidden_size, dtype=torch.float16, device='cuda')

    # Expert weights
    w1 = torch.randn(num_experts, hidden_size, ffn_hidden_size, dtype=torch.float16, device='cuda')
    w2 = torch.randn(num_experts, ffn_hidden_size, hidden_size, dtype=torch.float16, device='cuda')

    # Simulate routing (random assignment)
    # In practice, this comes from router
    tokens_per_expert = torch.randint(100, 600, (num_experts,), dtype=torch.int32, device='cuda')
    total_expert_tokens = tokens_per_expert.sum().item()

    # Create dispatched tokens (simulated - would come from AllToAll)
    dispatched_tokens = torch.randn(total_expert_tokens, hidden_size, dtype=torch.float16, device='cuda')

    # FFN1: [total_tokens, hidden] @ [hidden, ffn_hidden] = [total_tokens, ffn_hidden]
    intermediate = fluid_kernels.grouped_gemm(dispatched_tokens, w1, tokens_per_expert)
    print(f"  FFN1 output shape: {intermediate.shape}")

    # Activation (GeLU simulated as ReLU for simplicity)
    intermediate = torch.relu(intermediate)

    # FFN2: [total_tokens, ffn_hidden] @ [ffn_hidden, hidden] = [total_tokens, hidden]
    output = fluid_kernels.grouped_gemm(intermediate, w2, tokens_per_expert)
    print(f"  FFN2 output shape: {output.shape}")

    print("  MoE layer integration test passed!")


def benchmark_grouped_gemm():
    """Benchmark grouped GEMM performance"""
    import fluid_kernels
    import time

    print("Benchmarking grouped_gemm...")

    # Typical MoE dimensions
    configs = [
        # (num_experts, hidden, ffn_hidden, tokens_per_expert)
        (8, 1024, 4096, [256]*8),
        (8, 2048, 8192, [512]*8),
        (16, 1024, 4096, [128]*16),
        (32, 1024, 4096, [64]*32),
    ]

    for num_experts, hidden_size, ffn_size, tpe_list in configs:
        total_tokens = sum(tpe_list)

        input_act = torch.randn(total_tokens, hidden_size, dtype=torch.float16, device='cuda')
        weights = torch.randn(num_experts, hidden_size, ffn_size, dtype=torch.float16, device='cuda')
        tokens_per_expert = torch.tensor(tpe_list, dtype=torch.int32, device='cuda')

        # Warmup
        for _ in range(10):
            _ = fluid_kernels.grouped_gemm(input_act, weights, tokens_per_expert)
        torch.cuda.synchronize()

        # Benchmark
        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            _ = fluid_kernels.grouped_gemm(input_act, weights, tokens_per_expert)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_time = elapsed / num_iters * 1000  # ms
        flops = 2 * total_tokens * hidden_size * ffn_size  # FLOPs per GEMM
        tflops = flops / (avg_time / 1000) / 1e12

        print(f"  E={num_experts:2d}, H={hidden_size:4d}, FFN={ffn_size:4d}, "
              f"T={total_tokens:5d}: {avg_time:.3f}ms, {tflops:.2f} TFLOPS")


def main():
    print("=" * 60)
    print("Fluid Kernels Test Suite")
    print("=" * 60)

    test_barrier()
    print()

    test_grouped_gemm()
    print()

    test_moe_layer_integration()
    print()

    benchmark_grouped_gemm()
    print()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
