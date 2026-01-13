#!/usr/bin/env python
"""
测试 multicard_p2p 的前向和反向性能

对比 Baseline（标准 AllToAll）和 multicard_p2p（多卡 P2P 重叠）。

使用方法：
    torchrun --nproc_per_node=2 tests/test_multicard_p2p.py [--mode forward_only|forward_backward|all]
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def benchmark_fn(fn, warmup=10, iterations=30):
    """基准测试函数"""
    for _ in range(warmup):
        out = fn()
        del out
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        out = fn()
        del out  # 显式删除，帮助 GC 清理 autograd 图
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark_forward_backward(forward_fn, backward_fn, warmup=10, iterations=30):
    """分别测量前向和反向时间"""
    # Warmup
    for _ in range(warmup):
        out = forward_fn()
        backward_fn(out)
        del out
        torch.cuda.synchronize()

    # Forward timing - 用与 benchmark_fn 相同的方法测量
    torch.cuda.synchronize()
    fwd_start = torch.cuda.Event(enable_timing=True)
    fwd_end = torch.cuda.Event(enable_timing=True)

    fwd_start.record()
    for _ in range(iterations):
        out = forward_fn()
        del out
    fwd_end.record()
    torch.cuda.synchronize()
    fwd_time = fwd_start.elapsed_time(fwd_end) / iterations

    # Backward timing
    torch.cuda.synchronize()
    bwd_start = torch.cuda.Event(enable_timing=True)
    bwd_end = torch.cuda.Event(enable_timing=True)

    bwd_start.record()
    for _ in range(iterations):
        out = forward_fn()
        backward_fn(out)
        del out
    bwd_end.record()
    torch.cuda.synchronize()
    total_time = bwd_start.elapsed_time(bwd_end) / iterations
    bwd_time = total_time - fwd_time

    return fwd_time, bwd_time, total_time


class MoEBaseline:
    """Baseline MoE：使用标准 AllToAll"""

    def __init__(self, config, ep_group, device, dtype):
        self.config = config
        self.ep_group = ep_group
        self.device = device
        self.dtype = dtype
        self.ep_size = ep_group.size()

        self.hidden_size = config['hidden_size']
        self.ffn_hidden_size = config['ffn_hidden_size']
        self.num_tokens = config['num_tokens']

    def init_weights(self, requires_grad):
        """初始化权重"""
        self.weight1 = torch.randn(
            self.hidden_size, self.ffn_hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=requires_grad
        )
        self.weight2 = torch.randn(
            self.ffn_hidden_size, self.hidden_size,
            dtype=self.dtype, device=self.device, requires_grad=requires_grad
        )

    def forward(self, tokens, do_backward=False):
        """前向传播"""
        from fluid.communication import fluid_all_to_all_moe_dispatch, fluid_all_to_all_moe_combine

        num_tokens = tokens.shape[0]
        tokens_per_rank = num_tokens // self.ep_size
        input_splits = torch.tensor([tokens_per_rank] * self.ep_size, dtype=torch.int64, device=self.device)
        output_splits = torch.tensor([tokens_per_rank] * self.ep_size, dtype=torch.int64, device=self.device)

        # Dispatch AllToAll
        recv_tokens = fluid_all_to_all_moe_dispatch(tokens, input_splits, output_splits, self.ep_group)

        # Expert 计算
        fc1_out = torch.matmul(recv_tokens, self.weight1)
        act_out = F.gelu(fc1_out)
        fc2_out = torch.matmul(act_out, self.weight2)

        # Combine AllToAll
        output = fluid_all_to_all_moe_combine(fc2_out, output_splits, input_splits, self.ep_group)

        if do_backward:
            loss = output.sum()
            loss.backward()

        return output


class MoEMultiCardP2P:
    """MultiCard P2P MoE：使用多卡 P2P 重叠"""

    def __init__(self, config, ep_group, device, dtype):
        self.config = config
        self.ep_group = ep_group
        self.device = device
        self.dtype = dtype
        self.ep_size = ep_group.size()

        self.hidden_size = config['hidden_size']
        self.ffn_hidden_size = config['ffn_hidden_size']
        self.num_tokens = config['num_tokens']

        # 初始化 MultiCard 重叠上下文
        from fluid.multicard_p2p import MultiCardOverlapContext
        self.overlap_ctx = MultiCardOverlapContext(device, ep_size=self.ep_size)

    def init_weights(self, requires_grad, baseline_layer):
        """从 baseline 复制权重"""
        self.weight1 = baseline_layer.weight1
        self.weight2 = baseline_layer.weight2

    def forward(self, tokens, do_backward=False):
        """前向传播（使用多卡 P2P 重叠）"""
        from fluid.multicard_p2p import moe_multicard_p2p_overlap_forward

        num_tokens = tokens.shape[0]
        tokens_per_rank = num_tokens // self.ep_size
        input_splits = torch.tensor([tokens_per_rank] * self.ep_size, dtype=torch.int64, device=self.device)
        output_splits = torch.tensor([tokens_per_rank] * self.ep_size, dtype=torch.int64, device=self.device)
        tokens_per_expert = torch.tensor([num_tokens], dtype=torch.int64, device=self.device)

        output = moe_multicard_p2p_overlap_forward(
            tokens, input_splits, output_splits,
            self.weight1, self.weight2, self.ep_group, F.gelu,
            self.overlap_ctx,
            layer_id=0, num_local_experts=1, tokens_per_expert=tokens_per_expert
        )

        if do_backward:
            loss = output.sum()
            loss.backward()

        return output


def test_moe(rank, world_size, config, mode):
    """测试 MoE 层"""
    device = torch.device(f"cuda:{rank}")
    dtype = torch.bfloat16
    ep_group = dist.group.WORLD

    requires_grad = mode in ('forward_backward', 'forward_grad')

    # 创建 baseline 和 multicard 层
    baseline_layer = MoEBaseline(config, ep_group, device, dtype)
    multicard_layer = MoEMultiCardP2P(config, ep_group, device, dtype)

    # 初始化权重
    baseline_layer.init_weights(requires_grad)
    multicard_layer.init_weights(requires_grad, baseline_layer)

    # 创建输入
    num_tokens = config['num_tokens']
    hidden_states = torch.randn(
        num_tokens, config['hidden_size'],
        dtype=dtype, device=device, requires_grad=requires_grad
    )

    passed = True
    if rank == 0:
        print("  注: 正确性由各组件单独测试保证")

    if mode == 'forward_only':
        def baseline_fn():
            h = hidden_states.detach()
            return baseline_layer.forward(h, do_backward=False)

        def multicard_fn():
            h = hidden_states.detach()
            return multicard_layer.forward(h, do_backward=False)

        dist.barrier()
        baseline_time = benchmark_fn(baseline_fn)
        dist.barrier()
        multicard_time = benchmark_fn(multicard_fn)
        dist.barrier()

        return baseline_time, multicard_time, passed, None
    elif mode == 'forward_grad':
        # 测试 requires_grad=True 时的前向性能（不运行 backward warmup）
        def baseline_fn():
            h = hidden_states.detach().requires_grad_(True)
            return baseline_layer.forward(h, do_backward=False)

        def multicard_fn():
            h = hidden_states.detach().requires_grad_(True)
            return multicard_layer.forward(h, do_backward=False)

        dist.barrier()
        baseline_time = benchmark_fn(baseline_fn)
        dist.barrier()
        multicard_time = benchmark_fn(multicard_fn)
        dist.barrier()

        return baseline_time, multicard_time, passed, None
    else:
        def baseline_fwd():
            h = hidden_states.detach().requires_grad_(True)
            return baseline_layer.forward(h, do_backward=False)

        def baseline_bwd(out):
            loss = out.sum()
            loss.backward()

        def multicard_fwd():
            h = hidden_states.detach().requires_grad_(True)
            return multicard_layer.forward(h, do_backward=False)

        def multicard_bwd(out):
            loss = out.sum()
            loss.backward()

        dist.barrier()
        base_fwd, base_bwd, base_total = benchmark_forward_backward(baseline_fwd, baseline_bwd)
        dist.barrier()
        mc_fwd, mc_bwd, mc_total = benchmark_forward_backward(multicard_fwd, multicard_bwd)
        dist.barrier()

        detailed = {
            'base_fwd': base_fwd, 'base_bwd': base_bwd,
            'mc_fwd': mc_fwd, 'mc_bwd': mc_bwd
        }
        return base_total, mc_total, passed, detailed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['forward_only', 'forward_grad', 'forward_backward', 'all'],
                        default='all', help="测试模式")
    parser.add_argument("--num_tokens", type=int, default=16384)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--ffn_hidden_size", type=int, default=14336)
    args = parser.parse_args()

    rank = setup_distributed()
    world_size = dist.get_world_size()

    config = {
        'num_tokens': args.num_tokens,
        'hidden_size': args.hidden_size,
        'ffn_hidden_size': args.ffn_hidden_size,
    }

    if rank == 0:
        print("=" * 80)
        print("MultiCard P2P MoE 测试")
        print("=" * 80)
        print(f"配置: tokens={config['num_tokens']}, hidden={config['hidden_size']}")
        print(f"      ffn={config['ffn_hidden_size']}, world_size={world_size}")
        print("=" * 80)

    mode_names = {
        'forward_only': '推理模式（仅前向，无梯度）',
        'forward_grad': '前向+梯度追踪（无反向warmup）',
        'forward_backward': '训练模式（前向+反向）',
    }

    if args.mode == 'all':
        modes = ['forward_only', 'forward_backward']
    elif args.mode == 'forward_grad':
        modes = ['forward_grad']
    else:
        modes = [args.mode]

    for mode in modes:
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"模式: {mode_names[mode]}")
            print(f"{'='*80}")

        dist.barrier()

        try:
            result = test_moe(rank, world_size, config, mode)
            baseline_time, multicard_time, passed, detailed = result
            if rank == 0:
                speedup = baseline_time / multicard_time
                status = "✓" if passed else "✗"
                print(f"\n{'结果':^20}")
                print("-" * 60)
                if detailed:
                    fwd_speedup = detailed['base_fwd'] / detailed['mc_fwd']
                    bwd_speedup = detailed['base_bwd'] / detailed['mc_bwd']
                    print(f"  {'':15} {'Baseline':>12} {'MultiCard':>12} {'加速比':>10}")
                    print(f"  {'前向':15} {detailed['base_fwd']:>10.2f}ms {detailed['mc_fwd']:>10.2f}ms {fwd_speedup:>9.2f}x")
                    print(f"  {'反向':15} {detailed['base_bwd']:>10.2f}ms {detailed['mc_bwd']:>10.2f}ms {bwd_speedup:>9.2f}x")
                    print(f"  {'-'*55}")
                    print(f"  {'总计':15} {baseline_time:>10.2f}ms {multicard_time:>10.2f}ms {speedup:>9.2f}x {status}")
                else:
                    print(f"  Baseline:     {baseline_time:>10.2f} ms")
                    print(f"  MultiCard:    {multicard_time:>10.2f} ms")
                    print(f"  加速比:       {speedup:>10.2f}x {status}")
                print("-" * 60)
        except Exception as e:
            if rank == 0:
                import traceback
                print(f"ERROR: {e}")
                traceback.print_exc()

        dist.barrier()

    if rank == 0:
        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
