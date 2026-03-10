"""
Overlap ratio analyzer (single metric output).

输出: 仅打印一个 overlap_ratio 数值（rank0）

用法:
  torchrun --nproc_per_node=<N> tools/overlap_ratio_analyzer.py --model qwen_moe_a2_7b
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TOOLS_DIR)

from model_configs import get_model_config, list_model_names
from experiment_configs import get_block_benchmark_defaults
from fluid.layer import TransformerModel
from fluid.core.scheduler import get_backward_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Overlap ratio analyzer")
    parser.add_argument("--model", type=str, default="qwen_moe_a2_7b")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--mode", choices=["sync", "interleaved"], default="interleaved")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_models:
        for n in list_model_names():
            print(n)
        return

    cfg = get_model_config(args.model)
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    world_size = dist.get_world_size()

    hidden_size = int(cfg.get("hidden_size", 4096))
    num_heads = int(cfg.get("num_heads", 32))
    num_kv_heads = int(cfg.get("num_kv_heads", 8))
    ffn_hidden = int(cfg.get("ffn_hidden", 14336))
    num_experts = int(cfg.get("num_experts", 8))
    top_k = int(cfg.get("top_k", 2))
    num_layers = int(cfg.get("num_layers", 4))
    seq_len = int(cfg.get("seq_len", 4096))
    batch_size = int(cfg.get("batch_size", 4))
    capacity_factor = float(cfg.get("capacity_factor", 1.0))

    bench_defaults = get_block_benchmark_defaults()
    moe_combine_chunks = int(bench_defaults.get("moe_combine_chunks", 1))
    moe_dispatch_chunks = int(bench_defaults.get("moe_dispatch_chunks", 1))
    attn_proj_chunks = int(bench_defaults.get("attn_proj_chunks", 1))
    attn_qkv_chunks = int(bench_defaults.get("attn_qkv_chunks", 1))
    ar_trickle_sizes = bench_defaults.get("ar_trickle_sizes", {})
    dp_size = int(bench_defaults.get("dp_size", 1))
    cp_size = int(bench_defaults.get("cp_size", 2))
    ep_size = int(bench_defaults.get("ep_size", 2))

    assert ep_size == cp_size
    num_gpus = dp_size * cp_size
    assert world_size >= num_gpus
    assert seq_len % cp_size == 0
    seq_local = seq_len // cp_size

    if rank >= num_gpus:
        dist.barrier()
        dist.destroy_process_group()
        return

    if dp_size == 1 and num_gpus == world_size:
        all_group = cp_group = ep_group = dp_group = dist.group.WORLD
    else:
        if num_gpus == world_size:
            all_group = dist.group.WORLD
        else:
            all_group = dist.new_group(list(range(num_gpus)))
        for i in range(dp_size):
            ranks = list(range(i * cp_size, (i + 1) * cp_size))
            g = dist.new_group(ranks)
            if rank in ranks:
                cp_group = ep_group = g
        for i in range(cp_size):
            ranks = list(range(i, num_gpus, cp_size))
            g = dist.new_group(ranks)
            if rank in ranks:
                dp_group = g

    model = TransformerModel(
        num_layers=num_layers, hidden_size=hidden_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
        cp_group=cp_group, ep_group=ep_group,
        moe_combine_chunks=moe_combine_chunks, moe_dispatch_chunks=moe_dispatch_chunks,
        attn_proj_chunks=attn_proj_chunks, attn_qkv_chunks=attn_qkv_chunks,
        ar_trickle_sizes=ar_trickle_sizes,
        capacity_factor=capacity_factor,
        dtype=torch.bfloat16, device=device,
    )

    scheduler = get_backward_scheduler()
    scheduler.enable()
    scheduler.configure_allreduce(
        enabled=True,
        shared_dp_group=all_group,
        expert_dp_group=dp_group if dp_size > 1 else None,
    )
    model.setup_ar_buffer()
    scheduler.ar_enabled = (args.mode == "interleaved")

    x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    x_grad = x.clone().detach().requires_grad_(True)

    def run_step():
        x_grad.grad = None
        for p in model.parameters():
            p.grad = None
        model(x_grad).sum().backward()
        scheduler.finish_batch()
        scheduler.clear_iteration()

    for _ in range(args.warmup):
        run_step()
    torch.cuda.synchronize()
    scheduler.clear_iteration()
    dist.barrier()

    scheduler.reset_comm_metrics()
    scheduler.set_comm_metrics_enabled(True)
    for _ in range(args.iters):
        run_step()
    torch.cuda.synchronize()
    local_comm = scheduler.get_comm_metrics()
    scheduler.set_comm_metrics_enabled(False)
    scheduler.clear_iteration()

    comm_tensor = torch.tensor([
        local_comm["a2a_total_ms"] / args.iters,
        local_comm["a2a_visible_ms"] / args.iters,
    ], dtype=torch.float32, device=device)
    dist.all_reduce(comm_tensor, op=dist.ReduceOp.MAX, group=all_group)
    a2a_total = float(comm_tensor[0].item())
    a2a_visible = float(comm_tensor[1].item())
    overlap_ratio = 0.0 if a2a_total <= 1e-9 else max(0.0, min(1.0, 1.0 - a2a_visible / a2a_total))

    if rank == 0:
        print(f"{overlap_ratio:.6f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
