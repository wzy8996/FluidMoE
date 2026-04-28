"""Minimal nsys-friendly driver for FluidMoE.

Usage:
  CUDA_DEVICE_MAX_CONNECTIONS=8 nsys profile \\
      --trace=cuda,nvtx,nccl --force-overwrite=true \\
      --sample=none --capture-range=cudaProfilerApi \\
      -o /tmp/fluidmoe_nsys \\
      torchrun --nproc_per_node=8 tests/nsys_fluidmoe.py \\
          --warmup 10 --iters 5

The script uses cudaProfilerApi to bracket ONLY the measured iters.
Analysis:
  nsys stats --report gputrace,nvtxsum /tmp/fluidmoe_nsys.nsys-rep
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
for p in (ROOT_DIR, TOOLS_DIR, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch._dynamo
torch._dynamo.config.disable = True
import torch.distributed as dist
import torch.cuda.profiler as cuda_profiler

from experiment_configs import get_block_benchmark_defaults
from model_configs import get_model_config


def parse_args():
    d = get_block_benchmark_defaults()
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="dbrx_base")
    p.add_argument("--dp-size", type=int, default=d["dp_size"])
    p.add_argument("--cp-size", type=int, default=d["cp_size"])
    p.add_argument("--ep-size", type=int, default=d["ep_size"])
    p.add_argument("--warmup", type=int, default=d["warmup"])
    p.add_argument("--iters", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_model_config(args.model)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    world_size = dist.get_world_size()

    dp_size, cp_size, ep_size = args.dp_size, args.cp_size, args.ep_size
    num_gpus = dp_size * cp_size
    assert ep_size == cp_size
    assert world_size == num_gpus

    hidden_size = int(cfg["hidden_size"])
    num_heads = int(cfg["num_heads"])
    num_kv_heads = int(cfg["num_kv_heads"])
    ffn_hidden = int(cfg["ffn_hidden"])
    num_experts = int(cfg["num_experts"])
    top_k = int(cfg["top_k"])
    num_layers = int(cfg["num_layers"])
    seq_len = int(cfg["seq_len"])
    batch_size = int(cfg["batch_size"])
    capacity_factor = float(cfg.get("capacity_factor", 1.0))
    seq_local = seq_len // cp_size

    if dp_size == 1 and num_gpus == world_size:
        all_group = cp_group = ep_group = dp_group = dist.group.WORLD
    else:
        all_group = dist.group.WORLD if num_gpus == world_size else dist.new_group(list(range(num_gpus)))
        cp_group = ep_group = None
        dp_group = None
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

    if rank == 0:
        print("nsys driver", flush=True)

    from fluid.core.scheduler import get_backward_scheduler
    from fluid.layer import TransformerModel

    bench_defaults = get_block_benchmark_defaults()
    gap_budgets = bench_defaults.get("gap_budgets", {})
    shared_ar_bw = float(bench_defaults.get("shared_ar_bw", 0.0))
    expert_ar_bw = float(bench_defaults.get("expert_ar_bw", 0.0))

    model = TransformerModel(
        num_layers=num_layers, hidden_size=hidden_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
        cp_group=cp_group, ep_group=ep_group,
        moe_combine_chunks=1, moe_dispatch_chunks=1,
        attn_proj_chunks=2, attn_qkv_chunks=1,
        capacity_factor=capacity_factor,
        dtype=torch.bfloat16, device=device,
    )
    model.prepare_chunk_status(torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device))

    scheduler = get_backward_scheduler()
    scheduler.enable()
    scheduler.configure_allreduce(
        enabled=True,
        shared_dp_group=all_group,
        expert_dp_group=dp_group if dp_size > 1 else None,
        gap_budgets=gap_budgets,
        shared_ar_bw=shared_ar_bw,
        expert_ar_bw=expert_ar_bw,
    )
    model.setup_ar_buffer()

    x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    x_grad = x.clone().detach().requires_grad_(True)

    def run_step():
        x_grad.grad = None
        for p in model.parameters():
            p.grad = None
            if hasattr(p, "_ar_buf_written"):
                p._ar_buf_written = False
        model(x_grad).sum().backward()
        scheduler.finish_batch()
        scheduler.clear_iteration()

    # Warmup outside profile
    for _ in range(args.warmup):
        run_step()
    torch.cuda.synchronize()
    dist.barrier()

    # Profile only these iters
    cuda_profiler.start()
    for i in range(args.iters):
        torch.cuda.nvtx.range_push(f"iter_{i}")
        run_step()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    cuda_profiler.stop()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
