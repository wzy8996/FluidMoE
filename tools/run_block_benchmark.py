"""Unified benchmark runner for megatron and fluidmoe baselines."""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, TESTS_DIR)

import torch
import torch.distributed as dist

from experiment_configs import get_block_benchmark_defaults
from model_configs import get_model_config, list_model_names


def parse_args():
    defaults = get_block_benchmark_defaults()
    parser = argparse.ArgumentParser(description="FluidMoE Block Benchmark")
    parser.add_argument("--model", type=str, default="mixtral_8x7b", help="模型名称 (from tools/model_configs.py)")
    parser.add_argument("--impl", type=str, default="fluidmoe",
                        choices=["megatron", "fluidmoe"],
                        help="选择运行的实现")
    parser.add_argument("--list-models", action="store_true", help="打印可用模型并退出")
    parser.add_argument("--dp-size", type=int, default=defaults["dp_size"])
    parser.add_argument("--cp-size", type=int, default=defaults["cp_size"])
    parser.add_argument("--ep-size", type=int, default=defaults["ep_size"])
    parser.add_argument("--moe-combine-chunks", type=int, default=defaults["moe_combine_chunks"])
    parser.add_argument("--moe-dispatch-chunks", type=int, default=defaults["moe_dispatch_chunks"])
    parser.add_argument("--attn-proj-chunks", type=int, default=defaults["attn_proj_chunks"])
    parser.add_argument("--attn-qkv-chunks", type=int, default=defaults["attn_qkv_chunks"])
    parser.add_argument("--warmup", type=int, default=defaults["warmup"])
    parser.add_argument("--iters", type=int, default=defaults["iters"])
    return parser.parse_args()


def p0(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def allreduce_grads(model, dp_size, all_group, dp_group):
    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        is_expert_param = ".experts." in name
        if is_expert_param:
            if dp_size > 1:
                dist.all_reduce(p.grad, group=dp_group)
            continue

        dist.all_reduce(p.grad, group=all_group)


def bench(run_fn, scheduler, ev_s, ev_e, warmup, iters):
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    scheduler.clear_iteration()
    ev_s.record()
    for _ in range(iters):
        run_fn()
    ev_e.record()
    torch.cuda.synchronize()
    scheduler.clear_iteration()
    return ev_s.elapsed_time(ev_e) / iters


def main():
    args = parse_args()
    if args.list_models:
        print("Available models:")
        for name in list_model_names():
            print(" ", name)
        raise SystemExit(0)

    model_cfg = get_model_config(args.model)

    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    world_size = dist.get_world_size()

    hidden_size = int(model_cfg.get("hidden_size", 4096))
    num_heads = int(model_cfg.get("num_heads", 32))
    num_kv_heads = int(model_cfg.get("num_kv_heads", 8))
    ffn_hidden = int(model_cfg.get("ffn_hidden", 14336))
    num_experts = int(model_cfg.get("num_experts", 8))
    top_k = int(model_cfg.get("top_k", 2))
    num_layers = int(model_cfg.get("num_layers", 4))
    seq_len = int(model_cfg.get("seq_len", 4096))
    batch_size = int(model_cfg.get("batch_size", 4))
    capacity_factor = float(model_cfg.get("capacity_factor", 1.0))

    dp_size = args.dp_size
    cp_size = args.cp_size
    ep_size = args.ep_size
    num_gpus = dp_size * cp_size

    assert ep_size == cp_size, f"当前仅支持 ep=cp, 但 ep={ep_size}, cp={cp_size}"
    assert world_size >= num_gpus, f"需要至少 {num_gpus} GPU, 但只有 {world_size}"
    assert seq_len % cp_size == 0, f"seq_len={seq_len} 不能被 cp_size={cp_size} 整除"
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

    p0(rank, "=" * 60)
    p0(rank, "FluidMoE Block Benchmark")
    p0(rank, f"  model={args.model}, impl={args.impl}")
    p0(rank, f"  hidden={hidden_size}, heads={num_heads}, ffn={ffn_hidden}")
    p0(rank, f"  experts={num_experts}, top_k={top_k}, layers={num_layers}")
    p0(rank, f"  seq={seq_len}, batch={batch_size}, GPUs={num_gpus}")
    p0(rank, f"  dp={dp_size}, cp={cp_size}, ep={ep_size}")
    if args.impl == "fluidmoe":
        p0(rank, f"  chunks: combine={args.moe_combine_chunks}, dispatch={args.moe_dispatch_chunks}, "
                 f"proj={args.attn_proj_chunks}, qkv={args.attn_qkv_chunks}")
    p0(rank, "=" * 60)

    from fluid.core.scheduler import get_backward_scheduler
    from fluid.layer import TransformerModel
    from megatron_baseline import MegatronBaselineTransformerModel

    x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    x_grad = x.clone().detach().requires_grad_(True)
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)

    if args.impl == "megatron":
        model = MegatronBaselineTransformerModel(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ffn_hidden_size=ffn_hidden,
            num_experts=num_experts,
            top_k=top_k,
            cp_group=cp_group,
            ep_group=ep_group,
            shared_dp_group=all_group,
            expert_dp_group=dp_group if dp_size > 1 else None,
            capacity_factor=capacity_factor,
            dtype=torch.bfloat16,
            device=device,
        )
        scheduler = get_backward_scheduler()

        def run_megatron_bwd():
            x_grad.grad = None
            for p in model.parameters():
                p.grad = None
            model(x_grad).sum().backward()
            allreduce_grads(model, dp_size, all_group, dp_group)

        p0(rank, "Megatron warmup...")
        for _ in range(args.warmup):
            with torch.no_grad():
                model(x)
        torch.cuda.synchronize()
        dist.barrier()

        ev_s.record()
        for _ in range(args.iters):
            with torch.no_grad():
                model(x)
        ev_e.record()
        torch.cuda.synchronize()
        fwd_ms = ev_s.elapsed_time(ev_e) / args.iters
        iter_ms = bench(run_megatron_bwd, scheduler, ev_s, ev_e, args.warmup, args.iters)
        p0(rank, f"Megatron: forward={fwd_ms:.2f}ms  iter={iter_ms:.2f}ms")
        p0(rank, f"RESULT impl=megatron forward_ms={fwd_ms:.6f} iter_ms={iter_ms:.6f}")
        dist.destroy_process_group()
        p0(rank, "Done!")
        return

    ar_trickle_sizes = get_block_benchmark_defaults()["ar_trickle_sizes"]
    fluidmoe_model = TransformerModel(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        ffn_hidden_size=ffn_hidden,
        num_experts=num_experts,
        top_k=top_k,
        cp_group=cp_group,
        ep_group=ep_group,
        attn_proj_chunks=args.attn_proj_chunks,
        attn_qkv_chunks=args.attn_qkv_chunks,
        moe_combine_chunks=args.moe_combine_chunks,
        moe_dispatch_chunks=args.moe_dispatch_chunks,
        ar_trickle_sizes=ar_trickle_sizes,
        capacity_factor=capacity_factor,
        dtype=torch.bfloat16,
        device=device,
    )
    scheduler = get_backward_scheduler()
    scheduler.enable()
    scheduler.configure_allreduce(
        enabled=True,
        shared_dp_group=all_group,
        expert_dp_group=dp_group if dp_size > 1 else None,
    )
    fluidmoe_model.setup_ar_buffer()

    def run_fluid_bwd():
        x_grad.grad = None
        for p in fluidmoe_model.parameters():
            p.grad = None
        fluidmoe_model(x_grad).sum().backward()
        scheduler.finish_batch()
        scheduler.clear_iteration()

    p0(rank, "FluidMoE warmup...")
    for _ in range(args.warmup):
        with torch.no_grad():
            fluidmoe_model(x)
    torch.cuda.synchronize()
    dist.barrier()

    ev_s.record()
    for _ in range(args.iters):
        with torch.no_grad():
            fluidmoe_model(x)
    ev_e.record()
    torch.cuda.synchronize()
    fluidmoe_fwd = ev_s.elapsed_time(ev_e) / args.iters

    scheduler.ar_enabled = False
    fluidmoe_sync = bench(run_fluid_bwd, scheduler, ev_s, ev_e, args.warmup, args.iters)
    scheduler.ar_enabled = True
    fluidmoe_interleaved = bench(run_fluid_bwd, scheduler, ev_s, ev_e, args.warmup, args.iters)

    p0(rank, f"FluidMoE: forward={fluidmoe_fwd:.2f}ms  sync_iter={fluidmoe_sync:.2f}ms  interleaved_iter={fluidmoe_interleaved:.2f}ms")
    p0(rank, f"RESULT impl=fluidmoe forward_ms={fluidmoe_fwd:.6f} sync_iter_ms={fluidmoe_sync:.6f} interleaved_iter_ms={fluidmoe_interleaved:.6f}")

    dist.destroy_process_group()
    p0(rank, "Done!")


if __name__ == "__main__":
    main()
