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
megatron_path = os.environ.get('MEGATRON_PATH', '/home/zju/wzy/Megatron-LM')
if megatron_path not in sys.path:
    sys.path.insert(0, megatron_path)

import torch
import torch._dynamo
torch._dynamo.config.disable = True  # fair comparison: all frameworks run in eager mode
import torch.distributed as dist

from experiment_configs import get_block_benchmark_defaults
from model_configs import get_model_config, list_model_names


class _NullScheduler:
    def clear_iteration(self):
        return None


def parse_args():
    defaults = get_block_benchmark_defaults()
    parser = argparse.ArgumentParser(description="FluidMoE Block Benchmark")
    parser.add_argument("--model", type=str, default="mixtral_8x7b", help="模型名称 (from tools/model_configs.py)")
    parser.add_argument("--impl", type=str, default="fluidmoe",
                        choices=["megatron", "megatron-overlap", "megatron-overlap-dw", "fluidmoe", "deepspeed"],
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


_ar_bufs = {}  # cache: model_id -> result dict


def _build_ar_buffers(model):
    """按 dtype 分组构建 flat buffer，所有参数的 grad 都指向 buffer 视图，零额外分配。"""
    mid = id(model)
    if mid in _ar_bufs:
        return _ar_bufs[mid]

    device = next(model.parameters()).device

    # 按 is_expert 分组，buffer 统一用 fp32（matching Megatron's
    # accumulate_allreduce_grads_in_fp32 convention）
    groups = {}  # is_expert -> [params]
    for name, p in model.named_parameters():
        is_expert = "moe_w1" in name or "moe_w2" in name or ".experts." in name
        groups.setdefault(is_expert, []).append(p)

    # 为每组构建 fp32 flat buffer
    flat_bufs = {}  # (is_expert, dtype) -> (flat_tensor, [params])
    for is_expert, params in groups.items():
        numel = sum(p.numel() for p in params)
        flat = torch.zeros(numel, dtype=torch.float32, device=device)
        flat_bufs[(is_expert, torch.float32)] = (flat, params)

    _ar_bufs[mid] = flat_bufs
    return flat_bufs


def zero_grad(model):
    """Clear all param.grad."""
    for p in model.parameters():
        p.grad = None


def copy_grads_to_flat_and_allreduce(model, dp_size, all_group, dp_group):
    """Copy bf16 grads into fp32 flat buffer, pre-scale by 1/dp_cp_size, then allreduce.

    Matches Megatron DDP convention: accumulate_allreduce_grads_in_fp32 + pre-scale + SUM.
    """
    flat_bufs = _build_ar_buffers(model)
    dp_cp_size = dist.get_world_size(all_group)
    pre_scale = 1.0 / dp_cp_size if dp_cp_size > 1 else 1.0

    for (is_expert, _dt), (flat, params) in flat_bufs.items():
        # Copy bf16 grads → fp32 flat buffer
        offset = 0
        for p in params:
            n = p.numel()
            if p.grad is not None:
                flat[offset:offset + n].copy_(p.grad.view(-1))
            else:
                flat[offset:offset + n].zero_()
            offset += n

        # Pre-scale (Megatron convention)
        if pre_scale != 1.0:
            flat.mul_(pre_scale)

        # Allreduce
        if is_expert:
            if dp_size > 1:
                dist.all_reduce(flat, group=dp_group)
        else:
            dist.all_reduce(flat, group=all_group)

        # Write back fp32 grads as param.grad (bf16 for optimizer compatibility)
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = flat[offset:offset + n].view(p.shape).to(p.dtype)
            offset += n


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
    assert world_size == num_gpus, (
        f"world_size={world_size} != dp*cp={num_gpus}, "
        f"多余的 {world_size - num_gpus} 个进程会直接退出。"
        f"请设置 NPROC_PER_NODE={num_gpus} 或调整 dp/cp"
    )
    assert seq_len % cp_size == 0, f"seq_len={seq_len} 不能被 cp_size={cp_size} 整除"
    assert num_experts % ep_size == 0, (
        f"num_experts={num_experts} 不能被 ep_size={ep_size} 整除"
    )
    assert num_heads % cp_size == 0, (
        f"num_heads={num_heads} 不能被 cp_size={cp_size} 整除"
    )
    assert num_kv_heads % cp_size == 0, (
        f"num_kv_heads={num_kv_heads} 不能被 cp_size={cp_size} 整除"
    )
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
        p0(rank, f"  chunks: R1={args.moe_combine_chunks}, R2={args.moe_dispatch_chunks}, R3={args.attn_proj_chunks}, R4={args.attn_qkv_chunks}")
    p0(rank, "=" * 60)

    x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    x_grad = x.clone().detach().requires_grad_(True)
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)

    if args.impl in ("megatron", "megatron-overlap", "megatron-overlap-dw", "deepspeed"):
        if args.impl in ("megatron", "megatron-overlap", "megatron-overlap-dw"):
            from megatron_baseline import MegatronBaselineTransformerModel

            use_overlap = args.impl in ("megatron-overlap", "megatron-overlap-dw")
            use_delay_wgrad = args.impl == "megatron-overlap-dw"
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
                overlap_a2a=use_overlap,
                delay_wgrad=use_delay_wgrad,
                capacity_factor=capacity_factor,
                dtype=torch.bfloat16,
                device=device,
            )
        elif args.impl == "deepspeed":
            from deepspeed_ulysses_baseline import DeepSpeedBlockBaselineTransformerModel

            model = DeepSpeedBlockBaselineTransformerModel(
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ffn_hidden_size=ffn_hidden,
                num_experts=num_experts,
                top_k=top_k,
                cp_group=cp_group,
                ep_group=ep_group,
                capacity_factor=capacity_factor,
                dtype=torch.bfloat16,
                device=device,
            )
        impl_name = args.impl.capitalize()
        scheduler = _NullScheduler()

        # 预分配 flat buffer，将 param.grad 设为其视图，避免 backward 额外分配梯度
        _build_ar_buffers(model)

        def run_baseline_bwd():
            x_grad.grad = None
            zero_grad(model)
            model(x_grad).sum().backward()
            copy_grads_to_flat_and_allreduce(model, dp_size, all_group, dp_group)

        p0(rank, f"{impl_name} warmup...")
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
        iter_ms = bench(run_baseline_bwd, scheduler, ev_s, ev_e, args.warmup, args.iters)
        tokens_per_iter = seq_len * batch_size * dp_size
        tokps = tokens_per_iter / (iter_ms / 1000.0)
        p0(rank, f"{impl_name}: forward={fwd_ms:.2f}ms  iter={iter_ms:.2f}ms  throughput={tokps:.0f} tok/s")
        p0(rank, f"RESULT impl={args.impl} forward_ms={fwd_ms:.6f} iter_ms={iter_ms:.6f} tokens_per_sec={tokps:.6f}")
        dist.destroy_process_group()
        p0(rank, "Done!")
        return

    from fluid.core.scheduler import get_backward_scheduler
    from fluid.layer import TransformerModel

    _bench_defaults = get_block_benchmark_defaults()
    gap_budgets = _bench_defaults.get("gap_budgets", {})
    shared_ar_bw = float(_bench_defaults.get("shared_ar_bw", 0.0))
    expert_ar_bw = float(_bench_defaults.get("expert_ar_bw", 0.0))
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
        moe_combine_chunks=args.moe_combine_chunks,
        moe_dispatch_chunks=args.moe_dispatch_chunks,
        attn_proj_chunks=args.attn_proj_chunks,
        attn_qkv_chunks=args.attn_qkv_chunks,
        capacity_factor=capacity_factor,
        dtype=torch.bfloat16,
        device=device,
    )
    chunk_messages = fluidmoe_model.prepare_chunk_status(x)
    if chunk_messages:
        p0(rank, "[FluidMoE][chunk-check] " + " | ".join(chunk_messages))
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
    fluidmoe_model.setup_ar_buffer()

    def run_fluid_bwd():
        x_grad.grad = None
        for p in fluidmoe_model.parameters():
            p.grad = None
            if hasattr(p, '_ar_buf_written'):
                p._ar_buf_written = False
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

    tokens_per_iter = seq_len * batch_size * dp_size
    sync_tokps = tokens_per_iter / (fluidmoe_sync / 1000.0)
    interleaved_tokps = tokens_per_iter / (fluidmoe_interleaved / 1000.0)
    p0(rank, f"FluidMoE: forward={fluidmoe_fwd:.2f}ms  sync_iter={fluidmoe_sync:.2f}ms  interleaved_iter={fluidmoe_interleaved:.2f}ms")
    p0(rank, f"  throughput: sync={sync_tokps:.0f} tok/s  interleaved={interleaved_tokps:.0f} tok/s")
    p0(rank, f"RESULT impl=fluidmoe forward_ms={fluidmoe_fwd:.6f} sync_iter_ms={fluidmoe_sync:.6f} interleaved_iter_ms={fluidmoe_interleaved:.6f} sync_tokens_per_sec={sync_tokps:.6f} interleaved_tokens_per_sec={interleaved_tokps:.6f}")

    dist.destroy_process_group()
    p0(rank, "Done!")


if __name__ == "__main__":
    main()
