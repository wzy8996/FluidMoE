"""Block benchmark — each impl uses its native DDP / AR framework.

Variants (one impl per invocation):

  --impl megatron          : mcore_DDP, overlap_grad_reduce=False
                              (AR all stacked at end of backward)
  --impl megatron-overlap  : mcore_DDP, overlap_grad_reduce=True
                              (bucket-wise AR overlapped with backward)
  --impl deepspeed         : DeepSpeed-style flat-buffer AR with expert filter
                              (= DeepSpeed engine ZeRO-0 sync mode, AR at end)
  --impl fluidmoe-f        : FluidDDP, scheduler.enabled=False
  --impl fluidmoe-fb       : FluidDDP, scheduler on, ar_enabled=False
  --impl fluidmoe-full     : FluidDDP, scheduler on, ar_enabled=True

All wrappers expose the same per-iter API:
    ddp_model.zero_grad_buffer()
    ddp_model(x).sum().backward()
    ddp_model.finish_grad_sync()

Block test scope: only block-internal parameters (no embedding / output layer /
final layernorm), no optimizer step. Comparison is intentionally within a single
microbatch.
"""

import argparse
import os
import sys

os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, TESTS_DIR)

import torch
import torch._dynamo
torch._dynamo.config.disable = True  # fair comparison: all frameworks run in eager mode
import torch.distributed as dist
import torch.nn as nn

from experiment_configs import get_block_benchmark_defaults
from model_configs import get_model_config, list_model_names


# =============================================================================
# Wrappers — uniform (zero_grad_buffer / forward / finish_grad_sync) API
# =============================================================================

class _DeepSpeedZeRO0AR(nn.Module):
    """Approximation of DeepSpeed engine at ZeRO-0, sync AR at end of backward.

    DeepSpeed's actual engine requires an optimizer to initialize, which the
    block test deliberately skips. This wrapper replicates the AR semantics
    DeepSpeed exposes at ZeRO-0:
      - non-expert params: flat-buffer AR across DP+CP world
      - expert params (EP-distributed, different on each rank): AR only across
        the expert-DP-orthogonal group; in single-DP setups this is a no-op
      - all AR fired synchronously inside ``finish_grad_sync()`` (no within-
        backward overlap)
      - params split into ~500M-element buckets (DeepSpeed
        ``reduce_bucket_size`` default) so each AR matches DeepSpeed's
        per-bucket launch granularity rather than one giant AR per dtype

    Naive ``torch.nn.parallel.DistributedDataParallel`` would AR every param
    across the whole group, including experts — for mixtral_8x7b that means
    ~7.5 GB of wasted traffic per iter. This wrapper filters expert grads by
    parameter name, matching what DeepSpeed engine does.
    """

    _BUCKET_SIZE_ELEMS = int(5e8)  # DeepSpeed reduce_bucket_size default

    def __init__(self, module, *, all_group, dp_group, dp_size, dp_cp_size):
        super().__init__()
        self.module = module
        self._all_group = all_group
        self._dp_group = dp_group
        self._dp_size = dp_size
        self._dp_cp_size = dp_cp_size
        self._build_flat_buffers()

    @staticmethod
    def _is_expert_param(name: str) -> bool:
        # Match DeepSpeed MoE (deepspeed_moe.experts.*) and FluidMoE-style
        # expert weights (moe_w1 / moe_w2). Megatron MoE uses .experts.*.
        return ".experts." in name or "moe_w1" in name or "moe_w2" in name \
               or "deepspeed_moe.experts" in name

    def _build_flat_buffers(self):
        device = next(self.module.parameters()).device
        groups = {}  # (is_expert, dtype) -> [params]
        for name, p in self.module.named_parameters():
            is_expert = self._is_expert_param(name)
            groups.setdefault((is_expert, p.dtype), []).append(p)
        # Buckets: list of (flat, params, is_expert). Each bucket sits inside
        # a single (is_expert, dtype) group and holds <= _BUCKET_SIZE_ELEMS
        # elements (a single param exceeding the cap goes alone).
        self._buckets = []
        cap = self._BUCKET_SIZE_ELEMS
        for (is_expert, dtype), params in groups.items():
            cur, cur_n = [], 0
            for p in params:
                if cur and cur_n + p.numel() > cap:
                    self._buckets.append(self._make_bucket(cur, dtype, is_expert, device))
                    cur, cur_n = [], 0
                cur.append(p)
                cur_n += p.numel()
            if cur:
                self._buckets.append(self._make_bucket(cur, dtype, is_expert, device))

    @staticmethod
    def _make_bucket(params, dtype, is_expert, device):
        numel = sum(p.numel() for p in params)
        flat = torch.zeros(numel, dtype=dtype, device=device)
        offset = 0
        for p in params:
            n = p.numel()
            p.main_grad = flat[offset:offset + n].view(p.shape)
            offset += n
        return (flat, list(params), is_expert)

    def forward(self, *a, **kw):
        return self.module(*a, **kw)

    def zero_grad_buffer(self):
        for p in self.module.parameters():
            p.grad = None
        for flat, _, _ in self._buckets:
            flat.zero_()

    def finish_grad_sync(self):
        pre_scale = 1.0 / self._dp_cp_size if self._dp_cp_size > 1 else 1.0
        for flat, params, is_expert in self._buckets:
            offset = 0
            for p in params:
                n = p.numel()
                if p.grad is not None:
                    flat[offset:offset + n].copy_(p.grad.view(-1))
                offset += n
            if pre_scale != 1.0:
                flat.mul_(pre_scale)
            if is_expert:
                if self._dp_size > 1 and self._dp_group is not None:
                    dist.all_reduce(flat, group=self._dp_group)
            else:
                dist.all_reduce(flat, group=self._all_group)
            # Re-bind main_grad views (caching allocator may have moved them
            # under us if any operation reallocated).
            offset = 0
            for p in params:
                n = p.numel()
                p.main_grad = flat[offset:offset + n].view(p.shape)
                p.grad = None
                offset += n


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args():
    defaults = get_block_benchmark_defaults()
    parser = argparse.ArgumentParser(description="FluidMoE Block Benchmark")
    parser.add_argument("--model", type=str, default="mixtral_8x7b",
                        help="Model name (from tools/model_configs.py)")
    parser.add_argument("--impl", type=str, default="fluidmoe-full",
                        choices=["megatron", "megatron-overlap",
                                 "deepspeed",
                                 "fluidmoe-f", "fluidmoe-fb", "fluidmoe-full"],
                        help="Implementation to run. One per invocation. "
                             "Each impl uses its native DDP/AR mechanism:\n"
                             "  megatron          = mcore_DDP overlap=False (AR at end)\n"
                             "  megatron-overlap  = mcore_DDP overlap=True  (within-bwd AR overlap)\n"
                             "  deepspeed         = DeepSpeed ZeRO-0 sync AR at end (expert-aware)\n"
                             "  fluidmoe-f        = FluidDDP, scheduler off\n"
                             "  fluidmoe-fb       = FluidDDP, scheduler on, ar=False\n"
                             "  fluidmoe-full     = FluidDDP, scheduler on, ar=True")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")
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


# =============================================================================
# Timing helpers
# =============================================================================

def _bench_iter(run_fn, warmup, iters):
    """Time one full iter (zero_grad + forward + backward + AR). Returns ms/iter."""
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    ev_s.record()
    for _ in range(iters):
        run_fn()
    ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e) / iters


def _bench_fwd(model, x, warmup, iters):
    """Time forward-only pass under no_grad. Returns ms/iter."""
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    ev_s.record()
    for _ in range(iters):
        with torch.no_grad():
            model(x)
    ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e) / iters


# =============================================================================
# Main
# =============================================================================

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

    assert ep_size == cp_size, f"only ep=cp supported, got ep={ep_size}, cp={cp_size}"
    assert world_size == num_gpus, (
        f"world_size={world_size} != dp*cp={num_gpus}, "
        f"set NPROC_PER_NODE={num_gpus} or adjust dp/cp.")
    assert seq_len % cp_size == 0
    assert num_experts % ep_size == 0
    assert num_heads % cp_size == 0
    assert num_kv_heads % cp_size == 0
    seq_local = seq_len // cp_size

    if rank >= num_gpus:
        dist.barrier()
        dist.destroy_process_group()
        return

    # ---- Build process groups (manual; matches ProcessGroupCollection / Megatron) ----

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

    dp_cp_size = dist.get_world_size(all_group)

    p0(rank, "=" * 60)
    p0(rank, f"FluidMoE Block Benchmark  model={args.model} impl={args.impl}")
    p0(rank, f"  hidden={hidden_size} heads={num_heads} ffn={ffn_hidden}")
    p0(rank, f"  experts={num_experts} top_k={top_k} layers={num_layers}")
    p0(rank, f"  seq={seq_len} batch={batch_size} GPUs={num_gpus}")
    p0(rank, f"  dp={dp_size} cp={cp_size} ep={ep_size}")
    if args.impl.startswith("fluidmoe"):
        p0(rank, f"  chunks: R1={args.moe_combine_chunks} R2={args.moe_dispatch_chunks} "
                 f"R3={args.attn_proj_chunks} R4={args.attn_qkv_chunks}")
    p0(rank, "=" * 60)

    x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    x_grad = x.clone().detach().requires_grad_(True)

    # ---- Build model + DDP wrapper per --impl ----

    if args.impl in ("megatron", "megatron-overlap"):
        # mcore_DDP needs Megatron parallel_state initialized to find groups.
        from megatron.core import parallel_state
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=cp_size,
                expert_model_parallel_size=ep_size,
            )

        from megatron_baseline import MegatronBaselineTransformerModel
        model = MegatronBaselineTransformerModel(
            num_layers=num_layers, hidden_size=hidden_size,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
            cp_group=cp_group, ep_group=ep_group,
            shared_dp_group=all_group,
            expert_dp_group=dp_group if dp_size > 1 else None,
            overlap_a2a=False, delay_wgrad=False,
            capacity_factor=capacity_factor,
            dtype=torch.bfloat16, device=device,
        )

        from megatron.core.distributed import (
            DistributedDataParallel as McoreDDP,
            DistributedDataParallelConfig,
        )
        # Layers store the TransformerConfig; pull it from the first layer.
        transformer_config = model.layers[0].config
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=(args.impl == "megatron-overlap"),
            use_distributed_optimizer=False,
            bucket_size=None,
        )
        ddp_model = McoreDDP(transformer_config, ddp_config, model)

    elif args.impl == "deepspeed":
        from deepspeed_ulysses_baseline import DeepSpeedBlockBaselineTransformerModel
        model = DeepSpeedBlockBaselineTransformerModel(
            num_layers=num_layers, hidden_size=hidden_size,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
            cp_group=cp_group, ep_group=ep_group,
            capacity_factor=capacity_factor,
            dtype=torch.bfloat16, device=device,
        )
        ddp_model = _DeepSpeedZeRO0AR(
            model,
            all_group=all_group,
            dp_group=dp_group if dp_size > 1 else None,
            dp_size=dp_size,
            dp_cp_size=dp_cp_size,
        )
        n_shared = sum(1 for _, _, e in ddp_model._buckets if not e)
        n_expert = sum(1 for _, _, e in ddp_model._buckets if e)
        shared_elems = sum(f.numel() for f, _, e in ddp_model._buckets if not e)
        expert_elems = sum(f.numel() for f, _, e in ddp_model._buckets if e)
        p0(rank, f"  deepspeed buckets: shared={n_shared} ({shared_elems/1e6:.1f}M elems) "
                 f"expert={n_expert} ({expert_elems/1e6:.1f}M elems), cap={int(5e8/1e6)}M")

    elif args.impl in ("fluidmoe-f", "fluidmoe-fb", "fluidmoe-full"):
        from fluid.core.scheduler import get_backward_scheduler
        from fluid.layer import TransformerModel
        from fluid.setup import FluidDDP

        bench_defaults = get_block_benchmark_defaults()

        # Set FLUIDMOE_* env vars from bench_defaults so FluidDDP picks them
        # up via its env-var path (same mechanism production training uses).
        os.environ["FLUIDMOE_SHARED_AR_BW"] = str(bench_defaults.get("shared_ar_bw", 0.0))
        os.environ["FLUIDMOE_EXPERT_AR_BW"] = str(bench_defaults.get("expert_ar_bw", 0.0))
        for region, env_key in [('moe_combine', 'FLUIDMOE_GAP_MOE_COMBINE'),
                                ('moe_dispatch', 'FLUIDMOE_GAP_MOE_DISPATCH'),
                                ('attn_proj', 'FLUIDMOE_GAP_ATTN_PROJ'),
                                ('attn_qkv', 'FLUIDMOE_GAP_ATTN_QKV')]:
            os.environ[env_key] = str(bench_defaults.get("gap_budgets", {}).get(region, 0.0))

        fluidmoe_model = TransformerModel(
            num_layers=num_layers, hidden_size=hidden_size,
            num_heads=num_heads, num_kv_heads=num_kv_heads,
            ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
            cp_group=cp_group, ep_group=ep_group,
            moe_combine_chunks=args.moe_combine_chunks,
            moe_dispatch_chunks=args.moe_dispatch_chunks,
            attn_proj_chunks=args.attn_proj_chunks,
            attn_qkv_chunks=args.attn_qkv_chunks,
            capacity_factor=capacity_factor,
            dtype=torch.bfloat16, device=device,
        )
        chunk_messages = fluidmoe_model.prepare_chunk_status(x)
        if chunk_messages:
            p0(rank, "[FluidMoE][chunk-check] " + " | ".join(chunk_messages))

        scheduler = get_backward_scheduler()
        scheduler.enable()

        ddp_model = FluidDDP(
            config=None,
            module=fluidmoe_model,
            scheduler=scheduler,
            dp_cp_group=all_group if dp_cp_size > 1 else None,
            expert_dp_group=dp_group if dp_size > 1 else None,
        )

        # Variant selects the within-backward overlap flavor.
        if args.impl == "fluidmoe-f":
            scheduler.enabled = False
            # ar_enabled irrelevant when scheduler is off.
        elif args.impl == "fluidmoe-fb":
            scheduler.enabled = True
            scheduler.ar_enabled = False
        else:  # fluidmoe-full
            scheduler.enabled = True
            scheduler.ar_enabled = True

    else:
        raise ValueError(f"unexpected impl: {args.impl}")

    # ---- Benchmark ----

    p0(rank, f"warmup ({args.warmup} iters), then timed measurement ({args.iters} iters)...")
    dist.barrier()

    fwd_ms = _bench_fwd(ddp_model, x, args.warmup, args.iters)

    def run_iter():
        ddp_model.zero_grad_buffer()
        ddp_model(x_grad).sum().backward()
        ddp_model.finish_grad_sync()
        # FluidMoE: production drives ``scheduler.clear_iteration`` from
        # ``FluidOptimizerWrapper.step()``. Block test has no optimizer step,
        # so without this AR state can leak across iters and timing under-
        # counts. mcore_DDP / DeepSpeed wrappers self-sync inside
        # finish_grad_sync; FluidDDP doesn't (by design — production wants
        # AR to finish during optimizer.step instead).
        if args.impl.startswith("fluidmoe"):
            scheduler.clear_iteration()

    iter_ms = _bench_iter(run_iter, args.warmup, args.iters)

    tokens_per_iter = seq_len * batch_size * dp_size
    tokps = tokens_per_iter / (iter_ms / 1000.0)

    p0(rank, f"{args.impl}: forward={fwd_ms:.2f}ms  iter={iter_ms:.2f}ms  "
             f"throughput={tokps:.0f} tok/s")
    p0(rank, f"RESULT impl={args.impl} forward_ms={fwd_ms:.6f} "
             f"iter_ms={iter_ms:.6f} tokens_per_sec={tokps:.6f}")

    dist.destroy_process_group()
    p0(rank, "Done!")


if __name__ == "__main__":
    main()
