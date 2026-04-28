"""Measure exposed-communication share under Megatron baseline (paper Table 1).

For each model in --models, builds a MegatronBaselineTransformerModel + mcore_DDP
with overlap_grad_reduce=False, runs N timed iters, and prints a markdown table:

  | Model | Fwd A2A | Bwd A2A | DP sync tail | Σ exposed | Iter ms |

Methodology: because Megatron baseline has no compute-comm overlap, exposed
comm equals actual NCCL kernel time. We monkey-patch the three ``dist`` calls
Megatron uses (``all_to_all_single``, ``batch_isend_irecv``, ``all_reduce``)
to record CUDA events on the default stream around each call, tagged with the
currently-active phase ("fwd" / "bwd" / "ar"). At the end of each timed run
we sync and aggregate per-(phase, kind) wall-clock ms.

Usage:
  torchrun --nproc_per_node=2 tests/comm_ratio_table.py --dp-size 1 --cp-size 2 --ep-size 2

Notes:
- Block test scope (no embedding/output/optimizer); same as run_block_benchmark.
- One model that OOMs only skips itself; remaining models continue.
- Override ``--seq-len`` to match the paper's 32K setting if your hardware
  fits it (the model_configs defaults are smaller for 24GB cards).
"""

import argparse
import gc
import os
import sys
from collections import defaultdict

os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, TESTS_DIR)

import torch
import torch._dynamo
torch._dynamo.config.disable = True
import torch.distributed as dist

from model_configs import get_model_config


# ---------------------------------------------------------------------------
# CommTimer: monkey-patches dist.* with phase-tagged CUDA event timing
# ---------------------------------------------------------------------------

class CommTimer:
    """Wrap dist.all_to_all_single / batch_isend_irecv / all_reduce with CUDA
    event pairs on default stream. Caller sets ``phase`` before each section;
    each wrapped call appends (start, end) events to ``events[(phase, kind)]``.
    ``aggregate_ms`` syncs and returns total ms per (phase, kind)."""

    _A2A_FNS = ('all_to_all_single', 'batch_isend_irecv')
    _AR_FNS = ('all_reduce',)

    def __init__(self):
        self.phase = 'init'
        self.events = defaultdict(list)
        self._orig = {}

    def __enter__(self):
        for name in self._A2A_FNS:
            self._patch(name, kind='a2a')
        for name in self._AR_FNS:
            self._patch(name, kind='ar')
        return self

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(dist, name, orig)
        self._orig.clear()

    def _patch(self, name, kind):
        orig = getattr(dist, name)
        self._orig[name] = orig
        timer = self
        def wrapper(*args, **kwargs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = orig(*args, **kwargs)
            e.record()
            timer.events[(timer.phase, kind)].append((s, e))
            return out
        setattr(dist, name, wrapper)

    def aggregate_ms(self):
        torch.cuda.synchronize()
        out = defaultdict(float)
        for key, evts in self.events.items():
            for s, e in evts:
                out[key] += s.elapsed_time(e)
        return dict(out)


# ---------------------------------------------------------------------------
# Build Megatron baseline block + run iters
# ---------------------------------------------------------------------------

def _build_groups(rank, dp_size, cp_size, world_size, num_gpus):
    """Build (all_group, cp_group, ep_group, dp_group) — same scheme as
    run_block_benchmark.py:266-283. Returns dp_cp_size as well."""
    if dp_size == 1 and num_gpus == world_size:
        all_group = cp_group = ep_group = dp_group = dist.group.WORLD
    else:
        all_group = (dist.group.WORLD if num_gpus == world_size
                     else dist.new_group(list(range(num_gpus))))
        cp_group = ep_group = dp_group = None
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
    return all_group, cp_group, ep_group, dp_group, dp_cp_size


def measure_megatron_block(model_cfg, dp_size, cp_size, ep_size,
                           warmup, iters, rank, num_gpus,
                           all_group, cp_group, ep_group, dp_group, dp_cp_size):
    """Build Megatron baseline block, run warmup + timed iters, return per-iter
    metrics dict (fwd_a2a_ms, bwd_a2a_ms, ar_tail_ms, iter_ms).

    Process groups and ``parallel_state`` are passed in (initialized once in
    ``main()``); we never destroy/recreate them across models since that
    leaks NCCL communicator state and hangs the next model's first
    collective. Same parallel_state config (cp/ep/tp) works for every model
    because the parallelism layout is shared across the run.
    """
    device = torch.device(f'cuda:{rank}')

    hidden = int(model_cfg['hidden_size'])
    num_heads = int(model_cfg['num_heads'])
    num_kv_heads = int(model_cfg['num_kv_heads'])
    ffn_hidden = int(model_cfg['ffn_hidden'])
    num_experts = int(model_cfg['num_experts'])
    top_k = int(model_cfg['top_k'])
    num_layers = int(model_cfg['num_layers'])
    seq_len = int(model_cfg['seq_len'])
    batch_size = int(model_cfg['batch_size'])
    capacity_factor = float(model_cfg.get('capacity_factor', 1.0))
    seq_local = seq_len // cp_size

    # Wrap construction + run in try/finally so we ALWAYS free model + ddp
    # before returning, even on OOM. Without this, exception unwinds the
    # frame but autograd graph / NCCL workspaces may keep tensors alive,
    # leaving the next dist.barrier() with no headroom.
    model = None
    ddp_model = None
    x_grad = None
    try:
        from megatron_baseline import MegatronBaselineTransformerModel
        model = MegatronBaselineTransformerModel(
            num_layers=num_layers, hidden_size=hidden,
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
        transformer_config = model.layers[0].config
        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,         # match real-training default
            overlap_grad_reduce=False,        # paper "DP sync tail": all AR at end
            use_distributed_optimizer=False,
            bucket_size=None,
        )
        ddp_model = McoreDDP(transformer_config, ddp_config, model)

        x_grad = torch.randn(seq_local, batch_size, hidden,
                             dtype=torch.bfloat16, device=device, requires_grad=True)

        def run_iter():
            ddp_model.zero_grad_buffer()
            out = ddp_model(x_grad).sum()
            out.backward()
            ddp_model.finish_grad_sync()

        # Warmup (no timing)
        for _ in range(warmup):
            run_iter()
        torch.cuda.synchronize()

        # Timed run with comm timing patched. A2A in Megatron's MoE dispatcher
        # goes through ``dist.all_to_all_single`` directly, so the CommTimer
        # patch catches them. AR in mcore_DDP.finish_grad_sync goes through
        # ``_coalescing_manager`` which bypasses ``dist.all_reduce`` — so
        # we measure AR phase wall-clock with a dedicated CUDA event pair.
        # ``torch.cuda.synchronize()`` at each phase boundary forces
        # strict-serial execution so iter_ms = fwd + bwd + ar bit-additively.
        iter_evt_s = torch.cuda.Event(enable_timing=True)
        iter_evt_e = torch.cuda.Event(enable_timing=True)
        ar_phase_events = []
        with CommTimer() as timer:
            iter_evt_s.record()
            for _ in range(iters):
                ddp_model.zero_grad_buffer()

                timer.phase = 'fwd'
                out = ddp_model(x_grad).sum()
                torch.cuda.synchronize()

                timer.phase = 'bwd'
                out.backward()
                torch.cuda.synchronize()

                timer.phase = 'ar'
                ar_s = torch.cuda.Event(enable_timing=True)
                ar_e = torch.cuda.Event(enable_timing=True)
                ar_s.record()
                ddp_model.finish_grad_sync()
                ar_e.record()
                torch.cuda.synchronize()
                ar_phase_events.append((ar_s, ar_e))
            iter_evt_e.record()
            comm_ms = timer.aggregate_ms()

        iter_total_ms = iter_evt_s.elapsed_time(iter_evt_e)
        ar_phase_total_ms = sum(s.elapsed_time(e) for s, e in ar_phase_events)

        return {
            'fwd_a2a_ms': comm_ms.get(('fwd', 'a2a'), 0.0) / iters,
            'bwd_a2a_ms': comm_ms.get(('bwd', 'a2a'), 0.0) / iters,
            'ar_tail_ms': ar_phase_total_ms / iters,
            'fwd_ar_ms':  comm_ms.get(('fwd', 'ar'),  0.0) / iters,
            'bwd_ar_ms':  comm_ms.get(('bwd', 'ar'),  0.0) / iters,
            'ar_a2a_ms':  comm_ms.get(('ar',  'a2a'), 0.0) / iters,
            'iter_ms': iter_total_ms / iters,
        }
    finally:
        # ALWAYS release model + ddp + autograd graph before returning,
        # even on OOM. Without this, an exception leaves the (huge) ddp
        # main_grad buffer alive in stack frames the GC may not reach
        # before the next dist.barrier(), which itself needs CUDA workspace.
        del ddp_model, model, x_grad
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Markdown emission
# ---------------------------------------------------------------------------

def emit_table(rows):
    """rows: list of (model_name, metrics_or_None, err_or_None)."""
    print()
    print("## Exposed-communication share — Megatron baseline (paper Table 1)")
    print()
    print("| Model | Fwd A2A exp. | Bwd A2A exp. | DP sync tail | Σ exposed | Iter ms |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, m, err in rows:
        if err is not None or m is None:
            short = (err or "missing")[:80]
            print(f"| {name} | — | — | — | — | FAILED: {short} |")
            continue
        it = m['iter_ms']
        if it <= 0:
            print(f"| {name} | — | — | — | — | iter_ms=0 |")
            continue
        fwd_pct = 100 * m['fwd_a2a_ms'] / it
        bwd_pct = 100 * m['bwd_a2a_ms'] / it
        ar_pct  = 100 * m['ar_tail_ms'] / it
        total   = fwd_pct + bwd_pct + ar_pct
        print(f"| {name} "
              f"| {fwd_pct:.1f}% "
              f"| {bwd_pct:.1f}% "
              f"| {ar_pct:.1f}% "
              f"| **{total:.1f}%** "
              f"| {it:.1f} |")
    # Diagnostic: any phase mismatches (should all be ~0 ms)
    bad = []
    for name, m, _ in rows:
        if m is None:
            continue
        for k in ('fwd_ar_ms', 'bwd_ar_ms', 'ar_a2a_ms'):
            if m[k] > 0.05:
                bad.append((name, k, m[k]))
    if bad:
        print()
        print("> _Diagnostic: unexpected cross-phase comm (should be ~0ms):_")
        for name, k, v in bad:
            print(f"> - {name}: {k} = {v:.3f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    'dbrx_base',
    'deepseek_v3_mha_proxy',
    'glm4_5_air_mha_proxy',
    'qwen3_30b_a3b',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help='Model names from tools/model_configs.py')
    parser.add_argument('--dp-size', type=int, default=1)
    parser.add_argument('--cp-size', type=int, default=2)
    parser.add_argument('--ep-size', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--seq-len', type=int, default=None,
                        help='Override seq_len from model_config (paper used 32K).')
    args = parser.parse_args()

    rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    dist.init_process_group(backend='nccl', device_id=device)
    world_size = dist.get_world_size()

    assert args.ep_size == args.cp_size, \
        f"only ep=cp supported in block test (got ep={args.ep_size}, cp={args.cp_size})"
    num_gpus = args.dp_size * args.cp_size
    assert world_size == num_gpus, \
        f"world_size={world_size} != dp*cp={num_gpus}"

    if rank == 0:
        print(f"# comm_ratio_table — Megatron baseline")
        print(f"# dp={args.dp_size} cp={args.cp_size} ep={args.ep_size}  "
              f"world={world_size}  warmup={args.warmup} iters={args.iters}")
        if args.seq_len is not None:
            print(f"# seq_len override: {args.seq_len}")

    # Build process groups + initialize parallel_state ONCE for the whole run.
    # Re-initializing (via destroy_model_parallel + initialize_model_parallel)
    # between models leaks NCCL communicator state and reliably hangs the
    # second model's first collective. Same cp/ep config works for every
    # model since the parallelism layout is fixed across the run.
    all_group, cp_group, ep_group, dp_group, dp_cp_size = _build_groups(
        rank, args.dp_size, args.cp_size, world_size, num_gpus)
    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            context_parallel_size=args.cp_size,
            expert_model_parallel_size=args.ep_size,
        )

    rows = []
    for model_name in args.models:
        try:
            cfg = dict(get_model_config(model_name))
        except Exception as e:
            rows.append((model_name, None, f"unknown model: {e}"))
            continue
        if args.seq_len is not None:
            cfg['seq_len'] = args.seq_len

        if rank == 0:
            print(f"\n[run] {model_name}: hidden={cfg['hidden_size']} layers={cfg['num_layers']} "
                  f"experts={cfg['num_experts']} seq={cfg['seq_len']} batch={cfg['batch_size']}",
                  flush=True)

        try:
            metrics = measure_megatron_block(
                cfg, args.dp_size, args.cp_size, args.ep_size,
                args.warmup, args.iters, rank, num_gpus,
                all_group, cp_group, ep_group, dp_group, dp_cp_size)
            rows.append((model_name, metrics, None))
        except Exception as e:
            err = type(e).__name__ + ": " + str(e).split('\n')[0]
            rows.append((model_name, None, err))

        # Between models: aggressive cleanup + barrier. The finally inside
        # measure_megatron_block already del's the model and empties cache,
        # but we double-up here in case the failure path didn't reach
        # finally (e.g. OOM during construction before assignment).
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        emit_table(rows)

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
