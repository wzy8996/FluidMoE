"""
Analyze no-overlap baseline timing breakdown.

This script measures, per iteration:
1) Forward total time
2) Forward overlappable compute (QKV GEMM + OutProj GEMM + FC1+FC2 GEMM)
3) Forward communication time (CP/EP AllToAll + all_gather_object)
4) Forward non-overlappable compute = total - overlap - comm
5) Backward total time (excluding AR)
6) Backward overlappable compute (expert bwd + Router dW + LN2 dW + OutProj dW+dX + QKV dW+dX + LN1 dW)
7) Backward communication time (CP/EP AllToAll)
8) Backward non-overlappable compute = total - overlap - comm
9) AR total time
10) Last-layer qkv AR time

Usage:
    torchrun --nproc_per_node=2 tools/no_overlap_time_analyzer.py
"""

import os
import sys
import time
import argparse
from typing import Tuple

import torch
import torch.distributed as dist


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT_DIR, "tests")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import baseline as baseline_mod  # noqa: E402
from baseline import BaselineTransformerModel  # noqa: E402


class CommTracker:
    """Track communication time in forward/backward phases.

    反向通信分类:
      bwd_combine_comm_ms: R1 combine AllToAll BW，反向第一个 AllToAll，
                           chunk pipeline 第一个 chunk 无计算可重叠，1/n1 暴露
      bwd_comm_ms:         全部反向通信总量（含 combine）
    """

    def __init__(self):
        self.phase: str = "idle"
        self.fwd_comm_ms: float = 0.0
        self.bwd_comm_ms: float = 0.0
        self.bwd_combine_comm_ms: float = 0.0
        self._bwd_moe_count: int = 0  # 反向 MoE AllToAll 调用计数

    def reset_iter(self):
        self.fwd_comm_ms = 0.0
        self.bwd_comm_ms = 0.0
        self.bwd_combine_comm_ms = 0.0
        self._bwd_moe_count = 0

    def add_cuda_comm(self, ms: float):
        if self.phase == "fwd":
            self.fwd_comm_ms += ms
        elif self.phase == "bwd":
            self.bwd_comm_ms += ms

    def add_moe_comm(self, ms: float):
        """MoE AllToAll 计时：反向首次调用为 combine AllToAll BW (R1)。"""
        if self.phase == "fwd":
            self.fwd_comm_ms += ms
        elif self.phase == "bwd":
            self.bwd_comm_ms += ms
            self._bwd_moe_count += 1
            if self._bwd_moe_count == 1:
                self.bwd_combine_comm_ms += ms

    def add_host_comm(self, ms: float):
        if self.phase == "fwd":
            self.fwd_comm_ms += ms
        elif self.phase == "bwd":
            self.bwd_comm_ms += ms


class OverlapComputeTracker:
    """Track overlappable compute time in forward/backward phases.

    前向 (P2P Round-Robin, n=world_size, 四区域相同):
      fwd_qkv_ms, fwd_proj_ms, fwd_fc1_ms, fwd_fc2_ms

    后向分类:
      Pipeline dX (关键路径, 按 chunk 拆分为 AllToAll 重叠 + AR 重叠):
        bwd_fc2_dx_ms:  R1 FC2 dX matmul  (n1 chunks)
        bwd_fc1_dx_ms:  R2 FC1 dX matmul  (n2 chunks)
        bwd_proj_dx_ms: R3 OutProj dX     (n3 chunks)
        bwd_qkv_dx_ms:  R4 QKV dX         (n4 chunks)

      dW (全部 AllToAll 可重叠):
        bwd_fc2_dw_ms, bwd_fc1_dw_ms, bwd_proj_dw_ms, bwd_qkv_dw_ms
        bwd_router_dw_ms, bwd_ln2_dw_ms, bwd_ln1_dw_ms

      其他计算 (全部 AR 可重叠):
        bwd_act_recomp_ms: activation recompute (element-wise)
        bwd_act_bwd_ms:    gelu_backward
        bwd_router_dx_ms:  router backward dX
        bwd_ln2_dx_ms:     LN2 backward dX
        bwd_ln1_dx_ms:     LN1 backward dX
        bwd_sdpa_dx_ms:    SDPA backward

      第一反向层 dX (用于计算 R1/R2 不可重叠暴露):
        bwd_first_fc2_dx_ms: ÷n1 得 R1 暴露计算
        bwd_first_fc1_dx_ms: ÷n2 得 R2 暴露计算
    """

    # 所有后向计时 key (与 baseline.py _timed_overlap key 对应)
    _BWD_KEYS = [
        "fc2_dx", "fc2_dw", "act_recomp", "act_bwd",
        "fc1_dx", "fc1_dw",
        "router_dx", "router_dw", "ln2_dw", "ln2_dx",
        "proj_dw", "proj_dx", "sdpa_dx",
        "qkv_dw", "qkv_dx", "ln1_dw", "ln1_dx",
    ]

    def __init__(self):
        self.fwd_qkv_ms: float = 0.0
        self.fwd_proj_ms: float = 0.0
        self.fwd_fc1_ms: float = 0.0
        self.fwd_fc2_ms: float = 0.0
        for k in self._BWD_KEYS:
            setattr(self, f"bwd_{k}_ms", 0.0)
        self._bwd_fc2_dx_count: int = 0
        self.bwd_first_fc2_dx_ms: float = 0.0
        self._bwd_fc1_dx_count: int = 0
        self.bwd_first_fc1_dx_ms: float = 0.0

    @property
    def fwd_overlap_ms(self) -> float:
        return self.fwd_qkv_ms + self.fwd_proj_ms + self.fwd_fc1_ms + self.fwd_fc2_ms

    def reset_iter(self):
        self.fwd_qkv_ms = 0.0
        self.fwd_proj_ms = 0.0
        self.fwd_fc1_ms = 0.0
        self.fwd_fc2_ms = 0.0
        for k in self._BWD_KEYS:
            setattr(self, f"bwd_{k}_ms", 0.0)
        self._bwd_fc2_dx_count = 0
        self.bwd_first_fc2_dx_ms = 0.0
        self._bwd_fc1_dx_count = 0
        self.bwd_first_fc1_dx_ms = 0.0


def timed_cuda_call(fn, *args, **kwargs) -> Tuple[object, float]:
    """Measure a CUDA op call in ms using CUDA events."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    end.synchronize()
    return out, start.elapsed_time(end)


def timed_all_reduce(tensor: torch.Tensor, group) -> float:
    """Measure dist.all_reduce on CUDA tensor in ms."""
    _, ms = timed_cuda_call(dist.all_reduce, tensor, group=group)
    return ms


def parse_args():
    parser = argparse.ArgumentParser(description="No-overlap baseline timing analyzer")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--ffn-hidden", type=int, default=14336)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4)
    # Keep defaults aligned with tests/benchmark.py measurement style.
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--moe-combine-chunks", type=int, default=4, metavar="N1",
                        help="R1 combine AllToAll BW chunks")
    parser.add_argument("--moe-dispatch-chunks", type=int, default=2, metavar="N2",
                        help="R2 dispatch AllToAll BW chunks")
    parser.add_argument("--attn-proj-chunks", type=int, default=1, metavar="N3",
                        help="R3 sp2hp AllToAll BW chunks")
    parser.add_argument("--attn-qkv-chunks", type=int, default=4, metavar="N4",
                        help="R4 hp2sp AllToAll BW chunks")
    return parser.parse_args()


def p0(rank: int, *args):
    if rank == 0:
        print(*args, flush=True)


def main():
    args = parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    world_size = dist.get_world_size()

    seq_local = args.seq_len // world_size
    x = torch.randn(seq_local, args.batch_size, args.hidden_size, dtype=torch.bfloat16, device=device)

    model = BaselineTransformerModel(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        ffn_hidden_size=args.ffn_hidden,
        num_experts=args.num_experts,
        top_k=args.top_k,
        cp_group=dist.group.WORLD,
        ep_group=dist.group.WORLD,
        dtype=torch.bfloat16,
        device=device,
    )

    tracker = CommTracker()
    overlap_tracker = OverlapComputeTracker()

    # Patch baseline communication functions for comm timing.
    orig_sp2hp = baseline_mod._all_to_all_sp2hp
    orig_hp2sp = baseline_mod._all_to_all_hp2sp
    orig_moe_a2a = baseline_mod._moe_all_to_all
    orig_all_gather_object = dist.all_gather_object

    def patched_sp2hp(x_in, group):
        out, ms = timed_cuda_call(orig_sp2hp, x_in, group)
        tracker.add_cuda_comm(ms)
        return out

    def patched_hp2sp(x_in, group):
        out, ms = timed_cuda_call(orig_hp2sp, x_in, group)
        tracker.add_cuda_comm(ms)
        return out

    def patched_moe_a2a(x_in, send_splits, recv_splits, group):
        out, ms = timed_cuda_call(orig_moe_a2a, x_in, send_splits, recv_splits, group)
        tracker.add_moe_comm(ms)  # 区分 combine BW (第一次) 和 dispatch BW (第二次)
        return out

    def patched_all_gather_object(obj_list, obj, group=None):
        t0 = time.perf_counter()
        ret = orig_all_gather_object(obj_list, obj, group=group)
        ms = (time.perf_counter() - t0) * 1000.0
        tracker.add_host_comm(ms)
        return ret

    baseline_mod._all_to_all_sp2hp = patched_sp2hp
    baseline_mod._all_to_all_hp2sp = patched_hp2sp
    baseline_mod._moe_all_to_all = patched_moe_a2a
    dist.all_gather_object = patched_all_gather_object
    baseline_mod.set_compute_timer(overlap_tracker)

    n1 = args.moe_combine_chunks
    n2 = args.moe_dispatch_chunks
    n3 = args.attn_proj_chunks
    n4 = args.attn_qkv_chunks

    fwd_total_ms = 0.0
    fwd_overlap_ms = 0.0
    fwd_qkv_ms = 0.0
    fwd_proj_ms = 0.0
    fwd_fc1_ms = 0.0
    fwd_fc2_ms = 0.0
    fwd_comm_ms = 0.0
    bwd_total_ms = 0.0
    # 反向各分项累加器（与 OverlapComputeTracker._BWD_KEYS 对应）
    bwd_accum = {k: 0.0 for k in OverlapComputeTracker._BWD_KEYS}
    bwd_first_fc2_dx_ms = 0.0
    bwd_first_fc1_dx_ms = 0.0
    # 反向通信
    bwd_combine_eff_ms = 0.0
    bwd_combine_exp_ms = 0.0
    bwd_other_comm_ms = 0.0
    ar_total_ms = 0.0
    ar_last_qkv_ms = 0.0

    def allreduce_model_grads_timed() -> Tuple[float, float]:
        total = 0.0
        qkv_last = 0.0
        last_layer_idx = len(model.layers) - 1
        for li, layer in enumerate(model.layers):
            for name in (
                "qkv_weight",
                "proj_weight",
                "router_weight",
                "ln1_weight",
                "ln1_bias",
                "ln2_weight",
                "ln2_bias",
            ):
                param = getattr(layer, name, None)
                if param is None or param.grad is None:
                    continue
                ms = timed_all_reduce(param.grad, group=dist.group.WORLD)
                total += ms
                if li == last_layer_idx and name == "qkv_weight":
                    qkv_last += ms
        return total, qkv_last

    def zero_grads():
        for p in model.parameters():
            p.grad = None

    try:
        p0(rank, "=" * 70)
        p0(rank, "No-Overlap Baseline Timing Analyzer")
        p0(rank, "=" * 70)
        p0(rank, f"Config: h={args.hidden_size}, heads={args.num_heads}, kv={args.num_kv_heads}")
        p0(rank, f"        ffn={args.ffn_hidden}, experts={args.num_experts}, top_k={args.top_k}, layers={args.num_layers}")
        p0(rank, f"        seq={args.seq_len}, seq_local={seq_local}, batch={args.batch_size}, GPUs={world_size}")
        p0(rank, f"Forward  1/n: n=world_size={world_size} (P2P rounds)")
        p0(rank, f"Backward chunks: n1={n1} (R1), n2={n2} (R2), n3={n3} (R3), n4={n4} (R4)")
        p0(rank, "=" * 70)

        # Warmup
        for _ in range(args.warmup):
            zero_grads()
            overlap_tracker.reset_iter()
            with torch.no_grad():
                model(x)
            tracker.phase = "fwd"
            y = model(x)
            tracker.phase = "bwd"
            y.sum().backward()
            tracker.phase = "idle"
            allreduce_model_grads_timed()
        torch.cuda.synchronize()
        dist.barrier()

        # Measure
        for _ in range(args.iters):
            zero_grads()
            tracker.reset_iter()
            overlap_tracker.reset_iter()

            # Forward total
            tracker.phase = "fwd"
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()
            y = model(x)
            fwd_end.record()
            fwd_end.synchronize()
            fwd_ms = fwd_start.elapsed_time(fwd_end)

            # Backward total (exclude AR)
            tracker.phase = "bwd"
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start.record()
            y.sum().backward()
            bwd_end.record()
            bwd_end.synchronize()
            bwd_ms = bwd_start.elapsed_time(bwd_end)

            # AR timing
            tracker.phase = "idle"
            ar_ms, qkv_last_ms = allreduce_model_grads_timed()

            # 前向: (world_size-1)/world_size 修正（P2P Round-Robin 第一轮无配对）
            fwd_f = (world_size - 1) / world_size
            fwd_total_ms += fwd_ms
            fwd_overlap_ms += overlap_tracker.fwd_overlap_ms * fwd_f
            fwd_qkv_ms    += overlap_tracker.fwd_qkv_ms  * fwd_f
            fwd_proj_ms   += overlap_tracker.fwd_proj_ms * fwd_f
            fwd_fc1_ms    += overlap_tracker.fwd_fc1_ms  * fwd_f
            fwd_fc2_ms    += overlap_tracker.fwd_fc2_ms  * fwd_f
            fwd_comm_ms   += tracker.fwd_comm_ms

            # 反向: 累加所有分项计时
            bwd_total_ms += bwd_ms
            for k in OverlapComputeTracker._BWD_KEYS:
                bwd_accum[k] += getattr(overlap_tracker, f"bwd_{k}_ms")
            bwd_first_fc2_dx_ms += overlap_tracker.bwd_first_fc2_dx_ms
            bwd_first_fc1_dx_ms += overlap_tracker.bwd_first_fc1_dx_ms
            # R1 combine AllToAll BW: 1/n1 暴露通信
            bwd_combine_eff_ms += tracker.bwd_combine_comm_ms * (n1 - 1) / n1
            bwd_combine_exp_ms += tracker.bwd_combine_comm_ms / n1
            bwd_other_comm_ms  += tracker.bwd_comm_ms - tracker.bwd_combine_comm_ms
            ar_total_ms     += ar_ms
            ar_last_qkv_ms  += qkv_last_ms

        dist.barrier()
        torch.cuda.synchronize()

        ni = float(args.iters)
        fwd_total   = fwd_total_ms / ni
        fwd_overlap = fwd_overlap_ms / ni
        fwd_qkv     = fwd_qkv_ms / ni
        fwd_proj    = fwd_proj_ms / ni
        fwd_fc1     = fwd_fc1_ms / ni
        fwd_fc2     = fwd_fc2_ms / ni
        fwd_comm    = fwd_comm_ms / ni

        bwd_total = bwd_total_ms / ni
        # 各分项平均值
        b = {k: bwd_accum[k] / ni for k in OverlapComputeTracker._BWD_KEYS}
        bwd_first_fc2_dx = bwd_first_fc2_dx_ms / ni
        bwd_first_fc1_dx = bwd_first_fc1_dx_ms / ni

        # 暴露计算（第一反向层 pipeline 起始，不可 AR 重叠）
        bwd_fc2_dx_exp = bwd_first_fc2_dx / n1
        bwd_fc1_dx_exp = bwd_first_fc1_dx / n2

        # 反向通信
        bwd_combine_eff  = bwd_combine_eff_ms / ni
        bwd_combine_exp  = bwd_combine_exp_ms / ni
        bwd_other_comm   = bwd_other_comm_ms / ni
        bwd_comm_eff     = bwd_combine_eff + bwd_other_comm
        bwd_comm_total   = bwd_comm_eff + bwd_combine_exp

        ar_total = ar_total_ms / ni
        qkv_last = ar_last_qkv_ms / ni

        # ============================================================
        # AllToAll 计算: all dW + (n-1)/n pipeline dX
        # ============================================================
        all_dw = (b["fc2_dw"] + b["fc1_dw"] + b["proj_dw"] + b["qkv_dw"]
                  + b["router_dw"] + b["ln2_dw"] + b["ln1_dw"])
        pipeline_dx_alltoall = ((n1 - 1) / n1 * b["fc2_dx"]
                                + (n2 - 1) / n2 * b["fc1_dx"]
                                + (n3 - 1) / n3 * b["proj_dx"]
                                + (n4 - 1) / n4 * b["qkv_dx"])
        bwd_alltoall_compute = all_dw + pipeline_dx_alltoall

        # ============================================================
        # AR 计算: 1/n pipeline dX + other compute - 第一层暴露
        # ============================================================
        pipeline_dx_ar = (1 / n1 * b["fc2_dx"]
                          + 1 / n2 * b["fc1_dx"]
                          + 1 / n3 * b["proj_dx"]
                          + 1 / n4 * b["qkv_dx"])
        other_compute = (b["act_bwd"] + b["act_recomp"]
                         + b["router_dx"] + b["ln2_dx"] + b["ln1_dx"]
                         + b["sdpa_dx"])
        # 减去第一层暴露（该时刻 AR 不可能在执行）
        bwd_ar_compute = pipeline_dx_ar + other_compute - bwd_fc2_dx_exp - bwd_fc1_dx_exp

        # 通信
        bwd_alltoall_comm = bwd_comm_eff
        bwd_ar_comm = ar_total - qkv_last

        # 不可重叠
        bwd_no_overlap_compute = bwd_fc2_dx_exp + bwd_fc1_dx_exp
        bwd_no_overlap_comm = bwd_combine_exp + qkv_last
        # 残差（未计时的小操作）
        bwd_timed_total = bwd_alltoall_compute + bwd_ar_compute + bwd_no_overlap_compute
        bwd_residual = max(0.0, bwd_total - bwd_timed_total - bwd_comm_total)

        # ============================================================
        # 前向
        # ============================================================
        fwd_no_overlap = max(0.0, fwd_total - fwd_overlap - fwd_comm)
        fwd_fluidmoe_min = fwd_no_overlap + max(fwd_overlap, fwd_comm)
        speedup_fwd_ub = fwd_total / fwd_fluidmoe_min if fwd_fluidmoe_min > 0 else float("inf")

        # ============================================================
        # 总公式: fwd_min + bwd_no_overlap + bwd_residual + bwd_no_overlap_comm
        #          + max(bwd_alltoall_compute, bwd_alltoall_comm)
        #          + max(bwd_ar_compute, bwd_ar_comm)
        # ============================================================
        iter_total = fwd_total + bwd_total + ar_total

        fluidmoe_min = (fwd_fluidmoe_min
                        + bwd_no_overlap_compute + bwd_residual + bwd_no_overlap_comm
                        + max(bwd_alltoall_compute, bwd_alltoall_comm)
                        + max(bwd_ar_compute, bwd_ar_comm))
        speedup_ub = iter_total / fluidmoe_min if fluidmoe_min > 0 else float("inf")

        # ============================================================
        # Output
        # ============================================================
        p0(rank, "\n" + "=" * 70)
        p0(rank, "Average Per Iteration (No Overlap Baseline)")
        p0(rank, "=" * 70)
        p0(rank, f"Forward:  total={fwd_total:.3f} ms  [n=world_size={world_size}, factor={(world_size-1)/world_size:.3f}]")
        p0(rank, f"  overlap_compute={fwd_overlap:.3f} ms  "
                 f"(R4 qkv={fwd_qkv:.3f}, R3 proj={fwd_proj:.3f}, R2 fc1={fwd_fc1:.3f}, R1 fc2={fwd_fc2:.3f})")
        p0(rank, f"  no_overlap={fwd_no_overlap:.3f} ms, comm={fwd_comm:.3f} ms")
        p0(rank, f"  fwd_min={fwd_fluidmoe_min:.3f} ms  (speedup UB={speedup_fwd_ub:.3f}x)")
        p0(rank, f"Backward: total={bwd_total:.3f} ms (excl. AR)  [n1={n1}, n2={n2}, n3={n3}, n4={n4}]")
        p0(rank, f"  Pipeline dX (关键路径):")
        p0(rank, f"    R1 FC2 dX   = {b['fc2_dx']:.3f} ms  (1st_layer={bwd_first_fc2_dx:.3f}, exposed={bwd_fc2_dx_exp:.3f})")
        p0(rank, f"    R2 FC1 dX   = {b['fc1_dx']:.3f} ms  (1st_layer={bwd_first_fc1_dx:.3f}, exposed={bwd_fc1_dx_exp:.3f})")
        p0(rank, f"    R3 Proj dX  = {b['proj_dx']:.3f} ms")
        p0(rank, f"    R4 QKV dX   = {b['qkv_dx']:.3f} ms")
        p0(rank, f"  dW (AllToAll 可重叠):")
        p0(rank, f"    FC2={b['fc2_dw']:.3f}  FC1={b['fc1_dw']:.3f}  Proj={b['proj_dw']:.3f}  QKV={b['qkv_dw']:.3f}"
                 f"  Router={b['router_dw']:.3f}  LN2={b['ln2_dw']:.3f}  LN1={b['ln1_dw']:.3f}  total={all_dw:.3f}")
        p0(rank, f"  Other (AR 可重叠):")
        p0(rank, f"    act_bwd={b['act_bwd']:.3f}  act_recomp={b['act_recomp']:.3f}  router_dx={b['router_dx']:.3f}"
                 f"  sdpa_dx={b['sdpa_dx']:.3f}  ln2_dx={b['ln2_dx']:.3f}  ln1_dx={b['ln1_dx']:.3f}  total={other_compute:.3f}")
        p0(rank, f"  comm: AllToAll eff={bwd_comm_eff:.3f}, combine exposed(1/{n1})={bwd_combine_exp:.3f}")
        p0(rank, f"AR: total={ar_total:.3f} ms, last-layer-qkv={qkv_last:.3f} ms")
        p0(rank, "-" * 70)
        p0(rank, f"Iteration total (baseline)  = {iter_total:.3f} ms")
        p0(rank, f"  [1] fwd pair:  max(compute={fwd_overlap:.3f}, comm={fwd_comm:.3f}) = {max(fwd_overlap, fwd_comm):.3f}")
        p0(rank, f"      fwd no_overlap = {fwd_no_overlap:.3f}")
        p0(rank, f"  [2] bwd AllToAll pair:")
        p0(rank, f"      compute = {bwd_alltoall_compute:.3f}  (dW={all_dw:.3f} + pipeline_dx={pipeline_dx_alltoall:.3f})")
        p0(rank, f"      comm    = {bwd_alltoall_comm:.3f}")
        p0(rank, f"      max     = {max(bwd_alltoall_compute, bwd_alltoall_comm):.3f}")
        p0(rank, f"  [3] bwd AR pair:")
        p0(rank, f"      compute = {bwd_ar_compute:.3f}  (pipeline_dx={pipeline_dx_ar:.3f} + other={other_compute:.3f}"
                 f" - exposed={bwd_fc2_dx_exp + bwd_fc1_dx_exp:.3f})")
        p0(rank, f"      comm    = {bwd_ar_comm:.3f}  (AR={ar_total:.3f} - last_qkv={qkv_last:.3f})")
        p0(rank, f"      max     = {max(bwd_ar_compute, bwd_ar_comm):.3f}")
        p0(rank, f"  bwd no_overlap = {bwd_no_overlap_compute:.3f}  "
                 f"(fc2_dx_exp={bwd_fc2_dx_exp:.3f} + fc1_dx_exp={bwd_fc1_dx_exp:.3f})")
        p0(rank, f"  bwd no_overlap_comm = {bwd_no_overlap_comm:.3f}  "
                 f"(combine_exp={bwd_combine_exp:.3f} + last_qkv_ar={qkv_last:.3f})")
        p0(rank, f"  bwd residual = {bwd_residual:.3f}  (未计时小操作)")
        p0(rank, "-" * 70)
        p0(rank, f"FluidMoE theoretical min    = {fluidmoe_min:.3f} ms")
        p0(rank, f"  = fwd_min({fwd_fluidmoe_min:.3f})")
        p0(rank, f"    + bwd_no_overlap({bwd_no_overlap_compute:.3f}) + residual({bwd_residual:.3f})"
                 f" + no_overlap_comm({bwd_no_overlap_comm:.3f})")
        p0(rank, f"    + max(bwd_a2a_compute({bwd_alltoall_compute:.3f}), bwd_a2a_comm({bwd_alltoall_comm:.3f}))")
        p0(rank, f"    + max(bwd_ar_compute({bwd_ar_compute:.3f}), bwd_ar_comm({bwd_ar_comm:.3f}))")
        p0(rank, f"Speedup upper bound         = {speedup_ub:.3f}x")
        p0(rank, "=" * 70)

    finally:
        # Restore patched functions
        baseline_mod.set_compute_timer(None)
        baseline_mod._all_to_all_sp2hp = orig_sp2hp
        baseline_mod._all_to_all_hp2sp = orig_hp2sp
        baseline_mod._moe_all_to_all = orig_moe_a2a
        dist.all_gather_object = orig_all_gather_object

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
