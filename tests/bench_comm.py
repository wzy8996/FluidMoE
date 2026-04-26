"""Benchmark AllToAll vs P2P round-robin communication on current hardware.

Each method is **verified** before being timed: a rank-derived pattern is
written into the send buffer, the exchange is run once, and the recv buffer
is checked against the partner's pattern. If verification fails the method
is skipped (and a warning printed) so the printed bandwidth is never measured
on garbage / zero-filled buffers.

Fair comparison: all methods use the same per-peer chunk size.
- chunk_numel = elements sent to / received from each peer
- AllToAll total buffer = chunk_numel * ep_size  (includes self-copy, negligible)
- P2P total transfer   = chunk_numel * (ep_size - 1)  (no self)
- cross_gpu_bytes = chunk_numel * (ep_size - 1) * 2  (actual inter-GPU data, used for BW)
"""

import argparse
import os
import sys
import warnings

os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

# Make ``fluid.*`` importable when this script is launched from any cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.distributed as dist


# =========================================================================
# Helpers
# =========================================================================

def _tournament_partners(my_rank: int, ep_size: int) -> list:
    """Partner sequence matching production CPPlan/EPPlan (round-robin tournament).

    Each tournament round is a symmetric pair (a↔b) — both ranks see each
    other as partner in the same round. Mirrors compute_round_robin_schedule
    in fluid/core/comm.py and the partner-extraction loop in CPPlan.__init__.
    """
    from fluid.core.comm import compute_round_robin_schedule
    schedule = compute_round_robin_schedule(ep_size)
    partners = []
    for round_pairs in schedule:
        for a, b in round_pairs:
            if a == my_rank:
                partners.append(b)
                break
            if b == my_rank:
                partners.append(a)
                break
    return partners


def _fill_pattern(buf: torch.Tensor, sender_rank: int, receiver_rank: int) -> None:
    """Deterministic rank-pair pattern. Distinct across (sender, receiver) and
    chunk position so any swap / drop / partial copy is detectable."""
    n = buf.numel()
    base = (sender_rank * 1009 + receiver_rank * 31 + 7) & 0xFFFF
    idx = torch.arange(n, dtype=torch.int32, device=buf.device)
    pat = ((idx + base) & 0x7FFF).to(torch.float32) * (1.0 / 1024.0)
    buf.copy_(pat.to(buf.dtype))


def _expected_recv_pattern(send_buf: torch.Tensor, partner_rank: int,
                           my_rank: int) -> torch.Tensor:
    """Pattern we expect to receive when ``partner_rank`` filled their send_buf
    with ``_fill_pattern(buf, sender=partner_rank, receiver=my_rank)``."""
    expected = torch.empty_like(send_buf)
    _fill_pattern(expected, sender_rank=partner_rank, receiver_rank=my_rank)
    return expected


def _verify_or_warn(label: str, recv_buf: torch.Tensor,
                    expected: torch.Tensor, rank: int) -> bool:
    """Compare recv vs expected. Returns True iff exact match."""
    if recv_buf.shape != expected.shape:
        if rank == 0:
            warnings.warn(f"[{label}] recv shape {tuple(recv_buf.shape)} != "
                          f"expected {tuple(expected.shape)}; skipping.")
        return False
    diff = (recv_buf.to(torch.float32) - expected.to(torch.float32)).abs()
    max_err = float(diff.max().item())
    # bf16 round-trip should be exact (we generate the same bf16 values on
    # both sides; copy is bit-exact). Allow 0 to catch any layout / partial
    # transfer bugs.
    ok = max_err == 0.0
    if not ok and rank == 0:
        warnings.warn(f"[{label}] data integrity check FAILED "
                      f"(max abs diff = {max_err:.6e}). Skipping timing.")
    return ok


# =========================================================================
# AllToAll
# =========================================================================

def bench_alltoall(chunk_numel: int, ep_group, device, warmup=10, iters=50):
    """Bulk AllToAll. Verifies that recv chunks match partner-derived pattern."""
    ep_size = ep_group.size()
    my_rank = ep_group.rank()
    total_numel = chunk_numel * ep_size

    send_buf = torch.empty(total_numel, dtype=torch.bfloat16, device=device)
    recv_buf = torch.empty_like(send_buf)

    # send_buf is laid out as [chunk_for_peer_0 | chunk_for_peer_1 | ...]
    for peer in range(ep_size):
        _fill_pattern(send_buf[peer * chunk_numel:(peer + 1) * chunk_numel],
                      sender_rank=my_rank, receiver_rank=peer)

    send_splits = [chunk_numel] * ep_size
    recv_splits = [chunk_numel] * ep_size

    # Verify once before timing.
    dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits, group=ep_group)
    torch.cuda.synchronize()
    expected = torch.empty_like(recv_buf)
    for peer in range(ep_size):
        _fill_pattern(expected[peer * chunk_numel:(peer + 1) * chunk_numel],
                      sender_rank=peer, receiver_rank=my_rank)
    if not _verify_or_warn("AllToAll", recv_buf, expected, my_rank):
        return float('nan')

    for _ in range(warmup):
        dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits, group=ep_group)
    torch.cuda.synchronize()

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(iters):
        dist.all_to_all_single(recv_buf, send_buf, recv_splits, send_splits, group=ep_group)
    ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e) / iters


# =========================================================================
# NCCL P2P round-robin (matches production NCCLBackend.exchange exactly)
# =========================================================================

def bench_p2p_roundrobin(chunk_numel: int, ep_group, device, warmup=10, iters=50):
    """P2P round-robin matching production NCCLBackend.exchange.

    Production calls ``reqs[-1].wait()`` after each per-round
    ``batch_isend_irecv``. The previous version of this bench only waited the
    LAST batch's reqs — that's a different (and more pipelined) workload than
    what the production code actually pays for. We match production here so
    the printed numbers reflect the real iter-time cost.
    """
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    global_ranks = dist.get_process_group_ranks(ep_group)
    partners = _tournament_partners(my_rank, ep_size)
    n_rounds = len(partners)

    send_buf_list = []
    recv_buf_list = []
    global_partners = []
    for partner_local in partners:
        sb = torch.empty(chunk_numel, dtype=torch.bfloat16, device=device)
        _fill_pattern(sb, sender_rank=my_rank, receiver_rank=partner_local)
        send_buf_list.append(sb)
        recv_buf_list.append(torch.empty(chunk_numel, dtype=torch.bfloat16, device=device))
        global_partners.append(global_ranks[partner_local])

    def run_once():
        for i in range(n_rounds):
            gr = global_partners[i]
            ops = [
                dist.P2POp(dist.irecv, recv_buf_list[i], gr, group=ep_group),
                dist.P2POp(dist.isend, send_buf_list[i], gr, group=ep_group),
            ]
            reqs = dist.batch_isend_irecv(ops)
            # Per-round wait — production NCCLBackend.exchange does this.
            reqs[-1].wait()

    # Verify once.
    run_once()
    torch.cuda.synchronize()
    for i, partner_local in enumerate(partners):
        expected = _expected_recv_pattern(recv_buf_list[i], partner_local, my_rank)
        if not _verify_or_warn(f"P2P-RR round{i}", recv_buf_list[i], expected, my_rank):
            return float('nan')

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(iters):
        run_once()
    ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e) / iters


def bench_p2p_all_concurrent(chunk_numel: int, ep_group, device, warmup=10, iters=50):
    """P2P all-pairs concurrent: all send/recv launched in one batch, then
    wait. This is the "best case" pipelined NCCL P2P — useful for comparing
    against the production round-robin pattern."""
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    global_ranks = dist.get_process_group_ranks(ep_group)
    partners = _tournament_partners(my_rank, ep_size)
    n_partners = len(partners)

    send_buf_list = []
    recv_buf_list = []
    global_partner_list = []
    for partner_local in partners:
        sb = torch.empty(chunk_numel, dtype=torch.bfloat16, device=device)
        _fill_pattern(sb, sender_rank=my_rank, receiver_rank=partner_local)
        send_buf_list.append(sb)
        recv_buf_list.append(torch.empty(chunk_numel, dtype=torch.bfloat16, device=device))
        global_partner_list.append(global_ranks[partner_local])

    def run_once():
        ops = []
        for i in range(n_partners):
            ops.append(dist.P2POp(dist.irecv, recv_buf_list[i], global_partner_list[i], group=ep_group))
            ops.append(dist.P2POp(dist.isend, send_buf_list[i], global_partner_list[i], group=ep_group))
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # Verify.
    run_once()
    torch.cuda.synchronize()
    for i, partner_local in enumerate(partners):
        expected = _expected_recv_pattern(recv_buf_list[i], partner_local, my_rank)
        if not _verify_or_warn(f"P2P-Conc round{i}", recv_buf_list[i], expected, my_rank):
            return float('nan')

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(iters):
        run_once()
    ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e) / iters


# =========================================================================
# NVSHMEM
# =========================================================================

def _try_nvshmem_backend():
    """Try to create NVSHMEMBackend. Returns (backend, available)."""
    try:
        from fluid.core.p2p_backend import NVSHMEMBackend
        backend = NVSHMEMBackend()
        return backend, True
    except Exception:
        return None, False


def bench_nvshmem_roundrobin(chunk_numel: int, ep_group, device, nvshmem_backend,
                             warmup=10, iters=50):
    """NVSHMEM P2P round-robin via the production ``NVSHMEMBackend.exchange``
    path. We deliberately go through the public backend method so any signal
    protocol bug in production code surfaces here too."""
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    partners = _tournament_partners(my_rank, ep_size)
    n_rounds = len(partners)

    comm_stream = torch.cuda.Stream(device=device)

    send_bufs = []
    recv_bufs = []
    global_ranks = dist.get_process_group_ranks(ep_group)
    for i, partner_local in enumerate(partners):
        sb = torch.empty(chunk_numel, dtype=torch.bfloat16, device=device)
        _fill_pattern(sb, sender_rank=my_rank, receiver_rank=partner_local)
        send_bufs.append(sb)
        # Allocate symmetric recv via the production helper.
        recv_bufs.append(nvshmem_backend.alloc_recv_buffer(
            f"bench_nvshmem_recv_{i}", chunk_numel, torch.bfloat16, device))

    events = [torch.cuda.Event() for _ in range(n_rounds)]

    def run_once():
        for i in range(n_rounds):
            with torch.cuda.stream(comm_stream):
                nvshmem_backend.exchange(
                    send_buf=send_bufs[i], recv_buf=recv_bufs[i],
                    partner_global_rank=global_ranks[partners[i]],
                    partner_local_rank=partners[i],
                    group=ep_group, comm_stream=comm_stream, event=events[i],
                )
        torch.cuda.current_stream().wait_event(events[-1])

    # Verify.
    run_once()
    torch.cuda.synchronize()
    for i, partner_local in enumerate(partners):
        expected = _expected_recv_pattern(recv_bufs[i], partner_local, my_rank)
        if not _verify_or_warn(f"NVSHMEM-RR round{i}", recv_bufs[i], expected, my_rank):
            return float('nan')

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(iters):
        run_once()
    ev_e.record()
    torch.cuda.synchronize()
    return ev_s.elapsed_time(ev_e) / iters


def bench_nvshmem_all_concurrent(chunk_numel: int, ep_group, device, nvshmem_backend,
                                 warmup=10, iters=50):
    """NVSHMEM all-pairs truly concurrent: phase 1 issues all puts, phase 2
    waits all signals on the same comm_stream. Demonstrates how much of the
    round-robin time is wait-on-partner vs intrinsic put cost.

    Uses a **private** signal counter snapshot so it doesn't pollute the
    production backend's state if this bench is interleaved with real usage.
    """
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    partners_local = _tournament_partners(my_rank, ep_size)
    n_partners = len(partners_local)

    comm_stream = torch.cuda.Stream(device=device)

    nvshmem_backend._ensure_group(ep_group)
    gid = id(ep_group)
    partners_pe = [nvshmem_backend._pe_map[gid][p] for p in partners_local]

    _ops = nvshmem_backend._ops
    SIGNAL_SET = nvshmem_backend._SIGNAL_SET
    CMP_GE = nvshmem_backend._CMP_GE

    # Same put_signal protocol the production backend uses: both ``dest`` and
    # ``sig_addr`` are LOCAL symmetric heap pointers; NVSHMEM translates via
    # the ``pe`` argument. Cross-node correct.
    sig_put_addr = nvshmem_backend._get_sig_put_addr(gid)

    send_bufs = []
    recv_bufs = []
    local_recv_sym_ptrs = []
    local_sig_addrs = []
    for i, partner_local in enumerate(partners_local):
        sb = torch.empty(chunk_numel, dtype=torch.bfloat16, device=device)
        _fill_pattern(sb, sender_rank=my_rank, receiver_rank=partner_local)
        send_bufs.append(sb)

        recv_buf = nvshmem_backend.alloc_recv_buffer(
            f"bench_nvshmem_conc_recv_{i}", chunk_numel, torch.bfloat16, device)
        recv_bufs.append(recv_buf)
        local_recv_sym_ptrs.append(recv_buf.data_ptr())
        local_sig_addrs.append(
            nvshmem_backend._get_local_signal_addr(gid, partners_pe[i]))

    nbytes = chunk_numel * send_bufs[0].element_size()
    # PRIVATE counter snapshot per (gid, partner_pe). Production code's
    # counter is not touched.
    counters = [nvshmem_backend._signal_counter.get((gid, p_pe), 0)
                for p_pe in partners_pe]
    final_event = torch.cuda.Event()
    stream_ptr = comm_stream.cuda_stream

    def run_once():
        with torch.cuda.stream(comm_stream):
            for i in range(n_partners):
                send_bufs[i].record_stream(comm_stream)
                counters[i] += 1
                _ops.putmem_signal_on_stream(
                    local_recv_sym_ptrs[i], send_bufs[i], nbytes,
                    sig_put_addr, counters[i], SIGNAL_SET,
                    partners_pe[i], stream_ptr,
                )
            for i in range(n_partners):
                _ops.signal_wait_until_on_stream(
                    local_sig_addrs[i], CMP_GE, counters[i], stream_ptr,
                )
            final_event.record(comm_stream)
        torch.cuda.current_stream().wait_event(final_event)

    # Verify.
    run_once()
    torch.cuda.synchronize()
    for i, partner_local in enumerate(partners_local):
        expected = _expected_recv_pattern(recv_bufs[i], partner_local, my_rank)
        if not _verify_or_warn(f"NVSHMEM-Conc round{i}", recv_bufs[i], expected, my_rank):
            # Roll private counters back into the backend so a subsequent
            # production exchange to the same partner doesn't re-issue a
            # stale sig_val; even though we failed, our puts ran.
            for j, p_pe in enumerate(partners_pe):
                nvshmem_backend._signal_counter[(gid, p_pe)] = counters[j]
            return float('nan')

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(iters):
        run_once()
    ev_e.record()
    torch.cuda.synchronize()

    # Roll counters back into the backend — we have actually performed
    # ``warmup + iters + 1`` increments per partner here, so the production
    # state must reflect that or the next ``exchange`` to the same partner
    # would issue a sig_val behind the partner's local signal.
    for j, p_pe in enumerate(partners_pe):
        nvshmem_backend._signal_counter[(gid, p_pe)] = counters[j]

    return ev_s.elapsed_time(ev_e) / iters


# =========================================================================
# Driver
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="AllToAll vs P2P communication benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    ep_group = dist.group.WORLD
    ep_size = dist.get_world_size()

    nvshmem_backend, has_nvshmem = _try_nvshmem_backend()

    if rank == 0:
        print(f"{'='*100}")
        print(f"AllToAll vs P2P Communication Benchmark")
        print(f"  GPUs={ep_size}, warmup={args.warmup}, iters={args.iters}")
        print(f"  NVSHMEM: {'available' if has_nvshmem else 'not available (NCCL only)'}")
        print(f"  All methods are verified against rank-derived pattern before timing.")
        print(f"  BW = cross-GPU bytes / time  (excludes AllToAll self-copy)")
        print(f"  '   nan ms' = method failed verification — bandwidth not measured.")
        print(f"{'='*100}")
        print()

    cols = ["per-peer", "cross-GPU", "AllToAll", "P2P-RR", "P2P-Conc"]
    bw_cols = ["A2A BW", "RR BW", "Conc BW"]
    if has_nvshmem:
        cols += ["SHMEM-RR", "SHMEM-Conc"]
        bw_cols += ["ShmRR BW", "ShmCo BW"]
    if rank == 0:
        hdr = "  ".join(f"{c:>10s}" for c in cols + bw_cols)
        print(hdr)
        print("  ".join("-" * 10 for _ in cols + bw_cols))

    chunk_sizes_mb = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]

    def fmt_ms(x):
        return f"{x:>8.3f}ms" if x == x else f"{'nan':>8s}ms"  # nan check via x==x

    def fmt_bw(bytes_, ms):
        if ms != ms or ms <= 0:
            return f"{'nan':>7s}GB/s"
        return f"{bytes_ / (ms / 1000) / 1e9:>7.1f}GB/s"

    for chunk_mb in chunk_sizes_mb:
        chunk_bytes = int(chunk_mb * 1024 * 1024)
        chunk_numel = chunk_bytes // 2  # bf16 = 2 bytes
        cross_gpu_bytes = chunk_bytes * (ep_size - 1)
        cross_gpu_mb = cross_gpu_bytes / (1024 * 1024)

        a2a_ms = bench_alltoall(chunk_numel, ep_group, device, args.warmup, args.iters)
        p2p_rr_ms = bench_p2p_roundrobin(chunk_numel, ep_group, device, args.warmup, args.iters)
        p2p_conc_ms = bench_p2p_all_concurrent(chunk_numel, ep_group, device, args.warmup, args.iters)

        line = (f"{chunk_mb:>8.1f}MB  {cross_gpu_mb:>8.1f}MB  "
                f"{fmt_ms(a2a_ms)}  {fmt_ms(p2p_rr_ms)}  {fmt_ms(p2p_conc_ms)}  ")
        bw_line = f"{fmt_bw(cross_gpu_bytes, a2a_ms)}  {fmt_bw(cross_gpu_bytes, p2p_rr_ms)}  {fmt_bw(cross_gpu_bytes, p2p_conc_ms)}"

        if has_nvshmem:
            shm_rr_ms = bench_nvshmem_roundrobin(
                chunk_numel, ep_group, device, nvshmem_backend, args.warmup, args.iters)
            shm_conc_ms = bench_nvshmem_all_concurrent(
                chunk_numel, ep_group, device, nvshmem_backend, args.warmup, args.iters)
            line += f"{fmt_ms(shm_rr_ms)}  {fmt_ms(shm_conc_ms)}  "
            bw_line += f"  {fmt_bw(cross_gpu_bytes, shm_rr_ms)}  {fmt_bw(cross_gpu_bytes, shm_conc_ms)}"

        if rank == 0:
            print(line + bw_line)

    if rank == 0:
        print()
        print("per-peer  = data each rank sends to ONE other rank (chunk_numel * 2 bytes)")
        print("cross-GPU = per-peer * (ep_size-1) = total inter-GPU data per rank")
        print()
        print("AllToAll   = NCCL all_to_all_single (bulk, includes trivial self-copy)")
        print("P2P-RR    = NCCL P2P round-robin matching production NCCLBackend.exchange (per-round wait)")
        print("P2P-Conc  = NCCL P2P all partners launched concurrently in one batch")
        if has_nvshmem:
            print("SHMEM-RR  = NVSHMEM via production NVSHMEMBackend.exchange (per-group signal array)")
            print("SHMEM-Conc= NVSHMEM put_signal all partners back-to-back (private signal counter)")

    if has_nvshmem:
        nvshmem_backend.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
