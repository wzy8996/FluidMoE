"""Benchmark AllToAll vs P2P round-robin communication on current hardware.

Fair comparison: all methods use the same per-peer chunk size.
- chunk_numel = elements sent to / received from each peer
- AllToAll total buffer = chunk_numel * ep_size  (includes self-copy, negligible)
- P2P total transfer   = chunk_numel * (ep_size - 1)  (no self)
- cross_gpu_bytes = chunk_numel * (ep_size - 1) * 2  (actual inter-GPU data, used for BW)
"""

import argparse
import os

os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

import torch
import torch.distributed as dist


def bench_alltoall(chunk_numel: int, ep_group, device, warmup=10, iters=50):
    """Bulk AllToAll benchmark."""
    ep_size = ep_group.size()
    total_numel = chunk_numel * ep_size

    send_buf = torch.randn(total_numel, dtype=torch.bfloat16, device=device)
    recv_buf = torch.empty_like(send_buf)

    send_splits = [chunk_numel] * ep_size
    recv_splits = [chunk_numel] * ep_size

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


def bench_p2p_roundrobin(chunk_numel: int, ep_group, device, warmup=10, iters=50):
    """P2P round-robin benchmark (matching FluidMoE forward pattern).

    One partner at a time: start P2P_i, wait P2P_i, then next.
    """
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    global_ranks = dist.get_process_group_ranks(ep_group)

    # Pre-allocate persistent buffers and pre-compute per-round data (list, not dict)
    rounds = []
    send_buf_list = []
    recv_buf_list = []
    for r in range(1, ep_size):
        send_to = (my_rank + r) % ep_size
        recv_from = (my_rank - r + ep_size) % ep_size
        send_buf_list.append(torch.randn(chunk_numel, dtype=torch.bfloat16, device=device))
        recv_buf_list.append(torch.empty(chunk_numel, dtype=torch.bfloat16, device=device))
        rounds.append((global_ranks[recv_from], global_ranks[send_to]))
    n_rounds = len(rounds)

    def run_once():
        last_reqs = None
        for i in range(n_rounds):
            gr_recv, gr_send = rounds[i]
            ops = [
                dist.P2POp(dist.irecv, recv_buf_list[i], gr_recv, group=ep_group),
                dist.P2POp(dist.isend, send_buf_list[i], gr_send, group=ep_group),
            ]
            last_reqs = dist.batch_isend_irecv(ops)
        for req in last_reqs:
            req.wait()

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
    """P2P all-pairs concurrent (all send/recv launched at once)."""
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    global_ranks = dist.get_process_group_ranks(ep_group)

    # Pre-allocate persistent buffers and pre-compute partner data (list, not dict)
    partner_list = []
    send_buf_list = []
    recv_buf_list = []
    global_partner_list = []
    for r in range(1, ep_size):
        partner = (my_rank + r) % ep_size
        partner_list.append(partner)
        send_buf_list.append(torch.randn(chunk_numel, dtype=torch.bfloat16, device=device))
        recv_buf_list.append(torch.empty(chunk_numel, dtype=torch.bfloat16, device=device))
        global_partner_list.append(global_ranks[partner])
    n_partners = len(partner_list)

    def run_once():
        ops = []
        for i in range(n_partners):
            ops.append(dist.P2POp(dist.irecv, recv_buf_list[i], global_partner_list[i], group=ep_group))
            ops.append(dist.P2POp(dist.isend, send_buf_list[i], global_partner_list[i], group=ep_group))
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

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
    """NVSHMEM P2P round-robin benchmark (stream-ordered, no blocking wait)."""
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    n_rounds = ep_size - 1

    comm_stream = torch.cuda.Stream(device=device)

    send_bufs = []
    recv_bufs = []
    partners = []
    for r in range(1, ep_size):
        partner = (my_rank + r) % ep_size
        partners.append(partner)
        send_bufs.append(torch.randn(chunk_numel, dtype=torch.bfloat16, device=device))
        recv_bufs.append(nvshmem_backend.alloc_recv_buffer(
            f"bench_nvshmem_recv_{r}", chunk_numel, torch.bfloat16, device))

    global_ranks = dist.get_process_group_ranks(ep_group)
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
    """NVSHMEM all-pairs concurrent (all put+signal launched back-to-back)."""
    my_rank = ep_group.rank()
    ep_size = ep_group.size()
    n_partners = ep_size - 1

    comm_stream = torch.cuda.Stream(device=device)

    send_bufs = []
    recv_bufs = []
    partners = []
    for r in range(1, ep_size):
        partner = (my_rank + r) % ep_size
        partners.append(partner)
        send_bufs.append(torch.randn(chunk_numel, dtype=torch.bfloat16, device=device))
        recv_bufs.append(nvshmem_backend.alloc_recv_buffer(
            f"bench_nvshmem_conc_recv_{r}", chunk_numel, torch.bfloat16, device))

    global_ranks = dist.get_process_group_ranks(ep_group)
    final_event = torch.cuda.Event()

    def run_once():
        with torch.cuda.stream(comm_stream):
            for i in range(n_partners):
                evt = final_event if i == n_partners - 1 else None
                nvshmem_backend.exchange(
                    send_buf=send_bufs[i], recv_buf=recv_bufs[i],
                    partner_global_rank=global_ranks[partners[i]],
                    partner_local_rank=partners[i],
                    group=ep_group, comm_stream=comm_stream, event=evt,
                )
        torch.cuda.current_stream().wait_event(final_event)

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
        print(f"  BW = cross-GPU bytes / time  (excludes AllToAll self-copy)")
        print(f"{'='*100}")
        print()

    # Column headers
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

    for chunk_mb in chunk_sizes_mb:
        chunk_bytes = int(chunk_mb * 1024 * 1024)
        chunk_numel = chunk_bytes // 2  # bf16 = 2 bytes
        cross_gpu_bytes = chunk_bytes * (ep_size - 1)
        cross_gpu_mb = cross_gpu_bytes / (1024 * 1024)

        a2a_ms = bench_alltoall(chunk_numel, ep_group, device, args.warmup, args.iters)
        p2p_rr_ms = bench_p2p_roundrobin(chunk_numel, ep_group, device, args.warmup, args.iters)
        p2p_conc_ms = bench_p2p_all_concurrent(chunk_numel, ep_group, device, args.warmup, args.iters)

        a2a_bw = cross_gpu_bytes / (a2a_ms / 1000) / 1e9
        rr_bw = cross_gpu_bytes / (p2p_rr_ms / 1000) / 1e9
        conc_bw = cross_gpu_bytes / (p2p_conc_ms / 1000) / 1e9

        line = (f"{chunk_mb:>8.1f}MB  {cross_gpu_mb:>8.1f}MB  "
                f"{a2a_ms:>8.3f}ms  {p2p_rr_ms:>8.3f}ms  {p2p_conc_ms:>8.3f}ms  ")
        bw_line = f"{a2a_bw:>7.1f}GB/s  {rr_bw:>7.1f}GB/s  {conc_bw:>7.1f}GB/s"

        if has_nvshmem:
            shm_rr_ms = bench_nvshmem_roundrobin(
                chunk_numel, ep_group, device, nvshmem_backend, args.warmup, args.iters)
            shm_conc_ms = bench_nvshmem_all_concurrent(
                chunk_numel, ep_group, device, nvshmem_backend, args.warmup, args.iters)
            shm_rr_bw = cross_gpu_bytes / (shm_rr_ms / 1000) / 1e9
            shm_conc_bw = cross_gpu_bytes / (shm_conc_ms / 1000) / 1e9
            line += f"{shm_rr_ms:>8.3f}ms  {shm_conc_ms:>8.3f}ms  "
            bw_line += f"  {shm_rr_bw:>7.1f}GB/s  {shm_conc_bw:>7.1f}GB/s"

        if rank == 0:
            print(line + bw_line)

    if rank == 0:
        print()
        print("per-peer  = data each rank sends to ONE other rank (chunk_numel * 2 bytes)")
        print("cross-GPU = per-peer * (ep_size-1) = total inter-GPU data per rank")
        print()
        print("AllToAll   = NCCL all_to_all_single (bulk, includes trivial self-copy)")
        print("P2P-RR    = NCCL P2P round-robin, one partner at a time, blocking wait per round")
        print("P2P-Conc  = NCCL P2P all partners launched concurrently")
        if has_nvshmem:
            print("SHMEM-RR  = NVSHMEM put_signal round-robin (stream-ordered, no blocking wait)")
            print("SHMEM-Conc= NVSHMEM put_signal all partners back-to-back (stream-ordered)")

    if has_nvshmem:
        nvshmem_backend.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
