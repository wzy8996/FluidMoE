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

    if rank == 0:
        print(f"{'='*76}")
        print(f"AllToAll vs P2P Communication Benchmark")
        print(f"  GPUs={ep_size}, warmup={args.warmup}, iters={args.iters}")
        print(f"  Per-peer chunk size is identical across all methods")
        print(f"  BW = cross-GPU bytes / time  (excludes AllToAll self-copy)")
        print(f"{'='*76}")
        print()
        print(f"{'per-peer':>10s}  {'cross-GPU':>10s}  {'AllToAll':>10s}  {'P2P-RR':>10s}  {'P2P-Conc':>10s}  {'A2A BW':>10s}  {'RR BW':>10s}  {'Conc BW':>10s}")
        print(f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    # per-peer chunk sizes in MB
    chunk_sizes_mb = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]

    for chunk_mb in chunk_sizes_mb:
        chunk_bytes = int(chunk_mb * 1024 * 1024)
        chunk_numel = chunk_bytes // 2  # bf16 = 2 bytes
        cross_gpu_bytes = chunk_bytes * (ep_size - 1)  # actual inter-GPU data
        cross_gpu_mb = cross_gpu_bytes / (1024 * 1024)

        a2a_ms = bench_alltoall(chunk_numel, ep_group, device, args.warmup, args.iters)
        p2p_rr_ms = bench_p2p_roundrobin(chunk_numel, ep_group, device, args.warmup, args.iters)
        p2p_conc_ms = bench_p2p_all_concurrent(chunk_numel, ep_group, device, args.warmup, args.iters)

        # BW based on cross-GPU bytes (fair for all methods)
        a2a_bw = cross_gpu_bytes / (a2a_ms / 1000) / 1e9
        rr_bw = cross_gpu_bytes / (p2p_rr_ms / 1000) / 1e9
        conc_bw = cross_gpu_bytes / (p2p_conc_ms / 1000) / 1e9

        if rank == 0:
            print(
                f"{chunk_mb:>8.1f}MB  {cross_gpu_mb:>8.1f}MB  "
                f"{a2a_ms:>8.3f}ms  {p2p_rr_ms:>8.3f}ms  {p2p_conc_ms:>8.3f}ms  "
                f"{a2a_bw:>7.1f}GB/s  {rr_bw:>7.1f}GB/s  {conc_bw:>7.1f}GB/s"
            )

    if rank == 0:
        print()
        print("per-peer  = data each rank sends to ONE other rank (chunk_numel * 2 bytes)")
        print("cross-GPU = per-peer * (ep_size-1) = total inter-GPU data per rank")
        print()
        print("AllToAll  = NCCL all_to_all_single (bulk, includes trivial self-copy)")
        print("P2P-RR   = P2P round-robin, one partner at a time (FluidMoE forward pattern)")
        print("P2P-Conc = P2P all partners launched concurrently")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
