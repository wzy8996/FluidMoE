"""
nsys profiling script for FluidMoE vs Baseline.

Usage:
    nsys profile -t cuda,nvtx -o baseline --force-overwrite \
        torchrun --nproc_per_node=2 tests/profile_nsys.py --mode baseline

    nsys profile -t cuda,nvtx -o fluidmoe --force-overwrite \
        torchrun --nproc_per_node=2 tests/profile_nsys.py --mode fluidmoe

Then open baseline.nsys-rep / fluidmoe.nsys-rep in Nsight Systems GUI.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.cuda
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['baseline', 'fluidmoe'], required=True)
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--iters', type=int, default=3, help='profiled iterations')
args = parser.parse_args()

rank = int(os.environ.get('LOCAL_RANK', 0))
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl', device_id=device)
world_size = dist.get_world_size()

# ============================================================
# Config (same as benchmark.py)
# ============================================================
hidden_size = 4096
num_heads = 32
num_kv_heads = 32
ffn_hidden = 14336
num_experts = 8
top_k = 4
num_layers = 2
seq_len = 4096
batch_size = 4

moe_combine_chunks = 2
moe_dispatch_chunks = 2
attn_proj_chunks = 1
attn_qkv_chunks = 2

ar_trickle_sizes = {
    'moe_combine': 220 * 1024 * 1024,
    'moe_dispatch': 142 * 1024 * 1024,
    'attn_proj': 24 * 1024 * 1024,
    'attn_qkv': 26 * 1024 * 1024,
}

dp_size = 1
cp_size = 2
ep_size = 2
num_gpus = dp_size * cp_size
seq_local = seq_len // cp_size

if rank >= num_gpus:
    dist.barrier(); dist.destroy_process_group(); exit(0)

if dp_size == 1 and num_gpus == world_size:
    all_group = cp_group = ep_group = dp_group = dist.group.WORLD
else:
    all_group = dist.group.WORLD if num_gpus == world_size else dist.new_group(list(range(num_gpus)))
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

x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
x_grad = x.clone().detach().requires_grad_(True)


def allreduce_grads(model):
    for layer in model.layers:
        for name in ('qkv_weight', 'proj_weight', 'router_weight',
                     'ln1_weight', 'ln1_bias', 'ln2_weight', 'ln2_bias'):
            p = getattr(layer, name, None)
            if p is not None and p.grad is not None:
                dist.all_reduce(p.grad, group=all_group)
        if dp_size > 1:
            for name in ('moe_w1', 'moe_w2'):
                p = getattr(layer, name, None)
                if p is not None and p.grad is not None:
                    dist.all_reduce(p.grad, group=dp_group)


# ============================================================
# Baseline profiling
# ============================================================
if args.mode == 'baseline':
    from baseline import BaselineTransformerModel
    model = BaselineTransformerModel(
        num_layers=num_layers, hidden_size=hidden_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
        cp_group=cp_group, ep_group=ep_group,
        dtype=torch.bfloat16, device=device,
    )

    def run_iter():
        x_grad.grad = None
        for p in model.parameters():
            p.grad = None
        model(x_grad).sum().backward()
        allreduce_grads(model)

    # Warmup (outside profiler capture)
    for _ in range(args.warmup):
        run_iter()
    torch.cuda.synchronize()
    dist.barrier()

    # Profiled iterations
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.iters):
        torch.cuda.nvtx.range_push(f"baseline_iter_{i}")
        run_iter()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

# ============================================================
# FluidMoE profiling
# ============================================================
elif args.mode == 'fluidmoe':
    from fluid.layer import TransformerModel
    from fluid.core.scheduler import get_backward_scheduler

    model = TransformerModel(
        num_layers=num_layers, hidden_size=hidden_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
        cp_group=cp_group, ep_group=ep_group,
        moe_combine_chunks=moe_combine_chunks, moe_dispatch_chunks=moe_dispatch_chunks,
        attn_proj_chunks=attn_proj_chunks, attn_qkv_chunks=attn_qkv_chunks,
        ar_trickle_sizes=ar_trickle_sizes,
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

    def run_iter():
        x_grad.grad = None
        for p in model.parameters():
            p.grad = None
        torch.cuda.nvtx.range_push("forward")
        out = model(x_grad)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("backward")
        out.sum().backward()
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("finish_batch")
        scheduler.finish_batch()
        torch.cuda.nvtx.range_pop()
        scheduler.clear_iteration()

    # Warmup
    for _ in range(args.warmup):
        run_iter()
    torch.cuda.synchronize()
    dist.barrier()

    # Profiled iterations
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.iters):
        torch.cuda.nvtx.range_push(f"fluidmoe_iter_{i}")
        run_iter()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


torch.cuda.synchronize()
dist.barrier()
dist.destroy_process_group()
if rank == 0:
    print(f"Done. Profile captured for {args.mode} ({args.iters} iters).")
