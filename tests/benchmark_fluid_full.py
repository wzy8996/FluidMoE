"""
FluidMoE Full Transformer Benchmark (Matching Megatron Config)

用法:
    torchrun --nproc_per_node=2 tests/benchmark_fluid_full.py

说明:
    - 与 benchmark_megatron_full.py 完全相同的配置
    - 用于公平对比 FluidMoE vs Megatron 完整 Transformer 层
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

# Initialize dist
rank = int(os.environ.get('LOCAL_RANK', 0))
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl', device_id=device)
world_size = dist.get_world_size()

def p0(*args):
    if rank == 0:
        print(*args, flush=True)

# Import FluidMoE
from fluid.layer import TransformerModel
from fluid.core.scheduler import get_backward_scheduler

# ============================================================
# 配置（与 benchmark_megatron_full.py 完全相同）
# ============================================================
hidden_size = 2048
num_heads = 16
num_kv_heads = 16
ffn_hidden = 8192
num_experts = 8
top_k = 4
num_layers = 2
seq_local = 2048
batch_size = 4

p0("=" * 60)
p0("FluidMoE Full Transformer (Matching Megatron)")
p0("=" * 60)
p0(f"Config: hidden={hidden_size}, heads={num_heads}, ffn={ffn_hidden}")
p0(f"        experts={num_experts}, top_k={top_k}, layers={num_layers}")
p0(f"        seq_local={seq_local}, batch={batch_size}, GPUs={world_size}")
p0("=" * 60)

# Process groups (CP and EP)
ranks = list(range(world_size))
cp_group = dist.new_group(ranks=ranks, backend='nccl')
ep_group = dist.new_group(ranks=ranks, backend='nccl')

# ============================================================
# Test 1: FluidMoE WITHOUT overlap (baseline)
# ============================================================
p0("\n[Test 1/2] FluidMoE WITHOUT overlap (fair baseline)")

model_nooverlap = TransformerModel(
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden,
    num_experts=num_experts,
    top_k=top_k,
    cp_group=cp_group,
    ep_group=ep_group,
    moe_combine_chunks=1,  # No overlap
    moe_dispatch_chunks=1,
    attn_proj_chunks=1,
    attn_qkv_chunks=1,
    dtype=torch.bfloat16,
).to(device)

scheduler_nooverlap = get_backward_scheduler()
scheduler_nooverlap.enable()
scheduler_nooverlap.configure_allreduce(
    enabled=False,  # Disable AR interleaving
    dp_group=dist.group.WORLD,
    ep_group=ep_group,
)

p0(f"✓ Model created (no overlap)")
total_params = sum(p.numel() for p in model_nooverlap.parameters())
p0(f"  Total parameters: {total_params / 1e6:.1f}M")

x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
x_grad = x.clone().detach().requires_grad_(True)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
N = 5

# Warmup
p0("Warmup...")
for _ in range(3):
    with torch.no_grad():
        out = model_nooverlap(x)

for _ in range(2):
    x_grad.grad = None
    for p in model_nooverlap.parameters():
        p.grad = None
    out = model_nooverlap(x_grad)
    out.sum().backward()
    scheduler_nooverlap.finish_batch()

torch.cuda.synchronize()
dist.barrier()
p0("Warmup done.")

# Forward only
start_event.record()
for _ in range(N):
    with torch.no_grad():
        out = model_nooverlap(x)
end_event.record()
torch.cuda.synchronize()
fwd_time_nooverlap = start_event.elapsed_time(end_event) / N

# Forward + Backward
start_event.record()
for _ in range(N):
    x_grad.grad = None
    for p in model_nooverlap.parameters():
        p.grad = None
    out = model_nooverlap(x_grad)
    out.sum().backward()
end_event.record()
torch.cuda.synchronize()
fwdbwd_time_nooverlap = start_event.elapsed_time(end_event) / N

# Forward + Backward + AllReduce
start_event.record()
for _ in range(N):
    x_grad.grad = None
    for p in model_nooverlap.parameters():
        p.grad = None
    out = model_nooverlap(x_grad)
    out.sum().backward()
    scheduler_nooverlap.finish_batch()
end_event.record()
torch.cuda.synchronize()
fwdbwd_ar_time_nooverlap = start_event.elapsed_time(end_event) / N

p0(f"\nFluidMoE (No Overlap) Performance:")
p0(f"  Forward only:       {fwd_time_nooverlap:.2f} ms ({fwd_time_nooverlap/num_layers:.2f} ms/layer)")
p0(f"  Forward + Backward: {fwdbwd_time_nooverlap:.2f} ms ({fwdbwd_time_nooverlap/num_layers:.2f} ms/layer)")
p0(f"  Fwd+Bwd+AR:         {fwdbwd_ar_time_nooverlap:.2f} ms ({fwdbwd_ar_time_nooverlap/num_layers:.2f} ms/layer)")
p0(f"    (AR overhead:     {fwdbwd_ar_time_nooverlap - fwdbwd_time_nooverlap:.2f} ms)")

# ============================================================
# Test 2: FluidMoE WITH overlap
# ============================================================
p0("\n[Test 2/2] FluidMoE WITH overlap")

model_overlap = TransformerModel(
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden,
    num_experts=num_experts,
    top_k=top_k,
    cp_group=cp_group,
    ep_group=ep_group,
    moe_combine_chunks=2,  # Enable overlap
    moe_dispatch_chunks=2,
    attn_proj_chunks=2,
    attn_qkv_chunks=2,
    dtype=torch.bfloat16,
).to(device)

scheduler_overlap = get_backward_scheduler()
scheduler_overlap.enable()
scheduler_overlap.configure_allreduce(
    enabled=True,  # Enable AR interleaving
    dp_group=dist.group.WORLD,
    ep_group=ep_group,
)

p0(f"✓ Model created (with overlap)")

x2 = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
x2_grad = x2.clone().detach().requires_grad_(True)

# Warmup
p0("Warmup...")
for _ in range(3):
    with torch.no_grad():
        out = model_overlap(x2)

for _ in range(2):
    x2_grad.grad = None
    for p in model_overlap.parameters():
        p.grad = None
    out = model_overlap(x2_grad)
    out.sum().backward()
    scheduler_overlap.finish_batch()

torch.cuda.synchronize()
dist.barrier()
p0("Warmup done.")

# Forward only
start_event.record()
for _ in range(N):
    with torch.no_grad():
        out = model_overlap(x2)
end_event.record()
torch.cuda.synchronize()
fwd_time_overlap = start_event.elapsed_time(end_event) / N

# Forward + Backward + AllReduce
start_event.record()
for _ in range(N):
    x2_grad.grad = None
    for p in model_overlap.parameters():
        p.grad = None
    out = model_overlap(x2_grad)
    out.sum().backward()
    scheduler_overlap.finish_batch()
end_event.record()
torch.cuda.synchronize()
fwdbwd_ar_time_overlap = start_event.elapsed_time(end_event) / N

p0(f"\nFluidMoE (With Overlap) Performance:")
p0(f"  Forward only:       {fwd_time_overlap:.2f} ms ({fwd_time_overlap/num_layers:.2f} ms/layer)")
p0(f"  Fwd+Bwd+AR:         {fwdbwd_ar_time_overlap:.2f} ms ({fwdbwd_ar_time_overlap/num_layers:.2f} ms/layer)")

# ============================================================
# Summary
# ============================================================
p0(f"\n{'='*60}")
p0(f"SUMMARY")
p0(f"{'='*60}")
p0(f"FluidMoE (No Overlap):  {fwdbwd_ar_time_nooverlap/num_layers:.2f} ms/layer")
p0(f"FluidMoE (With Overlap): {fwdbwd_ar_time_overlap/num_layers:.2f} ms/layer")
p0(f"Speedup (overlap):       {fwdbwd_ar_time_nooverlap/fwdbwd_ar_time_overlap:.3f}x")
p0(f"{'='*60}")

dist.destroy_process_group()
p0("\nDone!")
