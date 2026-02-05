"""
对比 FluidMoE Baseline vs Megatron (相同配置)

用法:
    torchrun --nproc_per_node=2 tests/compare_baseline_megatron.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist

rank = int(os.environ.get('LOCAL_RANK', 0))
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl', device_id=device)
world_size = dist.get_world_size()

def p0(*args):
    if rank == 0:
        print(*args, flush=True)

# ============================================================
# 配置（与 Megatron 完全相同）
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
p0("Baseline vs Megatron (相同配置)")
p0("=" * 60)
p0(f"Config: hidden={hidden_size}, heads={num_heads}, ffn={ffn_hidden}")
p0(f"        experts={num_experts}, top_k={top_k}, layers={num_layers}")
p0(f"        seq_local={seq_local}, batch={batch_size}, GPUs={world_size}")
p0("=" * 60)

from tests.baseline import BaselineTransformerModel

# 创建 process groups
ranks = list(range(world_size))
cp_group = dist.new_group(ranks=ranks, backend='nccl')
ep_group = dist.new_group(ranks=ranks, backend='nccl')

# 创建 baseline 模型
baseline_model = BaselineTransformerModel(
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden,
    num_experts=num_experts,
    top_k=top_k,
    cp_group=cp_group,
    ep_group=ep_group,
    dtype=torch.bfloat16,
    device=device,
)

p0(f"\n✓ Baseline model created")
total_params = sum(p.numel() for p in baseline_model.parameters())
p0(f"  Total parameters: {total_params / 1e6:.1f}M")

# Count params that need AR (non-expert params)
ar_params = 0
for layer in baseline_model.layers:
    for name in ('qkv_weight', 'proj_weight', 'router_weight',
                 'ln1_weight', 'ln1_bias', 'ln2_weight', 'ln2_bias'):
        param = getattr(layer, name, None)
        if param is not None:
            ar_params += param.numel()

p0(f"  Parameters needing AR: {ar_params / 1e6:.1f}M ({100*ar_params/total_params:.1f}%)")

x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
x_grad = x.clone().detach().requires_grad_(True)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
N = 5

# Baseline AR helper (only AR non-expert params)
def allreduce_model_grads(model):
    """Only AllReduce non-expert parameters."""
    for layer in model.layers:
        for name in ('qkv_weight', 'proj_weight', 'router_weight',
                     'ln1_weight', 'ln1_bias', 'ln2_weight', 'ln2_bias'):
            param = getattr(layer, name, None)
            if param is not None and param.grad is not None:
                dist.all_reduce(param.grad, group=dist.group.WORLD)

# ============================================================
# Warmup
# ============================================================
p0("\nWarmup...")
for _ in range(3):
    with torch.no_grad():
        baseline_model(x)

for _ in range(2):
    x_grad.grad = None
    for p in baseline_model.parameters():
        p.grad = None
    baseline_model(x_grad).sum().backward()
    allreduce_model_grads(baseline_model)

torch.cuda.synchronize()
dist.barrier()
p0("Warmup done.")

# ============================================================
# Benchmark
# ============================================================

# Forward only
start_event.record()
for _ in range(N):
    with torch.no_grad():
        baseline_model(x)
end_event.record()
torch.cuda.synchronize()
baseline_fwd = start_event.elapsed_time(end_event) / N

# Forward + Backward
start_event.record()
for _ in range(N):
    x_grad.grad = None
    for p in baseline_model.parameters():
        p.grad = None
    baseline_model(x_grad).sum().backward()
end_event.record()
torch.cuda.synchronize()
baseline_fwdbwd = start_event.elapsed_time(end_event) / N

# Forward + Backward + AR
start_event.record()
for _ in range(N):
    x_grad.grad = None
    for p in baseline_model.parameters():
        p.grad = None
    baseline_model(x_grad).sum().backward()
    allreduce_model_grads(baseline_model)
end_event.record()
torch.cuda.synchronize()
baseline_fwdbwd_ar = start_event.elapsed_time(end_event) / N

# ============================================================
# 输出对比
# ============================================================
p0(f"\n{'='*60}")
p0(f"Baseline Performance ({num_layers} layers)")
p0(f"{'='*60}")
p0(f"Forward only:       {baseline_fwd:.2f} ms ({baseline_fwd/num_layers:.2f} ms/layer)")
p0(f"Forward + Backward: {baseline_fwdbwd:.2f} ms ({baseline_fwdbwd/num_layers:.2f} ms/layer)")
p0(f"Fwd+Bwd+AR:         {baseline_fwdbwd_ar:.2f} ms ({baseline_fwdbwd_ar/num_layers:.2f} ms/layer)")
p0(f"  (AR overhead:     {baseline_fwdbwd_ar - baseline_fwdbwd:.2f} ms)")
p0(f"{'='*60}")

# 对比 Megatron
megatron_fwdbwd_ar = 174.78  # ms/layer from benchmark_megatron_full_corrected.py
p0(f"\nComparison:")
p0(f"  Baseline:        {baseline_fwdbwd_ar/num_layers:.2f} ms/layer")
p0(f"  Megatron (TE):   {megatron_fwdbwd_ar:.2f} ms/layer")
p0(f"  Difference:      {abs(baseline_fwdbwd_ar/num_layers - megatron_fwdbwd_ar):.2f} ms")
p0(f"  Ratio:           {megatron_fwdbwd_ar/(baseline_fwdbwd_ar/num_layers):.3f}x")
p0(f"{'='*60}")

dist.destroy_process_group()
p0("\nDone!")
