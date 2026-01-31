"""
FluidMoE Benchmark

用法:
    torchrun --nproc_per_node=2 tests/benchmark.py

配置说明:
    - chunks: 反向传播分块数 (1=不分块, 2/4/8=分块重叠)
    - ar_enabled: True=AR插空执行, False=AR末尾同步执行
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist
import torch.nn.functional as F

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)

def p0(*args):
    if rank == 0:
        print(*args, flush=True)

# ============================================================
# 配置
# ============================================================
hidden_size = 4096
num_heads = 32
num_kv_heads = 32
ffn_hidden = 14336
num_experts = 8
top_k = 4
num_layers = 2
seq_local = 2048
batch_size = 4

chunks = 2
ar_enabled = True   # True=AR插空, False=AR末尾同步

p0("=" * 60)
p0("FluidMoE Benchmark")
p0("=" * 60)
p0(f"Config: hidden={hidden_size}, heads={num_heads}, ffn={ffn_hidden}")
p0(f"        experts={num_experts}, top_k={top_k}, layers={num_layers}")
p0(f"        seq_local={seq_local}, batch={batch_size}, GPUs={world_size}")
p0(f"        chunks={chunks}, ar_enabled={ar_enabled}")
p0("=" * 60)

from fluid.layer import TransformerModel
from baseline import BaselineTransformerModel
from fluid.core.scheduler import get_backward_scheduler

# 创建模型
baseline_model = BaselineTransformerModel(
    num_layers=num_layers, hidden_size=hidden_size,
    num_heads=num_heads, num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
    cp_group=dist.group.WORLD, ep_group=dist.group.WORLD,
    dtype=torch.bfloat16, device=device,
)

fluidmoe_model = TransformerModel(
    num_layers=num_layers, hidden_size=hidden_size,
    num_heads=num_heads, num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
    cp_group=dist.group.WORLD, ep_group=dist.group.WORLD,
    attn_proj_chunks=chunks, attn_qkv_chunks=chunks,
    moe_combine_chunks=chunks, moe_dispatch_chunks=chunks,
    dtype=torch.bfloat16, device=device,
)

# 启用调度器
scheduler = get_backward_scheduler()
scheduler.enable()
scheduler.configure_allreduce(
    enabled=ar_enabled,
    dp_group=dist.group.WORLD,
    ep_group=dist.group.WORLD,
)

x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
x_grad = x.clone().detach().requires_grad_(True)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
N = 5

# Baseline AR helper
def allreduce_model_grads(model):
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
for i in range(3):
    with torch.no_grad():
        baseline_model(x)
        fluidmoe_model(x)
    torch.cuda.synchronize()

for _ in range(2):
    x_grad.grad = None
    for p in baseline_model.parameters():
        p.grad = None
    baseline_model(x_grad).sum().backward()
    allreduce_model_grads(baseline_model)

    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    fluidmoe_model(x_grad).sum().backward()
    scheduler.finish_batch()
    scheduler.clear_iteration()
torch.cuda.synchronize()
p0("Warmup done.")

# ============================================================
# Forward Only
# ============================================================
start_event.record()
for _ in range(N):
    with torch.no_grad():
        baseline_model(x)
end_event.record()
torch.cuda.synchronize()
baseline_fwd = start_event.elapsed_time(end_event) / N

start_event.record()
for _ in range(N):
    with torch.no_grad():
        fluidmoe_model(x)
end_event.record()
torch.cuda.synchronize()
fluidmoe_fwd = start_event.elapsed_time(end_event) / N

# ============================================================
# Forward + Backward + AR
# ============================================================
start_event.record()
for _ in range(N):
    x_grad.grad = None
    for p in baseline_model.parameters():
        p.grad = None
    baseline_model(x_grad).sum().backward()
    allreduce_model_grads(baseline_model)
end_event.record()
torch.cuda.synchronize()
baseline_fwdbwd = start_event.elapsed_time(end_event) / N

# FluidMoE + no AR (measure pure fwd+bwd cost)
scheduler.clear_iteration()
scheduler.ar_enabled = False
orig_dp_world_size = scheduler.dp_world_size
scheduler.dp_world_size = 1  # skip AR entirely
for _ in range(2):
    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    fluidmoe_model(x_grad).sum().backward()
    scheduler.finish_batch()
torch.cuda.synchronize()
scheduler.clear_iteration()

start_event.record()
for _ in range(N):
    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    fluidmoe_model(x_grad).sum().backward()
    scheduler.finish_batch()
end_event.record()
torch.cuda.synchronize()
fluidmoe_noar = start_event.elapsed_time(end_event) / N
scheduler.clear_iteration()
scheduler.dp_world_size = orig_dp_world_size

# FluidMoE + sync AR (ar_enabled=False)
scheduler.clear_iteration()
scheduler.ar_enabled = False
for _ in range(2):  # warmup
    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    fluidmoe_model(x_grad).sum().backward()
    scheduler.finish_batch()
torch.cuda.synchronize()
scheduler.clear_iteration()

start_event.record()
for _ in range(N):
    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    fluidmoe_model(x_grad).sum().backward()
    scheduler.finish_batch()
end_event.record()
torch.cuda.synchronize()
fluidmoe_sync = start_event.elapsed_time(end_event) / N
scheduler.clear_iteration()

# FluidMoE + interleaved AR (ar_enabled=True)
scheduler.ar_enabled = True
for _ in range(2):  # warmup
    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    fluidmoe_model(x_grad).sum().backward()
    scheduler.finish_batch()
torch.cuda.synchronize()
scheduler.clear_iteration()

start_event.record()
for _ in range(N):
    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    fluidmoe_model(x_grad).sum().backward()
    scheduler.finish_batch()
end_event.record()
torch.cuda.synchronize()
fluidmoe_interleaved = start_event.elapsed_time(end_event) / N
scheduler.clear_iteration()

# ============================================================
# 输出
# ============================================================
p0(f"\nForward:     Baseline {baseline_fwd:.2f}ms  FluidMoE {fluidmoe_fwd:.2f}ms  Speedup {baseline_fwd/fluidmoe_fwd:.2f}x")
p0(f"Fwd+Bwd+AR:  Baseline {baseline_fwdbwd:.2f}ms")
p0(f"  sync AR:   FluidMoE {fluidmoe_sync:.2f}ms  Speedup {baseline_fwdbwd/fluidmoe_sync:.2f}x")
p0(f"  interl AR: FluidMoE {fluidmoe_interleaved:.2f}ms  Speedup {baseline_fwdbwd/fluidmoe_interleaved:.2f}x")

dist.destroy_process_group()
p0("Done!")
