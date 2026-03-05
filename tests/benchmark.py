"""
FluidMoE Benchmark

用法:
    torchrun --nproc_per_node=2 tests/benchmark.py   # dp=1, cp=2
    torchrun --nproc_per_node=4 tests/benchmark.py   # dp=2, cp=2
"""
import sys, os
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
# 配置
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

# 各 region 分块数 (R1=moe_combine, R2=moe_dispatch, R3=attn_proj, R4=attn_qkv)
moe_combine_chunks = 2
moe_dispatch_chunks = 2
attn_proj_chunks = 1
attn_qkv_chunks = 4

# Per-region AR trickle sizes (bytes), from tune.py BDP model.
# 0 = unlimited (submit all pending). Set by tune.py output.
ar_trickle_sizes = {
    'moe_combine': 185 * 1024 * 1024,
    'moe_dispatch': 147 * 1024 * 1024,
    'attn_proj': 24 * 1024 * 1024,
    'attn_qkv': 24 * 1024 * 1024,
}

# 并行度配置
dp_size = 1   # 数据并行度，dp > 1 时专家参数也需要 AR
cp_size = 2   # 上下文并行度（注意力 AllToAll）
ep_size = 2   # 专家并行度（MoE AllToAll），当前要求 ep = cp
num_gpus = dp_size * cp_size  # 参与的总 GPU 数

N_WARMUP = 5
N_ITER = 10

assert ep_size == cp_size, f"当前仅支持 ep=cp, 但 ep={ep_size}, cp={cp_size}"
assert world_size >= num_gpus, f"需要至少 {num_gpus} GPU, 但只有 {world_size}"
seq_local = seq_len // cp_size

# 多余的 rank 提前退出
if rank >= num_gpus:
    dist.barrier()
    dist.destroy_process_group()
    exit(0)

# 创建进程组
# all_group: 所有参与 rank（共享参数 AR 用，= dp + cp 两个维度的全集）
# cp_group = ep_group: 同一 dp 副本内的 rank（注意力/MoE AllToAll 用）
# dp_group: 持有相同专家分区的 rank（专家参数 AR 用，仅 dp > 1 时需要）
if dp_size == 1 and num_gpus == world_size:
    all_group = cp_group = ep_group = dp_group = dist.group.WORLD
else:
    # all_group: 所有参与的 rank
    if num_gpus == world_size:
        all_group = dist.group.WORLD
    else:
        all_group = dist.new_group(list(range(num_gpus)))
    # cp_group = ep_group: 连续 cp_size 个 rank
    for i in range(dp_size):
        ranks = list(range(i * cp_size, (i + 1) * cp_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            cp_group = ep_group = g
    # dp_group: 间隔 cp_size 的 rank（持有相同专家的 rank）
    for i in range(cp_size):
        ranks = list(range(i, num_gpus, cp_size))
        g = dist.new_group(ranks)
        if rank in ranks:
            dp_group = g

p0("=" * 60)
p0("FluidMoE Benchmark")
p0(f"  hidden={hidden_size}, heads={num_heads}, ffn={ffn_hidden}")
p0(f"  experts={num_experts}, top_k={top_k}, layers={num_layers}")
p0(f"  seq={seq_len}, batch={batch_size}, GPUs={num_gpus}")
p0(f"  dp={dp_size}, cp={cp_size}, ep={ep_size}")
p0(f"  chunks: combine={moe_combine_chunks}, dispatch={moe_dispatch_chunks}, "
   f"proj={attn_proj_chunks}, qkv={attn_qkv_chunks}")
p0("=" * 60)

from fluid.layer import TransformerModel
from baseline import BaselineTransformerModel
from fluid.core.scheduler import get_backward_scheduler

# 创建模型
baseline_model = BaselineTransformerModel(
    num_layers=num_layers, hidden_size=hidden_size,
    num_heads=num_heads, num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
    cp_group=cp_group, ep_group=ep_group,
    dtype=torch.bfloat16, device=device,
)

fluidmoe_model = TransformerModel(
    num_layers=num_layers, hidden_size=hidden_size,
    num_heads=num_heads, num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
    cp_group=cp_group, ep_group=ep_group,
    attn_proj_chunks=attn_proj_chunks, attn_qkv_chunks=attn_qkv_chunks,
    moe_combine_chunks=moe_combine_chunks, moe_dispatch_chunks=moe_dispatch_chunks,
    ar_trickle_sizes=ar_trickle_sizes,
    dtype=torch.bfloat16, device=device,
)

scheduler = get_backward_scheduler()
scheduler.enable()
scheduler.configure_allreduce(
    enabled=True,
    shared_dp_group=all_group,
)
fluidmoe_model.setup_ar_buffer()

x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
x_grad = x.clone().detach().requires_grad_(True)
ev_s = torch.cuda.Event(enable_timing=True)
ev_e = torch.cuda.Event(enable_timing=True)


def allreduce_grads(model):
    for layer in model.layers:
        # 共享参数: 所有 rank 都持有，AR 跨全部 rank
        for name in ('qkv_weight', 'proj_weight', 'router_weight',
                     'ln1_weight', 'ln1_bias', 'ln2_weight', 'ln2_bias'):
            p = getattr(layer, name, None)
            if p is not None and p.grad is not None:
                dist.all_reduce(p.grad, group=all_group)
        # 专家参数: 按 EP 分区，AR 仅跨持有相同专家的 dp rank
        if dp_size > 1:
            for name in ('moe_w1', 'moe_w2'):
                p = getattr(layer, name, None)
                if p is not None and p.grad is not None:
                    dist.all_reduce(p.grad, group=dp_group)


def run_baseline_bwd():
    x_grad.grad = None
    for p in baseline_model.parameters():
        p.grad = None
    baseline_model(x_grad).sum().backward()
    allreduce_grads(baseline_model)


def allreduce_expert_grads(model):
    """AR expert params across dp subgroup (only needed when dp > 1)."""
    if dp_size <= 1:
        return
    for layer in model.layers:
        for name in ('moe_w1', 'moe_w2'):
            p = getattr(layer, name, None)
            if p is not None and p.grad is not None:
                dist.all_reduce(p.grad, group=dp_group)


def run_fluid_bwd():
    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    fluidmoe_model(x_grad).sum().backward()
    scheduler.finish_batch()
    allreduce_expert_grads(fluidmoe_model)
    scheduler.clear_iteration()


def bench(run_fn, warmup=N_WARMUP, iters=N_ITER):
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()
    scheduler.clear_iteration()
    ev_s.record()
    for _ in range(iters):
        run_fn()
    ev_e.record()
    torch.cuda.synchronize()
    scheduler.clear_iteration()
    return ev_s.elapsed_time(ev_e) / iters


# ============================================================
# Warmup
# ============================================================
p0("Warmup...")
for _ in range(N_WARMUP):
    with torch.no_grad():
        baseline_model(x)
        fluidmoe_model(x)
torch.cuda.synchronize()
dist.barrier()
p0("Warmup done.")

# ============================================================
# Forward
# ============================================================
ev_s.record()
for _ in range(N_ITER):
    with torch.no_grad():
        baseline_model(x)
ev_e.record()
torch.cuda.synchronize()
baseline_fwd = ev_s.elapsed_time(ev_e) / N_ITER

ev_s.record()
for _ in range(N_ITER):
    with torch.no_grad():
        fluidmoe_model(x)
ev_e.record()
torch.cuda.synchronize()
fluidmoe_fwd = ev_s.elapsed_time(ev_e) / N_ITER

# ============================================================
# Backward + AR
# ============================================================
baseline_bwd = bench(run_baseline_bwd)

scheduler.ar_enabled = False
fluidmoe_sync = bench(run_fluid_bwd)

scheduler.ar_enabled = True
fluidmoe_interleaved = bench(run_fluid_bwd)

# ============================================================
# 输出
# ============================================================
p0(f"Forward:     Baseline {baseline_fwd:.2f}ms  FluidMoE {fluidmoe_fwd:.2f}ms  Speedup {baseline_fwd/fluidmoe_fwd:.2f}x")
p0(f"Fwd+Bwd+AR:  Baseline {baseline_bwd:.2f}ms")
p0(f"  sync AR:   FluidMoE {fluidmoe_sync:.2f}ms  Speedup {baseline_bwd/fluidmoe_sync:.2f}x")
p0(f"  interl AR: FluidMoE {fluidmoe_interleaved:.2f}ms  Speedup {baseline_bwd/fluidmoe_interleaved:.2f}x")

dist.destroy_process_group()
p0("Done!")
