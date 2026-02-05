"""
FluidMoE AR Chunk Size 搜索

搜索最优的 AllReduce 分块大小:
  1. 测量 AR bandwidth
  2. 测量 AllToAll 之间的空闲间隔
  3. 计算理论最优 chunk size
  4. 实测验证

用法:
    torchrun --nproc_per_node=2 tests/ar_tune.py
"""
import sys, os, gc, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)

def p0(*args):
    if rank == 0:
        print(*args, flush=True)


# ============================================================
# 配置 (可修改, 应与 chunk_tune.py 一致)
# ============================================================
hidden_size = 2048
num_heads = 16
num_kv_heads = 16
ffn_hidden = 14336
num_experts = 8
top_k = 4
seq_local = 2048
batch_size = 4


# ============================================================
# 主流程
# ============================================================
p0("=" * 60)
p0("FluidMoE AR Chunk Size 搜索")
p0(f"GPUs={world_size}")
p0("=" * 60)


# ============================================================
# Step 1: 测量 AR Bandwidth
# ============================================================
p0("\n=== Step 1: AR Bandwidth 测量 ===")

chunk_sizes_mb = [1, 4, 16, 64]
ar_times = {}

for size_mb in chunk_sizes_mb:
    size_bytes = int(size_mb * 1024 * 1024)
    num_elements = size_bytes // 2
    tensor = torch.randn(num_elements, dtype=torch.bfloat16, device=device)

    # warmup
    for _ in range(3):
        dist.all_reduce(tensor, group=dist.group.WORLD)
    torch.cuda.synchronize()

    # measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        dist.all_reduce(tensor, group=dist.group.WORLD)
    end.record()
    torch.cuda.synchronize()

    t = start.elapsed_time(end) / 10
    ar_times[size_mb] = t
    bandwidth = size_bytes / (t / 1000) / 1e9
    p0(f"  {size_mb:>5.1f} MB: {t:.3f} ms  ({bandwidth:.1f} GB/s)")

    del tensor
    torch.cuda.empty_cache()

# 估算 bandwidth (MB/ms)
s1, t1 = chunk_sizes_mb[0], ar_times[chunk_sizes_mb[0]]
s2, t2 = chunk_sizes_mb[-1], ar_times[chunk_sizes_mb[-1]]
B_est = (s2 - s1) / (t2 - t1)
t_launch = max(0, t1 - s1 / B_est)

p0(f"\n  Bandwidth ≈ {B_est:.2f} MB/ms")
p0(f"  Launch overhead ≈ {t_launch:.3f} ms")


# ============================================================
# Step 2: 测量 AllToAll 间隔
# ============================================================
p0("\n=== Step 2: AllToAll 间隔测量 ===")

from fluid.layer import TransformerModel
from fluid.core.scheduler import get_backward_scheduler

model_kwargs = dict(
    num_layers=1, hidden_size=hidden_size,
    num_heads=num_heads, num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
    cp_group=dist.group.WORLD, ep_group=dist.group.WORLD,
)

sched = get_backward_scheduler()

# 捕获 AllToAll events
a2a_events = []
_orig_wait = sched.wait_alltoall

def capturing_wait(task_id, num_tasks=1):
    task_data = sched._alltoall_results.get(task_id)
    if task_data is not None:
        done_event, result_holder = task_data
        done_event.wait()
        if len(result_holder) > 2:
            end_ev = result_holder[1]
            start_ev = result_holder[2]
            a2a_events.append((start_ev, end_ev))
    return _orig_wait(task_id, num_tasks)

sched.wait_alltoall = capturing_wait
sched.clear_iteration()

model = TransformerModel(
    **model_kwargs,
    attn_proj_chunks=1, attn_qkv_chunks=1,
    moe_combine_chunks=1, moe_dispatch_chunks=1,
    dtype=torch.bfloat16, device=device,
)
sched.enable()
sched.profiling = True
sched.configure_allreduce(enabled=True, dp_group=dist.group.WORLD)
sched.dp_world_size = 1

x_input = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
xg = x_input.clone().detach().requires_grad_(True)

# warmup
for _ in range(2):
    with torch.no_grad():
        model(x_input)
for _ in range(2):
    xg.grad = None
    for p in model.parameters():
        p.grad = None
    a2a_events.clear()
    model(xg).sum().backward()
    sched.finish_batch()
torch.cuda.synchronize()
sched.clear_iteration()

# 测量一次 backward
a2a_events.clear()
xg.grad = None
for p in model.parameters():
    p.grad = None
model(xg).sum().backward()
sched.finish_batch()
torch.cuda.synchronize()

p0(f"\n  检测到 {len(a2a_events)} 个 AllToAll")

# 计算间隔
if len(a2a_events) >= 2:
    a2a_times = []
    p0(f"\n  AllToAll 执行时间:")
    for i, (start_ev, end_ev) in enumerate(a2a_events):
        t = start_ev.elapsed_time(end_ev)
        a2a_times.append(t)
        p0(f"    A2A[{i}]: {t:.2f} ms")

    # gap = cycle_time - a2a_time
    p0(f"\n  AllToAll 间隔 (cycle - a2a_time):")
    gaps = []
    for i in range(len(a2a_events) - 1):
        _, end_ev_i = a2a_events[i]
        _, end_ev_j = a2a_events[i + 1]
        cycle = end_ev_i.elapsed_time(end_ev_j)
        gap = cycle - a2a_times[i + 1]
        gaps.append(gap)
        p0(f"    A2A[{i}]→A2A[{i+1}]: {cycle:.2f} - {a2a_times[i+1]:.2f} = {gap:.2f} ms")

    min_gap = min(gaps)
    total_gap = sum(gaps)
    p0(f"\n  最小间隔: {min_gap:.2f} ms")
    p0(f"  总间隔: {total_gap:.2f} ms")
else:
    min_gap = 0
    total_gap = 0
    gaps = []
    p0("  AllToAll 数量不足")

sched.wait_alltoall = _orig_wait
sched.profiling = False


# ============================================================
# Step 3: 计算最优 AR Chunk Size
# ============================================================
p0("\n=== Step 3: 最优 AR Chunk Size ===")

total_params = sum(p.numel() for p in model.parameters())
total_mb = total_params * 2 / 1024 / 1024
p0(f"  模型参数: {total_params/1e6:.1f}M, {total_mb:.1f} MB")

if min_gap > t_launch:
    S_max = B_est * (min_gap - t_launch)
    S_max = max(0.5, S_max)

    p0(f"\n  约束: AR chunk ≤ 最小间隔 ({min_gap:.2f} ms)")
    p0(f"  S_max = {B_est:.2f} × ({min_gap:.2f} - {t_launch:.3f}) = {S_max:.1f} MB")

    # 取 2 的幂
    S_optimal = 2 ** int(math.log2(S_max)) if S_max >= 1 else 1
    S_optimal = max(1, min(64, S_optimal))
    p0(f"  建议: {S_optimal} MB")
else:
    S_optimal = 1
    p0(f"  最小间隔太小，使用默认 1 MB")


# ============================================================
# Step 4: 实测验证
# ============================================================
p0("\n=== Step 4: 实测验证 ===")

del model, xg
gc.collect()
torch.cuda.empty_cache()
dist.barrier()

chunk_sizes_test = [2, 4, 8, 16]

for chunk_mb in chunk_sizes_test:
    dist.barrier()

    sched = get_backward_scheduler()
    sched.clear_iteration()
    sched.ar_chunk_size = chunk_mb * 1024 * 1024
    sched.profiling = False

    model = TransformerModel(
        **model_kwargs,
        attn_proj_chunks=1, attn_qkv_chunks=1,
        moe_combine_chunks=1, moe_dispatch_chunks=1,
        dtype=torch.bfloat16, device=device,
    )
    sched.enable()
    sched.configure_allreduce(enabled=True, dp_group=dist.group.WORLD)
    sched.ar_enabled = True

    xg = x_input.clone().detach().requires_grad_(True)

    # warmup
    for _ in range(2):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        sched.finish_batch()
    torch.cuda.synchronize()
    dist.barrier()
    sched.clear_iteration()

    # measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(3):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        sched.finish_batch()
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / 3

    marker = " <-- optimal" if chunk_mb == S_optimal else ""
    p0(f"  AR chunk = {chunk_mb:>2d} MB: {t:.2f} ms{marker}")

    del model, xg
    sched.clear_iteration()
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================
# 结果
# ============================================================
p0(f"\n{'=' * 60}")
p0("结果")
p0("=" * 60)
p0(f"  AR Bandwidth: {B_est:.2f} MB/ms")
p0(f"  最小 AllToAll 间隔: {min_gap:.2f} ms")
p0(f"  推荐 AR chunk size: {S_optimal} MB")
p0(f"\n  设置方式:")
p0(f"    scheduler.ar_chunk_size = {S_optimal} * 1024 * 1024")
p0(f"  或环境变量:")
p0(f"    export FLUID_AR_CHUNK_SIZE={S_optimal * 1024 * 1024}")

dist.barrier()
dist.destroy_process_group()
p0("\nDone!")
