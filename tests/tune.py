"""
FluidMoE 运行时调优

方法: 理论筛选 + 定向搜索
  Phase 1: 快速 profiling (chunks=2), 计算每个区域的 comm_ratio
           comm_ratio = T_comm / (T_comp + T_dW)
           ratio < 1 → 计算覆盖通信, 锁定 chunks=2
           ratio > 1 → 通信瓶颈, 标记为需要搜索
  Phase 2: 对瓶颈区域逐个尝试 chunks=4, 端到端计时比较
  Phase 3: 用最优 chunks 测量 AllToAll gap, 确定 AR chunk size

用法: torchrun --nproc_per_node=2 tests/tune.py
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
# 模型配置
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
N = 5

from fluid.layer import TransformerModel
from fluid.core.scheduler import get_backward_scheduler

x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
scheduler = get_backward_scheduler()

def make_model(mc, md, ap, aq):
    return TransformerModel(
        num_layers=num_layers, hidden_size=hidden_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
        cp_group=dist.group.WORLD, ep_group=dist.group.WORLD,
        attn_proj_chunks=ap, attn_qkv_chunks=aq,
        moe_combine_chunks=mc, moe_dispatch_chunks=md,
        dtype=torch.bfloat16, device=device,
    )

def benchmark(mc, md, ap, aq, ar_mb=16, label="", warmup_fwd=3, warmup_bwd=2):
    """端到端 benchmark, 返回平均 Fwd+Bwd+AR 时间 (ms)."""
    model = make_model(mc, md, ap, aq)
    scheduler.enable()
    scheduler.configure_allreduce(enabled=True, dp_group=dist.group.WORLD)
    scheduler.ar_chunk_size = ar_mb * 1024 * 1024

    xg = x.clone().detach().requires_grad_(True)
    for _ in range(warmup_fwd):
        with torch.no_grad():
            model(x)
    for _ in range(warmup_bwd):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        scheduler.finish_batch()
    torch.cuda.synchronize()
    scheduler.clear_iteration()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        scheduler.finish_batch()
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / N
    scheduler.clear_iteration()

    cleanup_model(model)
    del xg
    if label:
        p0(f"  {label}: {t:.2f}ms")
    return t

def profile_run(model):
    """Profile with no AR, return region profiles and wall time."""
    scheduler.enable()
    scheduler.configure_allreduce(enabled=True, dp_group=dist.group.WORLD)
    scheduler.dp_world_size = 1  # 跳过 AR

    xg = x.clone().detach().requires_grad_(True)
    for _ in range(3):
        with torch.no_grad():
            model(x)
    for _ in range(2):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        scheduler.finish_batch()
    torch.cuda.synchronize()
    scheduler.clear_iteration()

    scheduler.profiling = True
    scheduler._region_profiles.clear()
    for _ in range(N):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        scheduler.finish_batch()
    torch.cuda.synchronize()
    profiles = scheduler.get_region_profiles()
    scheduler.profiling = False
    scheduler.clear_iteration()

    # Wall time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        scheduler.finish_batch()
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / N
    scheduler.clear_iteration()

    scheduler.dp_world_size = dist.get_world_size()
    del xg
    return profiles, t

def cleanup_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

region_names = ['moe_combine', 'moe_dispatch', 'attn_proj', 'attn_qkv']

# ============================================================
# Phase 1: 快速 profiling, 计算 comm_ratio
# ============================================================
p0("=" * 60)
p0("Phase 1: Profile (chunks=2, no AR) → comm_ratio per region")
p0("=" * 60)

model = make_model(2, 2, 2, 2)
profiles, t_base = profile_run(model)
cleanup_model(model)

p0(f"\n  {'Region':>15s}  {'T_comm':>7s}  {'T_comp':>7s}  {'T_dW':>7s}  {'ratio':>6s}  {'Status'}")
bottleneck_regions = []
for name in region_names:
    if name not in profiles:
        continue
    p = profiles[name]
    cover = p['T_comp'] + p['T_dW']
    ratio = p['T_comm'] / cover if cover > 0.1 else 0.0
    is_bottleneck = ratio > 1.0
    status = "BOTTLENECK → search" if is_bottleneck else "OK → chunks=2"
    p0(f"  {name:>15s}  {p['T_comm']:>6.2f}ms  {p['T_comp']:>6.2f}ms  {p['T_dW']:>6.2f}ms  {ratio:>5.2f}  {status}")
    if is_bottleneck:
        bottleneck_regions.append(name)

p0(f"\n  Fwd+Bwd (no AR): {t_base:.2f}ms")
p0(f"  Bottleneck regions: {bottleneck_regions if bottleneck_regions else 'none'}")

# ============================================================
# Phase 2: 定向搜索瓶颈区域
# ============================================================
p0(f"\n{'=' * 60}")
p0("Phase 2: Targeted search for bottleneck regions")
p0("=" * 60)

# 基线: 全部 chunks=2
best_config = {'moe_combine': 2, 'moe_dispatch': 2, 'attn_proj': 2, 'attn_qkv': 2}
name_to_arg = {
    'moe_combine': 'mc', 'moe_dispatch': 'md',
    'attn_proj': 'ap', 'attn_qkv': 'aq',
}

if not bottleneck_regions:
    p0("  No bottleneck regions, keeping all chunks=2")
    t_best = benchmark(2, 2, 2, 2, 16, "baseline (2,2,2,2)")
else:
    t_best = benchmark(2, 2, 2, 2, 16, "baseline (2,2,2,2)")

    for name in bottleneck_regions:
        # 尝试 chunks=4 for this region
        test_config = best_config.copy()
        test_config[name] = 4
        mc = test_config['moe_combine']
        md = test_config['moe_dispatch']
        ap = test_config['attn_proj']
        aq = test_config['attn_qkv']

        label = f"{name}=4, rest={2} → ({mc},{md},{ap},{aq})"
        t = benchmark(mc, md, ap, aq, 16, label)

        if t < t_best - 1.0:  # 至少快 1ms 才切换 (避免噪声)
            p0(f"    → {name}=4 is better by {t_best - t:.1f}ms, adopting")
            best_config[name] = 4
            t_best = t
        else:
            p0(f"    → {name}=4 not better (delta={t_best - t:+.1f}ms), keeping chunks=2")

mc = best_config['moe_combine']
md = best_config['moe_dispatch']
ap = best_config['attn_proj']
aq = best_config['attn_qkv']
p0(f"\n  Best chunks: ({mc}, {md}, {ap}, {aq})")

# ============================================================
# Phase 3: AR chunk size (测量 AllToAll gap)
# ============================================================
p0(f"\n{'=' * 60}")
p0("Phase 3: Determine AR chunk size from AllToAll gaps")
p0("=" * 60)

model = make_model(mc, md, ap, aq)
scheduler.enable()
scheduler.configure_allreduce(enabled=True, dp_group=dist.group.WORLD)
scheduler.dp_world_size = 1  # no AR for clean gap measurement
xg = x.clone().detach().requires_grad_(True)

# Warmup
for _ in range(3):
    with torch.no_grad():
        model(x)
for _ in range(2):
    xg.grad = None
    for p in model.parameters():
        p.grad = None
    model(xg).sum().backward()
    scheduler.finish_batch()
torch.cuda.synchronize()
scheduler.clear_iteration()

# Collect gaps
scheduler._gap_times = []
for _ in range(N):
    xg.grad = None
    for p in model.parameters():
        p.grad = None
    model(xg).sum().backward()
    scheduler.finish_batch()
torch.cuda.synchronize()
gap_times = scheduler._gap_times.copy()
scheduler.dp_world_size = dist.get_world_size()
scheduler.clear_iteration()
cleanup_model(model)
del xg

ar_chunk_mb = 16  # default
if gap_times:
    bwd_gaps = [g for g in gap_times if g < 100]
    avg_bwd_gap = sum(bwd_gaps) / max(len(bwd_gaps), 1) if bwd_gaps else 0
    min_bwd_gap = min(bwd_gaps) if bwd_gaps else 0
    p0(f"  Backward gaps: {len(bwd_gaps)}, avg: {avg_bwd_gap:.2f}ms, min: {min_bwd_gap:.2f}ms")

    bw_gbps = 20  # conservative NVLink estimate
    if min_bwd_gap > 0.5:
        max_chunk_bytes = int(min_bwd_gap / 1000 * bw_gbps * 1e9)
        max_chunk_mb = max_chunk_bytes / (1024 * 1024)
        ar_chunk_mb = 2 ** int(math.log2(max(1, max_chunk_mb)))
        ar_chunk_mb = max(1, min(ar_chunk_mb, 64))
    p0(f"  AR chunk size: {ar_chunk_mb}MB")
else:
    p0("  No gap data, using default: 16MB")

# ============================================================
# Validation
# ============================================================
p0(f"\n{'=' * 60}")
p0("Validation")
p0("=" * 60)

t_default = benchmark(2, 2, 2, 2, 16, "default  (2,2,2,2 ar=16MB)")
t_tuned = benchmark(mc, md, ap, aq, ar_chunk_mb,
    f"tuned    ({mc},{md},{ap},{aq} ar={ar_chunk_mb}MB)")

diff = t_default - t_tuned
p0(f"\n  Delta: {diff:+.2f}ms ({diff/t_default*100:+.1f}%)")

# ============================================================
# 最终输出
# ============================================================
p0(f"\n{'=' * 60}")
p0("Final Configuration")
p0("=" * 60)
p0(f"  moe_combine_chunks  = {mc}")
p0(f"  moe_dispatch_chunks = {md}")
p0(f"  attn_proj_chunks    = {ap}")
p0(f"  attn_qkv_chunks     = {aq}")
p0(f"  ar_chunk_size        = {ar_chunk_mb}MB")

dist.destroy_process_group()
p0("Done!")
