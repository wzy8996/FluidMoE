"""
FluidMoE 完整参数搜索

搜索顺序:
  1. 同步 AR，逐 region 贪心搜索最优分块数 (R1~R4)
  2. BDP 模型自动计算最优 AR trickle size
  3. 网格搜索验证 BDP 预测

用法:
    torchrun --nproc_per_node=2 tools/tune.py
"""
import sys, os, gc
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
# 配置 (可修改, 应与 benchmark.py 一致)
# ============================================================
hidden_size = 4096
num_heads = 32
num_kv_heads = 32
ffn_hidden = 14336
num_experts = 8
top_k = 4
seq_local = 2048
batch_size = 4

# Chunk 搜索参数
N_WARMUP = 5            # 预热轮数 (forward+backward)
N_ITER = 10             # 测量轮数
MAX_C = 8               # 最大分块数
STOP_MIN_SAVING_MS = 0.5  # 早停阈值 (ms)

# AR 搜索参数
N_AR_WARMUP = 5         # 预热轮数
N_AR_ITER = 10          # 测量轮数

# ============================================================
# 初始化
# ============================================================
from fluid.layer import TransformerModel
from fluid.layer.transformer import TransformerLayerFunction
from fluid.core.scheduler import get_backward_scheduler, BackwardScheduler

p0("=" * 60)
p0("FluidMoE 完整参数搜索")
p0("=" * 60)
p0(f"Config: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}")
p0(f"        ffn={ffn_hidden}, experts={num_experts}, top_k={top_k}")
p0(f"        seq_local={seq_local}, batch={batch_size}, GPUs={world_size}")
p0(f"Chunk 搜索: warmup={N_WARMUP}, iter={N_ITER}, max_C={MAX_C}, stop<{STOP_MIN_SAVING_MS} ms")
p0(f"AR 搜索:    warmup={N_AR_WARMUP}, iter={N_AR_ITER}")
p0("=" * 60)

model_kwargs = dict(
    num_layers=2, hidden_size=hidden_size,
    num_heads=num_heads, num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
    cp_group=dist.group.WORLD, ep_group=dist.group.WORLD,
)

# 固定输入 (所有步骤共用)
x_input = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)


def get_valid_chunks(size, max_C=MAX_C):
    return [c for c in range(1, min(size, max_C) + 1) if size % c == 0]


# 在任何训练开始前调用 configure_allreduce，创建 ar_group
# (dist.new_group 是集合操作，必须在两个 rank 完全同步时调用一次)
# enabled=False: Step 1 使用同步 AR 模式 (ar_enabled=False + dp_world_size=world_size)
# Step 2/3 设 ar_enabled=True 切换为插入 AR 模式
sched = get_backward_scheduler()
sched.configure_allreduce(enabled=False, dp_group=dist.group.WORLD)


# ============================================================
# Step 1: Chunk 搜索 (无 AR)
# ============================================================
p0(f"\n{'=' * 60}")
p0("Step 1: Chunk 搜索 (同步 AR)")
p0("=" * 60)

valid_moe = get_valid_chunks(hidden_size)
valid_attn = get_valid_chunks(seq_local)
p0(f"有效因数: MoE(R1,R2)={valid_moe}, Attn(R3,R4)={valid_attn}")

sched.clear_iteration()
chunk_model = TransformerModel(
    **model_kwargs,
    moe_combine_chunks=1, moe_dispatch_chunks=1,
    attn_proj_chunks=1, attn_qkv_chunks=1,
    dtype=torch.bfloat16, device=device,
)
sched.enable()  # ar_enabled=False + dp_world_size=world_size → 同步 AR 模式
chunk_model.setup_ar_buffer()

xg = x_input.clone().detach().requires_grad_(True)


def bench_chunk(mc, md, ap, aq):
    """测量一个 chunk 配置的 backward 时间 (完整 forward+backward，固定输入)。"""
    # 直接修改模型各层的 chunk 属性，forward 时会传入 ctx 供 backward 使用
    for layer in chunk_model.layers:
        layer.moe_combine_chunks = mc
        layer.moe_dispatch_chunks = md
        layer.attn_proj_chunks = ap
        layer.attn_qkv_chunks = aq
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(N_ITER):
        xg.grad = None
        for p in chunk_model.parameters():
            p.grad = None
        loss = chunk_model(xg).sum()
        start.record()
        loss.backward()
        sched.finish_batch()
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
        sched.clear_iteration()
    return total / N_ITER


# 预热
p0("预热中...")
for _ in range(N_WARMUP):
    xg.grad = None
    for p in chunk_model.parameters():
        p.grad = None
    chunk_model(xg).sum().backward()
    sched.finish_batch()
torch.cuda.synchronize()
sched.clear_iteration()

# Baseline C=(1,1,1,1)
T_baseline = bench_chunk(1, 1, 1, 1)
p0(f"\nBaseline C=(1,1,1,1): {T_baseline:.3f} ms")

bench_data = {(1, 1, 1, 1): T_baseline}

# 贪心逐 region 扫描
region_names = ['R1 (moe_combine)', 'R2 (moe_dispatch)', 'R3 (attn_proj)', 'R4 (attn_qkv)']
region_valid = [valid_moe, valid_moe, valid_attn, valid_attn]
best_per_region = [1, 1, 1, 1]
best_speedup_per_region = [1.0, 1.0, 1.0, 1.0]

for ri in range(4):
    p0(f"\n  {region_names[ri]}")
    best_c = 1
    best_t_region = T_baseline
    prev_t = T_baseline

    for c in region_valid[ri]:
        cfg = list(best_per_region)
        cfg[ri] = c
        cfg_tuple = tuple(cfg)
        if cfg_tuple in bench_data:
            t = bench_data[cfg_tuple]
        else:
            t = bench_chunk(*cfg_tuple)
            bench_data[cfg_tuple] = t

        saving = T_baseline - t
        marginal = prev_t - t
        speedup = T_baseline / t

        marker = ""
        if t < best_t_region:
            best_t_region = t
            best_c = c
            marker = " <-- best"

        if c == 1:
            p0(f"    C={c}: {t:.3f} ms  (saving={saving:+.3f} ms, speedup={speedup:.4f}x){marker}")
        else:
            p0(f"    C={c}: {t:.3f} ms  (saving={saving:+.3f} ms, marginal={marginal:+.3f} ms, speedup={speedup:.4f}x){marker}")

        if c > 1 and marginal < STOP_MIN_SAVING_MS:
            p0(f"    >>> 早停 (marginal {marginal:+.3f} ms < 阈值 {STOP_MIN_SAVING_MS} ms)")
            break

        prev_t = t

    best_per_region[ri] = best_c
    best_speedup_per_region[ri] = T_baseline / best_t_region
    p0(f"  最优: C={best_c}, {best_t_region:.3f} ms  (speedup={best_speedup_per_region[ri]:.4f}x)")

final_cfg = tuple(best_per_region)
T_chunk_best = bench_data[final_cfg]
if T_baseline < T_chunk_best:
    p0(f"\n  Baseline ({T_baseline:.3f} ms) 优于最终配置 ({T_chunk_best:.3f} ms), 回退到 C=(1,1,1,1)")
    final_cfg = (1, 1, 1, 1)
    T_chunk_best = T_baseline

best_chunks = final_cfg
p0(f"\n最优分块配置: C={best_chunks}, {T_chunk_best:.3f} ms  (speedup={T_baseline/T_chunk_best:.4f}x)")

del chunk_model, xg
gc.collect()
torch.cuda.empty_cache()
sched.clear_iteration()
dist.barrier()  # 确保两个 rank 同步后再进入 Step 2


# ============================================================
# Step 2: 分区域 BDP 模型计算最优 AR trickle size
# ============================================================
p0(f"\n{'=' * 60}")
p0("Step 2: 分区域 BDP 模型计算最优 AR trickle size")
p0("=" * 60)
p0(f"  (使用最优分块配置 C={best_chunks})")

REGION_ORDER = ['moe_combine', 'moe_dispatch', 'attn_proj', 'attn_qkv']
REGION_LABELS = {
    'moe_combine':  'R1 (moe_combine)',
    'moe_dispatch': 'R2 (moe_dispatch)',
    'attn_proj':    'R3 (attn_proj)',
    'attn_qkv':     'R4 (attn_qkv)',
}
# Gap description: what compute fills the gap after each region
REGION_GAP_DESC = {
    'moe_combine':  'R1→R2: last FC2 dX + act_bwd + FC1 dX(c0)',
    'moe_dispatch': 'R2→R3: permute + router + LN2 dX + proj dX(c0)',
    'attn_proj':    'R3→R4: SDPA bwd + QKV reassemble',
    'attn_qkv':     'R4→R1: last QKV dX + LN1 dX + next layer',
}

# --- 2a: 测量 AR 带宽 ---
p0(f"\n  2a. 测量 AllReduce 带宽...")
ar_group = sched.ar_group if sched.ar_group is not None else sched.dp_group
bw_result = BackwardScheduler.measure_ar_bandwidth(
    ar_group=ar_group,
    sizes_mb=(1, 2, 4, 8, 16, 32, 64, 96, 128),
    warmup=5, repeat=20,
)
p0(f"  AR 带宽 profile:")
for i, sz in enumerate(bw_result['sizes_mb']):
    p0(f"    {sz:>4d} MB: {bw_result['bw_GBps'][i]:.2f} GB/s  "
       f"(latency {bw_result['latency_ms'][i]:.3f} ms)")
p0(f"  Peak BW: {bw_result['peak_bw_GBps']:.2f} GB/s")

# --- 2b: Profile per-region trickle windows (无 AR，纯净 gap) ---
p0(f"\n  2b. Profile 分区域 trickle windows (ar_enabled=False, 纯净 gap)...")
sched.clear_iteration()
ar_model = TransformerModel(
    **model_kwargs,
    moe_combine_chunks=best_chunks[0],
    moe_dispatch_chunks=best_chunks[1],
    attn_proj_chunks=best_chunks[2],
    attn_qkv_chunks=best_chunks[3],
    dtype=torch.bfloat16, device=device,
)
sched.enable()
sched.ar_enabled = False  # 无 AR，测纯净 gap
sched.ar_trickle_sizes = {}
ar_model.setup_ar_buffer()

xg_ar = x_input.clone().detach().requires_grad_(True)

# 预热
for _ in range(N_AR_WARMUP):
    xg_ar.grad = None
    for p in ar_model.parameters():
        p.grad = None
    ar_model(xg_ar).sum().backward()
    sched.finish_batch()
    sched.clear_iteration()
torch.cuda.synchronize()
sched.reset_gap_times()

# 采集 per-region gap times (CUDA events) — 无 AR 干扰
for _ in range(N_AR_ITER):
    xg_ar.grad = None
    for p in ar_model.parameters():
        p.grad = None
    ar_model(xg_ar).sum().backward()
    sched.finish_batch()
    sched.clear_iteration()
torch.cuda.synchronize()
sched.process_gap_events()  # convert CUDA event pairs → gap_ms

# --- 2c: 分区域 BDP 计算 (使用 size-dependent BW) ---
bdp_result = sched.compute_bdp_trickle_size(
    bw_profile=bw_result,
    percentile=10.0,
    safety_factor=0.9,
)

p0(f"\n  2c. 分区域 BDP 分析 (BW_AR = {bdp_result['bw_GBps']:.2f} GB/s, "
   f"safety = {bdp_result['safety_factor']}):")
p0(f"  {'Region':<22s} {'Gap description':<42s} {'T_p10':>8s} {'T_mean':>8s} {'BDP_MB':>8s} {'BW':>8s} {'n':>4s}")
p0(f"  {'-'*22} {'-'*42} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

bdp_per_region = {}  # region -> MB
for region in REGION_ORDER:
    rdata = bdp_result['per_region'].get(region)
    if rdata is None:
        p0(f"  {REGION_LABELS[region]:<22s} {'(no data)':<42s}")
        bdp_per_region[region] = 0
        continue
    label = REGION_LABELS[region]
    gap_desc = REGION_GAP_DESC[region]
    T_pct = rdata['T_window_percentile_ms']
    T_mean = rdata['T_window_mean_ms']
    bdp_mb = rdata['trickle_size_MB']
    bw_used = rdata.get('bw_GBps_used', bdp_result['bw_GBps'])
    n = rdata['n_windows']
    bdp_per_region[region] = bdp_mb
    p0(f"  {label:<22s} {gap_desc:<42s} {T_pct:>7.3f}ms {T_mean:>7.3f}ms {bdp_mb:>7d}MB {bw_used:>6.2f}G {n:>4d}")

# 也显示未预期的 region（如有）
for region, rdata in bdp_result['per_region'].items():
    if region not in REGION_ORDER:
        p0(f"  {region:<22s} {'(unexpected)':<42s} "
           f"{rdata['T_window_percentile_ms']:>7.3f}ms "
           f"{rdata['T_window_mean_ms']:>7.3f}ms "
           f"{rdata['trickle_size_MB']:>7d}MB "
           f"{rdata['n_windows']:>4d}")
        bdp_per_region[region] = rdata['trickle_size_MB']

dist.barrier()


# ============================================================
# Step 3: 网格搜索验证 (per-region BDP)
# ============================================================
p0(f"\n{'=' * 60}")
p0("Step 3: 网格搜索验证 (统一 trickle_size 对比 per-region BDP)")
p0("=" * 60)

grid_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 0]

trickle_bench = []
ev_s = torch.cuda.Event(enable_timing=True)
ev_e = torch.cuda.Event(enable_timing=True)


def bench_ar(label, setup_fn):
    """Benchmark a trickle configuration."""
    setup_fn()
    sched.clear_iteration()
    # 预热
    for _ in range(N_AR_WARMUP):
        xg_ar.grad = None
        for p in ar_model.parameters():
            p.grad = None
        ar_model(xg_ar).sum().backward()
        sched.finish_batch()
        sched.clear_iteration()
    torch.cuda.synchronize()
    # 测量
    total = 0.0
    for _ in range(N_AR_ITER):
        xg_ar.grad = None
        for p in ar_model.parameters():
            p.grad = None
        loss = ar_model(xg_ar).sum()
        ev_s.record()
        loss.backward()
        sched.finish_batch()
        ev_e.record()
        torch.cuda.synchronize()
        total += ev_s.elapsed_time(ev_e)
        sched.clear_iteration()
    return total / N_AR_ITER


# 3a: 统一 trickle size 网格搜索
p0(f"\n  3a. 统一 trickle_size 网格搜索:")
for trickle_mb in grid_sizes:
    def _setup(mb=trickle_mb):
        b = mb * 1024 * 1024
        sched.ar_trickle_sizes = {r: b for r in REGION_ORDER}
    t = bench_ar(f"{trickle_mb}MB", _setup)
    trickle_bench.append((trickle_mb, t))
    label = "unlimited" if trickle_mb == 0 else f"{trickle_mb} MB"
    p0(f"    trickle_size = {label:>12s}: {t:.2f} ms")

grid_best_mb, grid_best_t = min(trickle_bench, key=lambda x: x[1])

# 3b: Per-region BDP 配置
p0(f"\n  3b. Per-region BDP 配置:")
bdp_sizes_bytes = {r: mb * 1024 * 1024 for r, mb in bdp_per_region.items()}

def _setup_bdp():
    sched.ar_trickle_sizes = bdp_sizes_bytes

t_bdp = bench_ar("per-region BDP", _setup_bdp)
for region in REGION_ORDER:
    mb = bdp_per_region.get(region, 0)
    label = "unlimited" if mb == 0 else f"{mb} MB"
    p0(f"    {REGION_LABELS[region]}: {label}")
p0(f"    --> {t_bdp:.2f} ms")

del ar_model, xg_ar
gc.collect()
torch.cuda.empty_cache()
dist.barrier()

# 汇总
p0(f"\n  汇总:")
p0(f"    {'Config':<30s} {'Time':>10s} {'vs best':>10s}")
p0(f"    {'-'*30} {'-'*10} {'-'*10}")
for trickle_mb, t in trickle_bench:
    label = "unlimited" if trickle_mb == 0 else f"uniform {trickle_mb} MB"
    delta = (t - grid_best_t) / grid_best_t * 100
    marker = " <-- grid best" if trickle_mb == grid_best_mb else ""
    p0(f"    {label:<30s} {t:>9.2f}ms {delta:>+9.1f}%{marker}")

delta_bdp = (t_bdp - grid_best_t) / grid_best_t * 100
p0(f"    {'per-region BDP':<30s} {t_bdp:>9.2f}ms {delta_bdp:>+9.1f}%")
if abs(delta_bdp) < 2.0:
    p0(f"    VALIDATED: per-region BDP 与网格搜索一致 (< 2%)")


# ============================================================
# 最终结果
# ============================================================
p0(f"\n{'=' * 60}")
p0("最终结果")
p0("=" * 60)
p0(f"\n  分块配置:")
p0(f"    Baseline C=(1,1,1,1): {T_baseline:.3f} ms")
p0(f"    最优配置 C={best_chunks}: {T_chunk_best:.3f} ms  (speedup={T_baseline/T_chunk_best:.4f}x)")
for ri, name in enumerate(region_names):
    p0(f"    {name}: C={best_per_region[ri]}  (speedup={best_speedup_per_region[ri]:.4f}x)")

p0(f"\n  Per-region AR trickle size (BDP, size-dependent BW):")
p0(f"    safety = {bdp_result['safety_factor']}")
for region in REGION_ORDER:
    rdata = bdp_result['per_region'].get(region)
    mb = bdp_per_region.get(region, 0)
    label = "unlimited" if mb == 0 else f"{mb} MB"
    if rdata:
        bw_used = rdata.get('bw_GBps_used', bdp_result['bw_GBps'])
        p0(f"    {REGION_LABELS[region]}: {label}  (T_gap_p10={rdata['T_window_percentile_ms']:.3f}ms, BW={bw_used:.2f}GB/s)")
    else:
        p0(f"    {REGION_LABELS[region]}: {label}  (no data)")

p0(f"\n  性能对比:")
p0(f"    Grid best (uniform):   {grid_best_mb} MB → {grid_best_t:.2f} ms")
p0(f"    Per-region BDP:        {t_bdp:.2f} ms (delta={delta_bdp:+.1f}%)")

p0(f"\n  推荐设置:")
p0(f"    moe_combine_chunks  = {best_chunks[0]}")
p0(f"    moe_dispatch_chunks = {best_chunks[1]}")
p0(f"    attn_proj_chunks    = {best_chunks[2]}")
p0(f"    attn_qkv_chunks     = {best_chunks[3]}")
p0(f"    scheduler.ar_trickle_sizes = {{")
for region in REGION_ORDER:
    mb = bdp_per_region.get(region, 0)
    p0(f"        '{region}': {mb} * 1024 * 1024,  # {REGION_GAP_DESC[region]}")
p0(f"    }}")

dist.barrier()
dist.destroy_process_group()
p0("\nDone!")
