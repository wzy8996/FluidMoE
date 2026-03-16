"""
FluidMoE 完整参数搜索

搜索顺序:
  1. 同步 AR，逐 region 贪心搜索最优分块数 (R1~R4)
  2. BDP 模型自动计算最优 AR trickle size

用法:
    torchrun --nproc_per_node=<world_size> tools/tune.py --model <model_name>
"""
import argparse
import gc
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, TOOLS_DIR)
import torch
import torch.distributed as dist
from experiment_configs import get_tune_defaults, persist_block_benchmark_defaults
from model_configs import get_model_config, list_model_names


def parse_args():
    parser = argparse.ArgumentParser(description="FluidMoE 参数搜索")
    parser.add_argument("--model", type=str, default="mixtral_8x7b", help="模型名称 (from tools/model_configs.py)")
    parser.add_argument("--list-models", action="store_true", help="打印可用模型并退出")
    return parser.parse_args()


args = parse_args()
if args.list_models:
    print("Available models:")
    for n in list_model_names():
        print(" ", n)
    raise SystemExit(0)
model_name = args.model
model_cfg = get_model_config(model_name)
tune_defaults = get_tune_defaults()

rank = int(os.environ.get('LOCAL_RANK', 0))
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl', device_id=device)
world_size = dist.get_world_size()


def p0(*args):
    if rank == 0:
        print(*args, flush=True)


# ============================================================
# 模型结构参数（来自 tools/model_configs.py）
# ============================================================
hidden_size = int(model_cfg.get("hidden_size", 4096))
num_heads = int(model_cfg.get("num_heads", 32))
num_kv_heads = int(model_cfg.get("num_kv_heads", 8))
ffn_hidden = int(model_cfg.get("ffn_hidden", 14336))
num_experts = int(model_cfg.get("num_experts", 8))
top_k = int(model_cfg.get("top_k", 2))
num_layers = int(model_cfg.get("num_layers", 4))
seq_len = int(model_cfg.get("seq_len", 4096))
batch_size = int(model_cfg.get("batch_size", 4))
capacity_factor = float(model_cfg.get("capacity_factor", 1.0))

# 并行度配置（默认来自 tools/experiment_configs.py）
dp_size = int(tune_defaults["dp_size"])
cp_size = int(tune_defaults["cp_size"])
ep_size = int(tune_defaults["ep_size"])

assert dp_size > 0, f"dp_size 必须 > 0, 但 dp_size={dp_size}"
assert cp_size > 0, f"cp_size 必须 > 0, 但 cp_size={cp_size}"
assert ep_size > 0, f"ep_size 必须 > 0, 但 ep_size={ep_size}"
num_gpus = dp_size * cp_size

# Chunk 搜索参数
N_ITER = int(tune_defaults["chunk_search_iters"])              # 测量轮数
MAX_C = int(tune_defaults["chunk_search_max_c"])               # 最大候选分块数
MIN_SAVING = float(tune_defaults["chunk_stop_min_saving_ms"])  # 早停阈值(ms)
# 候选受限于本 rank 上的专家数 nle（expert-dim 切分要求 C | nle）
_nle = num_experts // ep_size
_effective_max = min(MAX_C, _nle)
CHUNK_CANDIDATES = [2**i for i in range(_effective_max.bit_length()) if 2**i <= _effective_max]
if 1 not in CHUNK_CANDIDATES:
    CHUNK_CANDIDATES.insert(0, 1)

# AR 搜索参数
N_AR_WARMUP = int(tune_defaults["ar_warmup"])                  # 预热轮数
N_AR_ITER = int(tune_defaults["ar_iters"])                     # 测量轮数

assert ep_size == cp_size, f"当前仅支持 ep=cp, 但 ep={ep_size}, cp={cp_size}"
assert world_size >= num_gpus, f"需要至少 {num_gpus} GPU, 但只有 {world_size}"
seq_local = seq_len // cp_size

# 多余的 rank 提前退出
if rank >= num_gpus:
    dist.barrier()
    dist.destroy_process_group()
    exit(0)

# 创建进程组
# all_group: 所有参与 rank（共享参数 AR 用）
# cp_group = ep_group: 同一 dp 副本内的 rank
# dp_group: 持有相同专家分区的 rank（专家参数 AR 用）
if dp_size == 1 and num_gpus == world_size:
    all_group = cp_group = ep_group = dp_group = dist.group.WORLD
else:
    if num_gpus == world_size:
        all_group = dist.group.WORLD
    else:
        all_group = dist.new_group(list(range(num_gpus)))
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

# ============================================================
# 初始化
# ============================================================
from fluid.layer import TransformerModel
from fluid.core.scheduler import get_backward_scheduler, BackwardScheduler

p0("=" * 60)
p0("FluidMoE 完整参数搜索")
p0("=" * 60)
p0(f"Model: {model_name} (from tools/model_configs.py)")
p0(f"Config: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}")
p0(f"        ffn={ffn_hidden}, experts={num_experts}, top_k={top_k}")
p0(f"        layers={num_layers}, seq={seq_len}, seq_local={seq_local}, batch={batch_size}, GPUs={num_gpus}")
p0(f"        dp={dp_size}, cp={cp_size}, ep={ep_size}")
p0(f"Chunk 搜索: iter={N_ITER}, candidates={CHUNK_CANDIDATES}, 早停阈值={MIN_SAVING}ms")
p0(f"AR 搜索:    warmup={N_AR_WARMUP}, iter={N_AR_ITER}")
p0("=" * 60)

model_kwargs = dict(
    num_layers=num_layers, hidden_size=hidden_size,
    num_heads=num_heads, num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
    cp_group=cp_group, ep_group=ep_group,
    capacity_factor=capacity_factor,
)

# 固定输入 (所有步骤共用)
x_input = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)


# 在任何训练开始前调用 configure_allreduce，创建 ar_group
# (dist.new_group 是集合操作，必须在所有 rank 同步时调用一次)
# shared_dp_group=all_group: 共享参数 AR 跨所有参与 rank (shared_dp_world_size=num_gpus)
# enabled=False: Step 1 使用同步 AR 模式
# Step 2 设 ar_enabled=True 切换为插入 AR 模式
sched = get_backward_scheduler()
sched.configure_allreduce(enabled=False, shared_dp_group=all_group,
                         expert_dp_group=dp_group if dp_size > 1 else None)


# ============================================================
# Step 1: 逐区域 Chunk 贪心搜索 (无 AR, 带早停)
# ============================================================
p0(f"\n{'=' * 60}")
p0("Step 1: 逐区域 Chunk 贪心搜索 (同步 AR, 带早停)")
p0("=" * 60)
p0(f"候选值: {CHUNK_CANDIDATES}, 早停阈值: {MIN_SAVING}ms")

SEARCH_ORDER = ['moe_combine', 'moe_dispatch', 'attn_proj', 'attn_qkv']
SEARCH_LABELS = {
    'moe_combine':  'R1 (moe_combine)',
    'moe_dispatch': 'R2 (moe_dispatch)',
    'attn_proj':    'R3 (attn_proj)',
    'attn_qkv':     'R4 (attn_qkv)',
}
REGION_ATTRS = {
    'moe_combine': 'moe_combine_chunks',
    'moe_dispatch': 'moe_dispatch_chunks',
    'attn_proj': 'attn_proj_chunks',
    'attn_qkv': 'attn_qkv_chunks',
}
MOE_REGIONS = {'moe_combine', 'moe_dispatch'}

sched.clear_iteration()
chunk_model = TransformerModel(
    **model_kwargs,
    dtype=torch.bfloat16, device=device,
)
sched.enable()  # ar_enabled=False + shared_dp_world_size=num_gpus → 同步 AR 模式
chunk_model.setup_ar_buffer()

xg = x_input.clone().detach().requires_grad_(True)


def bench_bwd():
    """测量当前 chunk 配置的 backward 时间 (avg across ranks)."""
    for _ in range(2):
        xg.grad = None
        for p in chunk_model.parameters():
            p.grad = None
        chunk_model(xg).sum().backward()
        sched.finish_batch()
        sched.clear_iteration()
    torch.cuda.synchronize()
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
    t_tensor = torch.tensor([total / N_ITER], device=device)
    dist.all_reduce(t_tensor, op=dist.ReduceOp.AVG)
    return t_tensor.item()


T_baseline = bench_bwd()  # 全 C=1 baseline
p0(f"  全 C=1 baseline: {T_baseline:.3f} ms")

best_chunks = {r: 1 for r in SEARCH_ORDER}

for region in SEARCH_ORDER:
    attr = REGION_ATTRS[region]
    label = SEARCH_LABELS[region]
    p0(f"\n  搜索 {label} (候选: {CHUNK_CANDIDATES})...")

    region_results = {}
    best_c = 1
    best_t = float('inf')

    for c in CHUNK_CANDIDATES:
        for layer in chunk_model.layers:
            setattr(layer, attr, c)
            if region in MOE_REGIONS:
                layer._moe_chunk_config = None
        t = bench_bwd()
        region_results[c] = t
        marker = ""
        if t < best_t:
            best_t = t
            best_c = c
            marker = " <-- best"
        p0(f"    C={c}: {t:.3f} ms{marker}")

    T_c1 = region_results[1]
    saving = T_c1 - best_t

    if saving < MIN_SAVING:
        p0(f"  {label}: 节省={saving:.3f}ms < {MIN_SAVING}ms, 使用 C=1")
        for layer in chunk_model.layers:
            setattr(layer, attr, 1)
            if region in MOE_REGIONS:
                layer._moe_chunk_config = None
        continue

    # 锁定最优值
    best_chunks[region] = best_c
    for layer in chunk_model.layers:
        setattr(layer, attr, best_c)
        if region in MOE_REGIONS:
            layer._moe_chunk_config = None
    p0(f"  {label}: 最优 C={best_c}, 节省={saving:.3f}ms")

T_chunk_best = bench_bwd()
p0(f"\n最优分块配置: {best_chunks}")
p0(f"  最终: {T_chunk_best:.3f} ms  (speedup={T_baseline/T_chunk_best:.4f}x)")

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
p0(f"  (使用最优分块配置: {best_chunks})")

REGION_ORDER = ['moe_combine', 'moe_dispatch', 'attn_proj', 'attn_qkv']
REGION_LABELS = SEARCH_LABELS
# Gap description: what compute fills the gap after each region
REGION_GAP_DESC = {
    'moe_combine':  'R1→R2: last FC2 dX + act_bwd + FC1 dX(c0)',
    'moe_dispatch': 'R2→R3: permute + router + LN2 dX + proj dX(c0)',
    'attn_proj':    'R3→R4: SDPA bwd + QKV reassemble',
    'attn_qkv':     'R4→R1: last QKV dX + LN1 dX + next layer',
}

# --- 2a: 测量 AR 带宽 ---
p0(f"\n  2a. 测量 AllReduce 带宽...")
ar_group = sched.ar_group if sched.ar_group is not None else sched.shared_dp_group
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
    moe_combine_chunks=best_chunks['moe_combine'],
    moe_dispatch_chunks=best_chunks['moe_dispatch'],
    attn_proj_chunks=best_chunks['attn_proj'],
    attn_qkv_chunks=best_chunks['attn_qkv'],
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
sched.profiling = True  # 启用轻量事件 profiling，仅记录 gap/A2A 时序

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
sched.profiling = False

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

del ar_model, xg_ar
gc.collect()
torch.cuda.empty_cache()
dist.barrier()


# ============================================================
# 最终结果
# ============================================================
p0(f"\n{'=' * 60}")
p0("最终结果")
p0("=" * 60)
p0(f"\n  分块配置:")
p0(f"    Baseline 全C=1: {T_baseline:.3f} ms")
p0(f"    最优配置: {T_chunk_best:.3f} ms  (speedup={T_baseline/T_chunk_best:.4f}x)")
for region in REGION_ORDER:
    p0(f"      {REGION_LABELS[region]}: C={best_chunks[region]}")

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

p0(f"\n  推荐设置:")
for region in REGION_ORDER:
    p0(f"    {REGION_ATTRS[region]} = {best_chunks[region]}")
p0(f"    scheduler.ar_trickle_sizes = {{")
for region in REGION_ORDER:
    mb = bdp_per_region.get(region, 0)
    p0(f"        '{region}': {mb} * 1024 * 1024,  # {REGION_GAP_DESC[region]}")
p0(f"    }}")

persist_updates = {
    "moe_combine_chunks": best_chunks['moe_combine'],
    "moe_dispatch_chunks": best_chunks['moe_dispatch'],
    "attn_proj_chunks": best_chunks['attn_proj'],
    "attn_qkv_chunks": best_chunks['attn_qkv'],
    "ar_trickle_sizes": {
        region: int(bdp_per_region.get(region, 0)) * 1024 * 1024
        for region in REGION_ORDER
    },
}

if rank == 0:
    persist_block_benchmark_defaults(persist_updates)
    p0(f"\n  已写回 tools/experiment_configs.py 的 BLOCK_BENCHMARK_DEFAULTS")

dist.barrier()
dist.destroy_process_group()
p0("\nDone!")
