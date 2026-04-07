"""
FluidMoE 完整参数搜索

搜索顺序:
  1. 同步 AR，逐 region 贪心搜索最优分块数 (R1~R4)
  2. 测量 AR 带宽 + Profile per-region gap times

用法:
    torchrun --nproc_per_node=<world_size> tools/tune.py --model <model_name>
"""
import argparse
import gc
import os
import sys

# Required for P2P + CP/TP correctness (same as Megatron's requirement)
os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')

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
# Chunk 候选列表 (powers of 2 up to MAX_C)
# MoE 和 Attention 使用相同候选范围，两级分解 (_decompose_chunks)
# 会自动处理无效值（返回有效的 C_expert × C_cap 或 fallback 到 C=1）
CHUNK_CANDIDATES = [2**i for i in range(MAX_C.bit_length()) if 2**i <= MAX_C]
if 1 not in CHUNK_CANDIDATES:
    CHUNK_CANDIDATES.insert(0, 1)
MOE_CHUNK_CANDIDATES = CHUNK_CANDIDATES
ATTN_CHUNK_CANDIDATES = CHUNK_CANDIDATES

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
p0(f"Chunk 搜索: iter={N_ITER}, MoE候选={MOE_CHUNK_CANDIDATES}, Attn候选={ATTN_CHUNK_CANDIDATES}, 早停阈值={MIN_SAVING}ms")
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
p0(f"MoE候选: {MOE_CHUNK_CANDIDATES}, Attn候选: {ATTN_CHUNK_CANDIDATES}, 早停阈值: {MIN_SAVING}ms")

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
    candidates = MOE_CHUNK_CANDIDATES if region in MOE_REGIONS else ATTN_CHUNK_CANDIDATES
    p0(f"\n  搜索 {label} (候选: {candidates})...")

    region_results = {}
    best_c = 1
    best_t = float('inf')

    for c in candidates:
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

# 复用 Step 1 的模型和 AR buffer，避免 delete + recreate 导致 OOM
# chunk_model 已设置为 best_chunks，AR buffer 已建好
ar_model = chunk_model
del bench_bwd  # 释放闭包对 chunk_model 的引用
gc.collect()
torch.cuda.empty_cache()
sched.clear_iteration()
dist.barrier()  # 确保两个 rank 同步后再进入 Step 2


# ============================================================
# Step 2: 测量 AR 带宽 + Profile per-region gap times
# ============================================================
p0(f"\n{'=' * 60}")
p0("Step 2: 测量 AR 带宽 + Profile per-region gap times")
p0("=" * 60)
p0(f"  (使用最优分块配置: {best_chunks})")

REGION_ORDER = ['moe_combine', 'moe_dispatch', 'attn_proj', 'attn_qkv']
REGION_LABELS = SEARCH_LABELS
REGION_GAP_DESC = {
    'moe_combine':  'R1→R2: last FC2 dX + act_bwd + FC1 dX(c0)',
    'moe_dispatch': 'R2→R3: permute + router + LN2 dX + proj dX(c0)',
    'attn_proj':    'R3→R4: SDPA bwd + QKV reassemble',
    'attn_qkv':     'R4→R1: last QKV dX + LN1 dX + next layer',
}

# --- 2a: 测量 AR 带宽 ---
p0(f"\n  2a. 测量 AllReduce 带宽...")
shared_ar_group = sched.shared_dp_group
bw_result = BackwardScheduler.measure_ar_bandwidth(
    ar_group=shared_ar_group,
    sizes_mb=(1, 2, 4, 8, 16, 32, 64, 96, 128),
    warmup=5, repeat=20,
)
# peak_bw_GBps → bytes/ms:  1 GB/s = 1e6 bytes/ms
shared_ar_bw = bw_result['peak_bw_GBps'] * 1e6  # bytes/ms
p0(f"  Shared AR 带宽 profile (world_size={sched.shared_dp_world_size}):")
for i, sz in enumerate(bw_result['sizes_mb']):
    p0(f"    {sz:>4d} MB: {bw_result['bw_GBps'][i]:.2f} GB/s  "
       f"(latency {bw_result['latency_ms'][i]:.3f} ms)")
p0(f"  Shared AR Peak BW: {bw_result['peak_bw_GBps']:.2f} GB/s ({shared_ar_bw:.0f} bytes/ms)")

expert_ar_bw = 0.0
expert_ar_group = sched.expert_dp_group
if expert_ar_group is not None and sched.expert_dp_world_size > 1:
    bw_result_expert = BackwardScheduler.measure_ar_bandwidth(
        ar_group=expert_ar_group,
        sizes_mb=(1, 2, 4, 8, 16, 32, 64, 96, 128),
        warmup=5, repeat=20,
    )
    expert_ar_bw = bw_result_expert['peak_bw_GBps'] * 1e6  # bytes/ms
    p0(f"  Expert AR 带宽 profile (world_size={sched.expert_dp_world_size}):")
    for i, sz in enumerate(bw_result_expert['sizes_mb']):
        p0(f"    {sz:>4d} MB: {bw_result_expert['bw_GBps'][i]:.2f} GB/s  "
           f"(latency {bw_result_expert['latency_ms'][i]:.3f} ms)")
    p0(f"  Expert AR Peak BW: {bw_result_expert['peak_bw_GBps']:.2f} GB/s ({expert_ar_bw:.0f} bytes/ms)")
else:
    p0(f"  Expert AR: dp=1, 无需 AR")

# --- 2b: Profile per-region gap times (无 AR，纯净 gap) ---
p0(f"\n  2b. Profile 分区域 gap times (ar_enabled=False, 纯净 gap)...")
sched.clear_iteration()
sched.ar_enabled = False  # 无 AR，测纯净 gap
sched.gap_budgets = {}

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
sched.profiling = True

# 采集 per-region gap times (CUDA events)
for _ in range(N_AR_ITER):
    xg_ar.grad = None
    for p in ar_model.parameters():
        p.grad = None
    ar_model(xg_ar).sum().backward()
    sched.finish_batch()
    sched.clear_iteration()
torch.cuda.synchronize()
sched.process_gap_events()
sched.profiling = False

# 提取 per-region gap 的 p10 作为 gap_budgets (ms)
SAFETY_FACTOR = 0.9
gap_budgets = {}
p0(f"\n  Per-region gap times (safety_factor={SAFETY_FACTOR}):")
p0(f"  {'Region':<22s} {'Gap description':<42s} {'T_p10':>8s} {'T_mean':>8s} {'budget':>8s} {'n':>4s}")
p0(f"  {'-'*22} {'-'*42} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")
for region in REGION_ORDER:
    gaps_ms = sched._region_gaps.get(region, [])
    if not gaps_ms:
        p0(f"  {REGION_LABELS[region]:<22s} {'(no data)':<42s}")
        gap_budgets[region] = 0.0
        continue
    gaps_sorted = sorted(gaps_ms)
    n = len(gaps_sorted)
    idx = max(0, int(n * 10.0 / 100.0) - 1)
    T_p10 = gaps_sorted[idx]
    T_mean = sum(gaps_ms) / n
    budget = T_p10 * SAFETY_FACTOR
    gap_budgets[region] = budget
    p0(f"  {REGION_LABELS[region]:<22s} {REGION_GAP_DESC[region]:<42s} "
       f"{T_p10:>7.3f}ms {T_mean:>7.3f}ms {budget:>7.3f}ms {n:>4d}")

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

p0(f"\n  AR 带宽:")
p0(f"    Shared: {shared_ar_bw:.0f} bytes/ms ({shared_ar_bw/1e6:.2f} GB/s)")
p0(f"    Expert: {expert_ar_bw:.0f} bytes/ms ({expert_ar_bw/1e6:.2f} GB/s)")

p0(f"\n  Per-region gap budgets (p10 * {SAFETY_FACTOR}):")
for region in REGION_ORDER:
    budget = gap_budgets.get(region, 0.0)
    p0(f"    {REGION_LABELS[region]}: {budget:.3f} ms")

p0(f"\n  推荐设置:")
for region in REGION_ORDER:
    p0(f"    {REGION_ATTRS[region]} = {best_chunks[region]}")
p0(f"    scheduler.gap_budgets = {gap_budgets}")
p0(f"    scheduler.shared_ar_bw = {shared_ar_bw}")
p0(f"    scheduler.expert_ar_bw = {expert_ar_bw}")

persist_updates = {
    "moe_combine_chunks": best_chunks['moe_combine'],
    "moe_dispatch_chunks": best_chunks['moe_dispatch'],
    "attn_proj_chunks": best_chunks['attn_proj'],
    "attn_qkv_chunks": best_chunks['attn_qkv'],
    "gap_budgets": {region: round(gap_budgets.get(region, 0.0), 4) for region in REGION_ORDER},
    "shared_ar_bw": round(shared_ar_bw, 2),
    "expert_ar_bw": round(expert_ar_bw, 2),
}

if rank == 0:
    persist_block_benchmark_defaults(persist_updates)
    p0(f"\n  已写回 tools/experiment_configs.py 的 BLOCK_BENCHMARK_DEFAULTS")

dist.barrier()
dist.destroy_process_group()
p0("\nDone!")
