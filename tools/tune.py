"""
FluidMoE 完整参数搜索

搜索顺序:
  1. 同步 AR，逐 region 贪心搜索最优分块数 (R1~R4)
  2. 保持最优分块数，启用异步 AR，搜索最优 AR chunk size

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

# AR chunk size 搜索参数
N_AR_WARMUP = 5         # 预热轮数
N_AR_ITER = 10           # 测量轮数

# ============================================================
# 初始化
# ============================================================
from fluid.layer import TransformerModel
from fluid.layer.transformer import TransformerLayerFunction
from fluid.core.scheduler import get_backward_scheduler

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
# Step 2 设 ar_enabled=True 切换为插入 AR 模式
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
# Step 2: 实测最优 AR chunk size (使用最优分块数)
# ============================================================
p0(f"\n{'=' * 60}")
p0("Step 2: 实测最优 AR chunk size")
p0("=" * 60)
p0(f"  (使用最优分块配置 C={best_chunks})")

chunk_sizes_test = [1, 2, 4, 8, 16, 32, 64]
ar_bench = []  # (chunk_mb, time_ms)

# 建一次模型，循环内只改 ar_chunk_size
# ar_group 已在脚本开头创建，直接启用 AR 即可
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
sched.ar_enabled = True
sched.profiling = False

xg_ar = x_input.clone().detach().requires_grad_(True)
ev_s = torch.cuda.Event(enable_timing=True)
ev_e = torch.cuda.Event(enable_timing=True)

for chunk_mb in chunk_sizes_test:
    sched.ar_chunk_size = chunk_mb * 1024 * 1024
    sched.clear_iteration()

    # 预热
    for _ in range(N_AR_WARMUP):
        xg_ar.grad = None
        for p in ar_model.parameters():
            p.grad = None
        ar_model(xg_ar).sum().backward()
        sched.finish_batch()
    torch.cuda.synchronize()
    sched.clear_iteration()

    # 测量 (与 bench_chunk 一致：forward 后开始计时，finish_batch 后停止)
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
    t = total / N_AR_ITER
    sched.clear_iteration()

    ar_bench.append((chunk_mb, t))
    p0(f"  AR chunk = {chunk_mb:>2d} MB: {t:.2f} ms")

del ar_model, xg_ar
gc.collect()
torch.cuda.empty_cache()
dist.barrier()

best_ar_mb, best_ar_t = min(ar_bench, key=lambda x: x[1])
p0(f"\n  汇总:")
for chunk_mb, t in ar_bench:
    marker = " <-- best" if chunk_mb == best_ar_mb else ""
    p0(f"    {chunk_mb:>2d} MB: {t:.2f} ms{marker}")


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
p0(f"\n  AR chunk size:")
p0(f"    实测最优: {best_ar_mb} MB ({best_ar_t:.2f} ms)")
p0(f"\n  推荐设置:")
p0(f"    moe_combine_chunks  = {best_chunks[0]}")
p0(f"    moe_dispatch_chunks = {best_chunks[1]}")
p0(f"    attn_proj_chunks    = {best_chunks[2]}")
p0(f"    attn_qkv_chunks     = {best_chunks[3]}")
p0(f"    scheduler.ar_chunk_size = {best_ar_mb} * 1024 * 1024")
p0(f"  或环境变量:")
p0(f"    export FLUID_AR_CHUNK_SIZE={best_ar_mb * 1024 * 1024}")

dist.barrier()
dist.destroy_process_group()
p0("\nDone!")
