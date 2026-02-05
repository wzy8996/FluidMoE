"""
FluidMoE 分块参数搜索

搜索四个区域的最优分块数:
  - R1: moe_combine (通信优先)
  - R2: moe_dispatch (计算优先)
  - R3: attn_proj (计算优先)
  - R4: attn_qkv (通信优先)

方法: 逐 region 扫描所有有效因数, 固定其他 region 为 C=1

用法:
    torchrun --nproc_per_node=2 tests/chunk_tune.py
"""
import sys, os, gc
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
# 配置 (可修改) - 与 benchmark.py 一致
# ============================================================
hidden_size = 2048
num_heads = 16
num_kv_heads = 16
ffn_hidden = 14336
num_experts = 8
top_k = 4
seq_local = 2048
batch_size = 4

N_ITER = 3  # 测量迭代数
MAX_C = 8   # 最大分块数


# ============================================================
# 工具函数
# ============================================================
from fluid.layer import TransformerModel
from fluid.core.scheduler import get_backward_scheduler


def get_valid_chunks(size, max_C=MAX_C):
    """返回 size 的因数列表 (1 ~ max_C)."""
    return [c for c in range(1, min(size, max_C) + 1) if size % c == 0]


def bench(model_kwargs, mc, md, ap, aq, x_input):
    """Benchmark 一个 chunks 配置."""
    sched = get_backward_scheduler()
    sched.clear_iteration()

    model = TransformerModel(
        **model_kwargs,
        attn_proj_chunks=ap, attn_qkv_chunks=aq,
        moe_combine_chunks=mc, moe_dispatch_chunks=md,
        dtype=torch.bfloat16, device=device,
    )
    sched.enable()
    sched.configure_allreduce(enabled=True, dp_group=dist.group.WORLD)
    sched.dp_world_size = 1  # 禁用 AR

    xg = x_input.clone().detach().requires_grad_(True)

    # warmup
    for _ in range(2):
        with torch.no_grad():
            model(x_input)
    for _ in range(2):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        sched.finish_batch()
    torch.cuda.synchronize()
    sched.clear_iteration()

    # measure
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N_ITER):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        sched.finish_batch()
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / N_ITER
    sched.clear_iteration()

    del model, xg
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    dist.barrier()
    return t


# ============================================================
# 主流程
# ============================================================
p0("=" * 60)
p0("FluidMoE 分块参数搜索")
p0("=" * 60)
p0(f"Config: hidden={hidden_size}, heads={num_heads}, kv_heads={num_kv_heads}")
p0(f"        ffn={ffn_hidden}, experts={num_experts}, top_k={top_k}")
p0(f"        seq_local={seq_local}, batch={batch_size}, GPUs={world_size}")
p0("=" * 60)

model_kwargs = dict(
    num_layers=1, hidden_size=hidden_size,
    num_heads=num_heads, num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
    cp_group=dist.group.WORLD, ep_group=dist.group.WORLD,
)

x_input = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)

# 有效因数
valid_moe = get_valid_chunks(hidden_size)
valid_attn = get_valid_chunks(seq_local)
p0(f"\n有效因数: MoE(R1,R2)={valid_moe}, Attn(R3,R4)={valid_attn}")

# Step 1: baseline (C=1,1,1,1)
p0(f"\n=== Step 1: Baseline ===")
T_baseline = bench(model_kwargs, 1, 1, 1, 1, x_input)
p0(f"  C=(1,1,1,1): {T_baseline:.2f} ms")

bench_data = {(1, 1, 1, 1): T_baseline}

# Step 2: 逐 region 扫描
region_names = ['R1 (moe_combine)', 'R2 (moe_dispatch)', 'R3 (attn_proj)', 'R4 (attn_qkv)']
region_valid = [valid_moe, valid_moe, valid_attn, valid_attn]
best_per_region = [1, 1, 1, 1]

total_bench = sum(len(v) - 1 for v in region_valid)
p0(f"\n=== Step 2: 逐 Region 扫描 (共 {total_bench} 次) ===")

for ri in range(4):
    p0(f"\n  --- {region_names[ri]} ---")
    best_c, best_t = 1, T_baseline
    for c in region_valid[ri]:
        cfg = [1, 1, 1, 1]
        cfg[ri] = c
        cfg_tuple = tuple(cfg)
        if cfg_tuple in bench_data:
            t = bench_data[cfg_tuple]
        else:
            t = bench(model_kwargs, *cfg_tuple, x_input)
            bench_data[cfg_tuple] = t
        delta = T_baseline - t
        marker = " <-- best" if t < best_t else ""
        if t < best_t:
            best_t = t
            best_c = c
        p0(f"    C={c}: {t:.2f} ms (delta={delta:+.2f} ms){marker}")
    best_per_region[ri] = best_c
    p0(f"  最优: C={best_c}, 节省 {T_baseline - best_t:.2f} ms")

# Step 3: 组合验证
final_cfg = tuple(best_per_region)
p0(f"\n=== Step 3: 组合验证 ===")
p0(f"  逐 region 最优组合: C={final_cfg}")

if final_cfg in bench_data:
    T_final = bench_data[final_cfg]
else:
    T_final = bench(model_kwargs, *final_cfg, x_input)

if T_baseline < T_final:
    p0(f"  Baseline ({T_baseline:.2f} ms) 优于组合 ({T_final:.2f} ms), 回退到 C=(1,1,1,1)")
    final_cfg = (1, 1, 1, 1)
    T_final = T_baseline

speedup = T_baseline / T_final if T_final > 0 else 1.0

# 结果
p0(f"\n{'=' * 60}")
p0("结果")
p0("=" * 60)
p0(f"  Baseline C=(1,1,1,1):       {T_baseline:.2f} ms")
p0(f"  最优配置 C={final_cfg}:  {T_final:.2f} ms  ({speedup:.3f}x)")
p0(f"\n  推荐设置:")
p0(f"    moe_combine_chunks  = {final_cfg[0]}")
p0(f"    moe_dispatch_chunks = {final_cfg[1]}")
p0(f"    attn_proj_chunks    = {final_cfg[2]}")
p0(f"    attn_qkv_chunks     = {final_cfg[3]}")

del x_input
gc.collect()
torch.cuda.empty_cache()
dist.destroy_process_group()
p0("\nDone!")
