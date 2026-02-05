"""快速验证: 单独跑一个配置看 chunks 是否合理."""
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

from fluid.layer import TransformerModel
from fluid.core.scheduler import get_backward_scheduler, BackwardScheduler

N = 5

# Mid-MHA config
hidden_size = 4096
num_heads = 32
num_kv_heads = 32
ffn_hidden = 14336
num_experts = 8
top_k = 4
seq_local = 1024
batch_size = 2

x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)

def bench_one(mc, md, ap, aq, label=""):
    BackwardScheduler.reset()
    sched = get_backward_scheduler()
    model = TransformerModel(
        num_layers=1, hidden_size=hidden_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        ffn_hidden_size=ffn_hidden, num_experts=num_experts, top_k=top_k,
        cp_group=dist.group.WORLD, ep_group=dist.group.WORLD,
        attn_proj_chunks=ap, attn_qkv_chunks=aq,
        moe_combine_chunks=mc, moe_dispatch_chunks=md,
        dtype=torch.bfloat16, device=device,
    )
    sched.enable()
    sched.configure_allreduce(enabled=True, dp_group=dist.group.WORLD)
    sched.dp_world_size = 1

    xg = x.clone().detach().requires_grad_(True)
    for _ in range(3):
        with torch.no_grad():
            model(x)
    for _ in range(3):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        sched.finish_batch()
    torch.cuda.synchronize()
    sched.clear_iteration()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N):
        xg.grad = None
        for p in model.parameters():
            p.grad = None
        model(xg).sum().backward()
        sched.finish_batch()
    end.record()
    torch.cuda.synchronize()
    t = start.elapsed_time(end) / N
    sched.clear_iteration()
    del model, xg
    gc.collect()
    torch.cuda.empty_cache()
    p0(f"  {label:>20s}: {t:.2f}ms")
    return t

p0(f"Mid-MHA: H={hidden_size}, heads={num_heads}, ffn={ffn_hidden}, E={num_experts}, top_k={top_k}")
p0(f"  seq={seq_local}, batch={batch_size}, GPUs={world_size}")

bench_one(1, 1, 1, 1, "C=(1,1,1,1)")
bench_one(2, 2, 2, 2, "C=(2,2,2,2)")
bench_one(2, 1, 1, 1, "C=(2,1,1,1)")
bench_one(1, 2, 1, 1, "C=(1,2,1,1)")
bench_one(1, 1, 2, 1, "C=(1,1,2,1)")
bench_one(1, 1, 1, 2, "C=(1,1,1,2)")
bench_one(4, 1, 1, 1, "C=(4,1,1,1)")

dist.destroy_process_group()
p0("Done!")
