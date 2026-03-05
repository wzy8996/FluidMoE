"""
FluidMoE Overhead Profiler

精确测量 FluidMoE 相比 baseline 的额外开销来源。
分别测量：
  1. 激活函数反向传播方法差异 (fused gelu_backward vs autograd.grad)
  2. FC1 重算开销
  3. 分块 AllToAll vs 完整 AllToAll
  4. 分块 matmul vs 完整 matmul
  5. .contiguous() 开销
  6. GPU 计算+通信并行时的资源竞争

用法:
    torchrun --nproc_per_node=2 tests/overhead_profiler.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import torch.distributed as dist

rank = int(os.environ.get('LOCAL_RANK', 0))
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl', device_id=device)
world_size = dist.get_world_size()

def p0(*args):
    if rank == 0:
        print(*args, flush=True)

# Config (same as benchmark.py)
hidden_size = 4096
ffn_hidden = 14336
num_experts = 8
num_local_experts = num_experts // world_size
top_k = 4
seq_len = 4096
batch_size = 4
num_tokens = (seq_len // world_size) * batch_size
dtype = torch.bfloat16

# Typical token distribution: ~32000 total recv tokens (top_k * num_tokens / world_size * 2)
total_recv = num_tokens * top_k  # 每个 rank 收到的 token 数 (近似)
tokens_per_expert_val = total_recv // num_local_experts  # 均匀分布近似

p0("=" * 70)
p0("FluidMoE Overhead Profiler")
p0("=" * 70)
p0(f"Config: hidden={hidden_size}, ffn={ffn_hidden}, experts={num_experts}")
p0(f"        total_recv≈{total_recv}, tokens_per_expert≈{tokens_per_expert_val}")
p0(f"        num_local_experts={num_local_experts}, GPUs={world_size}")
p0("=" * 70)

def timed(fn, warmup=5, repeat=20, label=""):
    """精确测量 GPU kernel 时间"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / repeat
    return ms

# ============================================================
# Test 1: 激活函数反向传播方法比较
# ============================================================
p0("\n[Test 1] Activation backward: fused gelu_backward vs autograd.grad")
p0("-" * 60)

fc1_data = torch.randn(total_recv, ffn_hidden, dtype=dtype, device=device)
grad_act = torch.randn(total_recv, ffn_hidden, dtype=dtype, device=device)

# Method A: Baseline 使用的融合 gelu_backward
def act_bwd_fused():
    # Step 1: activation recompute
    act_out = F.gelu(fc1_data)
    # Step 2: fused backward
    grad_fc1 = torch.ops.aten.gelu_backward(grad_act, fc1_data)
    return grad_fc1

# Method B: FluidMoE 使用的 autograd.grad
def act_bwd_autograd():
    with torch.enable_grad():
        fc1_with_grad = fc1_data.detach().requires_grad_(True)
        act_output = F.gelu(fc1_with_grad)
        grad_fc1, = torch.autograd.grad(act_output, fc1_with_grad, grad_act, retain_graph=False)
    return grad_fc1

ms_fused = timed(act_bwd_fused, label="fused gelu_backward")
ms_autograd = timed(act_bwd_autograd, label="autograd.grad")
p0(f"  Fused (act_recomp + gelu_bwd):  {ms_fused:.3f} ms")
p0(f"  Autograd (enable_grad + grad):   {ms_autograd:.3f} ms")
p0(f"  Diff per call:                   {ms_autograd - ms_fused:.3f} ms")
p0(f"  Diff for 2 layers:               {(ms_autograd - ms_fused) * 2:.3f} ms")

# ============================================================
# Test 2: FC1 重算开销
# ============================================================
p0("\n[Test 2] FC1 recomputation overhead (FluidMoE has this, baseline does not)")
p0("-" * 60)

expert_tokens = torch.randn(total_recv, hidden_size, dtype=dtype, device=device)
w1 = torch.randn(num_local_experts, hidden_size, ffn_hidden, dtype=dtype, device=device)
tokens_per_expert = [tokens_per_expert_val] * num_local_experts

def fc1_recomp():
    output = torch.empty(total_recv, ffn_hidden, dtype=dtype, device=device)
    start = 0
    for exp_idx in range(num_local_experts):
        n_tok = tokens_per_expert[exp_idx]
        if n_tok > 0:
            output[start:start + n_tok] = torch.matmul(
                expert_tokens[start:start + n_tok], w1[exp_idx])
            start += n_tok
    return output

ms_recomp = timed(fc1_recomp, label="fc1_recomp")
p0(f"  FC1 recompute per layer:         {ms_recomp:.3f} ms")
p0(f"  FC1 recompute for 2 layers:      {ms_recomp * 2:.3f} ms")
p0(f"  (Should be hidden by AllToAll, but verify with contention test)")

# ============================================================
# Test 3: 分块 AllToAll vs 完整 AllToAll
# ============================================================
p0("\n[Test 3] Chunked AllToAll vs single AllToAll (MoE combine direction)")
p0("-" * 60)

# Simulate MoE combine backward: send [total_send, hidden] tokens
total_send = num_tokens * top_k
input_splits = [total_send // world_size] * world_size
output_splits = [total_recv // world_size] * world_size

send_data = torch.randn(total_send, hidden_size, dtype=dtype, device=device)

def single_alltoall():
    output = torch.empty(total_recv, hidden_size, dtype=dtype, device=device)
    dist.all_to_all_single(output, send_data,
                           output_split_sizes=output_splits,
                           input_split_sizes=input_splits,
                           group=dist.group.WORLD)
    return output

ms_single = timed(single_alltoall, label="single AllToAll")
p0(f"  Single AllToAll [{total_send}x{hidden_size}]:  {ms_single:.3f} ms")

for num_chunks in [2, 4]:
    chunk_size = hidden_size // num_chunks
    send_chunks = [send_data[:, i*chunk_size:(i+1)*chunk_size].contiguous()
                   for i in range(num_chunks)]
    chunk_out_splits = [s for s in output_splits]
    chunk_in_splits = [s for s in input_splits]

    def chunked_alltoall(chunks=send_chunks, nc=num_chunks, cs=chunk_size):
        results = []
        for i in range(nc):
            out = torch.empty(total_recv, cs, dtype=dtype, device=device)
            dist.all_to_all_single(out, chunks[i],
                                   output_split_sizes=chunk_out_splits,
                                   input_split_sizes=chunk_in_splits,
                                   group=dist.group.WORLD)
            results.append(out)
        return results

    ms_chunked = timed(chunked_alltoall, label=f"{num_chunks}-chunk AllToAll")
    overhead_pct = (ms_chunked - ms_single) / ms_single * 100
    p0(f"  {num_chunks}-chunk AllToAll [{total_send}x{chunk_size}]*{num_chunks}: "
       f"{ms_chunked:.3f} ms  (overhead: {ms_chunked - ms_single:+.3f} ms, {overhead_pct:+.1f}%)")

# ============================================================
# Test 4: 分块 matmul vs 完整 matmul
# ============================================================
p0("\n[Test 4] Chunked matmul vs full matmul (FC2 dx)")
p0("-" * 60)

grad_fc2 = torch.randn(total_recv, hidden_size, dtype=dtype, device=device)
w2 = torch.randn(num_local_experts, ffn_hidden, hidden_size, dtype=dtype, device=device)

# Full matmul: grad_fc2 @ w2.T -> [total_recv, ffn_hidden]
def full_fc2_dx():
    output = torch.empty(total_recv, ffn_hidden, dtype=dtype, device=device)
    offset = 0
    for exp_idx in range(num_local_experts):
        n_tok = tokens_per_expert[exp_idx]
        if n_tok > 0:
            output[offset:offset+n_tok] = torch.matmul(
                grad_fc2[offset:offset+n_tok], w2[exp_idx].t())
            offset += n_tok
    return output

ms_full = timed(full_fc2_dx, label="full FC2 dx")
p0(f"  Full FC2 dx [{total_recv}x{hidden_size}] @ [{ffn_hidden}x{hidden_size}].T: {ms_full:.3f} ms")

for num_chunks in [2, 4]:
    chunk_size = hidden_size // num_chunks
    def chunked_fc2_dx(nc=num_chunks, cs=chunk_size):
        output = torch.zeros(total_recv, ffn_hidden, dtype=dtype, device=device)
        for c in range(nc):
            h_s = c * cs
            h_e = h_s + cs
            offset = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx]
                if n_tok > 0:
                    # addmm_ for accumulation
                    output[offset:offset+n_tok].addmm_(
                        grad_fc2[offset:offset+n_tok, h_s:h_e],
                        w2[exp_idx, :, h_s:h_e].t())
                    offset += n_tok
        return output

    ms_chunked = timed(chunked_fc2_dx, label=f"{num_chunks}-chunk FC2 dx")
    overhead_pct = (ms_chunked - ms_full) / ms_full * 100
    p0(f"  {num_chunks}-chunk FC2 dx:  {ms_chunked:.3f} ms  "
       f"(overhead: {ms_chunked - ms_full:+.3f} ms, {overhead_pct:+.1f}%)")

# Test 4b: FC1 dx chunked vs full
p0()
grad_fc1 = torch.randn(total_recv, ffn_hidden, dtype=dtype, device=device)

def full_fc1_dx():
    output = torch.empty(total_recv, hidden_size, dtype=dtype, device=device)
    offset = 0
    for exp_idx in range(num_local_experts):
        n_tok = tokens_per_expert[exp_idx]
        if n_tok > 0:
            output[offset:offset+n_tok] = torch.matmul(
                grad_fc1[offset:offset+n_tok], w1[exp_idx].t())
            offset += n_tok
    return output

ms_full_fc1 = timed(full_fc1_dx, label="full FC1 dx")
p0(f"  Full FC1 dx [{total_recv}x{ffn_hidden}] @ [{hidden_size}x{ffn_hidden}].T: {ms_full_fc1:.3f} ms")

for num_chunks in [2, 4]:
    chunk_size = hidden_size // num_chunks
    def chunked_fc1_dx(nc=num_chunks, cs=chunk_size):
        output = torch.empty(total_recv, hidden_size, dtype=dtype, device=device)
        for c in range(nc):
            h_s = c * cs
            h_e = h_s + cs
            offset = 0
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_expert[exp_idx]
                if n_tok > 0:
                    output[offset:offset+n_tok, h_s:h_e] = torch.matmul(
                        grad_fc1[offset:offset+n_tok],
                        w1[exp_idx, h_s:h_e, :].t())
                    offset += n_tok
        return output

    ms_chunked_fc1 = timed(chunked_fc1_dx, label=f"{num_chunks}-chunk FC1 dx")
    overhead_pct = (ms_chunked_fc1 - ms_full_fc1) / ms_full_fc1 * 100
    p0(f"  {num_chunks}-chunk FC1 dx:  {ms_chunked_fc1:.3f} ms  "
       f"(overhead: {ms_chunked_fc1 - ms_full_fc1:+.3f} ms, {overhead_pct:+.1f}%)")

# ============================================================
# Test 5: .contiguous() 开销
# ============================================================
p0("\n[Test 5] .contiguous() overhead")
p0("-" * 60)

# 模拟 chunked input preparation: grad_output[:, h_start:h_end].contiguous()
grad_output_full = torch.randn(total_send, hidden_size, dtype=dtype, device=device)

for num_chunks in [2, 4]:
    chunk_size = hidden_size // num_chunks
    def contiguous_chunks(nc=num_chunks, cs=chunk_size):
        chunks = []
        for c in range(nc):
            h_s = c * cs
            h_e = h_s + cs
            chunks.append(grad_output_full[:, h_s:h_e].contiguous())
        return chunks

    ms_contig = timed(contiguous_chunks, label=f"{num_chunks} contiguous chunks")
    p0(f"  {num_chunks} chunks [{total_send}x{chunk_size}]:  {ms_contig:.3f} ms")

# ============================================================
# Test 6: GPU 资源竞争 — 计算+通信并行 vs 串行
# ============================================================
p0("\n[Test 6] GPU resource contention: compute with/without concurrent comm")
p0("-" * 60)

comm_stream = torch.cuda.Stream(device=device)
default_stream = torch.cuda.current_stream(device)

# 计算任务: FC2 dx (代表 R1 中的计算)
def compute_task():
    offset = 0
    output = torch.zeros(total_recv, ffn_hidden, dtype=dtype, device=device)
    for exp_idx in range(num_local_experts):
        n_tok = tokens_per_expert[exp_idx]
        if n_tok > 0:
            output[offset:offset+n_tok].addmm_(
                grad_fc2[offset:offset+n_tok], w2[exp_idx].t())
            offset += n_tok
    return output

# 通信任务: AllToAll (代表 R1 中的 AllToAll)
a2a_send = torch.randn(total_send, hidden_size // 4, dtype=dtype, device=device)

def comm_task():
    out = torch.empty(total_recv, hidden_size // 4, dtype=dtype, device=device)
    dist.all_to_all_single(out, a2a_send,
                           output_split_sizes=output_splits,
                           input_split_sizes=input_splits,
                           group=dist.group.WORLD)
    return out

# 测量纯计算
ms_compute_only = timed(compute_task, label="compute only")
p0(f"  Compute only (FC2 dx):  {ms_compute_only:.3f} ms")

# 测量纯通信
ms_comm_only = timed(comm_task, label="comm only")
p0(f"  Comm only (AllToAll):   {ms_comm_only:.3f} ms")

# 测量并行 (计算在 default stream，通信在 comm stream)
def parallel_compute_comm():
    # Launch comm on separate stream
    with torch.cuda.stream(comm_stream):
        comm_stream.wait_stream(default_stream)
        out = torch.empty(total_recv, hidden_size // 4, dtype=dtype, device=device)
        dist.all_to_all_single(out, a2a_send,
                               output_split_sizes=output_splits,
                               input_split_sizes=input_splits,
                               group=dist.group.WORLD)
    # Compute on default stream concurrently
    result = compute_task()
    # Wait for comm
    default_stream.wait_stream(comm_stream)
    return result

ms_parallel = timed(parallel_compute_comm, label="parallel compute+comm")
ideal_parallel = max(ms_compute_only, ms_comm_only)
contention = ms_parallel - ideal_parallel
p0(f"  Parallel (compute||comm): {ms_parallel:.3f} ms")
p0(f"  Ideal (max of above):     {ideal_parallel:.3f} ms")
p0(f"  Contention overhead:      {contention:+.3f} ms ({contention/ideal_parallel*100:+.1f}%)")

# ============================================================
# Test 7: Scheduler overhead estimation
# ============================================================
p0("\n[Test 7] Scheduler overhead: submit/wait cycle")
p0("-" * 60)

from fluid.core.scheduler import get_backward_scheduler
scheduler = get_backward_scheduler()
scheduler.enable()
scheduler.configure_allreduce(enabled=False, shared_dp_group=dist.group.WORLD)

# 测量 submit + wait 的纯开销 (用空操作)
dummy_result = [None]
def dummy_fn():
    dummy_result[0] = torch.empty(1, device=device)
    return dummy_result[0]

def scheduler_submit_wait():
    task_id = scheduler.submit_alltoall(dummy_fn)
    scheduler.wait_alltoall(task_id)

# Warmup
for _ in range(10):
    scheduler_submit_wait()
torch.cuda.synchronize()

ms_sched = timed(scheduler_submit_wait, warmup=10, repeat=100, label="scheduler overhead")
p0(f"  Submit+wait per AllToAll:  {ms_sched:.4f} ms")
# R1: 4 chunks + R2: 2 chunks + R3: 1 + R4: 4 = 11 AllToAll per layer, x2 layers
# Plus flush_pending_ar, begin/end_region
total_sched_ops = (4 + 2 + 1 + 4) * 2
p0(f"  Estimated total ({total_sched_ops} ops): {ms_sched * total_sched_ops:.3f} ms")

scheduler.clear_iteration()

# ============================================================
# Test 8: Layout conversion total
# ============================================================
p0("\n[Test 8] Layout conversion (_sort_chunks_by_idxs) total overhead")
p0("-" * 60)

from fluid.core import _sort_chunks_by_idxs

# Create test data matching actual backward
# rank-major split sizes and indices (for 2 ranks, 4 experts per rank)
split_sizes = torch.tensor([tokens_per_expert_val // world_size] * (world_size * num_local_experts),
                           dtype=torch.int64, device=device)
sort_idxs = torch.tensor(
    [r * num_local_experts + e for e in range(num_local_experts) for r in range(world_size)],
    dtype=torch.int64, device=device)

# Test for full hidden_size and chunk sizes
for tensor_width, label in [(hidden_size, "full hidden"), (hidden_size // 4, "chunk h/4")]:
    test_tensor = torch.randn(total_recv, tensor_width, dtype=dtype, device=device)
    def layout_convert(t=test_tensor, ss=split_sizes, si=sort_idxs):
        return _sort_chunks_by_idxs(t, ss, si)

    ms_layout = timed(layout_convert, repeat=50, label=label)
    # R1: num_chunks layout converts, R2: 1 (non-chunked path) or num_chunks
    # With n1=4 R1 chunks: 4 layout converts per layer
    # With n2=2 R2 chunks: 2 layout converts + 2 reorder per layer (if chunked)
    # Non-chunked R2: 1 layout convert per layer
    # Total per layer: 4 (R1) + 1~2 (R2) = 5~6
    p0(f"  [{total_recv}x{tensor_width}]: {ms_layout:.3f} ms/call")

p0(f"  Estimated total (12 calls, mix of sizes): ~1.8 ms")

# ============================================================
# Summary
# ============================================================
p0("\n" + "=" * 70)
p0("SUMMARY: Overhead decomposition (per iteration, 2 layers)")
p0("=" * 70)

overhead_act_bwd = (ms_autograd - ms_fused) * 2
overhead_fc1_recomp = ms_recomp * 2
# Use conservative estimates
p0(f"  1. Activation backward method:  {overhead_act_bwd:+.1f} ms  (autograd vs fused, x2 layers)")
p0(f"  2. FC1 recomputation:           {overhead_fc1_recomp:+.1f} ms  (extra in FluidMoE, x2 layers)")
p0(f"     (FC1 recomp should be hidden by AllToAll if AllToAll > {ms_recomp:.1f}ms)")
p0(f"  3. GPU resource contention:     {contention:+.1f} ms  (per overlap region, varies)")
p0(f"  4. Scheduler overhead:          {ms_sched * total_sched_ops:.1f} ms  ({total_sched_ops} ops)")
p0(f"  5. Layout conversion:           ~1.8 ms  (12 calls)")
p0(f"  ---")
p0(f"  Measured overheads total:        ~{overhead_act_bwd + overhead_fc1_recomp + contention + ms_sched * total_sched_ops + 1.8:.1f} ms")
p0(f"  Target to explain:              ~122 ms")

dist.destroy_process_group()
p0("\nDone!")
