"""
FluidMoE Benchmark: 完整的性能测试

测试内容:
1. 完整Transformer Layer (Forward + Backward)
2. 组件分解 (Attention vs MoE)

用法:
    torchrun --nproc_per_node=2 tests/benchmark.py

配置说明:
    - chunks: 反向传播分块数 (1=不分块, 4/8=分块重叠)
    - ar_enabled: 是否启用AR interleaved
"""
import sys
sys.path.insert(0, '/home/zju/wzy/FluidMoE')
import torch
import torch.distributed as dist
import torch.nn.functional as F

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)

def p0(*args):
    if rank == 0:
        print(*args, flush=True)

# ============================================================
# 配置参数 (可手动修改)
# ============================================================
hidden_size = 4096
num_heads = 32
num_kv_heads = 32
head_dim = hidden_size // num_heads
ffn_hidden = 14336
num_experts = 8
top_k = 4
num_layers = 2
seq_local = 2048
batch_size = 2

# 调度配置
chunks = 4          # 反向分块数: 1=不分块, 4/8=分块
ar_enabled = True   # AR interleaved开关

p0("=" * 60)
p0("FluidMoE Benchmark")
p0("=" * 60)
p0(f"Config: hidden={hidden_size}, heads={num_heads}, ffn={ffn_hidden}")
p0(f"        experts={num_experts}, top_k={top_k}, layers={num_layers}")
p0(f"        seq_local={seq_local}, batch={batch_size}, GPUs={world_size}")
p0(f"        chunks={chunks}, ar_enabled={ar_enabled}")
p0("=" * 60)

from fluid.layer import TransformerLayer, TransformerModel
from baseline import BaselineTransformerLayer, BaselineTransformerModel
from fluid.core.scheduler import get_backward_scheduler

# 创建模型
baseline_model = BaselineTransformerModel(
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden,
    num_experts=num_experts,
    top_k=top_k,
    cp_group=dist.group.WORLD,
    ep_group=dist.group.WORLD,
    dtype=torch.bfloat16,
    device=device,
)

fluidmoe_model = TransformerModel(
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden,
    num_experts=num_experts,
    top_k=top_k,
    cp_group=dist.group.WORLD,
    ep_group=dist.group.WORLD,
    attn_proj_chunks=chunks,
    attn_qkv_chunks=chunks,
    moe_chunks=chunks,
    dtype=torch.bfloat16,
    device=device,
)

# 单层用于组件分析
config = dict(
    hidden_size=hidden_size,
    num_heads=num_heads,
    num_kv_heads=num_kv_heads,
    ffn_hidden_size=ffn_hidden,
    num_experts=num_experts,
    top_k=top_k,
    cp_group=dist.group.WORLD,
    ep_group=dist.group.WORLD,
    layer_id=0,
    dtype=torch.bfloat16,
    device=device,
)
baseline_layer = BaselineTransformerLayer(**config)
fluidmoe_layer = TransformerLayer(attn_proj_chunks=chunks, attn_qkv_chunks=chunks, moe_chunks=chunks, **config)

# 启用调度器
scheduler = get_backward_scheduler()
scheduler.enable()
scheduler.configure_allreduce(enabled=ar_enabled, dp_group=dist.group.WORLD)

x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Helper: AllReduce gradients
def allreduce_model_grads(model, group):
    for layer in model.layers:
        if layer.qkv_weight.grad is not None:
            dist.all_reduce(layer.qkv_weight.grad, group=group)
        if layer.proj_weight.grad is not None:
            dist.all_reduce(layer.proj_weight.grad, group=group)
        if layer.router_weight.grad is not None:
            dist.all_reduce(layer.router_weight.grad, group=group)

# ============================================================
# Warmup
# ============================================================
p0("\nWarmup...")
for _ in range(3):
    with torch.no_grad():
        baseline_model(x)
        fluidmoe_model(x)
torch.cuda.synchronize()

# ============================================================
# Test 1: Forward Only
# ============================================================
p0(f"\n[1] Forward Only ({num_layers} layers):")

start_event.record()
for _ in range(10):
    with torch.no_grad():
        baseline_model(x)
end_event.record()
torch.cuda.synchronize()
baseline_fwd = start_event.elapsed_time(end_event) / 10

start_event.record()
for _ in range(10):
    with torch.no_grad():
        fluidmoe_model(x)
end_event.record()
torch.cuda.synchronize()
fluidmoe_fwd = start_event.elapsed_time(end_event) / 10

p0(f"    Baseline: {baseline_fwd:.2f} ms")
p0(f"    FluidMoE: {fluidmoe_fwd:.2f} ms")
p0(f"    Speedup:  {baseline_fwd / fluidmoe_fwd:.2f}x")

# ============================================================
# Test 2: Forward + Backward
# ============================================================
p0(f"\n[2] Forward + Backward ({num_layers} layers, AR={ar_enabled}):")

x_grad = x.clone().detach().requires_grad_(True)

# Warmup
for _ in range(2):
    x_grad.grad = None
    for p in baseline_model.parameters():
        p.grad = None
    out = baseline_model(x_grad)
    out.sum().backward()
    allreduce_model_grads(baseline_model, dist.group.WORLD)

    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    out = fluidmoe_model(x_grad)
    out.sum().backward()
    scheduler.finish_batch()
    scheduler.clear_iteration()
torch.cuda.synchronize()

# Baseline
start_event.record()
for _ in range(5):
    x_grad.grad = None
    for p in baseline_model.parameters():
        p.grad = None
    out = baseline_model(x_grad)
    out.sum().backward()
    allreduce_model_grads(baseline_model, dist.group.WORLD)
end_event.record()
torch.cuda.synchronize()
baseline_fwdbwd = start_event.elapsed_time(end_event) / 5

# FluidMoE
scheduler.clear_iteration()
start_event.record()
for _ in range(5):
    x_grad.grad = None
    for p in fluidmoe_model.parameters():
        p.grad = None
    out = fluidmoe_model(x_grad)
    out.sum().backward()
    scheduler.finish_batch()
end_event.record()
torch.cuda.synchronize()
fluidmoe_fwdbwd = start_event.elapsed_time(end_event) / 5

p0(f"    Baseline: {baseline_fwdbwd:.2f} ms")
p0(f"    FluidMoE: {fluidmoe_fwdbwd:.2f} ms")
p0(f"    Speedup:  {baseline_fwdbwd / fluidmoe_fwdbwd:.2f}x")

stats = scheduler.get_stats()
p0(f"    AR stats: total={stats['total_ar_tasks']}, during_gap={stats['ar_during_gap']}")

# ============================================================
# Test 3: Component Breakdown (Attention vs MoE)
# ============================================================
p0(f"\n[3] Component Breakdown (single layer):")

from fluid.attention.forward import (
    qkv_projection_p2p_forward,
    scaled_dot_product_attention_forward,
    output_projection_p2p_forward,
)
from fluid.moe.forward import router_forward, dispatch_fc1_p2p_forward, fc2_combine_p2p_forward
from fluid.core import _all_to_all_sp2hp_forward, _all_to_all_hp2sp_forward
from fluid.core.comm import MultiCardOverlapContext

overlap_ctx = MultiCardOverlapContext(device, world_size, world_size)

# 权重
qkv_weight = baseline_layer.qkv_weight.detach()
proj_weight = baseline_layer.proj_weight.detach()
ln1_weight = baseline_layer.ln1_weight.detach()
ln1_bias = baseline_layer.ln1_bias.detach()
router_weight = baseline_layer.router_weight.detach()
moe_w1 = baseline_layer.moe_w1.detach()
moe_w2 = baseline_layer.moe_w2.detach()
ln2_weight = baseline_layer.ln2_weight.detach()
ln2_bias = baseline_layer.ln2_bias.detach()
num_local_experts = num_experts // world_size

q_per_kv = num_heads // num_kv_heads
group_size = (q_per_kv + 2) * head_dim
q_dim = q_per_kv * head_dim

# --- Attention Only ---
def baseline_attention(x):
    ln1_out = F.layer_norm(x, (hidden_size,), ln1_weight, ln1_bias)
    qkv = F.linear(ln1_out, qkv_weight)
    qkv = qkv.view(seq_local, batch_size, num_kv_heads, group_size)
    q_sp = qkv[:, :, :, :q_dim].reshape(seq_local, batch_size, num_heads, head_dim).contiguous()
    k_sp = qkv[:, :, :, q_dim:q_dim + head_dim].contiguous()
    v_sp = qkv[:, :, :, q_dim + head_dim:].contiguous()
    q_hp = _all_to_all_sp2hp_forward(q_sp, dist.group.WORLD)
    k_hp = _all_to_all_sp2hp_forward(k_sp, dist.group.WORLD)
    v_hp = _all_to_all_sp2hp_forward(v_sp, dist.group.WORLD)
    q_bf = q_hp.permute(1, 2, 0, 3)
    k_bf = k_hp.permute(1, 2, 0, 3)
    v_bf = v_hp.permute(1, 2, 0, 3)
    attn_out_bf = scaled_dot_product_attention_forward(q_bf, k_bf, v_bf, is_causal=True)
    attn_out = attn_out_bf.permute(2, 0, 1, 3).contiguous()
    attn_gathered = _all_to_all_hp2sp_forward(attn_out, dist.group.WORLD)
    attn_flat = attn_gathered.view(seq_local, batch_size, -1)
    return x + F.linear(attn_flat, proj_weight)

def fluidmoe_attention(x):
    ln1_out = F.layer_norm(x, (hidden_size,), ln1_weight, ln1_bias)
    q_hp, k_hp, v_hp = qkv_projection_p2p_forward(
        ln1_out, qkv_weight, num_heads, num_kv_heads, head_dim,
        dist.group.WORLD, overlap_ctx
    )
    q_bf = q_hp.permute(1, 2, 0, 3)
    k_bf = k_hp.permute(1, 2, 0, 3)
    v_bf = v_hp.permute(1, 2, 0, 3)
    attn_out_bf = scaled_dot_product_attention_forward(q_bf, k_bf, v_bf, is_causal=True)
    attn_out = attn_out_bf.permute(2, 0, 1, 3).contiguous()
    output, _ = output_projection_p2p_forward(
        attn_out, proj_weight, None, dist.group.WORLD, overlap_ctx
    )
    return x + output

# Warmup & Test Attention
for _ in range(5):
    baseline_attention(x)
    fluidmoe_attention(x)
torch.cuda.synchronize()

start_event.record()
for _ in range(20):
    baseline_attention(x)
end_event.record()
torch.cuda.synchronize()
baseline_attn = start_event.elapsed_time(end_event) / 20

start_event.record()
for _ in range(20):
    fluidmoe_attention(x)
end_event.record()
torch.cuda.synchronize()
fluidmoe_attn = start_event.elapsed_time(end_event) / 20

p0(f"    Attention: Baseline {baseline_attn:.2f}ms, FluidMoE {fluidmoe_attn:.2f}ms, Speedup {baseline_attn/fluidmoe_attn:.2f}x")

# --- MoE Only ---
def baseline_moe(x):
    ln2_out = F.layer_norm(x, (hidden_size,), ln2_weight, ln2_bias)
    ln2_flat = ln2_out.view(-1, hidden_size)
    num_tokens = ln2_flat.shape[0]

    router_logits = F.linear(ln2_flat.float(), router_weight.t())
    router_probs = F.softmax(router_logits, dim=-1)
    top_probs, top_indices = torch.topk(router_probs, k=top_k, dim=-1)
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

    expanded_tokens = ln2_flat.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
    expanded_probs = top_probs.reshape(-1)
    expanded_expert_indices = top_indices.reshape(-1)

    sorted_indices = torch.argsort(expanded_expert_indices, stable=True)
    permuted_tokens = expanded_tokens[sorted_indices]
    permuted_probs = expanded_probs[sorted_indices]
    sorted_expert_indices = expanded_expert_indices[sorted_indices]

    tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=num_experts)
    experts_per_rank = num_experts // world_size
    input_splits = [tokens_per_expert[i * experts_per_rank:(i + 1) * experts_per_rank].sum().item()
                    for i in range(world_size)]

    all_splits = [None] * world_size
    dist.all_gather_object(all_splits, input_splits)
    output_splits = [all_splits[r][rank] for r in range(world_size)]

    recv_tokens = torch.empty(sum(output_splits), hidden_size, dtype=x.dtype, device=device)
    input_list = list(permuted_tokens.split(input_splits, dim=0))
    output_list = list(recv_tokens.split(output_splits, dim=0))
    dist.all_to_all(output_list, input_list)
    recv_tokens = torch.cat(output_list, dim=0)

    all_tpe = [None] * world_size
    local_tpe = tokens_per_expert[rank * num_local_experts:(rank + 1) * num_local_experts].tolist()
    dist.all_gather_object(all_tpe, local_tpe)
    tokens_per_local_expert = [sum(all_tpe[r][e] for r in range(world_size)) for e in range(num_local_experts)]

    expert_output = torch.zeros(recv_tokens.shape[0], hidden_size, dtype=x.dtype, device=device)
    offset = 0
    for exp_idx in range(num_local_experts):
        n_tok = tokens_per_local_expert[exp_idx]
        if n_tok > 0:
            fc1 = torch.matmul(recv_tokens[offset:offset + n_tok], moe_w1[exp_idx])
            act = F.gelu(fc1)
            expert_output[offset:offset + n_tok] = torch.matmul(act, moe_w2[exp_idx])
            offset += n_tok

    combined = torch.empty(sum(input_splits), hidden_size, dtype=x.dtype, device=device)
    input_list = list(expert_output.split(output_splits, dim=0))
    output_list = list(combined.split(input_splits, dim=0))
    dist.all_to_all(output_list, input_list)
    combined = torch.cat(output_list, dim=0)

    restore_indices = torch.argsort(sorted_indices)
    weighted = combined * permuted_probs.unsqueeze(-1).to(x.dtype)
    restored = weighted[restore_indices]
    moe_output = restored.view(num_tokens, top_k, hidden_size).sum(dim=1)
    return x + moe_output.view(seq_local, batch_size, hidden_size)

def fluidmoe_moe(x):
    ln2_out = F.layer_norm(x, (hidden_size,), ln2_weight, ln2_bias)
    ln2_flat = ln2_out.view(-1, hidden_size)
    num_tokens = ln2_flat.shape[0]

    (permuted_tokens, permuted_probs, restore_indices, sorted_indices,
     input_splits, output_splits, tokens_per_expert,
     router_probs, top_indices, router_logits) = router_forward(
        ln2_flat, router_weight, num_experts, top_k, dist.group.WORLD
    )

    input_splits_list = input_splits.tolist()
    output_splits_list = output_splits.tolist()

    (local_tokens, local_act, recv_act_results, recv_buffers,
     partners, recv_offsets, tokens_cpu) = dispatch_fc1_p2p_forward(
        permuted_tokens, moe_w1, input_splits_list, output_splits_list,
        dist.group.WORLD, overlap_ctx, F.gelu, num_local_experts,
        tokens_per_expert, needs_backward=False,
    )

    (combined_output, _, _, _, _) = fc2_combine_p2p_forward(
        local_tokens, local_act, recv_act_results, recv_buffers,
        moe_w2, input_splits_list, output_splits_list,
        dist.group.WORLD, overlap_ctx, num_local_experts,
        partners, tokens_cpu, needs_backward=False,
    )

    weighted_output = combined_output * permuted_probs.unsqueeze(-1).to(x.dtype)
    restored_output = weighted_output[restore_indices]
    moe_output = restored_output.view(num_tokens, top_k, hidden_size).sum(dim=1)
    return x + moe_output.view(seq_local, batch_size, hidden_size)

# Warmup & Test MoE
for _ in range(5):
    baseline_moe(x)
    fluidmoe_moe(x)
torch.cuda.synchronize()

start_event.record()
for _ in range(20):
    baseline_moe(x)
end_event.record()
torch.cuda.synchronize()
baseline_moe_time = start_event.elapsed_time(end_event) / 20

start_event.record()
for _ in range(20):
    fluidmoe_moe(x)
end_event.record()
torch.cuda.synchronize()
fluidmoe_moe_time = start_event.elapsed_time(end_event) / 20

p0(f"    MoE:       Baseline {baseline_moe_time:.2f}ms, FluidMoE {fluidmoe_moe_time:.2f}ms, Speedup {baseline_moe_time/fluidmoe_moe_time:.2f}x")

# ============================================================
# Summary
# ============================================================
p0("\n" + "=" * 60)
p0("Summary")
p0("=" * 60)
p0(f"Forward ({num_layers}L):     {baseline_fwd:.2f}ms -> {fluidmoe_fwd:.2f}ms  ({baseline_fwd/fluidmoe_fwd:.2f}x)")
p0(f"Fwd+Bwd ({num_layers}L):     {baseline_fwdbwd:.2f}ms -> {fluidmoe_fwdbwd:.2f}ms  ({baseline_fwdbwd/fluidmoe_fwdbwd:.2f}x)")
p0("-" * 60)
p0(f"Attention:        {baseline_attn:.2f}ms -> {fluidmoe_attn:.2f}ms  ({baseline_attn/fluidmoe_attn:.2f}x)")
p0(f"MoE:              {baseline_moe_time:.2f}ms -> {fluidmoe_moe_time:.2f}ms  ({baseline_moe_time/fluidmoe_moe_time:.2f}x)")
p0("-" * 60)
p0(f"Attention占比:    {baseline_attn/(baseline_attn+baseline_moe_time)*100:.1f}%")
p0(f"MoE占比:          {baseline_moe_time/(baseline_attn+baseline_moe_time)*100:.1f}%")
p0("-" * 60)
p0(f"Config: chunks={chunks}, ar_enabled={ar_enabled}")
p0("=" * 60)

dist.destroy_process_group()
p0("\nDone!")
