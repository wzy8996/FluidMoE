"""
Megatron-Core Complete Transformer Layer Benchmark - CORRECTED

用法:
    torchrun --nproc_per_node=2 tests/benchmark_megatron_full_corrected.py

修正:
    只 AllReduce 有 allreduce=True 标记的参数（排除 EP 专家参数）
"""
import sys
import os

# Add paths
sys.path.insert(0, '/home/zju/wzy/Megatron-LM')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

# Initialize dist
rank = int(os.environ.get('LOCAL_RANK', 0))
device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(device)
dist.init_process_group(backend='nccl', device_id=device)
world_size = dist.get_world_size()

def p0(*args):
    if rank == 0:
        print(*args, flush=True)

# Initialize Megatron parallel_state
from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

if not parallel_state.is_initialized():
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,  # No TP
        pipeline_model_parallel_size=1,
        context_parallel_size=world_size,  # CP for attention
        expert_model_parallel_size=world_size,  # EP for MoE
    )
    model_parallel_cuda_manual_seed(1234)

# Import after parallel_state init
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)

# ============================================================
# 配置（正常规模，使用 CP=2 + EP=2）
# ============================================================
hidden_size = 2048
num_heads = 16
num_kv_heads = 16
ffn_hidden = 8192
num_experts = 8
top_k = 4
num_layers = 2
seq_local = 2048
batch_size = 4

p0("=" * 60)
p0("Megatron-Core Full Transformer Layer with TE (CORRECTED)")
p0("=" * 60)
p0(f"Config: hidden={hidden_size}, heads={num_heads}, ffn={ffn_hidden}")
p0(f"        experts={num_experts}, top_k={top_k}, layers={num_layers}")
p0(f"        seq_local={seq_local}, batch={batch_size}, GPUs={world_size}")
p0(f"        Parallelism: CP={world_size} (Attention), EP={world_size} (MoE)")
p0(f"        TransformerEngine: ENABLED (required for CP)")
p0(f"        moe_shared_expert_overlap=False (disabled)")
p0(f"CORRECTION: Only AllReduce params with allreduce=True")
p0("=" * 60)

# Megatron config
config = TransformerConfig(
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_attention_heads=num_heads,
    num_query_groups=num_kv_heads,  # GQA
    ffn_hidden_size=ffn_hidden,

    # MoE config
    num_moe_experts=num_experts,
    moe_router_topk=top_k,
    moe_token_dispatcher_type="alltoall",
    moe_router_load_balancing_type="aux_loss",
    moe_aux_loss_coeff=0.01,

    # Parallelism（CP=2 + EP=2，fold 在一起）
    tensor_model_parallel_size=1,  # No TP
    pipeline_model_parallel_size=1,
    context_parallel_size=world_size,  # CP=2 for attention
    expert_model_parallel_size=world_size,  # EP=2 for MoE (folded with CP)
    sequence_parallel=False,  # Don't enable (requires TP>1)

    # 禁用 overlap
    moe_shared_expert_overlap=False,

    # Other
    add_bias_linear=False,
    gated_linear_unit=False,  # 改为 False，与 FluidMoE 对齐（GELU）
    bias_activation_fusion=False,
    apply_residual_connection_post_layernorm=False,
    normalization='LayerNorm',  # FusedLayerNorm doesn't support RMSNorm
    attention_dropout=0.0,
    hidden_dropout=0.0,
    bf16=True,
    params_dtype=torch.bfloat16,
)

# Build Transformer layers using TE spec (required for CP)
layers = torch.nn.ModuleList()
for layer_id in range(num_layers):
    layer = TransformerLayer(
        config=config,
        submodules=get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=False,
            qk_layernorm=False,
        ).submodules,
        layer_number=layer_id + 1,
    )
    layers.append(layer)

# Move to device
for layer in layers:
    layer.to(device=device, dtype=torch.bfloat16)

p0(f"✓ {num_layers} Transformer layers created successfully")
total_params = sum(p.numel() for layer in layers for p in layer.parameters())
p0(f"  Total parameters: {total_params / 1e6:.1f}M")

# Count parameters that need AR
ar_params = sum(p.numel() for layer in layers for p in layer.parameters()
                if p.grad is not None or not hasattr(p, 'allreduce') or p.allreduce)
no_ar_params = sum(p.numel() for layer in layers for p in layer.parameters()
                   if hasattr(p, 'allreduce') and not p.allreduce)
p0(f"  Parameters needing AR: {ar_params / 1e6:.1f}M")
p0(f"  Parameters NOT needing AR (EP): {no_ar_params / 1e6:.1f}M")

# Input: [seq_local, batch, hidden] (sequence parallel format)
x = torch.randn(seq_local, batch_size, hidden_size, dtype=torch.bfloat16, device=device)
x_grad = x.clone().detach().requires_grad_(True)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
N = 5

# Warmup
p0("\nWarmup...")
for _ in range(3):
    with torch.no_grad():
        hidden_states = x
        for layer in layers:
            hidden_states, _ = layer(hidden_states)

for _ in range(2):
    x_grad.grad = None
    for layer in layers:
        for p in layer.parameters():
            p.grad = None

    hidden_states = x_grad
    for layer in layers:
        hidden_states, _ = layer(hidden_states)
    hidden_states.sum().backward()

torch.cuda.synchronize()
dist.barrier()
p0("Warmup done.")

# ============================================================
# Benchmark
# ============================================================

# Forward only
start_event.record()
for _ in range(N):
    with torch.no_grad():
        hidden_states = x
        for layer in layers:
            hidden_states, _ = layer(hidden_states)
end_event.record()
torch.cuda.synchronize()
fwd_time = start_event.elapsed_time(end_event) / N

# Forward + Backward
start_event.record()
for _ in range(N):
    x_grad.grad = None
    for layer in layers:
        for p in layer.parameters():
            p.grad = None

    hidden_states = x_grad
    for layer in layers:
        hidden_states, _ = layer(hidden_states)
    hidden_states.sum().backward()
end_event.record()
torch.cuda.synchronize()
fwdbwd_time = start_event.elapsed_time(end_event) / N

# Forward + Backward + AllReduce (CORRECTED: only AR params with allreduce=True)
def allreduce_grads_corrected(layers):
    """Only AllReduce parameters that need it (excluding EP expert params)."""
    for layer in layers:
        for param in layer.parameters():
            if param.grad is not None:
                # Check if parameter should be AllReduced
                # Default to True if attribute doesn't exist (for safety)
                should_ar = getattr(param, 'allreduce', True)
                if should_ar:
                    dist.all_reduce(param.grad, group=dist.group.WORLD)

start_event.record()
for _ in range(N):
    x_grad.grad = None
    for layer in layers:
        for p in layer.parameters():
            p.grad = None

    hidden_states = x_grad
    for layer in layers:
        hidden_states, _ = layer(hidden_states)
    hidden_states.sum().backward()
    allreduce_grads_corrected(layers)
end_event.record()
torch.cuda.synchronize()
fwdbwd_ar_time = start_event.elapsed_time(end_event) / N

# ============================================================
# 输出
# ============================================================
p0(f"\n{'='*60}")
p0(f"Megatron Full Transformer Performance ({num_layers} layers)")
p0(f"{'='*60}")
p0(f"Forward only:       {fwd_time:.2f} ms")
p0(f"Forward + Backward: {fwdbwd_time:.2f} ms")
p0(f"Fwd+Bwd+AR:         {fwdbwd_ar_time:.2f} ms")
p0(f"  (AR overhead:     {fwdbwd_ar_time - fwdbwd_time:.2f} ms)")
p0(f"\nPer-layer average:")
p0(f"  Forward:          {fwd_time / num_layers:.2f} ms/layer")
p0(f"  Fwd+Bwd+AR:       {fwdbwd_ar_time / num_layers:.2f} ms/layer")
p0(f"{'='*60}")

dist.destroy_process_group()
p0("\nDone!")
