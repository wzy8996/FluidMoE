"""DeepSpeed block-level baseline for the current FluidMoE benchmark.

This module intentionally does not use the official HF/Ulysses full-model path.
Instead it exposes a transformer-block wrapper with the same benchmark contract as
FluidMoE/Megatron:

- input/output shape: [seq_local, batch, hidden]
- context parallel attention via DeepSpeed DistributedAttention (Ulysses-style)
- expert parallel MoE with the same fixed-capacity padding path used by the
  local baseline/FluidMoE benchmark

Notes:
- DeepSpeed's public MoE layer only supports top-k in {1, 2}. The current
  FluidMoE model presets use top-k up to 8, so this block baseline keeps the
  MoE path compatible with FluidMoE's routing/capacity semantics instead of
  forcing a narrower DeepSpeed-MoE configuration.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from baseline import _get_fc1_buf, _timed_overlap, _moe_all_to_all_uniform
from fluid.attention.forward import scaled_dot_product_attention_forward
from fluid.moe.forward import pad_moe_dispatch


def _require_deepspeed_block_deps():
    try:
        import deepspeed
        import deepspeed.comm as ds_comm
        from deepspeed.sequence.layer import DistributedAttention
    except ImportError as exc:
        raise ImportError(
            "DeepSpeed block baseline requires `deepspeed` with "
            "`deepspeed.sequence.layer.DistributedAttention`."
        ) from exc

    if dist.is_initialized():
        is_init = False
        if hasattr(ds_comm, "is_initialized"):
            try:
                is_init = bool(ds_comm.is_initialized())
            except Exception:
                is_init = False
        if not is_init:
            deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

    return {"DistributedAttention": DistributedAttention}


class _UlyssesLocalAttention(nn.Module):
    """Local attention kernel used inside DeepSpeed DistributedAttention.

    Expects batch-first input: [batch, seq, heads, head_dim].
    """

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args, **kwargs):
        del args, kwargs
        # Input: [batch, seq, heads, head_dim] → SDPA expects [batch, heads, seq, head_dim]
        q_bf = query.permute(0, 2, 1, 3)
        k_bf = key.permute(0, 2, 1, 3)
        v_bf = value.permute(0, 2, 1, 3)
        enable_gqa = q_bf.shape[1] != k_bf.shape[1]
        out_bf = scaled_dot_product_attention_forward(
            q_bf,
            k_bf,
            v_bf,
            scale=self.scale,
            is_causal=True,
            enable_gqa=enable_gqa,
        )
        return out_bf.permute(0, 2, 1, 3).contiguous()


class DeepSpeedBlockBaselineFunction(torch.autograd.Function):
    """Manual block backward with DeepSpeed Ulysses attention + padded MoE."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        ln1_weight: torch.Tensor,
        ln1_bias: torch.Tensor,
        ln2_weight: torch.Tensor,
        ln2_bias: torch.Tensor,
        qkv_weight: torch.Tensor,
        proj_weight: torch.Tensor,
        router_weight: torch.Tensor,
        moe_w1: torch.Tensor,
        moe_w2: torch.Tensor,
        cp_group,
        ep_group,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int,
        top_k: int,
        activation_func: Callable,
        capacity_factor: float,
        ulysses_attn: nn.Module,
    ):
        seq_len, batch_size, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        ep_size = ep_group.size()
        head_dim = hidden_size // num_heads
        num_local_experts = num_experts // ep_size

        ln1_out = F.layer_norm(hidden_states, (hidden_size,), ln1_weight, ln1_bias)
        with _timed_overlap("fwd", "qkv"):
            qkv = F.linear(ln1_out, qkv_weight)

        q_per_kv = num_heads // num_kv_heads
        group_size = (q_per_kv + 2) * head_dim
        qkv = qkv.view(seq_len, batch_size, num_kv_heads, group_size)

        q_dim = q_per_kv * head_dim
        q_sp = qkv[:, :, :, :q_dim].reshape(seq_len, batch_size, num_heads, head_dim)
        k_sp = qkv[:, :, :, q_dim:q_dim + head_dim]
        v_sp = qkv[:, :, :, q_dim + head_dim:]

        # Use batch_dim_idx=0 to avoid DeepSpeed shape bug in reverse AllToAll
        q_bf = q_sp.permute(1, 0, 2, 3).contiguous()
        k_bf = k_sp.permute(1, 0, 2, 3).contiguous()
        v_bf = v_sp.permute(1, 0, 2, 3).contiguous()

        with torch.enable_grad():
            q_for_attn = q_bf.detach().requires_grad_(True)
            k_for_attn = k_bf.detach().requires_grad_(True)
            v_for_attn = v_bf.detach().requires_grad_(True)
            attn_out_bf = ulysses_attn(q_for_attn, k_for_attn, v_for_attn, batch_dim_idx=0)
            attn_out_sp = attn_out_bf.permute(1, 0, 2, 3).contiguous()

        with _timed_overlap("fwd", "proj"):
            proj_out = F.linear(attn_out_sp.view(seq_len, batch_size, -1), proj_weight)
        hidden_after_attn = hidden_states + proj_out

        ln2_out = F.layer_norm(hidden_after_attn, (hidden_size,), ln2_weight, ln2_bias)
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

        expert_capacity = int(math.ceil(num_tokens * top_k / num_experts * capacity_factor))
        offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=device)
        torch.cumsum(tokens_per_expert, dim=0, out=offsets[1:])
        within_expert_pos = torch.arange(num_tokens * top_k, device=device) - offsets[sorted_expert_indices]
        keep_mask = within_expert_pos < expert_capacity
        sorted_indices = sorted_indices[keep_mask]
        token_ids = sorted_indices // top_k
        permuted_tokens = expanded_tokens[sorted_indices]
        permuted_probs = expanded_probs[sorted_indices]
        tokens_per_expert = tokens_per_expert.clamp(max=expert_capacity)

        cap_per_rank = expert_capacity
        (
            padded_tokens,
            padded_probs,
            padded_sorted_indices,
            padded_token_ids,
            _padded_tpe,
            _real_mask,
        ) = pad_moe_dispatch(
            permuted_tokens,
            permuted_probs,
            sorted_indices,
            token_ids,
            tokens_per_expert,
            cap_per_rank,
            num_experts,
        )

        S = num_local_experts * cap_per_rank
        recv_tokens = _moe_all_to_all_uniform(padded_tokens, S, ep_size, ep_group)
        tokens_per_local_expert = [ep_size * cap_per_rank] * num_local_experts

        total_recv = recv_tokens.shape[0]
        expert_output = torch.zeros(total_recv, hidden_size, dtype=dtype, device=device)
        offset = 0
        for exp_idx in range(num_local_experts):
            n_tok = tokens_per_local_expert[exp_idx]
            if n_tok > 0:
                with _timed_overlap("fwd", "fc1"):
                    fc1 = torch.matmul(recv_tokens[offset:offset + n_tok], moe_w1[exp_idx])
                    act = activation_func(fc1)
                with _timed_overlap("fwd", "fc2"):
                    expert_output[offset:offset + n_tok] = torch.matmul(act, moe_w2[exp_idx])
                offset += n_tok

        combined_output = _moe_all_to_all_uniform(expert_output, S, ep_size, ep_group)

        weighted = combined_output * padded_probs.unsqueeze(-1).to(dtype)
        moe_output = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
        moe_output.scatter_add_(0, padded_token_ids.unsqueeze(1).expand_as(weighted), weighted)
        moe_output = moe_output.view(seq_len, batch_size, hidden_size)

        output = hidden_after_attn + moe_output

        ctx.save_for_backward(
            hidden_states,
            ln1_out,
            hidden_after_attn,
            ln2_flat,
            padded_tokens,
            padded_probs,
            padded_sorted_indices,
            padded_token_ids,
            router_probs,
            top_probs,
            top_indices,
            recv_tokens,
            combined_output,
            ln1_weight,
            ln1_bias,
            ln2_weight,
            ln2_bias,
            qkv_weight,
            proj_weight,
            router_weight,
            moe_w1,
            moe_w2,
        )
        ctx._q_for_attn = q_for_attn
        ctx._k_for_attn = k_for_attn
        ctx._v_for_attn = v_for_attn
        ctx._attn_out_sp = attn_out_sp
        ctx.cp_group = cp_group
        ctx.ep_group = ep_group
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.activation_func = activation_func
        ctx.tokens_per_local_expert = tokens_per_local_expert
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            hidden_states,
            ln1_out,
            hidden_after_attn,
            ln2_flat,
            permuted_tokens,
            permuted_probs,
            sorted_indices,
            token_ids,
            router_probs,
            top_probs,
            top_indices,
            recv_tokens,
            combined_output,
            ln1_weight,
            ln1_bias,
            ln2_weight,
            ln2_bias,
            qkv_weight,
            proj_weight,
            router_weight,
            moe_w1,
            moe_w2,
        ) = ctx.saved_tensors

        q_for_attn = ctx._q_for_attn
        k_for_attn = ctx._k_for_attn
        v_for_attn = ctx._v_for_attn
        attn_out_sp = ctx._attn_out_sp

        ep_group = ctx.ep_group
        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        num_experts = ctx.num_experts
        top_k = ctx.top_k
        activation_func = ctx.activation_func
        tokens_per_local_expert = ctx.tokens_per_local_expert

        seq_len, batch_size, hidden_size = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device
        ep_size = ep_group.size()
        head_dim = hidden_size // num_heads
        num_local_experts = num_experts // ep_size
        num_tokens = seq_len * batch_size
        S = recv_tokens.shape[0] // ep_size

        grad_output_flat = grad_output.view(num_tokens, hidden_size)
        grad_hidden_after_attn = grad_output.clone()

        grad_weighted = grad_output_flat[token_ids]
        grad_combined = grad_weighted * permuted_probs.unsqueeze(-1).to(dtype)
        grad_expert_output = _moe_all_to_all_uniform(grad_combined, S, ep_size, ep_group)

        grad_recv_tokens = torch.zeros_like(recv_tokens)
        grad_moe_w1 = torch.zeros_like(moe_w1)
        grad_moe_w2 = torch.zeros_like(moe_w2)

        expert_offsets = []
        off = 0
        for exp_idx in range(num_local_experts):
            expert_offsets.append(off)
            off += tokens_per_local_expert[exp_idx]

        with _timed_overlap("bwd", "fc1_recomp"):
            total = recv_tokens.shape[0]
            ffn = moe_w1.shape[2]
            if total == 0:
                fc1_all = torch.empty(0, ffn, dtype=dtype, device=device)
            else:
                fc1_all = _get_fc1_buf(total, ffn, dtype, device)
                for exp_idx in range(num_local_experts):
                    n_tok = tokens_per_local_expert[exp_idx]
                    if n_tok > 0:
                        o = expert_offsets[exp_idx]
                        torch.mm(recv_tokens[o:o + n_tok], moe_w1[exp_idx], out=fc1_all[o:o + n_tok])

        with _timed_overlap("bwd", "fc2_dx"):
            grad_exp_act = torch.empty_like(fc1_all)
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_local_expert[exp_idx]
                if n_tok > 0:
                    o = expert_offsets[exp_idx]
                    torch.mm(
                        grad_expert_output[o:o + n_tok],
                        moe_w2[exp_idx].t(),
                        out=grad_exp_act[o:o + n_tok],
                    )

        with _timed_overlap("bwd", "act_bwd"):
            if fc1_all.numel() == 0:
                grad_fc1_all = torch.empty_like(fc1_all)
                act_all_detached = torch.empty_like(fc1_all)
            else:
                with torch.enable_grad():
                    fc1_with_grad = fc1_all.detach().requires_grad_(True)
                    act_all = activation_func(fc1_with_grad)
                    (grad_fc1_all,) = torch.autograd.grad(
                        act_all,
                        fc1_with_grad,
                        grad_exp_act,
                        retain_graph=False,
                    )
                act_all_detached = act_all.detach()

        with _timed_overlap("bwd", "fc2_dw"):
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_local_expert[exp_idx]
                if n_tok > 0:
                    o = expert_offsets[exp_idx]
                    grad_moe_w2[exp_idx] = torch.matmul(
                        act_all_detached[o:o + n_tok].t(),
                        grad_expert_output[o:o + n_tok],
                    )

        with _timed_overlap("bwd", "fc1_dx"):
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_local_expert[exp_idx]
                if n_tok > 0:
                    o = expert_offsets[exp_idx]
                    grad_recv_tokens[o:o + n_tok] = torch.matmul(
                        grad_fc1_all[o:o + n_tok],
                        moe_w1[exp_idx].t(),
                    )

        with _timed_overlap("bwd", "fc1_dw"):
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_local_expert[exp_idx]
                if n_tok > 0:
                    o = expert_offsets[exp_idx]
                    grad_moe_w1[exp_idx] = torch.matmul(
                        recv_tokens[o:o + n_tok].t(),
                        grad_fc1_all[o:o + n_tok],
                    )

        grad_permuted_tokens = _moe_all_to_all_uniform(grad_recv_tokens, S, ep_size, ep_group)

        grad_ln2_flat_from_tokens = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
        grad_ln2_flat_from_tokens.scatter_add_(
            0, token_ids.unsqueeze(1).expand_as(grad_permuted_tokens), grad_permuted_tokens
        )

        with _timed_overlap("bwd", "router_dx"):
            grad_permuted_probs = (grad_weighted * combined_output.to(dtype)).sum(dim=-1)
            grad_top_probs = torch.zeros(num_tokens, top_k, dtype=grad_permuted_probs.dtype, device=device)
            slot_ids = sorted_indices % top_k
            grad_top_probs[token_ids, slot_ids] = grad_permuted_probs
            grad_dot = (grad_top_probs * top_probs).sum(dim=-1, keepdim=True)
            grad_raw_top_probs = (
                grad_top_probs - grad_dot * top_probs
            ) / top_probs.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            grad_router_probs = torch.zeros_like(router_probs)
            grad_router_probs.scatter_(1, top_indices, grad_raw_top_probs)
            sum_grad_probs = (grad_router_probs * router_probs).sum(dim=-1, keepdim=True)
            grad_router_logits = router_probs * (grad_router_probs - sum_grad_probs)
            grad_ln2_flat_from_router = torch.matmul(
                grad_router_logits.float(), router_weight.t().float()
            ).to(dtype)

        with _timed_overlap("bwd", "router_dw"):
            grad_router_weight = torch.matmul(ln2_flat.t().float(), grad_router_logits.float())

        grad_ln2_flat = grad_ln2_flat_from_tokens + grad_ln2_flat_from_router

        mean2 = hidden_after_attn.mean(dim=-1, keepdim=True)
        var2 = hidden_after_attn.var(dim=-1, unbiased=False, keepdim=True)
        std2 = (var2 + 1e-5).sqrt()
        normalized2 = (hidden_after_attn - mean2) / std2

        grad_ln2_out = grad_ln2_flat.view(seq_len, batch_size, hidden_size)
        with _timed_overlap("bwd", "ln2_dw"):
            grad_ln2_weight = (grad_ln2_out * normalized2).sum(dim=(0, 1))
            grad_ln2_bias = grad_ln2_out.sum(dim=(0, 1))
        with _timed_overlap("bwd", "ln2_dx"):
            grad_hidden_after_attn = grad_hidden_after_attn + grad_ln2_out * ln2_weight / std2

        grad_proj_out = grad_hidden_after_attn.view(seq_len, batch_size, hidden_size)
        with _timed_overlap("bwd", "proj_dw"):
            grad_proj_weight = torch.matmul(
                grad_proj_out.view(-1, hidden_size).t(),
                attn_out_sp.view(-1, num_heads * head_dim),
            )
        with _timed_overlap("bwd", "proj_dx"):
            grad_attn_out_sp = torch.matmul(
                grad_proj_out.view(-1, hidden_size), proj_weight
            ).view(seq_len, batch_size, num_heads, head_dim)

        with _timed_overlap("bwd", "sdpa_dx"):
            grad_q_bf, grad_k_bf, grad_v_bf = torch.autograd.grad(
                attn_out_sp,
                (q_for_attn, k_for_attn, v_for_attn),
                grad_attn_out_sp,
                retain_graph=False,
            )
            # q/k/v_for_attn are batch-first, permute grads back to [seq, batch, ...]
            grad_q_sp = grad_q_bf.permute(1, 0, 2, 3).contiguous()
            grad_k_sp = grad_k_bf.permute(1, 0, 2, 3).contiguous()
            grad_v_sp = grad_v_bf.permute(1, 0, 2, 3).contiguous()

        q_per_kv = num_heads // num_kv_heads
        grad_qkv = torch.zeros(
            seq_len,
            batch_size,
            num_kv_heads,
            (q_per_kv + 2) * head_dim,
            dtype=dtype,
            device=device,
        )
        q_dim = q_per_kv * head_dim
        grad_qkv[:, :, :, :q_dim] = grad_q_sp.view(seq_len, batch_size, num_kv_heads, q_dim)
        grad_qkv[:, :, :, q_dim:q_dim + head_dim] = grad_k_sp
        grad_qkv[:, :, :, q_dim + head_dim:] = grad_v_sp
        grad_qkv_flat = grad_qkv.view(seq_len * batch_size, -1)

        with _timed_overlap("bwd", "qkv_dw"):
            grad_qkv_weight = torch.matmul(grad_qkv_flat.t(), ln1_out.view(-1, hidden_size))
        with _timed_overlap("bwd", "qkv_dx"):
            grad_ln1_out = torch.matmul(grad_qkv_flat, qkv_weight).view(seq_len, batch_size, hidden_size)

        grad_hidden_states = grad_hidden_after_attn.clone()

        mean1 = hidden_states.mean(dim=-1, keepdim=True)
        var1 = hidden_states.var(dim=-1, unbiased=False, keepdim=True)
        std1 = (var1 + 1e-5).sqrt()
        normalized1 = (hidden_states - mean1) / std1

        with _timed_overlap("bwd", "ln1_dw"):
            grad_ln1_weight = (grad_ln1_out * normalized1).sum(dim=(0, 1))
            grad_ln1_bias = grad_ln1_out.sum(dim=(0, 1))
        with _timed_overlap("bwd", "ln1_dx"):
            grad_hidden_states = grad_hidden_states + grad_ln1_out * ln1_weight / std1

        return (
            grad_hidden_states,
            grad_ln1_weight,
            grad_ln1_bias,
            grad_ln2_weight,
            grad_ln2_bias,
            grad_qkv_weight,
            grad_proj_weight,
            grad_router_weight,
            grad_moe_w1,
            grad_moe_w2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DeepSpeedBlockBaselineLayer(nn.Module):
    """Transformer block with DeepSpeed Ulysses attention and padded-capacity MoE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_hidden_size: int,
        num_experts: int,
        top_k: int,
        cp_group: dist.ProcessGroup,
        ep_group: dist.ProcessGroup,
        layer_id: int = 0,
        activation_func: Optional[Callable] = None,
        capacity_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        deps = _require_deepspeed_block_deps()
        distributed_attention_cls = deps["DistributedAttention"]

        if device is None:
            device = torch.device(f"cuda:{dist.get_rank()}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_id = layer_id
        self.activation_func = activation_func or F.gelu
        self.capacity_factor = capacity_factor
        self.cp_group = cp_group
        self.ep_group = ep_group
        self.ep_size = ep_group.size()

        num_local_experts = num_experts // self.ep_size
        head_dim = hidden_size // num_heads
        q_per_kv = num_heads // num_kv_heads
        qkv_size = num_kv_heads * (q_per_kv + 2) * head_dim

        self.ln1_weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.ln1_bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.ln2_weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.ln2_bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

        self.qkv_weight = nn.Parameter(torch.empty(qkv_size, hidden_size, dtype=dtype, device=device))
        self.proj_weight = nn.Parameter(torch.empty(hidden_size, num_heads * head_dim, dtype=dtype, device=device))

        self.router_weight = nn.Parameter(torch.empty(hidden_size, num_experts, dtype=torch.float32, device=device))
        self.moe_w1 = nn.Parameter(
            torch.empty(num_local_experts, hidden_size, ffn_hidden_size, dtype=dtype, device=device)
        )
        self.moe_w2 = nn.Parameter(
            torch.empty(num_local_experts, ffn_hidden_size, hidden_size, dtype=dtype, device=device)
        )

        self.ulysses_attn = distributed_attention_cls(
            local_attention=_UlyssesLocalAttention(scale=1.0 / math.sqrt(head_dim)),
            sequence_process_group=cp_group,
            scatter_idx=2,
            gather_idx=0,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.xavier_uniform_(self.proj_weight)
        nn.init.xavier_uniform_(self.router_weight)
        nn.init.xavier_uniform_(self.moe_w1)
        nn.init.xavier_uniform_(self.moe_w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return DeepSpeedBlockBaselineFunction.apply(
            x,
            self.ln1_weight,
            self.ln1_bias,
            self.ln2_weight,
            self.ln2_bias,
            self.qkv_weight,
            self.proj_weight,
            self.router_weight,
            self.moe_w1,
            self.moe_w2,
            self.cp_group,
            self.ep_group,
            self.num_heads,
            self.num_kv_heads,
            self.num_experts,
            self.top_k,
            self.activation_func,
            self.capacity_factor,
            self.ulysses_attn,
        )


class DeepSpeedBlockBaselineTransformerModel(nn.Module):
    """Multi-layer block-level DeepSpeed baseline for the current benchmark."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        ffn_hidden_size: int,
        num_experts: int,
        top_k: int,
        cp_group: dist.ProcessGroup,
        ep_group: dist.ProcessGroup,
        activation_func: Optional[Callable] = None,
        capacity_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DeepSpeedBlockBaselineLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    num_experts=num_experts,
                    top_k=top_k,
                    cp_group=cp_group,
                    ep_group=ep_group,
                    layer_id=i,
                    activation_func=activation_func,
                    capacity_factor=capacity_factor,
                    dtype=dtype,
                    device=device,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
