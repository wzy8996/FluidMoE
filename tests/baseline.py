"""
Baseline Transformer Layer (No P2P Overlap)

Uses standard PyTorch AllToAll without overlap for fair comparison.
Uses attention recomputation (same as FluidMoE) for fair memory comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import contextmanager
from typing import Optional, Callable, List

from fluid.attention.forward import scaled_dot_product_attention_forward  # noqa: F401 (used in forward)


# ---------------------------------------------------------------------------
# 可重叠计算计时基础设施 (由 no_overlap_time_analyzer 设置)
# ---------------------------------------------------------------------------
_g_compute_timer = None


def set_compute_timer(timer) -> None:
    """设置可重叠计算计时器 (timer 需有 fwd_overlap_ms / bwd_overlap_ms 属性)。"""
    global _g_compute_timer
    _g_compute_timer = timer


@contextmanager
def _timed_overlap(phase: str, key: str = ""):
    """对一段可重叠计算块计时并累加到 timer。

    phase: 'fwd' 或 'bwd'
    key:   区域标识，对应 OverlapComputeTracker 的 {phase}_{key}_ms 字段。
           前向: 'qkv' / 'proj' / 'fc1' / 'fc2'
           后向: 'fc2_dx' / 'fc2_dw' / 'act_recomp' / 'act_bwd' /
                 'fc1_dx' / 'fc1_dw' / 'router_dx' / 'router_dw' /
                 'ln2_dw' / 'ln2_dx' / 'proj_dw' / 'proj_dx' /
                 'sdpa_dx' / 'qkv_dw' / 'qkv_dx' / 'ln1_dw' / 'ln1_dx'
    """
    timer = _g_compute_timer
    if timer is None:
        yield
        return
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    yield
    ev_e.record()
    ev_e.synchronize()
    ms = ev_s.elapsed_time(ev_e)
    attr = f"{phase}_{key}_ms"
    setattr(timer, attr, getattr(timer, attr) + ms)
    # 特殊追踪：反向第一层的 FC2 dX (R1, ÷n1) 和 FC1 dX (R2, ÷n2)，用于计算暴露计算量
    if phase == "bwd":
        if key == "fc2_dx":
            timer._bwd_fc2_dx_count += 1
            if timer._bwd_fc2_dx_count == 1:
                timer.bwd_first_fc2_dx_ms += ms
        elif key == "fc1_dx":
            timer._bwd_fc1_dx_count += 1
            if timer._bwd_fc1_dx_count == 1:
                timer.bwd_first_fc1_dx_ms += ms


class BaselineTransformerFunction(torch.autograd.Function):
    """Baseline transformer with attention recomputation (fair comparison with FluidMoE)."""

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        ln1_weight: torch.Tensor, ln1_bias: torch.Tensor,
        ln2_weight: torch.Tensor, ln2_bias: torch.Tensor,
        qkv_weight: torch.Tensor, proj_weight: torch.Tensor,
        router_weight: torch.Tensor,
        moe_w1: torch.Tensor, moe_w2: torch.Tensor,
        cp_group, ep_group,
        num_heads: int, num_kv_heads: int,
        num_experts: int, top_k: int,
        activation_func: Callable,
    ):
        seq_len, batch_size, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        cp_size = cp_group.size()
        ep_size = ep_group.size()
        ep_rank = ep_group.rank()
        head_dim = hidden_size // num_heads
        num_local_experts = num_experts // ep_size

        # ===================== Attention =====================
        ln1_out = F.layer_norm(hidden_states, (hidden_size,), ln1_weight, ln1_bias)
        with _timed_overlap("fwd", "qkv"):  # [可重叠] R4 QKV GEMM
            qkv = F.linear(ln1_out, qkv_weight)

        # Reshape QKV
        q_per_kv = num_heads // num_kv_heads
        group_size = (q_per_kv + 2) * head_dim
        qkv = qkv.view(seq_len, batch_size, num_kv_heads, group_size)

        q_dim = q_per_kv * head_dim
        q_sp = qkv[:, :, :, :q_dim].reshape(seq_len, batch_size, num_heads, head_dim)
        k_sp = qkv[:, :, :, q_dim:q_dim + head_dim]
        v_sp = qkv[:, :, :, q_dim + head_dim:]

        # AllToAll: sp2hp (combined QKV)
        qkv_sp = torch.cat([q_sp, k_sp, v_sp], dim=2)  # [seq, batch, heads+2*kv_heads, dim]
        qkv_hp = _all_to_all_sp2hp(qkv_sp, cp_group)
        q_heads_local = num_heads // cp_size
        kv_heads_local = num_kv_heads // cp_size
        q_hp = qkv_hp[:, :, :q_heads_local, :]
        k_hp = qkv_hp[:, :, q_heads_local:q_heads_local + kv_heads_local, :]
        v_hp = qkv_hp[:, :, q_heads_local + kv_heads_local:, :]

        # Attention with native GQA support (PyTorch 2.5+)
        q_bf = q_hp.permute(1, 2, 0, 3)  # [batch, q_heads_local, seq, head_dim]
        k_bf = k_hp.permute(1, 2, 0, 3)  # [batch, kv_heads_local, seq, head_dim]
        v_bf = v_hp.permute(1, 2, 0, 3)  # [batch, kv_heads_local, seq, head_dim]

        enable_gqa = (q_heads_local != kv_heads_local)
        scale = 1.0 / (head_dim ** 0.5)

        # Keep computation graph for backward (avoids redundant SDPA forward in backward)
        with torch.enable_grad():
            q_for_attn = q_bf.detach().requires_grad_(True)
            k_for_attn = k_bf.detach().requires_grad_(True)
            v_for_attn = v_bf.detach().requires_grad_(True)
            attn_out_bf = scaled_dot_product_attention_forward(
                q_for_attn, k_for_attn, v_for_attn, scale=scale, is_causal=True, enable_gqa=enable_gqa
            )
        attn_out = attn_out_bf.permute(2, 0, 1, 3).contiguous()

        # AllToAll: hp2sp
        attn_out_sp = _all_to_all_hp2sp(attn_out, cp_group)

        # Output projection
        with _timed_overlap("fwd", "proj"):  # [可重叠] R3 OutProj GEMM
            proj_out = F.linear(attn_out_sp.view(seq_len, batch_size, -1), proj_weight)
        hidden_after_attn = hidden_states + proj_out

        # ===================== MoE =====================
        ln2_out = F.layer_norm(hidden_after_attn, (hidden_size,), ln2_weight, ln2_bias)
        ln2_flat = ln2_out.view(-1, hidden_size)
        num_tokens = ln2_flat.shape[0]

        # Router
        router_logits = F.linear(ln2_flat.float(), router_weight.t())
        router_probs = F.softmax(router_logits, dim=-1)
        top_probs, top_indices = torch.topk(router_probs, k=top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        # Expand and sort
        expanded_tokens = ln2_flat.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
        expanded_probs = top_probs.reshape(-1)
        expanded_expert_indices = top_indices.reshape(-1)

        sorted_indices = torch.argsort(expanded_expert_indices, stable=True)
        permuted_tokens = expanded_tokens[sorted_indices]
        permuted_probs = expanded_probs[sorted_indices]
        sorted_expert_indices = expanded_expert_indices[sorted_indices]

        # Count and splits
        tokens_per_expert = torch.bincount(sorted_expert_indices, minlength=num_experts)
        experts_per_rank = num_experts // ep_size

        input_splits = [tokens_per_expert[i * experts_per_rank:(i + 1) * experts_per_rank].sum().item()
                        for i in range(ep_size)]

        # AllGather splits
        all_splits = [None] * ep_size
        dist.all_gather_object(all_splits, input_splits, group=ep_group)
        output_splits = [all_splits[r][ep_rank] for r in range(ep_size)]

        # Dispatch AllToAll
        recv_tokens = _moe_all_to_all(permuted_tokens, input_splits, output_splits, ep_group)

        # Get expert token counts
        all_tpe = [None] * ep_size
        local_tpe = tokens_per_expert[ep_rank * num_local_experts:(ep_rank + 1) * num_local_experts].tolist()
        dist.all_gather_object(all_tpe, local_tpe, group=ep_group)

        tokens_per_local_expert = [sum(all_tpe[r][e] for r in range(ep_size)) for e in range(num_local_experts)]

        # Expert compute (保存 fc1 供反向使用，避免 FC1 recomp)
        total_recv = recv_tokens.shape[0]
        expert_output = torch.zeros(total_recv, hidden_size, dtype=dtype, device=device)
        fc1_all = torch.empty(total_recv, moe_w1.shape[2], dtype=dtype, device=device)
        offset = 0
        for exp_idx in range(num_local_experts):
            n_tok = tokens_per_local_expert[exp_idx]
            if n_tok > 0:
                with _timed_overlap("fwd", "fc1"):  # [可重叠] R2 FC1 GEMM + activation
                    fc1 = torch.matmul(recv_tokens[offset:offset + n_tok], moe_w1[exp_idx])
                    fc1_all[offset:offset + n_tok] = fc1
                    act = activation_func(fc1)
                with _timed_overlap("fwd", "fc2"):  # [可重叠] R1 FC2 GEMM
                    expert_output[offset:offset + n_tok] = torch.matmul(act, moe_w2[exp_idx])
                offset += n_tok

        # Combine AllToAll
        combined_output = _moe_all_to_all(expert_output, output_splits, input_splits, ep_group)

        # Restore
        restore_indices = torch.argsort(sorted_indices)
        weighted = combined_output * permuted_probs.unsqueeze(-1).to(dtype)
        restored = weighted[restore_indices]
        moe_output = restored.view(num_tokens, top_k, hidden_size).sum(dim=1)
        moe_output = moe_output.view(seq_len, batch_size, hidden_size)

        output = hidden_after_attn + moe_output

        # Save for backward
        ctx.save_for_backward(
            hidden_states, ln1_out, hidden_after_attn, ln2_flat,
            permuted_tokens, permuted_probs, restore_indices, sorted_indices,
            router_probs, top_indices, recv_tokens, combined_output,
            ln1_weight, ln1_bias, ln2_weight, ln2_bias,
            qkv_weight, proj_weight, router_weight, moe_w1, moe_w2,
            fc1_all,
        )
        # Save SDPA computation graph (avoids redundant forward in backward)
        ctx._q_for_attn = q_for_attn
        ctx._k_for_attn = k_for_attn
        ctx._v_for_attn = v_for_attn
        ctx._attn_out_bf = attn_out_bf
        ctx._attn_out_sp = attn_out_sp  # Save SP format to avoid extra hp2sp in backward
        ctx._enable_gqa = enable_gqa
        ctx.cp_group = cp_group
        ctx.ep_group = ep_group
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.activation_func = activation_func
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.tokens_per_local_expert = tokens_per_local_expert
        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        (hidden_states, ln1_out, hidden_after_attn, ln2_flat,
         permuted_tokens, permuted_probs, restore_indices, sorted_indices,
         router_probs, top_indices, recv_tokens, combined_output,
         ln1_weight, ln1_bias, ln2_weight, ln2_bias,
         qkv_weight, proj_weight, router_weight, moe_w1, moe_w2,
         fc1_all) = ctx.saved_tensors

        # Retrieve SDPA computation graph
        q_for_attn = ctx._q_for_attn
        k_for_attn = ctx._k_for_attn
        v_for_attn = ctx._v_for_attn
        attn_out_bf = ctx._attn_out_bf
        attn_out_sp = ctx._attn_out_sp
        enable_gqa = ctx._enable_gqa

        cp_group = ctx.cp_group
        ep_group = ctx.ep_group
        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        num_experts = ctx.num_experts
        top_k = ctx.top_k
        activation_func = ctx.activation_func
        input_splits = ctx.input_splits
        output_splits = ctx.output_splits
        tokens_per_local_expert = ctx.tokens_per_local_expert
        scale = ctx.scale

        seq_len, batch_size, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        cp_size = cp_group.size()
        ep_size = ep_group.size()
        head_dim = hidden_size // num_heads
        num_local_experts = num_experts // ep_size
        num_tokens = seq_len * batch_size

        # ===================== MoE Backward =====================
        grad_output_flat = grad_output.view(num_tokens, hidden_size)
        grad_hidden_after_attn = grad_output.clone()

        # Restore backward - expand grad to match top_k expanded tokens
        grad_expanded = grad_output_flat.unsqueeze(1).expand(-1, top_k, -1).reshape(num_tokens * top_k, hidden_size)
        grad_weighted = grad_expanded[sorted_indices]  # reorder to match permuted order
        grad_combined = grad_weighted * permuted_probs.unsqueeze(-1).to(dtype)  # weight by routing probs

        # Combine AllToAll backward
        grad_expert_output = _moe_all_to_all(grad_combined, input_splits, output_splits, ep_group)

        # Expert backward: FC2 and FC1 timed separately (analogous to forward fwd_fc2/fwd_fc1)
        grad_recv_tokens = torch.zeros_like(recv_tokens)
        grad_moe_w1 = torch.zeros_like(moe_w1)
        grad_moe_w2 = torch.zeros_like(moe_w2)

        # Pre-compute per-expert offsets
        expert_offsets = []
        off = 0
        for exp_idx in range(num_local_experts):
            expert_offsets.append(off)
            off += tokens_per_local_expert[exp_idx]

        # FC2 dX matmul (pipeline 关键路径, 暴露: 1/n1)
        grad_act_per_expert = [None] * num_local_experts
        with _timed_overlap("bwd", "fc2_dx"):
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_local_expert[exp_idx]
                if n_tok > 0:
                    o = expert_offsets[exp_idx]
                    grad_act_per_expert[exp_idx] = torch.matmul(
                        grad_expert_output[o:o + n_tok], moe_w2[exp_idx].t())

        # Activation backward (AR 可重叠): single activation graph + autograd.grad
        # Also reuse detached act output for FC2 dW to avoid duplicate activation compute.
        with _timed_overlap("bwd", "act_bwd"):
            if fc1_all.numel() == 0:
                grad_fc1_all = torch.empty_like(fc1_all)
                act_all_detached = torch.empty_like(fc1_all)
            else:
                grad_act_all = torch.zeros_like(fc1_all)
                for exp_idx in range(num_local_experts):
                    grad_act_slice = grad_act_per_expert[exp_idx]
                    if grad_act_slice is None:
                        continue
                    o = expert_offsets[exp_idx]
                    n_valid = grad_act_slice.shape[0]
                    if n_valid > 0:
                        grad_act_all[o:o + n_valid] = grad_act_slice
                with torch.enable_grad():
                    fc1_with_grad = fc1_all.detach().requires_grad_(True)
                    act_all = activation_func(fc1_with_grad)
                    grad_fc1_all, = torch.autograd.grad(
                        act_all, fc1_with_grad, grad_act_all, retain_graph=False
                    )
                act_all_detached = act_all.detach()

        # FC2 dW matmul (AllToAll 可重叠) - reuse activation output from act_bwd stage
        with _timed_overlap("bwd", "fc2_dw"):
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_local_expert[exp_idx]
                if n_tok > 0:
                    o = expert_offsets[exp_idx]
                    act_slice = act_all_detached[o:o + n_tok]
                    grad_moe_w2[exp_idx] = torch.matmul(
                        act_slice.t(), grad_expert_output[o:o + n_tok])

        # FC1 dX matmul (pipeline 关键路径, 暴露: 1/n2)
        with _timed_overlap("bwd", "fc1_dx"):
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_local_expert[exp_idx]
                if n_tok > 0:
                    o = expert_offsets[exp_idx]
                    grad_fc1_slice = grad_fc1_all[o:o + n_tok]
                    grad_recv_tokens[o:o + n_tok] = torch.matmul(
                        grad_fc1_slice, moe_w1[exp_idx].t())

        # FC1 dW matmul (AllToAll 可重叠)
        with _timed_overlap("bwd", "fc1_dw"):
            for exp_idx in range(num_local_experts):
                n_tok = tokens_per_local_expert[exp_idx]
                if n_tok > 0:
                    o = expert_offsets[exp_idx]
                    grad_fc1_slice = grad_fc1_all[o:o + n_tok]
                    grad_moe_w1[exp_idx] = torch.matmul(
                        recv_tokens[o:o + n_tok].t(), grad_fc1_slice)

        # Dispatch AllToAll backward
        grad_permuted_tokens = _moe_all_to_all(grad_recv_tokens, output_splits, input_splits, ep_group)

        # Unsort
        grad_expanded_tokens = grad_permuted_tokens[torch.argsort(sorted_indices)]
        grad_ln2_flat_from_tokens = grad_expanded_tokens.view(num_tokens, top_k, hidden_size).sum(dim=1)

        # Router backward dX (AR 可重叠)
        with _timed_overlap("bwd", "router_dx"):
            grad_permuted_probs = (grad_weighted * combined_output.to(dtype)).sum(dim=-1)
            grad_expanded_probs = torch.zeros_like(grad_permuted_probs)
            grad_expanded_probs[sorted_indices] = grad_permuted_probs
            grad_top_probs = grad_expanded_probs.view(num_tokens, top_k)
            top_probs_saved = permuted_probs[restore_indices].view(num_tokens, top_k)
            grad_dot = (grad_top_probs * top_probs_saved).sum(dim=-1, keepdim=True)
            grad_raw_top_probs = (grad_top_probs - grad_dot * top_probs_saved) / top_probs_saved.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            grad_router_probs = torch.zeros_like(router_probs)
            grad_router_probs.scatter_(1, top_indices, grad_raw_top_probs)
            sum_grad_probs = (grad_router_probs * router_probs).sum(dim=-1, keepdim=True)
            grad_router_logits = router_probs * (grad_router_probs - sum_grad_probs)
            grad_ln2_flat_from_router = torch.matmul(grad_router_logits.float(), router_weight.t().float()).to(dtype)

        # Router dW (AllToAll 可重叠)
        with _timed_overlap("bwd", "router_dw"):
            grad_router_weight = torch.matmul(ln2_flat.t().float(), grad_router_logits.float())

        # Combine gradients from tokens and router
        grad_ln2_flat = grad_ln2_flat_from_tokens + grad_ln2_flat_from_router

        # LayerNorm2 backward
        mean2 = hidden_after_attn.mean(dim=-1, keepdim=True)
        var2 = hidden_after_attn.var(dim=-1, unbiased=False, keepdim=True)
        std2 = (var2 + 1e-5).sqrt()
        normalized2 = (hidden_after_attn - mean2) / std2

        grad_ln2_out = grad_ln2_flat.view(seq_len, batch_size, hidden_size)
        with _timed_overlap("bwd", "ln2_dw"):  # LN2 dW (AllToAll 可重叠)
            grad_ln2_weight = (grad_ln2_out * normalized2).sum(dim=(0, 1))
            grad_ln2_bias = grad_ln2_out.sum(dim=(0, 1))
        with _timed_overlap("bwd", "ln2_dx"):  # LN2 dX (AR 可重叠)
            grad_hidden_after_attn = grad_hidden_after_attn + grad_ln2_out * ln2_weight / std2

        # ===================== Attention Backward =====================
        # OutProj dW (AllToAll 可重叠)
        grad_proj_out = grad_hidden_after_attn.view(seq_len, batch_size, hidden_size)
        with _timed_overlap("bwd", "proj_dw"):
            grad_proj_weight = torch.matmul(
                grad_proj_out.view(-1, hidden_size).t(),
                attn_out_sp.view(-1, num_heads * head_dim)
            )
        # OutProj dX (pipeline 关键路径, 暴露: 1/n3)
        with _timed_overlap("bwd", "proj_dx"):
            grad_attn_out_sp = torch.matmul(grad_proj_out.view(-1, hidden_size), proj_weight)
            grad_attn_out_sp = grad_attn_out_sp.view(seq_len, batch_size, num_heads, head_dim)

        # sp2hp AllToAll backward
        grad_attn_out = _all_to_all_sp2hp(grad_attn_out_sp, cp_group)

        # SDPA backward (AR 可重叠)
        grad_attn_bf = grad_attn_out.permute(1, 2, 0, 3)
        with _timed_overlap("bwd", "sdpa_dx"):
            grad_q_bf, grad_k_bf, grad_v_bf = torch.autograd.grad(
                attn_out_bf, (q_for_attn, k_for_attn, v_for_attn),
                grad_attn_bf, retain_graph=False,
            )

        grad_q_hp = grad_q_bf.permute(2, 0, 1, 3)
        grad_k_hp = grad_k_bf.permute(2, 0, 1, 3)
        grad_v_hp = grad_v_bf.permute(2, 0, 1, 3)

        # hp2sp AllToAll backward (combined QKV)
        grad_qkv_hp = torch.cat([grad_q_hp, grad_k_hp, grad_v_hp], dim=2)
        grad_qkv_sp = _all_to_all_hp2sp(grad_qkv_hp, cp_group)
        grad_q_sp = grad_qkv_sp[:, :, :num_heads, :]
        grad_k_sp = grad_qkv_sp[:, :, num_heads:num_heads + num_kv_heads, :]
        grad_v_sp = grad_qkv_sp[:, :, num_heads + num_kv_heads:, :]

        # QKV reshape (shared setup)
        q_per_kv = num_heads // num_kv_heads
        grad_qkv = torch.zeros(seq_len, batch_size, num_kv_heads, (q_per_kv + 2) * head_dim,
                               dtype=dtype, device=device)
        q_dim = q_per_kv * head_dim
        grad_qkv[:, :, :, :q_dim] = grad_q_sp.view(seq_len, batch_size, num_kv_heads, q_dim)
        grad_qkv[:, :, :, q_dim:q_dim + head_dim] = grad_k_sp
        grad_qkv[:, :, :, q_dim + head_dim:] = grad_v_sp
        grad_qkv_flat = grad_qkv.view(seq_len * batch_size, -1)

        # QKV dW (AllToAll 可重叠)
        with _timed_overlap("bwd", "qkv_dw"):
            grad_qkv_weight = torch.matmul(grad_qkv_flat.t(), ln1_out.view(-1, hidden_size))
        # QKV dX (pipeline 关键路径, 暴露: 1/n4)
        with _timed_overlap("bwd", "qkv_dx"):
            grad_ln1_out = torch.matmul(grad_qkv_flat, qkv_weight).view(seq_len, batch_size, hidden_size)

        # Residual
        grad_hidden_states = grad_hidden_after_attn.clone()

        # LayerNorm1 backward
        mean1 = hidden_states.mean(dim=-1, keepdim=True)
        var1 = hidden_states.var(dim=-1, unbiased=False, keepdim=True)
        std1 = (var1 + 1e-5).sqrt()
        normalized1 = (hidden_states - mean1) / std1

        # LN1 dW (AllToAll 可重叠)
        with _timed_overlap("bwd", "ln1_dw"):
            grad_ln1_weight = (grad_ln1_out * normalized1).sum(dim=(0, 1))
            grad_ln1_bias = grad_ln1_out.sum(dim=(0, 1))
        # LN1 dX (AR 可重叠)
        with _timed_overlap("bwd", "ln1_dx"):
            grad_hidden_states = grad_hidden_states + grad_ln1_out * ln1_weight / std1

        return (
            grad_hidden_states,
            grad_ln1_weight, grad_ln1_bias,
            grad_ln2_weight, grad_ln2_bias,
            grad_qkv_weight, grad_proj_weight,
            grad_router_weight,
            grad_moe_w1, grad_moe_w2,
            None, None,  # groups
            None, None, None, None, None,  # config
        )


def _all_to_all_sp2hp(x: torch.Tensor, group) -> torch.Tensor:
    """sp2hp AllToAll (Ulysses style)."""
    cp = group.size()
    seq_local, batch, heads, dim = x.shape

    x = x.contiguous().view(seq_local, batch, cp, heads // cp, dim)
    x = x.permute(2, 0, 1, 3, 4).contiguous().view(seq_local * cp, -1)

    output = torch.empty_like(x)
    dist.all_to_all_single(output, x,
                           output_split_sizes=[seq_local] * cp,
                           input_split_sizes=[seq_local] * cp,
                           group=group)

    return output.view(seq_local * cp, batch, heads // cp, dim)


def _all_to_all_hp2sp(x: torch.Tensor, group) -> torch.Tensor:
    """hp2sp AllToAll (Ulysses style)."""
    cp = group.size()
    seq_full, batch, heads_local, dim = x.shape
    seq_local = seq_full // cp

    x = x.contiguous().view(seq_full, batch * heads_local * dim)

    output = torch.empty_like(x)
    dist.all_to_all_single(output, x,
                           output_split_sizes=[seq_local] * cp,
                           input_split_sizes=[seq_local] * cp,
                           group=group)

    output = output.view(cp, seq_local, batch, heads_local, dim)
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    return output.view(seq_local, batch, heads_local * cp, dim)


def _moe_all_to_all(x: torch.Tensor, send_splits: List[int], recv_splits: List[int], group) -> torch.Tensor:
    """MoE AllToAll."""
    output = torch.empty(sum(recv_splits), x.shape[1], dtype=x.dtype, device=x.device)
    input_list = list(x.split(send_splits, dim=0))
    output_list = list(output.split(recv_splits, dim=0))
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=0)


class BaselineTransformerLayer(nn.Module):
    """Baseline Transformer layer with attention recomputation (fair comparison)."""

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
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device(f'cuda:{dist.get_rank()}')

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_id = layer_id
        self.activation_func = activation_func or F.gelu

        self.cp_group = cp_group
        self.ep_group = ep_group
        self.ep_size = ep_group.size()

        num_local_experts = num_experts // self.ep_size
        head_dim = hidden_size // num_heads

        # LayerNorm
        self.ln1_weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.ln1_bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.ln2_weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.ln2_bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

        # Attention weights
        q_per_kv = num_heads // num_kv_heads
        qkv_size = num_kv_heads * (q_per_kv + 2) * head_dim
        self.qkv_weight = nn.Parameter(torch.empty(qkv_size, hidden_size, dtype=dtype, device=device))
        self.proj_weight = nn.Parameter(torch.empty(hidden_size, num_heads * head_dim, dtype=dtype, device=device))

        # MoE weights
        self.router_weight = nn.Parameter(torch.empty(hidden_size, num_experts, dtype=torch.float32, device=device))
        self.moe_w1 = nn.Parameter(torch.empty(num_local_experts, hidden_size, ffn_hidden_size, dtype=dtype, device=device))
        self.moe_w2 = nn.Parameter(torch.empty(num_local_experts, ffn_hidden_size, hidden_size, dtype=dtype, device=device))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.xavier_uniform_(self.proj_weight)
        nn.init.xavier_uniform_(self.router_weight)
        nn.init.xavier_uniform_(self.moe_w1)
        nn.init.xavier_uniform_(self.moe_w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return BaselineTransformerFunction.apply(
            x,
            self.ln1_weight, self.ln1_bias,
            self.ln2_weight, self.ln2_bias,
            self.qkv_weight, self.proj_weight,
            self.router_weight,
            self.moe_w1, self.moe_w2,
            self.cp_group, self.ep_group,
            self.num_heads, self.num_kv_heads,
            self.num_experts, self.top_k,
            self.activation_func,
        )


class BaselineTransformerModel(nn.Module):
    """Baseline Transformer model."""

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
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BaselineTransformerLayer(
                hidden_size, num_heads, num_kv_heads, ffn_hidden_size,
                num_experts, top_k, cp_group, ep_group, i,
                activation_func, dtype, device,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
