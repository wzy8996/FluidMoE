"""
FluidMoE Unified Transformer Layer

Single autograd.Function for complete Transformer layer (Attention + MoE).
Minimizes Python overhead by combining all operations into one Function.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Callable, Tuple, List

from fluid.core.comm import MultiCardOverlapContext
from fluid.core.scheduler import get_backward_scheduler
from fluid.core import _sort_chunks_by_idxs

# Import forward operations
from fluid.attention.forward import (
    qkv_projection_p2p_forward,
    scaled_dot_product_attention_forward,
    output_projection_p2p_forward,
)
from fluid.moe.forward import (
    router_forward,
    dispatch_fc1_p2p_forward,
    fc2_combine_p2p_forward,
)

# Import backward operations
from fluid.attention.backward import (
    output_projection_backward_chunked,
    attention_backward_chunked,
    qkv_projection_backward,
    output_projection_register_dw,
)
from fluid.moe.backward import (
    combine_backward,
    expert_dispatch_backward,
    router_backward,
    register_router_dw_task,
)


class TransformerLayerFunction(torch.autograd.Function):
    """
    Unified autograd.Function for complete Transformer layer.

    Combines Attention and MoE into single Function to minimize Python overhead.

    Forward: LN1 -> Attention -> Residual -> LN2 -> MoE -> Residual
    Backward: Reverse order with P2P overlap and dW scheduling
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        # LayerNorm weights
        ln1_weight: torch.Tensor,
        ln1_bias: torch.Tensor,
        ln2_weight: torch.Tensor,
        ln2_bias: torch.Tensor,
        # Attention weights
        qkv_weight: torch.Tensor,
        proj_weight: torch.Tensor,
        # MoE weights
        router_weight: torch.Tensor,
        moe_w1: torch.Tensor,
        moe_w2: torch.Tensor,
        # Groups and context
        cp_group: dist.ProcessGroup,
        ep_group: dist.ProcessGroup,
        attn_overlap_ctx: MultiCardOverlapContext,
        moe_overlap_ctx: MultiCardOverlapContext,
        # Config
        layer_id: int,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int,
        top_k: int,
        # Chunk configs for different backward passes
        attn_proj_chunks: int,  # output projection backward (sp2hp)
        attn_qkv_chunks: int,   # attention/qkv backward (hp2sp)
        moe_chunks: int,        # MoE dispatch backward
        activation_func: Callable,
    ) -> torch.Tensor:
        """Forward pass for complete Transformer layer."""
        needs_grad = hidden_states.requires_grad
        ctx.needs_grad = needs_grad

        device = hidden_states.device
        dtype = hidden_states.dtype
        seq_len, batch_size, hidden_size = hidden_states.shape

        cp_size = cp_group.size()
        ep_size = ep_group.size()
        my_ep_rank = ep_group.rank()
        num_local_experts = num_experts // ep_size
        head_dim = hidden_size // num_heads

        # Detach weights for forward computation
        qkv_weight_d = qkv_weight.detach()
        proj_weight_d = proj_weight.detach()
        moe_w1_d = moe_w1.detach()
        moe_w2_d = moe_w2.detach()

        # =====================================================================
        # Attention Block: LN1 -> Attention -> Residual
        # =====================================================================

        # LayerNorm 1
        ln1_out = F.layer_norm(hidden_states, (hidden_size,), ln1_weight, ln1_bias)

        # QKV Projection with P2P overlap
        # Returns q, k, v in format [seq_full, batch, heads_local, head_dim]
        q_hp, k_hp, v_hp = qkv_projection_p2p_forward(
            ln1_out, qkv_weight_d, num_heads, num_kv_heads, head_dim,
            cp_group, attn_overlap_ctx
        )

        # GQA expansion if needed (expand K/V heads to match Q heads for attention)
        kv_heads_local = num_kv_heads // cp_size
        q_heads_local = num_heads // cp_size
        if q_heads_local > kv_heads_local:
            expand_ratio = q_heads_local // kv_heads_local
            k_hp_expanded = k_hp.repeat_interleave(expand_ratio, dim=2)
            v_hp_expanded = v_hp.repeat_interleave(expand_ratio, dim=2)
        else:
            k_hp_expanded = k_hp
            v_hp_expanded = v_hp

        # Convert to batch-first format for attention: [seq, batch, heads, dim] -> [batch, heads, seq, dim]
        # Note: permute creates a view, matmul works with non-contiguous tensors
        q_bf = q_hp.permute(1, 2, 0, 3)
        k_bf = k_hp_expanded.permute(1, 2, 0, 3)
        v_bf = v_hp_expanded.permute(1, 2, 0, 3)

        # Compute scale for attention
        scale = 1.0 / (head_dim ** 0.5)

        # Scaled Dot-Product Attention
        # Input/Output: [batch, heads, seq, head_dim]
        attn_out_bf = scaled_dot_product_attention_forward(
            q_bf, k_bf, v_bf, scale=scale, is_causal=True
        )

        # Convert back to seq-first: [batch, heads, seq, dim] -> [seq, batch, heads, dim]
        # Need contiguous for output_projection_p2p_forward's P2P slicing
        attn_out = attn_out_bf.permute(2, 0, 1, 3).contiguous()

        # Output Projection with P2P overlap
        proj_out, attn_input_full = output_projection_p2p_forward(
            attn_out, proj_weight_d, None, cp_group, attn_overlap_ctx
        )

        # Residual connection
        hidden_after_attn = hidden_states + proj_out

        # =====================================================================
        # MoE Block: LN2 -> MoE -> Residual
        # =====================================================================

        # LayerNorm 2
        ln2_out = F.layer_norm(hidden_after_attn, (hidden_size,), ln2_weight, ln2_bias)

        # Flatten for MoE: [seq, batch, hidden] -> [seq*batch, hidden]
        ln2_flat = ln2_out.view(-1, hidden_size)
        num_tokens = ln2_flat.shape[0]

        # Router forward (AllGather for tokens_per_expert_2d removed - metadata piggybacked on P2P)
        (permuted_tokens, permuted_probs, restore_indices, sorted_indices,
         input_splits, output_splits, tokens_per_expert,
         router_probs, top_indices, router_logits) = router_forward(
            ln2_flat, router_weight, num_experts, top_k, ep_group
        )

        input_splits_list = input_splits.tolist()
        output_splits_list = output_splits.tolist()

        # Dispatch + FC1 with P2P overlap (tokens_cpu built from P2P metadata)
        (local_tokens, local_act, recv_act_results, recv_buffers,
         moe_partners, recv_offsets, tokens_cpu) = dispatch_fc1_p2p_forward(
            permuted_tokens, moe_w1_d, input_splits_list, output_splits_list,
            ep_group, moe_overlap_ctx, activation_func, num_local_experts,
            tokens_per_expert, needs_backward=needs_grad,
        )

        # FC2 + Combine with P2P overlap (uses tokens_cpu from dispatch)
        (combined_output, local_fc2, all_expert_tokens,
         all_tokens_per_expert, backward_indices) = fc2_combine_p2p_forward(
            local_tokens, local_act, recv_act_results, recv_buffers,
            moe_w2_d, input_splits_list, output_splits_list,
            ep_group, moe_overlap_ctx, num_local_experts,
            moe_partners, tokens_cpu, needs_backward=needs_grad,
        )

        # Apply probs and restore order
        weighted_output = combined_output * permuted_probs.unsqueeze(-1).to(dtype)
        restored_output = weighted_output[restore_indices]
        moe_output = restored_output.view(num_tokens, top_k, hidden_size).sum(dim=1)

        # Reshape back: [seq*batch, hidden] -> [seq, batch, hidden]
        moe_output = moe_output.view(seq_len, batch_size, hidden_size)

        # Residual connection
        output = hidden_after_attn + moe_output

        # =====================================================================
        # Save for backward
        # =====================================================================
        if needs_grad:
            # Compute ffn_hidden for backward
            total_ffn_hidden = moe_w1.shape[-1]
            ffn_hidden = total_ffn_hidden // num_local_experts

            ctx.save_for_backward(
                # Input and intermediate states
                hidden_states, ln1_out, hidden_after_attn, ln2_flat,
                # Attention states (save before permute for consistency)
                q_hp, k_hp, v_hp, attn_input_full,
                # MoE states
                permuted_tokens, permuted_probs, restore_indices, sorted_indices,
                router_probs, top_indices, all_expert_tokens, combined_output,
                # Weights (detached)
                ln1_weight, ln1_bias, ln2_weight, ln2_bias,
                qkv_weight.detach(), proj_weight.detach(),
                router_weight.detach(), moe_w1.detach(), moe_w2.detach(),
            )
            # Store original weights for gradient assignment
            ctx._orig_ln1_weight = ln1_weight
            ctx._orig_ln1_bias = ln1_bias
            ctx._orig_ln2_weight = ln2_weight
            ctx._orig_ln2_bias = ln2_bias
            ctx._orig_qkv_weight = qkv_weight
            ctx._orig_proj_weight = proj_weight
            ctx._orig_router_weight = router_weight
            ctx._orig_moe_w1 = moe_w1
            ctx._orig_moe_w2 = moe_w2

            # Store config
            ctx.cp_group = cp_group
            ctx.ep_group = ep_group
            ctx.layer_id = layer_id
            ctx.num_heads = num_heads
            ctx.num_kv_heads = num_kv_heads
            ctx.num_experts = num_experts
            ctx.top_k = top_k
            ctx.attn_proj_chunks = attn_proj_chunks
            ctx.attn_qkv_chunks = attn_qkv_chunks
            ctx.moe_chunks = moe_chunks
            ctx.activation_func = activation_func
            ctx.num_local_experts = num_local_experts
            ctx.head_dim = head_dim
            ctx.ffn_hidden = ffn_hidden
            ctx.scale = scale

            # MoE routing info
            ctx.input_splits_list = input_splits_list
            ctx.output_splits_list = output_splits_list
            ctx.backward_indices = backward_indices
            ctx.all_tokens_per_expert = all_tokens_per_expert

            # Shape info
            ctx.seq_len = seq_len
            ctx.batch_size = batch_size
            ctx.hidden_size = hidden_size
            ctx.num_tokens = num_tokens

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass with P2P overlap and dW scheduling."""
        if not ctx.needs_grad:
            return (None,) * 23  # 10 weights + 4 groups/contexts + 9 config params

        # Retrieve saved tensors
        (hidden_states, ln1_out, hidden_after_attn, ln2_flat,
         q_hp, k_hp, v_hp, attn_input_full,
         permuted_tokens, permuted_probs, restore_indices, sorted_indices,
         router_probs, top_indices, all_expert_tokens, combined_output,
         ln1_weight, ln1_bias, ln2_weight, ln2_bias,
         qkv_weight, proj_weight,
         router_weight, moe_w1_2d, moe_w2_2d) = ctx.saved_tensors

        # Retrieve config
        cp_group = ctx.cp_group
        ep_group = ctx.ep_group
        layer_id = ctx.layer_id
        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        num_experts = ctx.num_experts
        top_k = ctx.top_k
        attn_proj_chunks = ctx.attn_proj_chunks
        attn_qkv_chunks = ctx.attn_qkv_chunks
        moe_chunks = ctx.moe_chunks
        activation_func = ctx.activation_func
        num_local_experts = ctx.num_local_experts
        head_dim = ctx.head_dim
        ffn_hidden = ctx.ffn_hidden
        scale = ctx.scale

        input_splits_list = ctx.input_splits_list
        output_splits_list = ctx.output_splits_list
        backward_indices = ctx.backward_indices
        all_tokens_per_expert = ctx.all_tokens_per_expert

        seq_len = ctx.seq_len
        batch_size = ctx.batch_size
        hidden_size = ctx.hidden_size
        num_tokens = ctx.num_tokens

        orig_ln1_weight = ctx._orig_ln1_weight
        orig_ln1_bias = ctx._orig_ln1_bias
        orig_ln2_weight = ctx._orig_ln2_weight
        orig_ln2_bias = ctx._orig_ln2_bias
        orig_qkv_weight = ctx._orig_qkv_weight
        orig_proj_weight = ctx._orig_proj_weight
        orig_router_weight = ctx._orig_router_weight
        orig_moe_w1 = ctx._orig_moe_w1
        orig_moe_w2 = ctx._orig_moe_w2

        scheduler = get_backward_scheduler()
        dtype = hidden_states.dtype
        device = grad_output.device
        cp_size = cp_group.size()

        # MoE weights are already 3D: [num_local_experts, hidden/ffn, ffn/hidden]
        moe_w1 = moe_w1_2d  # Already in shape [E, hidden, ffn]
        moe_w2 = moe_w2_2d  # Already in shape [E, ffn, hidden]

        # Compute actual chunk values (1 if scheduler disabled)
        is_enabled = scheduler.is_enabled()
        attn_proj_chunks_actual = attn_proj_chunks if is_enabled else 1
        attn_qkv_chunks_actual = attn_qkv_chunks if is_enabled else 1
        moe_chunks_actual = moe_chunks if is_enabled else 1

        # =====================================================================
        # MoE Backward (reverse order)
        # =====================================================================

        # Gradient through residual: grad flows to both hidden_after_attn and moe_output
        grad_hidden_after_attn = grad_output.clone()
        grad_moe_output = grad_output.view(num_tokens, hidden_size)

        # Step 1: Backward through sum and restore
        grad_restored = grad_moe_output.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
        inverse_restore_indices = torch.argsort(restore_indices)
        grad_weighted = grad_restored[inverse_restore_indices]

        # Step 2: Backward through prob weighting
        grad_combined = grad_weighted * permuted_probs.unsqueeze(-1).to(grad_weighted.dtype)
        grad_permuted_probs = (grad_weighted * combined_output.to(grad_weighted.dtype)).sum(dim=-1)

        # Step 3: Combine Backward AllToAll with FC1 recomputation
        grad_combined_recv, all_fc1 = combine_backward(
            grad_combined, input_splits_list, output_splits_list, ep_group, layer_id,
            all_expert_tokens=all_expert_tokens,
            weight1=moe_w1,
            num_local_experts=num_local_experts,
            all_tokens_per_expert=all_tokens_per_expert,
        )

        # Step 4: Convert layout: rank-major -> expert-major
        if 'split_sizes_rank_major' in backward_indices:
            grad_all_fc2 = _sort_chunks_by_idxs(
                grad_combined_recv,
                backward_indices['split_sizes_rank_major'],
                backward_indices['sorted_idxs_rank_to_exp'],
            )
        else:
            grad_all_fc2 = grad_combined_recv

        # Step 5-7: Combined expert backward + dW registration + dispatch AllToAll
        # This computes grad_fc1, registers dW tasks, then dX with chunking overlapped with AllToAll
        split_sizes_exp_major = backward_indices.get('split_sizes_exp_major', all_tokens_per_expert)
        sorted_idxs_exp_to_rank = backward_indices.get('sorted_idxs_exp_to_rank', list(range(len(all_tokens_per_expert))))

        grad_all_fc1, act_output, grad_permuted_tokens, grad_moe_w1, grad_moe_w2 = expert_dispatch_backward(
            grad_all_fc2, all_fc1, moe_w1, moe_w2,
            activation_func, num_local_experts, all_tokens_per_expert,
            split_sizes_exp_major, sorted_idxs_exp_to_rank,
            input_splits_list, output_splits_list, ep_group,
            layer_id=layer_id, num_chunks=moe_chunks_actual,
            comm_stream=scheduler.comm_stream if is_enabled else None,
            all_expert_tokens=all_expert_tokens,
            orig_weight1=orig_moe_w1,
            orig_weight2=orig_moe_w2,
        )

        # Step 8: Backward through sort/permute
        grad_expanded_tokens = torch.zeros_like(grad_permuted_tokens)
        grad_expanded_tokens[sorted_indices] = grad_permuted_tokens

        # Step 9: Backward through expand (sum over top_k copies)
        grad_hidden_from_moe_tokens = grad_expanded_tokens.view(num_tokens, top_k, hidden_size).sum(dim=1)

        # Step 10: Router backward
        grad_hidden_from_router, grad_router_logits = router_backward(
            grad_permuted_probs=grad_permuted_probs,
            sorted_indices=sorted_indices,
            restore_indices=restore_indices,
            permuted_probs=permuted_probs,
            router_probs=router_probs,
            top_indices=top_indices,
            router_weight=router_weight,
            num_tokens=num_tokens,
            top_k=top_k,
            dtype=dtype,
        )

        # Step 11: Combine MoE gradients for ln2_flat
        grad_ln2_flat = grad_hidden_from_moe_tokens + grad_hidden_from_router

        # Step 12: Register router dW task
        grad_router_weight = register_router_dw_task(
            hidden_states=ln2_flat,
            grad_router_logits=grad_router_logits,
            router_weight=orig_router_weight,
            layer_id=layer_id,
        )

        # Step 13: LayerNorm 2 backward
        grad_ln2_out = grad_ln2_flat.view(seq_len, batch_size, hidden_size)

        # Compute dX immediately (needed for gradient propagation)
        mean = hidden_after_attn.mean(dim=-1, keepdim=True)
        var = hidden_after_attn.var(dim=-1, unbiased=False, keepdim=True)
        std = (var + 1e-5).sqrt()
        normalized = (hidden_after_attn - mean) / std

        # dX: gradient through LayerNorm
        grad_hidden_after_attn = grad_hidden_after_attn + grad_ln2_out * ln2_weight / std

        # Register dW tasks for LN2 (deferred, overlaps with AllToAll)
        scheduler = get_backward_scheduler()
        if scheduler.is_enabled():
            grad_ln2_out_saved = grad_ln2_out.detach()
            normalized_saved = normalized.detach()

            def compute_ln2_weight_dw():
                return (grad_ln2_out_saved * normalized_saved).sum(dim=(0, 1))

            def compute_ln2_bias_dw():
                return grad_ln2_out_saved.sum(dim=(0, 1))

            scheduler.register_dw_task(
                layer_name=f"ln2_weight_L{layer_id}",
                layer_id=layer_id,
                compute_fn=compute_ln2_weight_dw,
                weight_param=orig_ln2_weight,
                needs_ar=True,
            )
            scheduler.register_dw_task(
                layer_name=f"ln2_bias_L{layer_id}",
                layer_id=layer_id,
                compute_fn=compute_ln2_bias_dw,
                weight_param=orig_ln2_bias,
                needs_ar=True,
            )
            grad_ln2_weight = None
            grad_ln2_bias = None
        else:
            grad_ln2_weight = (grad_ln2_out * normalized).sum(dim=(0, 1))
            grad_ln2_bias = grad_ln2_out.sum(dim=(0, 1))

        # =====================================================================
        # Attention Backward
        # =====================================================================

        # Step 14: Register output projection dW task
        grad_proj_weight = output_projection_register_dw(
            grad_hidden_after_attn, attn_input_full, proj_weight, layer_id
        )

        # Prepare Q, K, V in attention format for recomputation overlap
        q_perm = q_hp.permute(1, 2, 0, 3)  # [batch, heads_local, seq_full, head_dim]
        k_perm = k_hp.permute(1, 2, 0, 3)
        v_perm = v_hp.permute(1, 2, 0, 3)

        # GQA expansion for attention recomputation (K/V have fewer heads than Q)
        kv_heads_local = num_kv_heads // cp_size
        q_heads_local = num_heads // cp_size
        if q_heads_local > kv_heads_local:
            expand_ratio = q_heads_local // kv_heads_local
            k_perm_expanded = k_perm.repeat_interleave(expand_ratio, dim=1)
            v_perm_expanded = v_perm.repeat_interleave(expand_ratio, dim=1)
        else:
            k_perm_expanded = k_perm
            v_perm_expanded = v_perm

        # Step 15: Output projection backward (chunked dX + sp2hp AllToAll + attention recompute)
        # Attention recomputation overlaps with AllToAll communication
        grad_attn_output, attn_probs, _ = output_projection_backward_chunked(
            grad_hidden_after_attn, proj_weight, num_heads, head_dim, cp_group,
            num_chunks=attn_proj_chunks_actual,
            query=q_perm, key=k_perm_expanded, value=v_perm_expanded, scale=scale,
        )

        # Step 16: Attention backward (uses precomputed attn_probs)
        grad_attn_hp = grad_attn_output.permute(1, 2, 0, 3)

        # Attention backward with chunked hp2sp AllToAll
        # Use expanded K/V for GQA (same as forward attention)
        grad_q, grad_k_expanded, grad_v_expanded = attention_backward_chunked(
            grad_attn_hp, q_perm, k_perm_expanded, v_perm_expanded, scale, cp_group,
            num_chunks=attn_qkv_chunks_actual,
            attn_probs_precomputed=attn_probs,
        )

        # Note: grad_k_expanded/grad_v_expanded are in expanded form [seq_local, batch, num_heads, head_dim]
        # qkv_projection_backward will handle GQA contraction internally

        # Step 17: QKV projection backward
        grad_ln1_out, grad_qkv_weight = qkv_projection_backward(
            grad_q, grad_k_expanded, grad_v_expanded, ln1_out, qkv_weight, cp_group,
            num_heads, num_kv_heads, head_dim, layer_id
        )

        # Step 18: Residual backward - grad flows to hidden_states
        grad_hidden_states = grad_hidden_after_attn.clone()

        # Step 19: LayerNorm 1 backward
        mean1 = hidden_states.mean(dim=-1, keepdim=True)
        var1 = hidden_states.var(dim=-1, unbiased=False, keepdim=True)
        std1 = (var1 + 1e-5).sqrt()
        normalized1 = (hidden_states - mean1) / std1

        # dX: gradient through LayerNorm
        grad_hidden_states = grad_hidden_states + grad_ln1_out * ln1_weight / std1

        # Register dW tasks for LN1 (deferred, overlaps with AllToAll)
        if scheduler.is_enabled():
            grad_ln1_out_saved = grad_ln1_out.detach()
            normalized1_saved = normalized1.detach()

            def compute_ln1_weight_dw():
                return (grad_ln1_out_saved * normalized1_saved).sum(dim=(0, 1))

            def compute_ln1_bias_dw():
                return grad_ln1_out_saved.sum(dim=(0, 1))

            scheduler.register_dw_task(
                layer_name=f"ln1_weight_L{layer_id}",
                layer_id=layer_id,
                compute_fn=compute_ln1_weight_dw,
                weight_param=orig_ln1_weight,
                needs_ar=True,
            )
            scheduler.register_dw_task(
                layer_name=f"ln1_bias_L{layer_id}",
                layer_id=layer_id,
                compute_fn=compute_ln1_bias_dw,
                weight_param=orig_ln1_bias,
                needs_ar=True,
            )
            grad_ln1_weight = None
            grad_ln1_bias = None
        else:
            grad_ln1_weight = (grad_ln1_out * normalized1).sum(dim=(0, 1))
            grad_ln1_bias = grad_ln1_out.sum(dim=(0, 1))

        # Return gradients in same order as forward inputs
        return (
            grad_hidden_states,   # hidden_states
            grad_ln1_weight,      # ln1_weight
            grad_ln1_bias,        # ln1_bias
            grad_ln2_weight,      # ln2_weight
            grad_ln2_bias,        # ln2_bias
            grad_qkv_weight,      # qkv_weight (or None if using dW task)
            grad_proj_weight,     # proj_weight (or None if using dW task)
            grad_router_weight,   # router_weight
            grad_moe_w1,          # moe_w1
            grad_moe_w2,          # moe_w2
            None, None, None, None,  # groups and contexts
            None, None, None, None, None,  # layer_id, num_heads, num_kv_heads, num_experts, top_k
            None, None, None,     # attn_proj_chunks, attn_qkv_chunks, moe_chunks
            None,                 # activation_func
        )


class TransformerLayer(nn.Module):
    """
    Transformer layer using unified autograd.Function.

    Args:
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        ffn_hidden_size: FFN hidden dimension
        num_experts: Total number of experts
        top_k: Number of experts per token
        cp_group: Context parallel group
        ep_group: Expert parallel group
        layer_id: Layer index
        attn_proj_chunks: Chunks for output projection backward (sp2hp AllToAll)
        attn_qkv_chunks: Chunks for attention/qkv backward (hp2sp AllToAll)
        moe_chunks: Chunks for MoE dispatch backward (expert AllToAll)
        activation_func: MoE activation function
        dtype: Parameter dtype
        device: Parameter device
    """

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
        attn_proj_chunks: int = 4,
        attn_qkv_chunks: int = 4,
        moe_chunks: int = 1,
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
        self.attn_proj_chunks = attn_proj_chunks
        self.attn_qkv_chunks = attn_qkv_chunks
        self.moe_chunks = moe_chunks
        self.activation_func = activation_func or F.gelu

        self.cp_group = cp_group
        self.ep_group = ep_group

        cp_size = cp_group.size()
        ep_size = ep_group.size()
        num_local_experts = num_experts // ep_size
        head_dim = hidden_size // num_heads

        # LayerNorm 1 & 2
        self.ln1_weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.ln1_bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.ln2_weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.ln2_bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

        # Attention weights (QKV packed)
        q_per_kv = num_heads // num_kv_heads
        qkv_size = num_kv_heads * (q_per_kv + 2) * head_dim
        self.qkv_weight = nn.Parameter(torch.empty(qkv_size, hidden_size, dtype=dtype, device=device))
        self.proj_weight = nn.Parameter(torch.empty(hidden_size, num_heads * head_dim, dtype=dtype, device=device))

        # MoE weights (stored in 3D shape to avoid permute overhead)
        # w1: [num_local_experts, hidden_size, ffn_hidden_size] for matmul(tokens, w1[exp])
        # w2: [num_local_experts, ffn_hidden_size, hidden_size] for matmul(act, w2[exp])
        self.router_weight = nn.Parameter(torch.empty(hidden_size, num_experts, dtype=torch.float32, device=device))
        self.moe_w1 = nn.Parameter(torch.empty(num_local_experts, hidden_size, ffn_hidden_size, dtype=dtype, device=device))
        self.moe_w2 = nn.Parameter(torch.empty(num_local_experts, ffn_hidden_size, hidden_size, dtype=dtype, device=device))

        # Overlap contexts
        self.attn_overlap_ctx = MultiCardOverlapContext(device, cp_size, cp_size)
        self.moe_overlap_ctx = MultiCardOverlapContext(device, ep_size, ep_size)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.xavier_uniform_(self.proj_weight)
        nn.init.xavier_uniform_(self.router_weight)
        nn.init.xavier_uniform_(self.moe_w1)
        nn.init.xavier_uniform_(self.moe_w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [seq, batch, hidden] input tensor

        Returns:
            [seq, batch, hidden] output tensor
        """
        return TransformerLayerFunction.apply(
            x,
            self.ln1_weight, self.ln1_bias,
            self.ln2_weight, self.ln2_bias,
            self.qkv_weight, self.proj_weight,
            self.router_weight, self.moe_w1, self.moe_w2,
            self.cp_group, self.ep_group,
            self.attn_overlap_ctx, self.moe_overlap_ctx,
            self.layer_id, self.num_heads, self.num_kv_heads,
            self.num_experts, self.top_k,
            self.attn_proj_chunks, self.attn_qkv_chunks, self.moe_chunks,
            self.activation_func,
        )


class TransformerModel(nn.Module):
    """Complete Transformer model with unified layer Function."""

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
        attn_proj_chunks: int = 4,
        attn_qkv_chunks: int = 4,
        moe_chunks: int = 1,
        activation_func: Optional[Callable] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                ffn_hidden_size=ffn_hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                cp_group=cp_group,
                ep_group=ep_group,
                layer_id=i,
                attn_proj_chunks=attn_proj_chunks,
                attn_qkv_chunks=attn_qkv_chunks,
                moe_chunks=moe_chunks,
                activation_func=activation_func,
                dtype=dtype,
                device=device,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
