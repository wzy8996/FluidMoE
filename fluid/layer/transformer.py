"""
FluidMoE Unified Transformer Layer

Single autograd.Function for complete Transformer layer (Attention + MoE).
Minimizes Python overhead by combining all operations into one Function.
"""

import math

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Callable

from fluid.core.comm import MultiCardOverlapContext
from fluid.core.scheduler import get_backward_scheduler
from fluid.core.te_ops import (
    te_layernorm_fwd_with_stats, te_layernorm_bwd,
    create_te_dpa, create_te_linear,
)
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
    outproj_sp2hp_backward,
    hp2sp_qkv_backward,
    output_projection_register_dw,
)
from fluid.moe.backward import (
    combine_fc2_backward,
    fc1_dispatch_backward,
    register_moe_dw_tasks,
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
        # Chunk config for backward overlap regions
        moe_combine_chunks: int,
        moe_dispatch_chunks: int,
        attn_proj_chunks: int,
        attn_qkv_chunks: int,
        activation_func: Callable,
        capacity_factor: float,
        chunk_config: Optional[dict] = None,
        ep_plan=None,
        cp_plan=None,
        # TE modules (None if TE unavailable)
        te_qkv_linear=None,
        te_proj_linear=None,
        te_attn=None,
        # MoE stability options (Megatron-aligned).
        router_dtype: torch.dtype = torch.bfloat16,
        aux_loss_coeff: float = 0.0,
        z_loss_coeff: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass for complete Transformer layer."""
        needs_grad = hidden_states.requires_grad
        ctx.needs_grad = needs_grad
        ctx.dtype = hidden_states.dtype

        device = hidden_states.device
        dtype = hidden_states.dtype
        seq_len, batch_size, hidden_size = hidden_states.shape

        cp_size = cp_group.size()
        ep_size = ep_group.size()
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

        # LayerNorm 1 (save stats for TE fused backward)
        ln1_out, ln1_mu, ln1_rsigma = te_layernorm_fwd_with_stats(
            hidden_states, ln1_weight, ln1_bias)

        # QKV Projection with P2P overlap (uses TE Linear when available)
        # Returns q, k, v in format [seq_full, batch, heads_local, head_dim]
        # Layout: rank-major-contiguous (FluidMoE's natural P2P assembly).
        q_hp, k_hp, v_hp = qkv_projection_p2p_forward(
            ln1_out, qkv_weight_d, num_heads, num_kv_heads, head_dim,
            cp_group, attn_overlap_ctx, te_qkv_linear=te_qkv_linear,
            cp_plan=cp_plan,
        )

        # Compute scale for attention
        scale = 1.0 / (head_dim ** 0.5)

        # Check if GQA is needed (q_heads != kv_heads)
        q_heads_local = num_heads // cp_size
        kv_heads_local = num_kv_heads // cp_size
        enable_gqa = (q_heads_local != kv_heads_local)

        # Scaled Dot-Product Attention
        # Keep computation graph for backward: avoids redundant forward in backward.
        if te_attn is not None:
            # TE DotProductAttention: uses sbhd format [seq, batch, heads, dim]
            # Output is 3D [seq, batch, hidden] (heads*head_dim flattened)
            if needs_grad:
                with torch.enable_grad():
                    q_for_attn = q_hp.detach().requires_grad_(True)
                    k_for_attn = k_hp.detach().requires_grad_(True)
                    v_for_attn = v_hp.detach().requires_grad_(True)
                    attn_out_te = te_attn(
                        q_for_attn, k_for_attn, v_for_attn,
                        attention_mask=None,
                    )
            else:
                attn_out_te = te_attn(q_hp, k_hp, v_hp, attention_mask=None)
            # Reshape TE output from 3D [seq, batch, hidden] to 4D [seq, batch, heads, dim]
            attn_out = attn_out_te.view(
                q_hp.shape[0], q_hp.shape[1], q_heads_local, head_dim
            ).contiguous()
        else:
            # PyTorch SDPA: needs bhsd format [batch, heads, seq, dim]
            q_bf = q_hp.permute(1, 2, 0, 3)
            k_bf = k_hp.permute(1, 2, 0, 3)
            v_bf = v_hp.permute(1, 2, 0, 3)
            if needs_grad:
                with torch.enable_grad():
                    q_for_attn = q_bf.detach().requires_grad_(True)
                    k_for_attn = k_bf.detach().requires_grad_(True)
                    v_for_attn = v_bf.detach().requires_grad_(True)
                    attn_out_bf = scaled_dot_product_attention_forward(
                        q_for_attn, k_for_attn, v_for_attn, scale=scale, is_causal=True, enable_gqa=enable_gqa
                    )
            else:
                attn_out_bf = scaled_dot_product_attention_forward(
                    q_bf, k_bf, v_bf, scale=scale, is_causal=True, enable_gqa=enable_gqa
                )
            attn_out = attn_out_bf.permute(2, 0, 1, 3).contiguous()

        # Output Projection with P2P overlap (uses TE Linear when available)
        proj_out, attn_input_full = output_projection_p2p_forward(
            attn_out, proj_weight_d, None, cp_group, attn_overlap_ctx,
            te_proj_linear=te_proj_linear, cp_plan=cp_plan,
        )

        # Residual connection
        hidden_after_attn = hidden_states + proj_out

        # =====================================================================
        # MoE Block: LN2 -> MoE -> Residual
        # =====================================================================

        # LayerNorm 2 (save stats for TE fused backward)
        ln2_out, ln2_mu, ln2_rsigma = te_layernorm_fwd_with_stats(
            hidden_after_attn, ln2_weight, ln2_bias)

        # Flatten for MoE: [seq, batch, hidden] -> [seq*batch, hidden]
        ln2_flat = ln2_out.view(-1, hidden_size)
        num_tokens = ln2_flat.shape[0]

        # Router forward (Megatron-aligned: post-softmax routing with TE fused permute)
        (permuted_tokens, permuted_probs, sorted_indices,
         tokens_per_expert, top_indices, top_probs, router_logits,
         routing_map, row_id_map) = router_forward(
            ln2_flat, router_weight, num_experts, top_k, ep_group,
            capacity_factor=capacity_factor, ep_plan=ep_plan,
            router_dtype=router_dtype,
        )

        # Dispatch + FC1 with P2P overlap (Megatron-aligned: probs applied between act and FC2)
        (local_tokens, local_act, recv_act_results, recv_buffers,
         moe_partners, _recv_offsets, tokens_cpu,
         local_probs, recv_probs_dict,
         local_fc1, recv_fc1_results) = dispatch_fc1_p2p_forward(
            permuted_tokens, moe_w1_d, ep_group, moe_overlap_ctx,
            activation_func, num_local_experts, ep_plan,
            permuted_probs=permuted_probs,
        )

        # FC2 + Combine with P2P overlap.
        (combined_output, local_fc2, all_expert_tokens, all_tokens_per_expert,
         backward_indices, all_expert_probs, all_fc1) = fc2_combine_p2p_forward(
            local_tokens, local_act, recv_act_results, recv_buffers,
            moe_w2_d, ep_group, moe_overlap_ctx, num_local_experts, ep_plan,
            needs_backward=needs_grad,
            local_probs=local_probs, recv_probs=recv_probs_dict,
            local_fc1=local_fc1, recv_fc1_results=recv_fc1_results,
        )

        # Unpermute: scatter FC2 output back to token positions (NO probs — already applied inside experts)
        moe_output = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
        moe_output.scatter_add_(
            0, sorted_indices.unsqueeze(1).expand_as(combined_output), combined_output)

        # Reshape back: [seq*batch, hidden] -> [seq, batch, hidden]
        moe_output = moe_output.view(seq_len, batch_size, hidden_size)

        # Residual connection
        output = hidden_after_attn + moe_output

        # =====================================================================
        # Save for backward
        # =====================================================================
        if needs_grad:
            # all_expert_tokens is pre-merged in forward (expert-major order)
            if all_expert_tokens is None:
                all_expert_tokens = torch.empty(0, hidden_size, dtype=dtype, device=hidden_states.device)
            # all_expert_probs for backward probs gradient (small 1D tensor)
            if all_expert_probs is None:
                all_expert_probs = torch.empty(0, dtype=torch.float32, device=hidden_states.device)
            ctx.save_for_backward(
                # Input and intermediate states
                hidden_states, ln1_out, hidden_after_attn, ln2_flat,
                # Attention states - only attn_input_full needed (q/k/v saved with graph below)
                attn_input_full,
                # MoE states (Megatron-aligned format)
                permuted_tokens, sorted_indices,
                top_indices, top_probs,
                all_expert_tokens,
                all_expert_probs,
                # Weights (detached)
                ln1_weight, ln1_bias, ln2_weight, ln2_bias,
                qkv_weight.detach(), proj_weight.detach(),
                router_weight.detach(), moe_w1.detach(), moe_w2.detach(),
                # LN stats for TE fused backward
                ln1_mu, ln1_rsigma, ln2_mu, ln2_rsigma,
            )
            # Saved here so backward can skip recompute_fc1_gemm.
            ctx._all_fc1_saved = all_fc1
            # Store non-tensor merge results on ctx
            ctx.all_tokens_per_expert = all_tokens_per_expert
            ctx.backward_indices = backward_indices
            # tokens_per_expert for router backward (in fused path, includes padding entries)
            ctx.tokens_per_expert = tokens_per_expert
            # Router states for aux / z loss backward (injected manually since we're
            # in a custom autograd.Function and Megatron's MoEAuxLossAutoScaler
            # can't reach inside us).
            ctx.router_dtype = router_dtype
            ctx.aux_loss_coeff = float(aux_loss_coeff)
            ctx.z_loss_coeff = float(z_loss_coeff)
            if aux_loss_coeff > 0 or z_loss_coeff > 0:
                ctx._router_logits = router_logits  # [T, E] in router_dtype
                ctx._routing_map = routing_map      # [T, E] bool (post capacity)
            # Save SDPA computation graph (avoids redundant forward in backward)
            # FlashAttention only stores logsumexp O(seq), not attention matrix O(seq²)
            ctx._q_for_attn = q_for_attn
            ctx._k_for_attn = k_for_attn
            ctx._v_for_attn = v_for_attn
            ctx._used_te_attn = (te_attn is not None)
            if te_attn is not None:
                ctx._attn_out_bf = attn_out_te  # TE DPA output: 3D [seq, batch, hidden]
            else:
                ctx._attn_out_bf = attn_out_bf  # PyTorch SDPA: bhsd [batch, heads, seq, dim]
            ctx._enable_gqa = enable_gqa
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
            ctx.moe_combine_chunks = moe_combine_chunks
            ctx.moe_dispatch_chunks = moe_dispatch_chunks
            ctx.attn_proj_chunks = attn_proj_chunks
            ctx.attn_qkv_chunks = attn_qkv_chunks
            ctx.activation_func = activation_func
            ctx.num_local_experts = num_local_experts
            ctx.head_dim = head_dim
            ctx.scale = scale

            # MoE routing info (all from EPPlan; static in padded mode).
            ctx.input_splits_list = ep_plan.input_splits_list
            ctx.output_splits_list = ep_plan.output_splits_list
            # Pre-built padded expert_ids for router_backward.
            ctx.expert_ids_gpu = ep_plan.expert_ids_gpu

            # Shape info
            ctx.seq_len = seq_len
            ctx.batch_size = batch_size
            ctx.hidden_size = hidden_size
            ctx.num_tokens = num_tokens

            # Pre-computed static chunk config (None when capacity_factor <= 0)
            ctx.chunk_config = chunk_config

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass with P2P overlap and dW scheduling."""
        if not ctx.needs_grad:
            return (None,) * 31  # 10 weights + 4 groups/contexts + 17 config params

        # Ensure grad_output matches forward dtype (e.g. loss computed in float32 → cast back to bf16)
        if grad_output.dtype != ctx.dtype:
            grad_output = grad_output.to(ctx.dtype)

        # Retrieve saved tensors (Megatron-aligned format: post-softmax top_indices/top_probs)
        (hidden_states, ln1_out, hidden_after_attn, ln2_flat,
         attn_input_full,
         permuted_tokens, sorted_indices,
         top_indices, top_probs,
         all_expert_tokens,
         all_expert_probs,
         ln1_weight, ln1_bias, ln2_weight, ln2_bias,
         qkv_weight, proj_weight,
         router_weight, moe_w1_2d, moe_w2_2d,
         ln1_mu, ln1_rsigma, ln2_mu, ln2_rsigma) = ctx.saved_tensors

        # Retrieve pre-computed merge results
        all_tokens_per_expert = ctx.all_tokens_per_expert
        backward_indices = ctx.backward_indices
        tokens_per_expert = ctx.tokens_per_expert

        # Retrieve SDPA computation graph for direct backward (no redundant forward)
        q_for_attn = ctx._q_for_attn
        k_for_attn = ctx._k_for_attn
        v_for_attn = ctx._v_for_attn
        attn_out_bf = ctx._attn_out_bf

        # Retrieve config
        cp_group = ctx.cp_group
        ep_group = ctx.ep_group
        layer_id = ctx.layer_id
        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        num_experts = ctx.num_experts
        top_k = ctx.top_k
        moe_combine_chunks = ctx.moe_combine_chunks
        moe_dispatch_chunks = ctx.moe_dispatch_chunks
        attn_proj_chunks = ctx.attn_proj_chunks
        attn_qkv_chunks = ctx.attn_qkv_chunks
        activation_func = ctx.activation_func
        num_local_experts = ctx.num_local_experts
        head_dim = ctx.head_dim

        input_splits_list = ctx.input_splits_list
        output_splits_list = ctx.output_splits_list
        chunk_config = ctx.chunk_config

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

        # MoE weights are already 3D: [num_local_experts, hidden/ffn, ffn/hidden]
        moe_w1 = moe_w1_2d  # Already in shape [E, hidden, ffn]
        moe_w2 = moe_w2_2d  # Already in shape [E, ffn, hidden]

        # Compute actual chunk values (1 if scheduler disabled)
        is_enabled = scheduler.is_enabled()
        if not is_enabled:
            moe_combine_chunks = moe_dispatch_chunks = attn_proj_chunks = attn_qkv_chunks = 1

        # =====================================================================
        # MoE Backward (reverse order)
        # =====================================================================

        # Gradient through residual: grad flows to both hidden_after_attn and moe_output
        grad_hidden_after_attn = grad_output
        grad_moe_output = grad_output.view(num_tokens, hidden_size)

        # Step 1: Backward through unpermute (NO probs — probs applied inside expert computation)
        # sorted_indices = original token indices (Megatron format)
        grad_combined = grad_moe_output[sorted_indices]

        # Region 1: Combine AllToAll → FC2 dx → probs backward → activation backward
        # Probs backward (act * probs) is handled inside combine_fc2_backward
        scheduler.begin_region('moe_combine')
        (grad_all_fc1, act_output, all_fc1, grad_all_fc2,
         all_expert_tokens, all_tokens_per_expert, backward_indices,
         grad_expert_probs) = combine_fc2_backward(
            grad_combined, input_splits_list, output_splits_list, ep_group, layer_id,
            weight1=moe_w1,
            weight2=moe_w2,
            activation_func=activation_func,
            num_local_experts=num_local_experts,
            num_chunks=moe_combine_chunks,
            all_expert_tokens=all_expert_tokens,
            all_tokens_per_expert=all_tokens_per_expert,
            backward_indices=backward_indices,
            chunk_config=chunk_config,
            all_expert_probs=all_expert_probs,
            all_fc1_saved=getattr(ctx, '_all_fc1_saved', None),
        )

        scheduler.end_region()

        moe_needs_ar = scheduler.expert_dp_world_size > 1
        grad_moe_w1, grad_moe_w2 = register_moe_dw_tasks(
            moe_w1, moe_w2, all_expert_tokens, act_output,
            grad_all_fc2,
            grad_all_fc1,
            num_local_experts, all_tokens_per_expert, layer_id,
            orig_moe_w1, orig_moe_w2,
            needs_ar=moe_needs_ar,
        )

        # =================================================================
        # Transport probs gradient back via A2A (small 1D tensor, negligible overhead)
        # grad_expert_probs is in expert-major order on this rank.
        # Reverse dispatch A2A to send probs grads back to original sender ranks.
        # =================================================================
        if grad_expert_probs is not None:
            row_idx_exp_to_rank_probs = backward_indices['row_idx_exp_to_rank']
            grad_probs_rank_major = grad_expert_probs[row_idx_exp_to_rank_probs]
            from fluid.core import _all_to_all
            grad_probs_dispatched = _all_to_all(
                grad_probs_rank_major.unsqueeze(-1),
                output_split_sizes=input_splits_list,
                input_split_sizes=output_splits_list,
                group=ep_group,
                debug_tag="moe_probs_grad_a2a",
            ).squeeze(-1)
            # grad_probs_dispatched is now in same order as sorted_indices
            grad_permuted_probs = grad_probs_dispatched
        else:
            grad_permuted_probs = torch.zeros(
                sorted_indices.shape[0], dtype=torch.float32, device=device)

        # =================================================================
        # Register router_backward + router_dw as scheduler tasks
        # (execute during R2's Dispatch A2A overlap window)
        # Padding entries have grad_probs=0 (from probs=0 gate), no real_mask needed.
        # =================================================================
        router_result = [None, None]  # [grad_hidden_from_router, grad_router_logits]
        grad_router_weight = None
        grad_ln2_weight = None
        grad_ln2_bias = None

        if is_enabled:
            # Router backward as a queued task — runs DURING moe_dispatch's A2A
            # overlap window on default_stream, not before it. Inline (pre-region)
            # would push moe_dispatch's A2A start later by router_backward's
            # duration on default_stream (~5-10ms regression).
            _grad_permuted_probs = grad_permuted_probs.detach()
            _sorted_indices = sorted_indices
            _tokens_per_expert = tokens_per_expert
            _top_indices = top_indices
            _top_probs = top_probs
            _router_weight = router_weight
            _expert_ids_gpu = ctx.expert_ids_gpu
            _aux_loss_coeff = getattr(ctx, 'aux_loss_coeff', 0.0)
            _z_loss_coeff = getattr(ctx, 'z_loss_coeff', 0.0)
            _router_logits_saved = getattr(ctx, '_router_logits', None)
            _routing_map_saved = getattr(ctx, '_routing_map', None)

            def _router_bwd_task():
                router_result[0], router_result[1] = router_backward(
                    grad_permuted_probs=_grad_permuted_probs,
                    sorted_indices=_sorted_indices,
                    tokens_per_expert=_tokens_per_expert,
                    top_indices=_top_indices,
                    top_probs=_top_probs,
                    router_weight=_router_weight,
                    num_tokens=num_tokens,
                    num_experts=num_experts,
                    top_k=top_k,
                    dtype=dtype,
                    expert_ids=_expert_ids_gpu,
                    aux_loss_coeff=_aux_loss_coeff,
                    z_loss_coeff=_z_loss_coeff,
                    router_logits=_router_logits_saved,
                    routing_map=_routing_map_saved,
                )
                return None  # no gradient to write

            scheduler.register_dw_task(
                layer_name=f"router_bwd_L{layer_id}",
                layer_id=layer_id,
                compute_fn=_router_bwd_task,
                weight_param=None,
                defer=False,  # Python-side consumer: router_result[0] feeds grad_ln2_flat
            )

            # Router dW: depends on router_result[1] from above task.
            _ln2_flat_saved = ln2_flat.detach()

            def _router_dw_task():
                g = router_result[1]
                return torch.matmul(_ln2_flat_saved.to(g.dtype).t(), g)

            scheduler.register_dw_task(
                layer_name=f"router_weight_L{layer_id}",
                layer_id=layer_id,
                compute_fn=_router_dw_task,
                weight_param=orig_router_weight,
            )

        # Region 2: FC1 dx → Dispatch AllToAll (compute-first pipeline)
        # FC1 dx computed in chunks, each chunk submitted to AllToAll on completion
        # dW window overlaps: router_dw → moe dW (router_bwd ran inline above)
        row_idx_exp_to_rank = backward_indices['row_idx_exp_to_rank']

        scheduler.begin_region('moe_dispatch')
        grad_dispatched = fc1_dispatch_backward(
            grad_all_fc1, moe_w1,
            num_local_experts, all_tokens_per_expert,
            row_idx_exp_to_rank,
            input_splits_list, output_splits_list, ep_group,
            layer_id=layer_id, num_chunks=moe_dispatch_chunks,
            chunk_config=chunk_config,
        )
        # Unpermute gradients back to original token positions
        grad_hidden_from_moe_tokens = torch.zeros(
            num_tokens, hidden_size, dtype=dtype, device=device)
        grad_hidden_from_moe_tokens.scatter_add_(
            0, sorted_indices.unsqueeze(1).expand_as(grad_dispatched), grad_dispatched)

        scheduler.end_region()

        # FB/F mode: scheduler.enabled or ar_enabled is False, so router_bwd_task
        # was either not queued or executed synchronously. Run inline as fallback
        # so router_result is populated for the grad_ln2_flat addition below.
        if not is_enabled:
            router_result[0], router_result[1] = router_backward(
                grad_permuted_probs=grad_permuted_probs,
                sorted_indices=sorted_indices,
                tokens_per_expert=tokens_per_expert,
                top_indices=top_indices,
                top_probs=top_probs,
                router_weight=router_weight,
                num_tokens=num_tokens,
                num_experts=num_experts,
                top_k=top_k,
                dtype=dtype,
                expert_ids=ctx.expert_ids_gpu,
                aux_loss_coeff=getattr(ctx, 'aux_loss_coeff', 0.0),
                z_loss_coeff=getattr(ctx, 'z_loss_coeff', 0.0),
                router_logits=getattr(ctx, '_router_logits', None),
                routing_map=getattr(ctx, '_routing_map', None),
            )
            grad_router_weight = register_router_dw_task(
                hidden_states=ln2_flat,
                grad_router_logits=router_result[1],
                router_weight=orig_router_weight,
                layer_id=layer_id,
            )

        # Combine MoE gradients.
        grad_ln2_flat = grad_hidden_from_moe_tokens + router_result[0]

        # LayerNorm 2 backward (TE fused kernel: computes dx, dw, db in one launch)
        grad_ln2_out = grad_ln2_flat.view(seq_len, batch_size, hidden_size)
        grad_ln2_dx, grad_ln2_weight_val, grad_ln2_bias_val = te_layernorm_bwd(
            grad_ln2_out, hidden_after_attn, ln2_mu, ln2_rsigma, ln2_weight)
        grad_hidden_after_attn = grad_hidden_after_attn + grad_ln2_dx

        # Register dW tasks for LN2. NSYS-verified: keeping LN as queued tasks
        # (vs immediate flat-buffer write) yields more, smaller AR kernels in
        # dp_tail's commit-per-task path. NCCL is markedly less efficient on
        # medium-to-large AllReduce — empirically 8.6ms/100MB vs 19ms/200MB —
        # so smaller AR granularity wins by ~22ms/iter overall.
        if is_enabled:
            _ln2_dw = grad_ln2_weight_val.detach()
            _ln2_db = grad_ln2_bias_val.detach()
            scheduler.register_dw_task(
                layer_name=f"ln2_weight_L{layer_id}",
                layer_id=layer_id,
                compute_fn=lambda: _ln2_dw,
                weight_param=orig_ln2_weight,
                needs_ar=True,
            )
            scheduler.register_dw_task(
                layer_name=f"ln2_bias_L{layer_id}",
                layer_id=layer_id,
                compute_fn=lambda: _ln2_db,
                weight_param=orig_ln2_bias,
                needs_ar=True,
            )
        else:
            grad_ln2_weight = grad_ln2_weight_val
            grad_ln2_bias = grad_ln2_bias_val

        # =====================================================================
        # Attention Backward
        # =====================================================================

        # Step 14: Register output projection dW task
        grad_proj_weight = output_projection_register_dw(
            grad_hidden_after_attn, attn_input_full, orig_proj_weight, layer_id
        )

        # Step 15: Output projection backward (chunked dX + sp2hp AllToAll)
        scheduler.begin_region('attn_proj')
        grad_attn_output = outproj_sp2hp_backward(
            grad_hidden_after_attn, proj_weight, num_heads, head_dim, cp_group,
            num_chunks=attn_proj_chunks,
        )

        scheduler.end_region()

        # Step 16: Attention score backward using saved computation graph
        # Direct autograd.grad on saved SDPA output - no redundant forward pass
        # FlashAttention internally does tiled recomputation (unavoidable by design)
        used_te_attn = getattr(ctx, '_used_te_attn', False)
        if used_te_attn:
            # TE DPA output is 3D [seq, batch, hidden]; grad_attn_output is 4D [seq, batch, heads, dim]
            # Flatten grad to 3D to match TE DPA output shape
            grad_attn_3d = grad_attn_output.view(
                grad_attn_output.shape[0], grad_attn_output.shape[1], -1
            )
            grad_q, grad_k, grad_v = torch.autograd.grad(
                attn_out_bf, (q_for_attn, k_for_attn, v_for_attn),
                grad_attn_3d, retain_graph=False
            )
            # grad_q/k/v are sbhd 4D [seq, batch, heads, dim]
            grad_q = grad_q.permute(1, 2, 0, 3).contiguous()
            grad_k = grad_k.permute(1, 2, 0, 3).contiguous()
            grad_v = grad_v.permute(1, 2, 0, 3).contiguous()
        else:
            # PyTorch SDPA: attn_out_bf is bhsd [batch, heads, seq, dim]
            grad_attn_hp = grad_attn_output.permute(1, 2, 0, 3)
            grad_q, grad_k, grad_v = torch.autograd.grad(
                attn_out_bf, (q_for_attn, k_for_attn, v_for_attn),
                grad_attn_hp, retain_graph=False
            )

        grad_ln1_weight = None
        grad_ln1_bias = None

        # Region 4: hp2sp AllToAll → QKV dX (communication-first pipeline)
        scheduler.begin_region('attn_qkv')
        grad_ln1_out, grad_qkv_weight = hp2sp_qkv_backward(
            grad_q, grad_k, grad_v, cp_group,
            tokens=ln1_out, weight_qkv=orig_qkv_weight,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            layer_id=layer_id, num_chunks=attn_qkv_chunks,
        )

        scheduler.end_region()

        # Step 18: Residual backward - grad flows to hidden_states
        grad_hidden_states = grad_hidden_after_attn

        # Step 19: LayerNorm 1 backward (TE fused kernel)
        grad_ln1_dx, grad_ln1_weight_val, grad_ln1_bias_val = te_layernorm_bwd(
            grad_ln1_out, hidden_states, ln1_mu, ln1_rsigma, ln1_weight)
        grad_hidden_states = grad_hidden_states + grad_ln1_dx

        # Register dW tasks for LN1 (same nsys-verified rationale as LN2 above).
        if is_enabled:
            _ln1_dw = grad_ln1_weight_val.detach()
            _ln1_db = grad_ln1_bias_val.detach()
            scheduler.register_dw_task(
                layer_name=f"ln1_weight_L{layer_id}",
                layer_id=layer_id,
                compute_fn=lambda: _ln1_dw,
                weight_param=orig_ln1_weight,
                needs_ar=True,
            )
            scheduler.register_dw_task(
                layer_name=f"ln1_bias_L{layer_id}",
                layer_id=layer_id,
                compute_fn=lambda: _ln1_db,
                weight_param=orig_ln1_bias,
                needs_ar=True,
            )
        else:
            grad_ln1_weight = grad_ln1_weight_val
            grad_ln1_bias = grad_ln1_bias_val

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
            None, None, None, None,  # 4 chunk params
            None,                 # activation_func
            None,                 # capacity_factor
            None,                 # chunk_config
            None, None,           # ep_plan, cp_plan
            None, None, None,     # te_qkv_linear, te_proj_linear, te_attn
            None, None, None,     # router_dtype, aux_loss_coeff, z_loss_coeff
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
        moe_combine_chunks: R1 (combine AllToAll) chunk count
        moe_dispatch_chunks: R2 (dispatch AllToAll) chunk count
        attn_proj_chunks: R3 (sp2hp AllToAll) chunk count
        attn_qkv_chunks: R4 (hp2sp AllToAll) chunk count
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
        moe_combine_chunks: int = 1,
        moe_dispatch_chunks: int = 1,
        attn_proj_chunks: int = 1,
        attn_qkv_chunks: int = 1,
        activation_func: Optional[Callable] = None,
        capacity_factor: float = 1.0,
        init_std: float = 0.02,
        output_init_std: Optional[float] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        # MoE stability options (Megatron-aligned).
        router_dtype: torch.dtype = torch.bfloat16,
        aux_loss_coeff: float = 0.0,
        z_loss_coeff: float = 0.0,
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
        self.moe_combine_chunks = moe_combine_chunks
        self.moe_dispatch_chunks = moe_dispatch_chunks
        self.attn_proj_chunks = attn_proj_chunks
        self.attn_qkv_chunks = attn_qkv_chunks
        from fluid.core.te_ops import te_gelu
        self.activation_func = activation_func or te_gelu
        self.capacity_factor = capacity_factor
        self.router_dtype = router_dtype
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.init_std = init_std
        self.output_init_std = output_init_std if output_init_std is not None else init_std
        self._moe_chunk_config = None  # lazily computed on first forward
        self._moe_chunk_signature = None
        self._ep_plan = None  # created on first forward (pad mode only)
        self._cp_plan = None  # created on first forward (cp>1 only)
        self._plan_sig = None  # (seq, batch, dtype, device) for plan invalidation
        self._shared_qkv_send_slots = None
        self._shared_proj_recv_slots = None

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

        # TE modules for attention (DotProductAttention + Linear)
        # Match Megatron init semantics:
        # - qkv uses init_method
        # - proj uses output_layer_init_method
        import functools
        _qkv_init = functools.partial(nn.init.normal_, mean=0.0, std=self.init_std)
        _proj_init = functools.partial(nn.init.normal_, mean=0.0, std=self.output_init_std)
        try:
            from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
            _rng_tracker = get_cuda_rng_tracker if get_cuda_rng_tracker().is_initialized() else None
        except Exception:
            _rng_tracker = None
        self.te_qkv_linear = create_te_linear(
            hidden_size, qkv_size, bias=False,
            params_dtype=dtype, device=device,
            init_method=_qkv_init,
            get_rng_state_tracker=_rng_tracker,
            parallel_mode="column",
        )
        self.te_proj_linear = create_te_linear(
            num_heads * head_dim, hidden_size, bias=False,
            params_dtype=dtype, device=device,
            init_method=_proj_init,
            get_rng_state_tracker=_rng_tracker,
            parallel_mode="row",
        )
        q_heads_local = num_heads // cp_size
        kv_heads_local = num_kv_heads // cp_size
        self.te_attn = create_te_dpa(
            q_heads_local, head_dim,
            num_kv_heads=kv_heads_local,
            layer_number=layer_id,
        )

        # If TE unavailable, fall back to raw nn.Parameter
        if self.te_qkv_linear is None:
            self.qkv_weight = nn.Parameter(torch.empty(qkv_size, hidden_size, dtype=dtype, device=device))
        if self.te_proj_linear is None:
            self.proj_weight = nn.Parameter(torch.empty(hidden_size, num_heads * head_dim, dtype=dtype, device=device))

        # MoE weights (stored in 3D shape to avoid permute overhead)
        # w1: [num_local_experts, hidden_size, ffn_hidden_size] for matmul(tokens, w1[exp])
        # w2: [num_local_experts, ffn_hidden_size, hidden_size] for matmul(act, w2[exp])
        self.router_weight = nn.Parameter(torch.empty(hidden_size, num_experts, dtype=dtype, device=device))
        self.moe_w1 = nn.Parameter(torch.empty(num_local_experts, hidden_size, ffn_hidden_size, dtype=dtype, device=device))
        self.moe_w2 = nn.Parameter(torch.empty(num_local_experts, ffn_hidden_size, hidden_size, dtype=dtype, device=device))
        # Match Megatron MoE optimizer grouping: expert weights are not dense-allreduced.
        # Scheduler still owns their gradient synchronization; this flag is for
        # Megatron optimizer param-group classification only.
        setattr(self.moe_w1, 'allreduce', False)
        setattr(self.moe_w2, 'allreduce', False)

        # Overlap contexts
        self.attn_overlap_ctx = MultiCardOverlapContext(device, cp_size, cp_size)
        self.moe_overlap_ctx = MultiCardOverlapContext(device, ep_size, ep_size)

        self._reset_parameters()

    def _get_qkv_weight(self):
        """Get QKV weight tensor (from TE Linear or raw Parameter)."""
        if self.te_qkv_linear is not None:
            return self.te_qkv_linear.weight
        return self.qkv_weight

    def _get_proj_weight(self):
        """Get output projection weight tensor (from TE Linear or raw Parameter)."""
        if self.te_proj_linear is not None:
            return self.te_proj_linear.weight
        return self.proj_weight

    def _reset_parameters(self):
        """Initialize parameters to match Megatron's init/output init split.

        Uses Megatron's RNG tracker to fork to the correct RNG stream for each
        parameter type, ensuring identical weights with the same seed:
          - QKV: model-parallel-rng + init_method
          - Proj: model-parallel-rng + output_layer_init_method
          - Router: global CUDA RNG (no fork, same as Megatron)
          - Expert FC1: expert-parallel-rng + init_method
          - Expert FC2: expert-parallel-rng + output_layer_init_method

        MoE weights are initialized in Megatron's layout then transposed.
        """
        from contextlib import nullcontext
        try:
            from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
        except ImportError:
            get_cuda_rng_tracker = None
        try:
            from megatron.core.tensor_parallel.random import get_expert_parallel_rng_tracker_name
            expert_rng_name = get_expert_parallel_rng_tracker_name()
        except (ImportError, Exception):
            expert_rng_name = 'expert-parallel-rng'

        # TE Linear handles its own init + RNG fork via get_rng_state_tracker
        def _model_rng_ctx():
            if get_cuda_rng_tracker is None:
                return nullcontext()
            try:
                return get_cuda_rng_tracker().fork()
            except Exception:
                return nullcontext()

        if self.te_qkv_linear is None:
            with _model_rng_ctx():
                nn.init.normal_(self.qkv_weight, mean=0.0, std=self.init_std)
        if self.te_proj_linear is None:
            with _model_rng_ctx():
                nn.init.normal_(self.proj_weight, mean=0.0, std=self.output_init_std)

        # Router: Megatron uses global CUDA RNG (no fork), layout [E, H]
        tmp_router = torch.empty(
            self.num_experts, self.hidden_size,
            dtype=self.router_weight.dtype, device=self.router_weight.device)
        nn.init.normal_(tmp_router, mean=0.0, std=self.init_std)
        self.router_weight.data.copy_(tmp_router.t())

        # Expert FC1/FC2: Megatron uses expert-parallel-rng.
        # Megatron's GroupedMLP inits weight1 as [hidden, num_local*ffn] (ColumnParallel)
        # and weight2 as [num_local*ffn, hidden] (RowParallel) in one shot.
        # Fallback to default RNG when expert-parallel-rng is not registered
        # (e.g., block benchmark without full Megatron pretrain() init).
        num_local = self.moe_w1.shape[0]
        ffn = self.moe_w1.shape[2]

        def _has_rng_state(name):
            if get_cuda_rng_tracker is None:
                return False
            try:
                tracker = get_cuda_rng_tracker()
                return name in tracker.states_
            except Exception:
                return False

        expert_ctx = (lambda: get_cuda_rng_tracker().fork(expert_rng_name)) if _has_rng_state(expert_rng_name) else nullcontext

        with expert_ctx():
            tmp_w1 = torch.empty(num_local * ffn, self.hidden_size,
                dtype=self.moe_w1.dtype, device=self.moe_w1.device)
            nn.init.normal_(tmp_w1, mean=0.0, std=self.init_std)
            self.moe_w1.data.copy_(tmp_w1.view(num_local, ffn, self.hidden_size).transpose(1, 2))
        with expert_ctx():
            tmp_w2 = torch.empty(self.hidden_size, num_local * ffn,
                dtype=self.moe_w2.dtype, device=self.moe_w2.device)
            nn.init.normal_(tmp_w2, mean=0.0, std=self.output_init_std)
            self.moe_w2.data.copy_(tmp_w2.view(self.hidden_size, num_local, ffn).permute(1, 2, 0))

    def _build_moe_chunk_config(self, x: torch.Tensor):
        """Build static MoE chunk config for the current input shape."""
        signature = (int(x.shape[0]), int(x.shape[1]))
        seq_len, batch_size, _ = x.shape
        num_tokens = seq_len * batch_size
        nle = self.num_experts // self.ep_group.size()
        cap = int(math.ceil(
            num_tokens * self.top_k / self.num_experts * self.capacity_factor))
        from fluid.moe.backward import build_moe_chunk_config
        cfg = build_moe_chunk_config(
            num_local_experts=nle,
            ep_size=self.ep_group.size(),
            cap=cap,
            moe_combine_chunks=self.moe_combine_chunks,
            moe_dispatch_chunks=self.moe_dispatch_chunks,
            device=x.device,
        )
        return cfg, cap, signature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [seq, batch, hidden] input tensor

        Returns:
            [seq, batch, hidden] output tensor
        """
        # Plan init with shape/dtype/device signature check.
        # Invalidate and rebuild if input geometry changes.
        _sig = (x.shape[0], x.shape[1], x.dtype, x.device)
        if self._plan_sig != _sig:
            self._moe_chunk_config = None
            self._ep_plan = None
            self._cp_plan = None
            self._plan_sig = _sig

        if self._moe_chunk_config is None:
            self._moe_chunk_config, _, _ = self._build_moe_chunk_config(x)

        if self._ep_plan is None:
            from fluid.moe.forward import EPPlan
            self._ep_plan = EPPlan(
                num_tokens=x.shape[0] * x.shape[1],
                num_experts=self.num_experts,
                top_k=self.top_k, capacity_factor=self.capacity_factor,
                ep_group=self.ep_group, overlap_ctx=self.moe_overlap_ctx,
                device=x.device, hidden_size=x.shape[2],
            )

        if self._cp_plan is None and self.cp_group.size() > 1:
            from fluid.attention.forward import CPPlan
            self._cp_plan = CPPlan(
                seq_local=x.shape[0], batch=x.shape[1],
                hidden_size=x.shape[2],
                num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                head_dim=x.shape[2] // self.num_heads,
                cp_group=self.cp_group, overlap_ctx=self.attn_overlap_ctx,
            )

        if self._cp_plan is not None and self._shared_qkv_send_slots is not None:
            self._cp_plan.set_shared_slots(
                self._shared_qkv_send_slots,
                self._shared_proj_recv_slots,
            )

        return TransformerLayerFunction.apply(
            x,
            self.ln1_weight, self.ln1_bias,
            self.ln2_weight, self.ln2_bias,
            self._get_qkv_weight(), self._get_proj_weight(),
            self.router_weight, self.moe_w1, self.moe_w2,
            self.cp_group, self.ep_group,
            self.attn_overlap_ctx, self.moe_overlap_ctx,
            self.layer_id, self.num_heads, self.num_kv_heads,
            self.num_experts, self.top_k,
            self.moe_combine_chunks,
            self.moe_dispatch_chunks,
            self.attn_proj_chunks,
            self.attn_qkv_chunks,
            self.activation_func,
            self.capacity_factor,
            self._moe_chunk_config,
            self._ep_plan, self._cp_plan,
            self.te_qkv_linear, self.te_proj_linear, self.te_attn,
            self.router_dtype, self.aux_loss_coeff, self.z_loss_coeff,
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
        moe_combine_chunks: int = 1,
        moe_dispatch_chunks: int = 1,
        attn_proj_chunks: int = 1,
        attn_qkv_chunks: int = 1,
        activation_func: Optional[Callable] = None,
        capacity_factor: float = 1.0,
        init_std: float = 0.02,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        # MoE stability options (Megatron-aligned).
        router_dtype: torch.dtype = torch.bfloat16,
        aux_loss_coeff: float = 0.0,
        z_loss_coeff: float = 0.0,
    ):
        super().__init__()
        self._chunk_check_signature = None
        self._shared_cp_slot_signature = None
        self._shared_qkv_send_slots = None
        self._shared_proj_recv_slots = None
        output_init_std = init_std / math.sqrt(2.0 * max(num_layers, 1))

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
                moe_combine_chunks=moe_combine_chunks,
                moe_dispatch_chunks=moe_dispatch_chunks,
                attn_proj_chunks=attn_proj_chunks,
                attn_qkv_chunks=attn_qkv_chunks,
                activation_func=activation_func,
                capacity_factor=capacity_factor,
                init_std=init_std,
                output_init_std=output_init_std,
                dtype=dtype,
                device=device,
                router_dtype=router_dtype,
                aux_loss_coeff=aux_loss_coeff,
                z_loss_coeff=z_loss_coeff,
            )
            for i in range(num_layers)
        ])

    def prepare_chunk_status(self, x: torch.Tensor):
        """Precompute chunk config for this input shape and return one-time fallback messages."""
        if not self.layers:
            return []

        signature = (int(x.shape[0]), int(x.shape[1]))
        if self._chunk_check_signature == signature:
            return []

        layer0 = self.layers[0]
        seq_local, batch_size, _ = x.shape
        cp_size = layer0.cp_group.size()
        seq_full = seq_local * cp_size
        messages = []

        moe_cfg, cap, _ = layer0._build_moe_chunk_config(x)
        for layer in self.layers:
            layer._moe_chunk_config = moe_cfg
            layer._moe_chunk_signature = signature

        if layer0.moe_combine_chunks > 1 and (moe_cfg is None or moe_cfg.get("r1") is None):
            messages.append(
                f"R1 moe_combine_chunks={layer0.moe_combine_chunks} will fallback to 1 "
                f"(cap={cap} is not divisible)"
            )
        if layer0.moe_dispatch_chunks > 1 and (moe_cfg is None or moe_cfg.get("r2") is None):
            messages.append(
                f"R2 moe_dispatch_chunks={layer0.moe_dispatch_chunks} will fallback to 1 "
                f"(cap={cap} is not divisible)"
            )

        if layer0.attn_proj_chunks > 1 and seq_local % layer0.attn_proj_chunks != 0:
            messages.append(
                f"R3 attn_proj_chunks={layer0.attn_proj_chunks} will fallback to 1 "
                f"(seq_local={seq_local} is not divisible)"
            )
        if layer0.attn_qkv_chunks > 1 and seq_full % layer0.attn_qkv_chunks != 0:
            messages.append(
                f"R4 attn_qkv_chunks={layer0.attn_qkv_chunks} will fallback to 1 "
                f"(seq_full={seq_full} is not divisible)"
            )

        self._chunk_check_signature = signature
        return messages

    def _init_shared_cp_slots(self, x: torch.Tensor):
        return

    def setup_ar_buffer(self):
        """Set up flat AR buffers for zero-copy trickle slicing.

        Registers shared parameters in backward execution order on the
        shared AR buffer, and expert parameters on the expert AR buffer
        (separate NCCL group for dp subgroup AR when dp > 1).
        """
        sched = get_backward_scheduler()
        shared_params = []
        expert_params = []
        for layer in reversed(self.layers):
            # Expert params first (they execute earlier in MoE backward)
            expert_params.extend([layer.moe_w2, layer.moe_w1])
            # Then shared params (use accessors to handle TE Linear weights)
            shared_params.extend([
                layer.router_weight,
                layer.ln2_weight, layer.ln2_bias,
                layer._get_proj_weight(),
                layer._get_qkv_weight(),
                layer.ln1_weight, layer.ln1_bias,
            ])
        sched.setup_ar_buffer(shared_params)
        # Always register expert params in buffer (even when expert_dp=1)
        # so that finish_batch can apply uniform 1/dp_cp_size scaling.
        if expert_params:
            sched.setup_expert_ar_buffer(expert_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._init_shared_cp_slots(x)
        for layer in self.layers:
            x = layer(x)
        return x
