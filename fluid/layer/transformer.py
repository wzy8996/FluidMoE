"""
FluidMoE Unified Transformer Layer

Single autograd.Function for complete Transformer layer (Attention + MoE).
Minimizes Python overhead by combining all operations into one Function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Callable, Tuple, List

from fluid.core.comm import MultiCardOverlapContext
from fluid.core.scheduler import get_backward_scheduler
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
        # Chunk configs for 4 backward overlap regions
        attn_proj_chunks: int,       # Region 3: output projection dX + sp2hp AllToAll
        attn_qkv_chunks: int,        # Region 4: hp2sp AllToAll + QKV dX
        moe_combine_chunks: int,     # Region 1: Combine AllToAll + FC2 dX
        moe_dispatch_chunks: int,    # Region 2: FC1 dX + Dispatch AllToAll
        activation_func: Callable,
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

        # Convert to batch-first format for attention: [seq, batch, heads, dim] -> [batch, heads, seq, dim]
        # Note: permute creates a view, matmul works with non-contiguous tensors
        q_bf = q_hp.permute(1, 2, 0, 3)  # [batch, q_heads_local, seq, head_dim]
        k_bf = k_hp.permute(1, 2, 0, 3)  # [batch, kv_heads_local, seq, head_dim]
        v_bf = v_hp.permute(1, 2, 0, 3)  # [batch, kv_heads_local, seq, head_dim]

        # Compute scale for attention
        scale = 1.0 / (head_dim ** 0.5)

        # Check if GQA is needed (q_heads != kv_heads)
        q_heads_local = num_heads // cp_size
        kv_heads_local = num_kv_heads // cp_size
        enable_gqa = (q_heads_local != kv_heads_local)

        # Scaled Dot-Product Attention with native GQA support (PyTorch 2.5+)
        # Input: Q [batch, q_heads, seq, dim], K/V [batch, kv_heads, seq, dim]
        # Output: [batch, q_heads, seq, head_dim]
        #
        # Keep computation graph for backward: avoids redundant SDPA forward in backward.
        # FlashAttention only stores logsumexp (O(seq) memory), not full attention matrix.
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
        # Merge+sort is done here in forward, overlapping with last combine P2P.
        (combined_output, local_fc2, all_expert_tokens, all_tokens_per_expert,
         backward_indices) = fc2_combine_p2p_forward(
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
            # all_expert_tokens is pre-merged in forward (expert-major order)
            if all_expert_tokens is None:
                all_expert_tokens = torch.empty(0, hidden_size, dtype=dtype, device=hidden_states.device)

            ctx.save_for_backward(
                # Input and intermediate states
                hidden_states, ln1_out, hidden_after_attn, ln2_flat,
                # Attention states - only attn_input_full needed (q/k/v saved with graph below)
                attn_input_full,
                # MoE states
                permuted_tokens, permuted_probs, restore_indices, sorted_indices,
                router_probs, top_indices,
                all_expert_tokens,
                combined_output,
                # Weights (detached)
                ln1_weight, ln1_bias, ln2_weight, ln2_bias,
                qkv_weight.detach(), proj_weight.detach(),
                router_weight.detach(), moe_w1.detach(), moe_w2.detach(),
            )
            # Store non-tensor merge results on ctx
            ctx.all_tokens_per_expert = all_tokens_per_expert
            ctx.backward_indices = backward_indices
            # Save SDPA computation graph (avoids redundant forward in backward)
            # FlashAttention only stores logsumexp O(seq), not attention matrix O(seq²)
            ctx._q_for_attn = q_for_attn
            ctx._k_for_attn = k_for_attn
            ctx._v_for_attn = v_for_attn
            ctx._attn_out_bf = attn_out_bf
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
            ctx.attn_proj_chunks = attn_proj_chunks
            ctx.attn_qkv_chunks = attn_qkv_chunks
            ctx.moe_combine_chunks = moe_combine_chunks
            ctx.moe_dispatch_chunks = moe_dispatch_chunks
            ctx.activation_func = activation_func
            ctx.num_local_experts = num_local_experts
            ctx.head_dim = head_dim
            ctx.scale = scale

            # MoE routing info
            ctx.input_splits_list = input_splits_list
            ctx.output_splits_list = output_splits_list

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
            return (None,) * 24  # 10 weights + 4 groups/contexts + 10 config params

        # Ensure grad_output matches forward dtype (e.g. loss computed in float32 → cast back to bf16)
        if grad_output.dtype != ctx.dtype:
            grad_output = grad_output.to(ctx.dtype)

        # Retrieve saved tensors
        (hidden_states, ln1_out, hidden_after_attn, ln2_flat,
         attn_input_full,
         permuted_tokens, permuted_probs, restore_indices, sorted_indices,
         router_probs, top_indices,
         all_expert_tokens,
         combined_output,
         ln1_weight, ln1_bias, ln2_weight, ln2_bias,
         qkv_weight, proj_weight,
         router_weight, moe_w1_2d, moe_w2_2d) = ctx.saved_tensors

        # Retrieve pre-computed merge results
        all_tokens_per_expert = ctx.all_tokens_per_expert
        backward_indices = ctx.backward_indices

        # Retrieve SDPA computation graph for direct backward (no redundant forward)
        q_for_attn = ctx._q_for_attn
        k_for_attn = ctx._k_for_attn
        v_for_attn = ctx._v_for_attn
        attn_out_bf = ctx._attn_out_bf
        enable_gqa = ctx._enable_gqa

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
        moe_combine_chunks = ctx.moe_combine_chunks
        moe_dispatch_chunks = ctx.moe_dispatch_chunks
        activation_func = ctx.activation_func
        num_local_experts = ctx.num_local_experts
        head_dim = ctx.head_dim
        scale = ctx.scale

        input_splits_list = ctx.input_splits_list
        output_splits_list = ctx.output_splits_list

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
        moe_combine_chunks_actual = moe_combine_chunks if is_enabled else 1
        moe_dispatch_chunks_actual = moe_dispatch_chunks if is_enabled else 1

        # =====================================================================
        # MoE Backward (reverse order)
        # =====================================================================

        # Gradient through residual: grad flows to both hidden_after_attn and moe_output
        grad_hidden_after_attn = grad_output.clone()
        grad_moe_output = grad_output.view(num_tokens, hidden_size)

        # Step 1: Backward through sum and restore
        grad_restored = grad_moe_output.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_size)
        # restore_indices = argsort(sorted_indices), so inverse_restore_indices == sorted_indices.
        grad_weighted = grad_restored[sorted_indices]

        # Step 2: Backward through prob weighting
        grad_combined = grad_weighted * permuted_probs.unsqueeze(-1).to(grad_weighted.dtype)
        grad_permuted_probs = (grad_weighted * combined_output.to(grad_weighted.dtype)).sum(dim=-1)

        # Region 1: Combine AllToAll → FC2 dx (communication-first pipeline)
        # Chunked AllToAll submitted first, FC2 dx computed as each chunk arrives
        scheduler.begin_region('moe_combine')
        (grad_all_fc1, act_output, all_fc1, grad_all_fc2,
         all_expert_tokens, all_tokens_per_expert, backward_indices) = combine_fc2_backward(
            grad_combined, input_splits_list, output_splits_list, ep_group, layer_id,
            weight1=moe_w1,
            weight2=moe_w2,
            activation_func=activation_func,
            num_local_experts=num_local_experts,
            num_chunks=moe_combine_chunks_actual,
            all_expert_tokens=all_expert_tokens,
            all_tokens_per_expert=all_tokens_per_expert,
            backward_indices=backward_indices,
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
        # Register router_backward + router_dw + LN2 normalize as scheduler
        # tasks so they execute during R2's Dispatch A2A overlap window.
        # All inputs are available NOW (before R2): they only depend on
        # forward-saved tensors and pre-R1 grad computations.
        # =================================================================
        router_result = [None, None]  # [grad_hidden_from_router, grad_router_logits]
        ln2_cache = [None, None]      # [std, normalized]
        grad_router_weight = None
        grad_ln2_weight = None
        grad_ln2_bias = None

        if is_enabled:
            # --- router backward (no weight, just compute) ---
            _grad_permuted_probs = grad_permuted_probs.detach()
            _sorted_indices = sorted_indices
            _restore_indices = restore_indices
            _permuted_probs = permuted_probs
            _router_probs = router_probs
            _top_indices = top_indices
            _router_weight = router_weight

            def _router_bwd_task():
                router_result[0], router_result[1] = router_backward(
                    grad_permuted_probs=_grad_permuted_probs,
                    sorted_indices=_sorted_indices,
                    restore_indices=_restore_indices,
                    permuted_probs=_permuted_probs,
                    router_probs=_router_probs,
                    top_indices=_top_indices,
                    router_weight=_router_weight,
                    num_tokens=num_tokens,
                    top_k=top_k,
                    dtype=dtype,
                )
                return None  # no gradient to write

            scheduler.register_dw_task(
                layer_name=f"router_bwd_L{layer_id}",
                layer_id=layer_id,
                compute_fn=_router_bwd_task,
                weight_param=None,
            )

            # --- router dW (depends on router_result[1] from above) ---
            _ln2_flat_saved = ln2_flat.detach()

            def _router_dw_task():
                return torch.matmul(_ln2_flat_saved.t().float(),
                                    router_result[1].float())

            scheduler.register_dw_task(
                layer_name=f"router_weight_L{layer_id}",
                layer_id=layer_id,
                compute_fn=_router_dw_task,
                weight_param=orig_router_weight,
            )

            # --- LN2 normalize precompute (no weight, just cache std & normalized) ---
            _hidden_after_attn = hidden_after_attn.detach()

            def _ln2_norm_task():
                mean = _hidden_after_attn.mean(dim=-1, keepdim=True)
                var = _hidden_after_attn.var(dim=-1, unbiased=False, keepdim=True)
                ln2_cache[0] = (var + 1e-5).sqrt()
                ln2_cache[1] = (_hidden_after_attn - mean) / ln2_cache[0]
                return None

            scheduler.register_dw_task(
                layer_name=f"ln2_norm_L{layer_id}",
                layer_id=layer_id,
                compute_fn=_ln2_norm_task,
                weight_param=None,
            )

        # Region 2: FC1 dx → Dispatch AllToAll (compute-first pipeline)
        # FC1 dx computed in chunks, each chunk submitted to AllToAll on completion
        # dW window now also executes: router_bwd → router_dw → ln2_norm → moe dW
        split_sizes_exp_major = backward_indices.get('split_sizes_exp_major', all_tokens_per_expert)
        sorted_idxs_exp_to_rank = backward_indices.get('sorted_idxs_exp_to_rank', list(range(len(all_tokens_per_expert))))
        row_idx_exp_to_rank = backward_indices.get('row_idx_exp_to_rank', None)

        scheduler.begin_region('moe_dispatch')
        grad_hidden_from_moe_tokens = fc1_dispatch_backward(
            grad_all_fc1, moe_w1,
            num_local_experts, all_tokens_per_expert,
            split_sizes_exp_major, sorted_idxs_exp_to_rank, row_idx_exp_to_rank,
            input_splits_list, output_splits_list, ep_group,
            restore_indices=restore_indices,
            num_tokens=num_tokens,
            top_k=top_k,
            layer_id=layer_id, num_chunks=moe_dispatch_chunks_actual,
        )

        scheduler.end_region()

        # After R2: router_backward + LN2 normalize already executed in R2 overlap.
        # Fallback: compute synchronously if scheduler disabled.
        if not is_enabled:
            router_result[0], router_result[1] = router_backward(
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
            grad_router_weight = register_router_dw_task(
                hidden_states=ln2_flat,
                grad_router_logits=router_result[1],
                router_weight=orig_router_weight,
                layer_id=layer_id,
            )

        # Combine MoE gradients
        grad_ln2_flat = grad_hidden_from_moe_tokens + router_result[0]

        # LayerNorm 2 backward
        grad_ln2_out = grad_ln2_flat.view(seq_len, batch_size, hidden_size)

        if is_enabled:
            # LN2 normalize was precomputed in R2 overlap window
            std = ln2_cache[0]
            normalized = ln2_cache[1]
        else:
            mean = hidden_after_attn.mean(dim=-1, keepdim=True)
            var = hidden_after_attn.var(dim=-1, unbiased=False, keepdim=True)
            std = (var + 1e-5).sqrt()
            normalized = (hidden_after_attn - mean) / std

        # dX: gradient through LayerNorm
        grad_hidden_after_attn = grad_hidden_after_attn + grad_ln2_out * ln2_weight / std

        # Register dW tasks for LN2 (execute during R3 overlap)
        if is_enabled:
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
        else:
            grad_ln2_weight = (grad_ln2_out * normalized).sum(dim=(0, 1))
            grad_ln2_bias = grad_ln2_out.sum(dim=(0, 1))

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
            num_chunks=attn_proj_chunks_actual,
        )

        scheduler.end_region()

        # Step 16: Attention score backward using saved computation graph
        # Direct autograd.grad on saved SDPA output - no redundant forward pass
        # FlashAttention internally does tiled recomputation (unavoidable by design)
        grad_attn_hp = grad_attn_output.permute(1, 2, 0, 3)
        grad_q, grad_k, grad_v = torch.autograd.grad(
            attn_out_bf, (q_for_attn, k_for_attn, v_for_attn),
            grad_attn_hp, retain_graph=False
        )

        # =================================================================
        # Register LN1 normalize precompute as scheduler task.
        # R4's dW window is otherwise empty (R3 already drained the queue).
        # LN1 normalize only depends on hidden_states (forward-saved).
        # =================================================================
        ln1_cache = [None, None]  # [std1, normalized1]
        grad_ln1_weight = None
        grad_ln1_bias = None

        if is_enabled:
            _hidden_states_saved = hidden_states.detach()

            def _ln1_norm_task():
                mean1 = _hidden_states_saved.mean(dim=-1, keepdim=True)
                var1 = _hidden_states_saved.var(dim=-1, unbiased=False, keepdim=True)
                ln1_cache[0] = (var1 + 1e-5).sqrt()
                ln1_cache[1] = (_hidden_states_saved - mean1) / ln1_cache[0]
                return None

            scheduler.register_dw_task(
                layer_name=f"ln1_norm_L{layer_id}",
                layer_id=layer_id,
                compute_fn=_ln1_norm_task,
                weight_param=None,
            )

        # Region 4: hp2sp AllToAll → QKV dX (communication-first pipeline)
        scheduler.begin_region('attn_qkv')
        grad_ln1_out, grad_qkv_weight = hp2sp_qkv_backward(
            grad_q, grad_k, grad_v, cp_group,
            tokens=ln1_out, weight_qkv=orig_qkv_weight,
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
            layer_id=layer_id, num_chunks=attn_qkv_chunks_actual,
        )

        scheduler.end_region()

        # Step 18: Residual backward - grad flows to hidden_states
        grad_hidden_states = grad_hidden_after_attn.clone()

        # Step 19: LayerNorm 1 backward
        if is_enabled:
            # LN1 normalize was precomputed in R4 overlap window
            std1 = ln1_cache[0]
            normalized1 = ln1_cache[1]
        else:
            mean1 = hidden_states.mean(dim=-1, keepdim=True)
            var1 = hidden_states.var(dim=-1, unbiased=False, keepdim=True)
            std1 = (var1 + 1e-5).sqrt()
            normalized1 = (hidden_states - mean1) / std1

        # dX: gradient through LayerNorm
        grad_hidden_states = grad_hidden_states + grad_ln1_out * ln1_weight / std1

        # Register dW tasks for LN1 (execute during next layer's R1)
        if is_enabled:
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
            None, None, None, None,  # attn_proj_chunks, attn_qkv_chunks, moe_combine_chunks, moe_dispatch_chunks
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
        attn_proj_chunks: Region 3 chunks - output projection dX + sp2hp AllToAll
        attn_qkv_chunks: Region 4 chunks - hp2sp AllToAll + QKV dX
        moe_combine_chunks: Region 1 chunks - Combine AllToAll + FC2 dX
        moe_dispatch_chunks: Region 2 chunks - FC1 dX + Dispatch AllToAll
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
        moe_combine_chunks: int = 1,
        moe_dispatch_chunks: int = 1,
        ar_trickle_sizes: Optional[dict] = None,
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
        self.moe_combine_chunks = moe_combine_chunks
        self.moe_dispatch_chunks = moe_dispatch_chunks
        self.activation_func = activation_func or F.gelu

        # Apply per-region AR trickle sizes to scheduler singleton
        if ar_trickle_sizes is not None:
            from fluid.core.scheduler import BackwardScheduler
            sched = BackwardScheduler()
            sched.ar_trickle_sizes = {
                r: sz if sz > 0 else 0 for r, sz in ar_trickle_sizes.items()
            }

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
            self.attn_proj_chunks, self.attn_qkv_chunks,
            self.moe_combine_chunks, self.moe_dispatch_chunks,
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
        attn_proj_chunks: int = 1,
        attn_qkv_chunks: int = 1,
        moe_combine_chunks: int = 1,
        moe_dispatch_chunks: int = 1,
        ar_trickle_sizes: Optional[dict] = None,
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
                moe_combine_chunks=moe_combine_chunks,
                moe_dispatch_chunks=moe_dispatch_chunks,
                ar_trickle_sizes=ar_trickle_sizes if i == 0 else None,  # set once
                activation_func=activation_func,
                dtype=dtype,
                device=device,
            )
            for i in range(num_layers)
        ])

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
            # Then shared params
            shared_params.extend([
                layer.router_weight,
                layer.ln2_weight, layer.ln2_bias,
                layer.proj_weight,
                layer.qkv_weight,
                layer.ln1_weight, layer.ln1_bias,
            ])
        sched.setup_ar_buffer(shared_params)
        if sched.expert_dp_world_size > 1:
            sched.setup_expert_ar_buffer(expert_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
