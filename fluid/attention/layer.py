"""
Attention Layer - Complete Autograd Functions

This module provides complete attention autograd functions that combine:
- Forward: P2P overlap for QKV + sp2hp, attention, hp2sp + output projection
- Backward: AllToAll + chunked compute overlap + dW scheduling

Key classes:
- AttentionP2PChunkedFunction: Full autograd function with P2P forward + chunked AllToAll backward
- AttentionLayer: High-level nn.Module wrapper

Design principles:
- Forward uses P2P overlap for compute-communication overlap
- Backward uses AllToAll with chunked compute overlap
- Memory-efficient: save Q, K, V instead of attention matrix
- dW tasks are registered and executed during AllToAll communication
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple

from fluid.core.forward_comm import AttentionMultiCardOverlapContext, MultiCardOverlapContext
from fluid.attention.forward import (
    qkv_projection_p2p_forward,
    scaled_dot_product_attention_forward,
    output_projection_p2p_forward,
)
from fluid.attention.backward import (
    output_projection_backward_chunked,
    attention_backward_chunked,
    qkv_projection_backward,
    output_projection_register_dw,
)
from fluid.core.scheduler import get_backward_scheduler


class AttentionP2PChunkedFunction(torch.autograd.Function):
    """
    Complete attention autograd function with:
    - Forward: P2P overlap for QKV + sp2hp, attention, hp2sp + output projection
    - Backward: AllToAll + chunked compute overlap + dW scheduling

    This is the main entry point for context-parallel attention.

    Forward timeline:
        QKV phase: Each round computes QKV for partner while P2P runs
        Attention: Standard scaled dot-product attention
        Output phase: Each round computes partial output while receiving next data

    Backward timeline:
        Output proj: Chunked dX + sp2hp AllToAll overlap
        Attention: Recompute + chunked grad_Q/K/V + hp2sp AllToAll overlap
        QKV proj: dX + dW task registration
    """

    @staticmethod
    def forward(
        ctx,
        tokens: torch.Tensor,
        weight_qkv: torch.Tensor,
        weight_proj: torch.Tensor,
        bias_proj: Optional[torch.Tensor],
        cp_group: dist.ProcessGroup,
        qkv_overlap_ctx: MultiCardOverlapContext,
        proj_overlap_ctx: MultiCardOverlapContext,
        layer_id: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_chunks: int,
    ) -> torch.Tensor:
        """
        Forward pass with P2P overlap.

        Args:
            tokens: [seq_local, batch, hidden] - input tokens
            weight_qkv: [total_proj, hidden] - QKV weight
            weight_proj: [hidden, total_heads * head_dim] - output projection weight
            bias_proj: [hidden] or None - output projection bias
            cp_group: context parallel process group
            qkv_overlap_ctx: P2P context for QKV phase
            proj_overlap_ctx: P2P context for output projection phase
            layer_id: layer ID
            num_heads: total Q heads
            num_kv_heads: total K/V heads
            head_dim: dimension per head
            num_chunks: number of chunks for backward

        Returns:
            output: [seq_local, batch, hidden] - final output
        """
        needs_grad = tokens.requires_grad
        ctx.needs_grad = needs_grad

        # ============================================
        # Step 1: QKV projection with P2P overlap
        # ============================================
        q, k, v = qkv_projection_p2p_forward(
            tokens, weight_qkv, num_heads, num_kv_heads, head_dim,
            cp_group, qkv_overlap_ctx
        )
        # q: [seq_full, batch, q_heads_local, head_dim]
        # k, v: [seq_full, batch, kv_heads_local, head_dim] - NOT expanded

        # ============================================
        # Step 1.5: GQA expansion (expand K/V to match Q heads)
        # ============================================
        q_per_group = num_heads // num_kv_heads
        cp_size = cp_group.size()
        kv_heads_local = num_kv_heads // cp_size
        q_heads_local = num_heads // cp_size

        if q_per_group > 1:
            k = k.repeat_interleave(q_per_group, dim=2)
            v = v.repeat_interleave(q_per_group, dim=2)
        else:
            k = k.view(k.shape[0], k.shape[1], q_heads_local, head_dim)
            v = v.view(v.shape[0], v.shape[1], q_heads_local, head_dim)

        # ============================================
        # Save for backward BEFORE permute (optimization)
        # ============================================
        # IMPORTANT: Save Q, K, V before permute to avoid hidden memory copy.
        # After permute, tensors become non-contiguous views, and save_for_backward
        # would need to make contiguous copies (~3.5ms overhead for large tensors).
        # By saving before permute, we save contiguous tensors directly.
        if needs_grad:
            # Save Q, K, V in [seq_full, batch, heads_local, head_dim] format (before permute, contiguous)
            q_saved, k_saved, v_saved = q, k, v

        # ============================================
        # Step 2: Attention computation
        # ============================================
        # Permute to [batch, heads_local, seq_full, head_dim] for attention
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        scale = 1.0 / (head_dim ** 0.5)
        attn_out = scaled_dot_product_attention_forward(q, k, v, scale, is_causal=True)

        # Permute back to [seq_full, batch, heads_local, head_dim]
        attn_out = attn_out.permute(2, 0, 1, 3)

        # ============================================
        # Step 3: Output projection with P2P overlap
        # ============================================
        output, attn_input_full = output_projection_p2p_forward(
            attn_out, weight_proj, bias_proj, cp_group, proj_overlap_ctx
        )

        # ============================================
        # Save for backward
        # ============================================
        if needs_grad:
            ctx.save_for_backward(tokens, weight_qkv, weight_proj, q_saved, k_saved, v_saved, attn_input_full)

        ctx.cp_group = cp_group
        ctx.layer_id = layer_id
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim
        ctx.num_chunks = num_chunks
        ctx.has_bias = bias_proj is not None
        ctx.scale = scale

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass with AllToAll + chunked compute overlap.

        Args:
            grad_output: [seq_local, batch, hidden] - gradient w.r.t. output

        Returns:
            Gradients for all inputs (most are None for non-tensor inputs)
        """
        if not ctx.needs_grad:
            return (None,) * 12

        tokens, weight_qkv, weight_proj, q, k, v, attn_input_full = ctx.saved_tensors
        cp_group = ctx.cp_group
        layer_id = ctx.layer_id
        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        head_dim = ctx.head_dim
        num_chunks = ctx.num_chunks
        scale = ctx.scale
        has_bias = ctx.has_bias

        cp_size = cp_group.size()
        total_heads = num_heads

        # ============================================
        # Step 1: Register output projection dW task first (so it can execute during AllToAll)
        # ============================================
        grad_weight_proj = output_projection_register_dw(
            grad_output, attn_input_full, weight_proj, layer_id
        )

        # ============================================
        # Step 2: Output projection backward (chunked dX + sp2hp AllToAll)
        # ============================================
        grad_attn_output = output_projection_backward_chunked(
            grad_output, weight_proj, total_heads, head_dim, cp_group, num_chunks
        )

        # ============================================
        # Step 3: Attention backward (recompute + chunked grad_Q/K/V + hp2sp AllToAll)
        # ============================================
        # Reshape grad_attn_output: [seq_full, batch, heads_local, head_dim]
        #                        -> [batch, heads_local, seq_full, head_dim]
        grad_attn_hp = grad_attn_output.permute(1, 2, 0, 3)

        # Q, K, V were saved BEFORE permute in [seq_full, batch, heads_local, head_dim] format
        # Permute them to [batch, heads_local, seq_full, head_dim] for attention backward
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        grad_q, grad_k, grad_v = attention_backward_chunked(
            grad_attn_hp, q, k, v, scale, cp_group, num_chunks
        )

        # ============================================
        # Step 4: QKV projection backward (dX + dW)
        # ============================================
        grad_tokens, grad_weight_qkv = qkv_projection_backward(
            grad_q, grad_k, grad_v, tokens, weight_qkv, cp_group,
            num_heads, num_kv_heads, head_dim, layer_id
        )

        # Bias gradient
        if has_bias:
            grad_bias = grad_output.sum(dim=(0, 1))
        else:
            grad_bias = None

        # Return gradients in same order as forward inputs
        return (
            grad_tokens,      # tokens
            grad_weight_qkv,  # weight_qkv
            grad_weight_proj, # weight_proj
            grad_bias,        # bias_proj
            None,             # cp_group
            None,             # qkv_overlap_ctx
            None,             # proj_overlap_ctx
            None,             # layer_id
            None,             # num_heads
            None,             # num_kv_heads
            None,             # head_dim
            None,             # num_chunks
        )


def attention_p2p_chunked(
    tokens: torch.Tensor,
    weight_qkv: torch.Tensor,
    weight_proj: torch.Tensor,
    bias_proj: Optional[torch.Tensor],
    cp_group: dist.ProcessGroup,
    qkv_overlap_ctx: MultiCardOverlapContext,
    proj_overlap_ctx: MultiCardOverlapContext,
    layer_id: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    num_chunks: int = 4,
) -> torch.Tensor:
    """
    Context-parallel attention with P2P forward overlap and chunked AllToAll backward.

    This is the main API for using the optimized attention layer.

    Args:
        tokens: [seq_local, batch, hidden] - input tokens in SP format
        weight_qkv: [total_proj, hidden] - QKV projection weight (interleaved layout)
        weight_proj: [hidden, total_heads * head_dim] - output projection weight
        bias_proj: [hidden] or None - output projection bias
        cp_group: context parallel process group
        qkv_overlap_ctx: P2P context for QKV + sp2hp phase
        proj_overlap_ctx: P2P context for hp2sp + output projection phase
        layer_id: layer ID for dW task naming
        num_heads: total number of Q heads
        num_kv_heads: total number of K/V heads (groups for GQA)
        head_dim: dimension per head
        num_chunks: number of chunks for backward overlap

    Returns:
        output: [seq_local, batch, hidden] - final output in SP format

    Example:
        >>> # Setup
        >>> cp_group = dist.new_group(ranks=[0, 1, 2, 3])
        >>> qkv_ctx = AttentionMultiCardOverlapContext(cp_group)
        >>> proj_ctx = AttentionMultiCardOverlapContext(cp_group)
        >>>
        >>> # Forward
        >>> output = attention_p2p_chunked(
        ...     tokens, weight_qkv, weight_proj, None,
        ...     cp_group, qkv_ctx, proj_ctx,
        ...     layer_id=0, num_heads=32, num_kv_heads=8, head_dim=128,
        ...     num_chunks=4
        ... )
        >>>
        >>> # Backward (automatically uses chunked AllToAll overlap)
        >>> loss = output.sum()
        >>> loss.backward()
    """
    return AttentionP2PChunkedFunction.apply(
        tokens, weight_qkv, weight_proj, bias_proj,
        cp_group, qkv_overlap_ctx, proj_overlap_ctx,
        layer_id, num_heads, num_kv_heads, head_dim, num_chunks
    )


class AttentionLayer(nn.Module):
    """
    High-level attention layer module with P2P forward and chunked AllToAll backward.

    This module wraps the low-level autograd function with proper weight initialization
    and context management.

    Args:
        hidden_size: model hidden dimension
        num_heads: total number of Q heads
        num_kv_heads: total number of K/V heads (for GQA)
        head_dim: dimension per head
        cp_group: context parallel process group
        layer_id: layer ID for dW task naming
        num_chunks: number of chunks for backward overlap
        bias: whether to use bias in output projection

    Example:
        >>> layer = AttentionLayer(
        ...     hidden_size=4096,
        ...     num_heads=32,
        ...     num_kv_heads=8,
        ...     head_dim=128,
        ...     cp_group=cp_group,
        ...     layer_id=0,
        ... )
        >>> output = layer(tokens)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cp_group: dist.ProcessGroup,
        layer_id: int = 0,
        num_chunks: int = 4,
        bias: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.cp_group = cp_group
        self.layer_id = layer_id
        self.num_chunks = num_chunks

        # QKV projection: interleaved layout [Q_g0, K_g0, V_g0, Q_g1, K_g1, V_g1, ...]
        q_per_group = num_heads // num_kv_heads
        group_size = (q_per_group + 2) * head_dim
        total_proj = num_kv_heads * group_size

        self.weight_qkv = nn.Parameter(torch.empty(total_proj, hidden_size))

        # Output projection
        total_head_dim = num_heads * head_dim
        self.weight_proj = nn.Parameter(torch.empty(hidden_size, total_head_dim))

        if bias:
            self.bias_proj = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('bias_proj', None)

        # P2P overlap contexts
        cp_size = cp_group.size()
        device = torch.device(f'cuda:{dist.get_rank(cp_group)}')
        self.qkv_overlap_ctx = AttentionMultiCardOverlapContext(device, cp_size)
        self.proj_overlap_ctx = AttentionMultiCardOverlapContext(device, cp_size)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight_qkv)
        nn.init.xavier_uniform_(self.weight_proj)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens: [seq_local, batch, hidden] - input tokens in SP format

        Returns:
            output: [seq_local, batch, hidden] - output in SP format
        """
        return attention_p2p_chunked(
            tokens,
            self.weight_qkv,
            self.weight_proj,
            self.bias_proj,
            self.cp_group,
            self.qkv_overlap_ctx,
            self.proj_overlap_ctx,
            self.layer_id,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.num_chunks,
        )

    def extra_repr(self) -> str:
        return (
            f'hidden_size={self.hidden_size}, '
            f'num_heads={self.num_heads}, '
            f'num_kv_heads={self.num_kv_heads}, '
            f'head_dim={self.head_dim}, '
            f'num_chunks={self.num_chunks}'
        )


__all__ = [
    'AttentionP2PChunkedFunction',
    'attention_p2p_chunked',
    'AttentionLayer',
]
