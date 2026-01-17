"""
Attention Baseline Implementation

This module implements a simplified baseline attention layer with Ulysses-style
sequence parallel AllToAll. The backward pass uses the scheduler for dW overlap.

Key features:
- Ulysses SP: sp2hp AllToAll before attention, hp2sp AllToAll after
- dW tasks registered for overlap during backward
- Compatible with scheduler-based backward optimization
- Merged autograd functions (2 boundaries) for fair comparison with Overlap

Note: This is a standalone implementation for testing/benchmarking.
For production use with Megatron-LM, see attention_module.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from fluid.core import _all_to_all_sp2hp_forward, _all_to_all_hp2sp_forward
from fluid.core.scheduler import get_backward_scheduler


class _QKVWithSP2HPFunction(torch.autograd.Function):
    """
    Combined QKV projection + sp2hp AllToAll autograd function.

    Forward:
        1. QKV projection: tokens @ weight_qkv.T -> qkv
        2. sp2hp AllToAll: exchange sequence for heads
        3. Split into Q, K, V

    Backward:
        1. hp2sp AllToAll (with scheduler dW overlap)
        2. Compute dX for QKV projection
        3. Register dW task
    """

    @staticmethod
    def forward(ctx, tokens, weight_qkv, cp_group, layer_id, num_heads, num_kv_heads, head_dim):
        """
        Args:
            tokens: [seq_local, batch, hidden]
            weight_qkv: [total_proj, hidden] - interleaved layout
            cp_group: context parallel group
            layer_id: layer ID for dW task naming
            num_heads: total Q heads
            num_kv_heads: total K/V heads (groups)
            head_dim: dimension per head

        Returns:
            q, k, v: [seq_full, batch, heads_local, head_dim]
        """
        seq_local, batch, hidden_size = tokens.shape
        cp_size = cp_group.size()

        q_per_group = num_heads // num_kv_heads
        group_size = (q_per_group + 2) * head_dim
        total_proj = num_kv_heads * group_size

        # 1. QKV Projection
        # tokens: [seq_local, batch, hidden] @ weight_qkv.T: [hidden, total_proj] -> [seq_local, batch, total_proj]
        qkv = torch.matmul(tokens, weight_qkv.t())

        # Extract Q, K, V from interleaved layout
        qkv = qkv.view(seq_local, batch, num_kv_heads, group_size)

        q_dim = q_per_group * head_dim
        q = qkv[:, :, :, :q_dim]
        k = qkv[:, :, :, q_dim:q_dim + head_dim]
        v = qkv[:, :, :, q_dim + head_dim:]

        # Reshape Q to [seq_local, batch, num_heads, head_dim]
        q = q.reshape(seq_local, batch, num_heads, head_dim)

        # For GQA, expand K/V; for MHA they're already correct
        if q_per_group > 1:
            k = k.repeat_interleave(q_per_group, dim=2)
            v = v.repeat_interleave(q_per_group, dim=2)
        else:
            k = k.view(seq_local, batch, num_heads, head_dim)
            v = v.view(seq_local, batch, num_heads, head_dim)

        # 2. sp2hp AllToAll (merge Q, K, V for single communication)
        if cp_size > 1:
            qkv_merged = torch.cat([q, k, v], dim=-1)  # [seq_local, batch, num_heads, 3*head_dim]
            qkv_merged = _all_to_all_sp2hp_forward(qkv_merged, cp_group)
            # Split: [seq_full, batch, heads_local, 3*head_dim] -> 3 x [seq_full, batch, heads_local, head_dim]
            q, k, v = torch.split(qkv_merged, head_dim, dim=-1)

        # Save for backward
        needs_grad = tokens.requires_grad
        ctx.needs_grad = needs_grad
        if needs_grad:
            ctx.save_for_backward(tokens, weight_qkv)
        ctx.cp_group = cp_group
        ctx.layer_id = layer_id
        ctx.num_heads = num_heads
        ctx.num_kv_heads = num_kv_heads
        ctx.head_dim = head_dim

        return q, k, v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        """Backward with hp2sp AllToAll and dW overlap."""
        if not ctx.needs_grad:
            return None, None, None, None, None, None, None

        tokens, weight_qkv = ctx.saved_tensors
        cp_group = ctx.cp_group
        layer_id = ctx.layer_id
        num_heads = ctx.num_heads
        num_kv_heads = ctx.num_kv_heads
        head_dim = ctx.head_dim

        cp_size = cp_group.size()
        seq_local, batch, hidden_size = tokens.shape

        q_per_group = num_heads // num_kv_heads
        group_size = (q_per_group + 2) * head_dim
        heads_local = num_heads // cp_size

        scheduler = get_backward_scheduler()

        # 1. hp2sp AllToAll for gradients (with scheduler dW overlap)
        if cp_size > 1:
            # Merge grad_q, grad_k, grad_v
            grad_qkv_merged = torch.cat([grad_q, grad_k, grad_v], dim=-1)

            if scheduler.is_enabled():
                comm_stream = scheduler.comm_stream
                default_stream = torch.cuda.current_stream()

                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_stream(default_stream)
                    grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_merged, cp_group)
                    scheduler.record_alltoall_end(comm_stream)  # Reusable event

                scheduler.on_alltoall_start(comm_type=f"baseline_qkv_hp2sp_L{layer_id}")
                default_stream.wait_stream(comm_stream)
            else:
                grad_qkv_sp = _all_to_all_hp2sp_forward(grad_qkv_merged, cp_group)

            # Split back: [seq_local, batch, num_heads, 3*head_dim] -> grad_q, grad_k, grad_v
            grad_q_sp, grad_k_sp, grad_v_sp = torch.split(grad_qkv_sp, head_dim, dim=-1)
        else:
            grad_q_sp = grad_q
            grad_k_sp = grad_k
            grad_v_sp = grad_v

        # 2. Handle GQA: sum K/V gradients back to kv_heads
        if q_per_group > 1:
            # grad_k_sp, grad_v_sp: [seq_local, batch, num_heads, head_dim]
            # Need to reduce to [seq_local, batch, num_kv_heads, head_dim]
            grad_k_sp = grad_k_sp.view(seq_local, batch, num_kv_heads, q_per_group, head_dim).sum(dim=3)
            grad_v_sp = grad_v_sp.view(seq_local, batch, num_kv_heads, q_per_group, head_dim).sum(dim=3)

        # 3. Reassemble to interleaved grad_qkv format
        # grad_q_sp: [seq_local, batch, num_heads, head_dim] -> [seq_local, batch, num_kv_heads, q_per_group * head_dim]
        grad_q_grouped = grad_q_sp.view(seq_local, batch, num_kv_heads, q_per_group * head_dim)
        # grad_k_sp, grad_v_sp: [seq_local, batch, num_kv_heads, head_dim]
        grad_qkv = torch.cat([grad_q_grouped, grad_k_sp, grad_v_sp], dim=-1)
        # grad_qkv: [seq_local, batch, num_kv_heads, group_size] -> [seq_local, batch, total_proj]
        grad_qkv = grad_qkv.view(seq_local, batch, -1)

        # 4. Compute dX: grad_qkv @ weight_qkv
        grad_tokens = torch.matmul(grad_qkv, weight_qkv)

        # 5. Register dW task
        if scheduler.is_enabled():
            tokens_saved = tokens.detach()
            grad_qkv_saved = grad_qkv.detach()
            weight_qkv_saved = weight_qkv

            def compute_dw():
                # dW = grad_qkv.T @ tokens
                # [total_proj, seq*batch] @ [seq*batch, hidden] -> [total_proj, hidden]
                grad_flat = grad_qkv_saved.reshape(-1, grad_qkv_saved.shape[-1])
                tokens_flat = tokens_saved.reshape(-1, tokens_saved.shape[-1])
                return torch.matmul(grad_flat.t(), tokens_flat)

            scheduler.register_dw_task(
                layer_name=f"baseline_qkv_L{layer_id}",
                layer_id=layer_id,
                compute_fn=compute_dw,
                priority=100,
                weight_param=weight_qkv_saved,
            )
            grad_weight = None
        else:
            grad_flat = grad_qkv.reshape(-1, grad_qkv.shape[-1])
            tokens_flat = tokens.reshape(-1, tokens.shape[-1])
            grad_weight = torch.matmul(grad_flat.t(), tokens_flat)

        return grad_tokens, grad_weight, None, None, None, None, None


class _HP2SPWithOutputFunction(torch.autograd.Function):
    """
    Combined hp2sp AllToAll + output projection autograd function.

    Forward:
        1. hp2sp AllToAll: exchange heads for sequence
        2. Output projection: attn_out @ weight_proj

    Backward:
        1. Compute dX for output projection
        2. sp2hp AllToAll (with scheduler dW overlap)
        3. Register dW task
    """

    @staticmethod
    def forward(ctx, attn_out, weight_proj, cp_group, layer_id, num_heads, head_dim):
        """
        Args:
            attn_out: [seq_full, batch, heads_local, head_dim]
            weight_proj: [hidden, num_heads * head_dim]
            cp_group: context parallel group
            layer_id: layer ID for dW task naming
            num_heads: total heads
            head_dim: dimension per head

        Returns:
            output: [seq_local, batch, hidden]
        """
        cp_size = cp_group.size()
        seq_full = attn_out.shape[0]
        batch = attn_out.shape[1]
        hidden_size = weight_proj.shape[0]
        seq_local = seq_full // cp_size

        # 1. hp2sp AllToAll
        if cp_size > 1:
            attn_sp = _all_to_all_hp2sp_forward(attn_out, cp_group)
        else:
            attn_sp = attn_out

        # Reshape: [seq_local, batch, num_heads, head_dim] -> [seq_local, batch, num_heads * head_dim]
        attn_flat = attn_sp.reshape(seq_local, batch, -1)

        # 2. Output projection
        output = torch.matmul(attn_flat, weight_proj.t())

        # Save for backward
        needs_grad = attn_out.requires_grad
        ctx.needs_grad = needs_grad
        if needs_grad:
            ctx.save_for_backward(attn_flat, weight_proj)
        ctx.cp_group = cp_group
        ctx.layer_id = layer_id
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim
        ctx.seq_full = seq_full

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward with sp2hp AllToAll and dW overlap."""
        if not ctx.needs_grad:
            return None, None, None, None, None, None

        attn_flat, weight_proj = ctx.saved_tensors
        cp_group = ctx.cp_group
        layer_id = ctx.layer_id
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim
        seq_full = ctx.seq_full

        cp_size = cp_group.size()
        seq_local, batch, hidden_size = grad_output.shape

        scheduler = get_backward_scheduler()

        # 1. Compute dX for output projection
        # grad_output: [seq_local, batch, hidden] @ weight_proj: [hidden, num_heads * head_dim]
        grad_attn_flat = torch.matmul(grad_output, weight_proj)
        # Reshape: [seq_local, batch, num_heads * head_dim] -> [seq_local, batch, num_heads, head_dim]
        grad_attn_sp = grad_attn_flat.view(seq_local, batch, num_heads, head_dim)

        # 2. sp2hp AllToAll (with scheduler dW overlap)
        if cp_size > 1:
            if scheduler.is_enabled():
                comm_stream = scheduler.comm_stream
                default_stream = torch.cuda.current_stream()

                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_stream(default_stream)
                    grad_attn_out = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
                    scheduler.record_alltoall_end(comm_stream)  # Reusable event

                scheduler.on_alltoall_start(comm_type=f"baseline_proj_sp2hp_L{layer_id}")
                default_stream.wait_stream(comm_stream)
            else:
                grad_attn_out = _all_to_all_sp2hp_forward(grad_attn_sp, cp_group)
        else:
            grad_attn_out = grad_attn_sp

        # 3. Register dW task
        if scheduler.is_enabled():
            attn_flat_saved = attn_flat.detach()
            grad_output_saved = grad_output.detach()
            weight_proj_saved = weight_proj

            def compute_dw():
                # dW = grad_output.T @ attn_flat
                # [hidden, seq*batch] @ [seq*batch, num_heads*head_dim] -> [hidden, num_heads*head_dim]
                grad_flat = grad_output_saved.reshape(-1, grad_output_saved.shape[-1])
                attn_flat_2d = attn_flat_saved.reshape(-1, attn_flat_saved.shape[-1])
                return torch.matmul(grad_flat.t(), attn_flat_2d)

            scheduler.register_dw_task(
                layer_name=f"baseline_proj_L{layer_id}",
                layer_id=layer_id,
                compute_fn=compute_dw,
                priority=99,
                weight_param=weight_proj_saved,
            )
            grad_weight = None
        else:
            grad_flat = grad_output.reshape(-1, grad_output.shape[-1])
            attn_flat_2d = attn_flat.reshape(-1, attn_flat.shape[-1])
            grad_weight = torch.matmul(grad_flat.t(), attn_flat_2d)

        return grad_attn_out, grad_weight, None, None, None, None


def scaled_dot_product_attention(query, key, value, scale=None):
    """
    Simple scaled dot-product attention.

    Args:
        query: [batch, heads, seq, head_dim]
        key: [batch, heads, seq, head_dim]
        value: [batch, heads, seq, head_dim]
        scale: Optional scale factor (default: 1/sqrt(head_dim))

    Returns:
        output: [batch, heads, seq, head_dim]
    """
    if scale is None:
        scale = 1.0 / (query.shape[-1] ** 0.5)

    # Compute attention scores
    # [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq] -> [batch, heads, seq, seq]
    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Causal mask
    seq_len = query.shape[2]
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
        diagonal=1
    )
    attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Softmax
    attn_probs = F.softmax(attn_scores, dim=-1)

    # Apply attention to values
    # [batch, heads, seq, seq] @ [batch, heads, seq, head_dim] -> [batch, heads, seq, head_dim]
    output = torch.matmul(attn_probs, value)

    return output


class AttentionBaseline:
    """
    Baseline Attention layer with Ulysses-style sequence parallel.

    Uses scheduler-based dW overlap in backward pass.
    Merged autograd functions (2 boundaries) for fair comparison with Overlap.

    Communication pattern:
        Forward:
            1. [_QKVWithSP2HPFunction] QKV projection + sp2hp AllToAll
            2. Attention computation
            3. [_HP2SPWithOutputFunction] hp2sp AllToAll + Output projection

        Backward:
            Same pattern in reverse with dW overlap
    """

    def __init__(self, config, cp_group, device, dtype, layer_id=0):
        self.config = config
        self.cp_group = cp_group
        self.device = device
        self.dtype = dtype
        self.cp_size = cp_group.size()
        self.my_rank = cp_group.rank()
        self.layer_id = layer_id

        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.num_kv_heads = config.get('num_kv_heads', self.num_heads)  # Support GQA
        self.head_dim = config.get('head_dim', self.hidden_size // self.num_heads)
        self.heads_per_rank = self.num_heads // self.cp_size

        # GQA parameters
        self.q_per_group = self.num_heads // self.num_kv_heads
        self.group_size = (self.q_per_group + 2) * self.head_dim  # Q heads + K + V per group
        self.total_proj = self.num_kv_heads * self.group_size

    def init_weights(self, requires_grad=True):
        """
        Initialize weights using Overlap-compatible interleaved layout.

        Weight layout (same as Overlap/Megatron):
            weight_qkv: [total_proj, hidden] - interleaved by KV groups
                Each group: [Q0..Qn, K, V] where n = q_per_group
            weight_proj: [hidden, num_heads * head_dim]

        For MHA (num_heads == num_kv_heads):
            - q_per_group = 1
            - group_size = 3 * head_dim
            - total_proj = num_heads * 3 * head_dim = 3 * hidden
        """
        # QKV projection weight: [total_proj, hidden] interleaved layout
        self.weight_qkv = (torch.randn(
            self.total_proj, self.hidden_size,
            dtype=self.dtype, device=self.device,
        ) * 0.02).requires_grad_(requires_grad)

        # Output projection weight: [hidden, num_heads * head_dim]
        self.weight_proj = (torch.randn(
            self.hidden_size, self.num_heads * self.head_dim,
            dtype=self.dtype, device=self.device,
        ) * 0.02).requires_grad_(requires_grad)

    def forward(self, tokens, do_backward=False):
        """
        Forward pass.

        Args:
            tokens: [seq_local, batch, hidden_size]
            do_backward: Whether to run backward immediately

        Returns:
            output: [seq_local, batch, hidden_size]
        """
        seq_local, batch, hidden = tokens.shape

        # 1. QKV Projection + sp2hp AllToAll (merged)
        q, k, v = _QKVWithSP2HPFunction.apply(
            tokens, self.weight_qkv,
            self.cp_group, self.layer_id,
            self.num_heads, self.num_kv_heads, self.head_dim
        )

        # After sp2hp: [seq_full, batch, heads_local, head_dim]
        seq_full = seq_local * self.cp_size

        # 2. Core Attention
        # Reshape for attention: [batch, heads_local, seq_full, head_dim]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        attn_out = scaled_dot_product_attention(q, k, v)

        # Reshape back: [seq_full, batch, heads_local, head_dim]
        attn_out = attn_out.permute(2, 0, 1, 3)

        # 3. hp2sp AllToAll + Output Projection (merged)
        output = _HP2SPWithOutputFunction.apply(
            attn_out, self.weight_proj,
            self.cp_group, self.layer_id,
            self.num_heads, self.head_dim
        )

        if do_backward:
            loss = output.sum()
            loss.backward()

        return output
