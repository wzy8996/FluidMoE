"""
Attention Baseline Implementation

This module implements a simplified baseline attention layer with Ulysses-style
sequence parallel AllToAll. The backward pass uses the scheduler for dW overlap.

Key features:
- Ulysses SP: sp2hp AllToAll before attention, hp2sp AllToAll after
- dW tasks registered for overlap during backward
- Compatible with scheduler-based backward optimization

Note: This is a standalone implementation for testing/benchmarking.
For production use with Megatron-LM, see attention_module.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from fluid.core import _all_to_all_sp2hp_forward, _all_to_all_hp2sp_forward
from fluid.core.scheduler import get_backward_scheduler


class _QKVProjectionFunction(torch.autograd.Function):
    """
    QKV projection autograd function with dW scheduling.

    Forward: x @ weight (where weight may be transposed from original)
    Backward: dX = grad @ weight.T, register dW for overlap

    Supports two weight layouts:
    - Standard: weight [hidden, out_dim], orig_weight same
    - Interleaved: weight [hidden, out_dim] (transposed), orig_weight [out_dim, hidden]
    """

    @staticmethod
    def forward(ctx, input, weight, layer_name="attn_qkv", layer_id=0, orig_weight=None):
        # [seq, batch, hidden] @ [hidden, out_dim] -> [seq, batch, out_dim]
        output = torch.matmul(input, weight)

        ctx.save_for_backward(input, weight)
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id
        # Use orig_weight if provided (for transposed case), otherwise use weight
        ctx._orig_weight = orig_weight if orig_weight is not None else weight
        ctx._transposed = orig_weight is not None

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_saved, weight = ctx.saved_tensors
        layer_name = ctx.layer_name
        layer_id = ctx.layer_id
        orig_weight = ctx._orig_weight
        transposed = ctx._transposed

        scheduler = get_backward_scheduler()

        # CRITICAL PATH: Compute dX immediately
        grad_input = torch.matmul(grad_output, weight.t())

        # LAZY REGISTRATION: Register dW for overlap
        grad_output_saved = grad_output.detach()
        input_for_dw = input_saved.detach()

        def compute_dw():
            # dW = input.T @ grad_output
            # [hidden, seq*batch] @ [seq*batch, out_dim] -> [hidden, out_dim]
            inp_flat = input_for_dw.reshape(-1, input_for_dw.shape[-1])
            grad_flat = grad_output_saved.reshape(-1, grad_output_saved.shape[-1])
            grad_w = torch.matmul(inp_flat.t(), grad_flat)
            if transposed:
                # Original weight was [out_dim, hidden], grad_w is [hidden, out_dim]
                return grad_w.t()
            return grad_w

        scheduler.register_dw_task(
            layer_name=layer_name,
            layer_id=layer_id,
            compute_fn=compute_dw,
            priority=10,  # QKV weight priority
            weight_param=orig_weight,
        )

        return grad_input, None, None, None, None


class _OutputProjectionFunction(torch.autograd.Function):
    """
    Output projection autograd function with dW scheduling.

    Forward: x @ weight_proj
    Backward: dX = grad @ weight.T, register dW for overlap
    """

    @staticmethod
    def forward(ctx, input, weight, layer_name="attn_proj", layer_id=0, orig_weight=None):
        # [seq, batch, hidden] @ [hidden, hidden] -> [seq, batch, hidden]
        output = torch.matmul(input, weight)

        ctx.save_for_backward(input, weight)
        ctx.layer_name = layer_name
        ctx.layer_id = layer_id
        # Use orig_weight if provided (for detached case), otherwise use weight
        ctx._orig_weight = orig_weight if orig_weight is not None else weight

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_saved, weight = ctx.saved_tensors
        layer_name = ctx.layer_name
        layer_id = ctx.layer_id
        orig_weight = ctx._orig_weight

        scheduler = get_backward_scheduler()

        # CRITICAL PATH: Compute dX immediately
        grad_input = torch.matmul(grad_output, weight.t())

        # LAZY REGISTRATION: Register dW for overlap
        grad_output_saved = grad_output.detach()
        input_for_dw = input_saved.detach()

        def compute_dw():
            # dW = input.T @ grad_output
            inp_flat = input_for_dw.reshape(-1, input_for_dw.shape[-1])
            grad_flat = grad_output_saved.reshape(-1, grad_output_saved.shape[-1])
            return torch.matmul(inp_flat.t(), grad_flat)

        scheduler.register_dw_task(
            layer_name=layer_name,
            layer_id=layer_id,
            compute_fn=compute_dw,
            priority=5,  # Output projection priority (lower = earlier)
            weight_param=orig_weight,
        )

        return grad_input, None, None, None, None


class _SP2HPFunction(torch.autograd.Function):
    """AllToAll sp2hp forward / hp2sp backward."""

    @staticmethod
    def forward(ctx, input, cp_group, layer_id=0):
        ctx.cp_group = cp_group
        ctx.layer_id = layer_id
        return _all_to_all_sp2hp_forward(input, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        scheduler = get_backward_scheduler()
        cp_group = ctx.cp_group
        layer_id = ctx.layer_id

        if scheduler.is_enabled():
            comm_stream = scheduler.comm_stream
            default_stream = torch.cuda.current_stream()

            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_input = _all_to_all_hp2sp_forward(grad_output, cp_group)
                event = torch.cuda.Event()
                event.record(comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type=f"attn_hp2sp_L{layer_id}")
            default_stream.wait_stream(comm_stream)
        else:
            grad_input = _all_to_all_hp2sp_forward(grad_output, cp_group)

        return grad_input, None, None


class _HP2SPFunction(torch.autograd.Function):
    """AllToAll hp2sp forward / sp2hp backward."""

    @staticmethod
    def forward(ctx, input, cp_group, layer_id=0):
        ctx.cp_group = cp_group
        ctx.layer_id = layer_id
        return _all_to_all_hp2sp_forward(input, cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        scheduler = get_backward_scheduler()
        cp_group = ctx.cp_group
        layer_id = ctx.layer_id

        if scheduler.is_enabled():
            comm_stream = scheduler.comm_stream
            default_stream = torch.cuda.current_stream()

            with torch.cuda.stream(comm_stream):
                comm_stream.wait_stream(default_stream)
                grad_input = _all_to_all_sp2hp_forward(grad_output, cp_group)
                event = torch.cuda.Event()
                event.record(comm_stream)
                scheduler.set_alltoall_end_event(event)

            scheduler.on_alltoall_start(comm_type=f"attn_sp2hp_L{layer_id}")
            default_stream.wait_stream(comm_stream)
        else:
            grad_input = _all_to_all_sp2hp_forward(grad_output, cp_group)

        return grad_input, None, None


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

    Communication pattern:
        Forward:
            1. QKV projection
            2. sp2hp AllToAll: [seq/CP, B, heads, dim] -> [seq, B, heads/CP, dim]
            3. Attention computation
            4. hp2sp AllToAll: [seq, B, heads/CP, dim] -> [seq/CP, B, heads, dim]
            5. Output projection

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
        # NOTE: Create tensor first, then multiply by scale, then set requires_grad
        # to ensure the weight is a leaf tensor (non-leaf tensors cause graph retention issues)
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

        # 1. QKV Projection with interleaved weight layout
        # weight_qkv: [total_proj, hidden], input: [seq, batch, hidden]
        # output: [seq, batch, total_proj]
        # Detach + contiguous to avoid retaining computation graph across iterations
        # (we compute dW manually in backward, so no need for autograd to track weight)
        qkv = _QKVProjectionFunction.apply(
            tokens, self.weight_qkv.t().detach().contiguous(),  # Transpose for matmul: [hidden, total_proj]
            f"attn_qkv_L{self.layer_id}", self.layer_id,
            self.weight_qkv  # Pass original weight for gradient
        )

        # Extract Q, K, V from interleaved layout
        # qkv: [seq, batch, total_proj] = [seq, batch, num_kv_heads * group_size]
        # Each group: [Q0..Qn, K, V] where group_size = (q_per_group + 2) * head_dim
        qkv = qkv.view(seq_local, batch, self.num_kv_heads, self.group_size)

        q_dim = self.q_per_group * self.head_dim
        k_dim = self.head_dim
        v_dim = self.head_dim

        # Split each group into Q, K, V
        q = qkv[:, :, :, :q_dim]  # [seq, batch, num_kv_heads, q_per_group * head_dim]
        k = qkv[:, :, :, q_dim:q_dim + k_dim]  # [seq, batch, num_kv_heads, head_dim]
        v = qkv[:, :, :, q_dim + k_dim:]  # [seq, batch, num_kv_heads, head_dim]

        # Reshape Q to [seq, batch, num_heads, head_dim]
        q = q.view(seq_local, batch, self.num_heads, self.head_dim)
        # For GQA, K/V need to be expanded; for MHA (q_per_group=1), they're already correct
        if self.q_per_group > 1:
            # Expand K/V for GQA: [seq, batch, num_kv_heads, head_dim] -> [seq, batch, num_heads, head_dim]
            k = k.repeat_interleave(self.q_per_group, dim=2)
            v = v.repeat_interleave(self.q_per_group, dim=2)
        else:
            k = k.view(seq_local, batch, self.num_heads, self.head_dim)
            v = v.view(seq_local, batch, self.num_heads, self.head_dim)

        # 2. sp2hp AllToAll
        if self.cp_size > 1:
            q = _SP2HPFunction.apply(q, self.cp_group, self.layer_id)
            k = _SP2HPFunction.apply(k, self.cp_group, self.layer_id)
            v = _SP2HPFunction.apply(v, self.cp_group, self.layer_id)

        # After sp2hp: [seq_full, batch, heads_local, head_dim]
        seq_full = seq_local * self.cp_size

        # 3. Core Attention
        # Reshape for attention: [batch, heads_local, seq_full, head_dim]
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        attn_out = scaled_dot_product_attention(q, k, v)

        # Reshape back: [seq_full, batch, heads_local, head_dim]
        attn_out = attn_out.permute(2, 0, 1, 3)

        # 4. hp2sp AllToAll
        if self.cp_size > 1:
            attn_out = _HP2SPFunction.apply(attn_out, self.cp_group, self.layer_id)

        # After hp2sp: [seq_local, batch, heads, head_dim]
        # Reshape to [seq_local, batch, hidden]
        attn_out = attn_out.reshape(seq_local, batch, self.hidden_size)

        # 5. Output Projection
        # Detach weight to avoid retaining computation graph across iterations
        output = _OutputProjectionFunction.apply(
            attn_out, self.weight_proj.detach(),
            f"attn_proj_L{self.layer_id}", self.layer_id,
            self.weight_proj  # Pass original weight for gradient
        )

        if do_backward:
            loss = output.sum()
            loss.backward()

        return output
