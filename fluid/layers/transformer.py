"""
Transformer Layer Implementation

This module implements a complete Transformer layer combining:
- Self-Attention with Ulysses-style Context Parallel
- MoE (Mixture of Experts) with Expert Parallel

Communication pattern per layer (backward):
  1. Attention hp2sp AllToAll (backward of sp2hp) - overlap with dW
  2. Attention sp2hp AllToAll (backward of hp2sp) - overlap with dW
  3. MoE Combine AllToAll - overlap with dW
  4. MoE Dispatch AllToAll - overlap with dW

The scheduler manages all dW tasks and executes them during AllToAll communication
to hide computation under communication.
"""

import torch
import torch.nn as nn
from typing import Optional

from fluid.moe import MoEBaseline, compute_routing
from fluid.attention import AttentionBaseline
from fluid.core.scheduler import get_backward_scheduler


class TransformerLayer:
    """
    Complete Transformer layer: Attention + MoE.

    This is a simplified standalone implementation for testing/benchmarking
    the communication-computation overlap strategy.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + MoE(LayerNorm(x))

    Communication:
        - Attention uses Context Parallel (CP) with Ulysses-style AllToAll
        - MoE uses Expert Parallel (EP) with dispatch/combine AllToAll

    Backward optimization:
        - All dW tasks are registered with the scheduler
        - dW tasks execute during AllToAll communication
        - dX computation is chunked and pipelined with AllToAll
    """

    def __init__(
        self,
        config,
        ep_group,
        cp_group,
        device,
        dtype,
        layer_id=0,
    ):
        """
        Initialize TransformerLayer.

        Args:
            config: Configuration dict with:
                - hidden_size: Model hidden dimension
                - ffn_hidden_size: FFN hidden dimension
                - num_experts: Number of experts
                - top_k: Number of experts to route to
                - num_heads: Number of attention heads
                - head_dim: Dimension per attention head (optional)
            ep_group: Expert parallel process group
            cp_group: Context parallel process group
            device: torch device
            dtype: torch dtype
            layer_id: Layer index for identification
        """
        self.config = config
        self.ep_group = ep_group
        self.cp_group = cp_group
        self.device = device
        self.dtype = dtype
        self.layer_id = layer_id

        self.hidden_size = config['hidden_size']

        # Initialize sub-layers
        self.attention = AttentionBaseline(config, cp_group, device, dtype, layer_id)
        self.moe = MoEBaseline(config, ep_group, device, dtype, layer_id)

        # Layer norms (simple implementation)
        self.ln1_weight = None
        self.ln1_bias = None
        self.ln2_weight = None
        self.ln2_bias = None

    def init_weights(self, requires_grad=True):
        """Initialize all weights."""
        # Attention weights
        self.attention.init_weights(requires_grad)

        # MoE weights
        self.moe.init_weights(requires_grad)

        # LayerNorm weights
        self.ln1_weight = torch.ones(
            self.hidden_size, dtype=self.dtype, device=self.device,
            requires_grad=requires_grad
        )
        self.ln1_bias = torch.zeros(
            self.hidden_size, dtype=self.dtype, device=self.device,
            requires_grad=requires_grad
        )
        self.ln2_weight = torch.ones(
            self.hidden_size, dtype=self.dtype, device=self.device,
            requires_grad=requires_grad
        )
        self.ln2_bias = torch.zeros(
            self.hidden_size, dtype=self.dtype, device=self.device,
            requires_grad=requires_grad
        )

    def layer_norm(self, x, weight, bias, eps=1e-5):
        """Simple layer normalization."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + eps)
        return x_norm * weight + bias

    def forward(self, tokens, do_backward=False):
        """
        Forward pass.

        Args:
            tokens: [seq_local, batch, hidden_size]
            do_backward: Whether to run backward immediately

        Returns:
            output: [seq_local, batch, hidden_size]

        Note:
            seq_local = full_seq / cp_size (Context Parallel)
            Each rank processes different sequence portions for attention,
            but same tokens (after routing) for MoE.
        """
        # Pre-attention LayerNorm
        normed = self.layer_norm(tokens, self.ln1_weight, self.ln1_bias)

        # Self-Attention with residual
        attn_out = self.attention.forward(normed)
        hidden = tokens + attn_out

        # Pre-MoE LayerNorm
        normed = self.layer_norm(hidden, self.ln2_weight, self.ln2_bias)

        # MoE with residual
        # Note: MoE expects [num_tokens, hidden], but attention outputs [seq, batch, hidden]
        # For simplicity, we reshape here
        seq_local, batch, hidden = normed.shape
        normed_flat = normed.view(seq_local * batch, hidden)

        moe_out = self.moe.forward(normed_flat)
        moe_out = moe_out.view(seq_local, batch, hidden)

        output = hidden + moe_out

        if do_backward:
            loss = output.sum()
            loss.backward()

        return output

    def get_all_weights(self):
        """Get all trainable weights for optimizer."""
        weights = []

        # Attention weights
        weights.append(self.attention.weight_qkv)
        weights.append(self.attention.weight_proj)

        # MoE weights
        weights.append(self.moe.router_weight)
        weights.append(self.moe.weight1)
        weights.append(self.moe.weight2)

        # LayerNorm weights
        weights.extend([self.ln1_weight, self.ln1_bias, self.ln2_weight, self.ln2_bias])

        return weights


class TransformerModel:
    """
    Multi-layer Transformer model.

    Stacks multiple TransformerLayers and manages the backward scheduler
    for optimal communication-computation overlap.
    """

    def __init__(
        self,
        config,
        num_layers,
        ep_group,
        cp_group,
        device,
        dtype,
    ):
        """
        Initialize TransformerModel.

        Args:
            config: Configuration dict
            num_layers: Number of transformer layers
            ep_group: Expert parallel process group
            cp_group: Context parallel process group
            device: torch device
            dtype: torch dtype
        """
        self.config = config
        self.num_layers = num_layers
        self.ep_group = ep_group
        self.cp_group = cp_group
        self.device = device
        self.dtype = dtype

        # Create layers
        self.layers = []
        for layer_id in range(num_layers):
            layer = TransformerLayer(
                config, ep_group, cp_group, device, dtype, layer_id
            )
            self.layers.append(layer)

    def init_weights(self, requires_grad=True):
        """Initialize all layer weights."""
        for layer in self.layers:
            layer.init_weights(requires_grad)

    def forward(self, tokens, do_backward=False):
        """
        Forward pass through all layers.

        Args:
            tokens: [seq_local, batch, hidden_size]
            do_backward: Whether to run backward after forward

        Returns:
            output: [seq_local, batch, hidden_size]
        """
        hidden = tokens

        for layer in self.layers:
            hidden = layer.forward(hidden, do_backward=False)

        if do_backward:
            # Enable scheduler for backward
            scheduler = get_backward_scheduler()
            scheduler.enable()

            loss = hidden.sum()
            loss.backward()

            # Flush remaining dW tasks
            scheduler.flush_dw_tasks()
            scheduler.disable()

        return hidden

    def get_all_weights(self):
        """Get all trainable weights for optimizer."""
        weights = []
        for layer in self.layers:
            weights.extend(layer.get_all_weights())
        return weights
