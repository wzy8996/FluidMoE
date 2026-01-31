"""
FluidMoE GPT Model for Megatron Integration

Provides FluidMoEGPTModel that wraps FluidMoE's TransformerModel with
embedding and output layers, compatible with Megatron's pretrain() loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig

from fluid.layer import TransformerModel


class _CastToFloat32(torch.autograd.Function):
    """Cast tensor to float32 in forward, cast grad back to input dtype in backward."""
    @staticmethod
    def forward(ctx, x):
        ctx.input_dtype = x.dtype
        return x.float()

    @staticmethod
    def backward(ctx, grad):
        return grad.to(ctx.input_dtype)


class FluidMoEGPTModel(nn.Module):
    """GPT model using FluidMoE transformer layers.

    Architecture:
        - Word embedding + position embedding (simple, no TP)
        - FluidMoE TransformerModel (CP + EP)
        - Final LayerNorm + output projection

    Compatible with Megatron's pretrain() forward_step interface:
        model(input_ids, position_ids, attention_mask, labels=labels)
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = vocab_size
        self.pre_process = pre_process
        self.post_process = post_process

        # Embedding (only on first pipeline stage)
        if self.pre_process:
            self.word_embeddings = nn.Embedding(vocab_size, config.hidden_size)
            self.position_embeddings = nn.Embedding(max_sequence_length, config.hidden_size)
            self.embedding_dropout = nn.Dropout(config.hidden_dropout)

        # FluidMoE Transformer layers
        cp_group = parallel_state.get_context_parallel_group()
        ep_group = parallel_state.get_expert_model_parallel_group()

        num_kv_heads = config.num_attention_heads
        if hasattr(config, 'num_query_groups') and config.num_query_groups is not None:
            num_kv_heads = config.num_query_groups

        self.decoder = TransformerModel(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            ffn_hidden_size=config.ffn_hidden_size,
            num_experts=config.num_moe_experts,
            top_k=config.moe_router_topk,
            cp_group=cp_group,
            ep_group=ep_group,
            moe_combine_chunks=2,
            moe_dispatch_chunks=2,
            attn_proj_chunks=2,
            attn_qkv_chunks=2,
            dtype=config.params_dtype if hasattr(config, 'params_dtype') else torch.bfloat16,
        )

        # Final layer norm + output projection (only on last pipeline stage)
        if self.post_process:
            self.final_layernorm = nn.LayerNorm(config.hidden_size, dtype=config.params_dtype if hasattr(config, 'params_dtype') else torch.bfloat16)
            self.output_layer = nn.Linear(config.hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass compatible with Megatron's forward_step.

        Args:
            input_ids: [batch, seq] token ids
            position_ids: [batch, seq] position ids
            attention_mask: [1, 1, seq, seq] attention mask (ignored, using causal)
            labels: [batch, seq] target token ids for loss computation
            loss_mask: [batch, seq] mask for loss computation

        Returns:
            If labels provided: loss tensor [batch, seq] (per-token losses)
            If no labels: logits [batch, seq, vocab]
        """
        # 1. Embedding: [batch, seq] -> [seq, batch, hidden]
        if self.pre_process:
            hidden_states = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
            hidden_states = self.embedding_dropout(hidden_states)
            # [batch, seq, hidden] -> [seq, batch, hidden]
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            hidden_states = input_ids  # PP intermediate: already [seq, batch, hidden]

        # 2. Transformer layers: [seq, batch, hidden] -> [seq, batch, hidden]
        hidden_states = self.decoder(hidden_states)

        # 3. Output + loss
        if not self.post_process:
            return hidden_states

        hidden_states = self.final_layernorm(hidden_states)
        logits = self.output_layer(hidden_states)  # [seq, batch, vocab]

        if labels is None:
            # [seq, batch, vocab] -> [batch, seq, vocab]
            return logits.transpose(0, 1).contiguous()

        # Compute per-token cross-entropy loss
        # logits: [seq, batch, vocab] -> [batch*seq, vocab]
        # labels: [batch, seq] -> [batch*seq]
        logits_2d = logits.transpose(0, 1).contiguous().view(-1, self.vocab_size)
        # Cast to float32 for cross_entropy precision; grad flows back in original dtype
        logits_2d = _CastToFloat32.apply(logits_2d)
        labels_1d = labels.view(-1)
        loss = F.cross_entropy(logits_2d, labels_1d, reduction='none')
        return loss
