"""
FluidMoE model wrapper for Megatron integration.

Provides `FluidMoEModel`, which wraps FluidMoE's TransformerModel
with embedding and output layers, compatible with Megatron-style pretrain loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine.pytorch as te
    _HAS_TE = True
except ImportError:
    _HAS_TE = False

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


class FluidMoEModel(nn.Module):
    """Decoder-only language model using FluidMoE transformer layers.

    Architecture:
        - Word embedding + position embedding (simple, no TP)
        - FluidMoE TransformerModel (CP + EP)
        - Final LayerNorm + output projection

    Compatible with Megatron-style pretrain forward_step interface:
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

        if self.pre_process:
            self.word_embeddings = nn.Embedding(vocab_size, config.hidden_size)
            self.position_embeddings = nn.Embedding(max_sequence_length, config.hidden_size)
            self.embedding_dropout = nn.Dropout(config.hidden_dropout)

        cp_group = parallel_state.get_context_parallel_group()
        ep_group = parallel_state.get_expert_model_parallel_group()

        num_kv_heads = config.num_attention_heads
        if hasattr(config, 'num_query_groups') and config.num_query_groups is not None:
            num_kv_heads = config.num_query_groups

        capacity_factor = getattr(config, 'moe_capacity_factor', 1.0)

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
            capacity_factor=capacity_factor,
            dtype=config.params_dtype if hasattr(config, 'params_dtype') else torch.bfloat16,
        )

        if self.post_process:
            _ln_dtype = config.params_dtype if hasattr(config, 'params_dtype') else torch.bfloat16
            if _HAS_TE:
                self.final_layernorm = te.LayerNorm(config.hidden_size, params_dtype=_ln_dtype)
            else:
                self.final_layernorm = nn.LayerNorm(config.hidden_size, dtype=_ln_dtype)
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
        if self.pre_process:
            hidden_states = self.word_embeddings(input_ids) + self.position_embeddings(position_ids)
            hidden_states = self.embedding_dropout(hidden_states)
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            hidden_states = input_ids

        hidden_states = self.decoder(hidden_states)

        if not self.post_process:
            return hidden_states

        hidden_states = self.final_layernorm(hidden_states)
        logits = self.output_layer(hidden_states)

        if labels is None:
            return logits.transpose(0, 1).contiguous()

        logits_2d = logits.transpose(0, 1).contiguous().view(-1, self.vocab_size)
        logits_2d = _CastToFloat32.apply(logits_2d)
        labels_1d = labels.view(-1)
        loss = F.cross_entropy(logits_2d, labels_1d, reduction='none')
        return loss

__all__ = ["FluidMoEModel"]
