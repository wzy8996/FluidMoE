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
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = vocab_size
        self.pre_process = pre_process
        self.post_process = post_process
        # Required by Megatron's get_attr_wrapped_model()
        from megatron.core.enums import ModelType
        self.model_type = ModelType.encoder_or_decoder

        # Megatron-aligned init: use init_method_std (default 0.02) with
        # Megatron's RNG tracker so that identical seeds produce identical weights.
        init_std = getattr(config, 'init_method_std', 0.02)
        _dtype = config.params_dtype if hasattr(config, 'params_dtype') else torch.bfloat16
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

        if self.pre_process:
            # Word embedding: Megatron's VocabParallelEmbedding uses
            # torch.empty() + fork(model-parallel-rng) + init_method
            self.word_embeddings = nn.Embedding(vocab_size, config.hidden_size, device='meta')
            self.word_embeddings.weight = nn.Parameter(
                torch.empty(vocab_size, config.hidden_size, device='cuda', dtype=_dtype))
            with get_cuda_rng_tracker().fork():
                nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=init_std)

            # Position embedding: Megatron creates nn.Embedding on CPU (default init
            # consumes CPU RNG only), then reinits with init_method (CPU RNG).
            # Neither touches CUDA RNG. We replicate this pattern.
            self.position_embeddings = nn.Embedding(max_sequence_length, config.hidden_size)
            nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_std)
            self.position_embeddings = self.position_embeddings.cuda().to(_dtype)

            self.embedding_dropout = nn.Dropout(config.hidden_dropout)

        cp_group = parallel_state.get_context_parallel_group()
        ep_group = parallel_state.get_expert_model_parallel_group()

        num_kv_heads = config.num_attention_heads
        if hasattr(config, 'num_query_groups') and config.num_query_groups is not None:
            num_kv_heads = config.num_query_groups

        capacity_factor = getattr(config, 'moe_capacity_factor', 1.0)

        # Chunk parameters: from env (set by run_training.sh from experiment_configs.py)
        # or constructor kwargs, with sensible defaults.
        import os as _os
        _env_or = lambda key, default: int(_os.environ.get(key, str(default)))

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
            moe_combine_chunks=kwargs.get('moe_combine_chunks',
                _env_or('FLUIDMOE_MOE_COMBINE_CHUNKS', 4)),
            moe_dispatch_chunks=kwargs.get('moe_dispatch_chunks',
                _env_or('FLUIDMOE_MOE_DISPATCH_CHUNKS', 4)),
            attn_proj_chunks=kwargs.get('attn_proj_chunks',
                _env_or('FLUIDMOE_ATTN_PROJ_CHUNKS', 2)),
            attn_qkv_chunks=kwargs.get('attn_qkv_chunks',
                _env_or('FLUIDMOE_ATTN_QKV_CHUNKS', 4)),
            capacity_factor=capacity_factor,
            init_std=init_std,
            dtype=config.params_dtype if hasattr(config, 'params_dtype') else torch.bfloat16,
        )

        if self.post_process:
            _ln_dtype = config.params_dtype if hasattr(config, 'params_dtype') else torch.bfloat16
            if _HAS_TE:
                self.final_layernorm = te.LayerNorm(config.hidden_size, params_dtype=_ln_dtype)
            else:
                self.final_layernorm = nn.LayerNorm(config.hidden_size, dtype=_ln_dtype)
            # Output layer: Megatron's ColumnParallelLinear uses
            # torch.empty() + fork(model-parallel-rng) + init_method
            self.output_layer = nn.Linear(config.hidden_size, vocab_size, bias=False, device='meta')
            out_w = torch.empty(vocab_size, config.hidden_size, device='cuda', dtype=_dtype)
            with get_cuda_rng_tracker().fork():
                nn.init.normal_(out_w, mean=0.0, std=init_std)
            self.output_layer.weight = nn.Parameter(out_w)

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

    def shared_embedding_or_output_weight(self):
        """Required by Megatron's pretrain() for weight tying."""
        if self.pre_process:
            return self.word_embeddings.weight
        if self.post_process:
            return self.output_layer.weight
        return None

    def copy_weights_from_megatron(self, megatron_model):
        """Copy weights from a Megatron GPTModel to this FluidMoEModel.

        Maps Megatron's parameter layout to FluidMoE's:
          - word/position embeddings: direct copy
          - output_layer: direct copy
          - final_layernorm: direct copy
          - per-layer QKV/proj: direct copy (same TE Linear format)
          - per-layer LN: direct copy
          - per-layer router: transpose ([E, H] → [H, E])
          - per-layer MoE w1: stack per-expert weights + transpose
          - per-layer MoE w2: stack per-expert weights + transpose
        """
        with torch.no_grad():
            meg = megatron_model
            # Embeddings
            if self.pre_process:
                self.word_embeddings.weight.copy_(meg.embedding.word_embeddings.weight)
                self.position_embeddings.weight.copy_(meg.embedding.position_embeddings.weight)
            # Final LN + output
            if self.post_process:
                self.final_layernorm.weight.copy_(meg.decoder.final_layernorm.weight)
                if hasattr(meg.decoder.final_layernorm, 'bias') and meg.decoder.final_layernorm.bias is not None:
                    self.final_layernorm.bias.copy_(meg.decoder.final_layernorm.bias)
                self.output_layer.weight.copy_(meg.output_layer.weight)
            # Per-layer
            for i, fluid_layer in enumerate(self.decoder.layers):
                meg_layer = meg.decoder.layers[i]
                # LN1 (fused in QKV linear for Megatron)
                fluid_layer.ln1_weight.copy_(meg_layer.self_attention.linear_qkv.layer_norm_weight)
                fluid_layer.ln1_bias.copy_(meg_layer.self_attention.linear_qkv.layer_norm_bias)
                # LN2
                fluid_layer.ln2_weight.copy_(meg_layer.pre_mlp_layernorm.weight)
                fluid_layer.ln2_bias.copy_(meg_layer.pre_mlp_layernorm.bias)
                # QKV
                fluid_layer._get_qkv_weight().copy_(meg_layer.self_attention.linear_qkv.weight)
                # Proj
                fluid_layer._get_proj_weight().copy_(meg_layer.self_attention.linear_proj.weight)
                # Router: Megatron [E, H] → FluidMoE [H, E]
                fluid_layer.router_weight.copy_(meg_layer.mlp.router.weight.t())
                # MoE w1: Megatron per-expert [FFN, H] → FluidMoE [E, H, FFN]
                num_local = fluid_layer.moe_w1.shape[0]
                for e in range(num_local):
                    # Megatron uses LOCAL indices (weight0, weight1, ...) on each EP rank
                    w_key = f'weight{e}'
                    meg_w1 = getattr(meg_layer.mlp.experts.linear_fc1, w_key)
                    meg_w2 = getattr(meg_layer.mlp.experts.linear_fc2, w_key)
                    # Megatron fc1: [FFN, H], FluidMoE w1: [H, FFN]
                    fluid_layer.moe_w1.data[e].copy_(meg_w1.t())
                    # Megatron fc2: [H, FFN], FluidMoE w2: [FFN, H]
                    fluid_layer.moe_w2.data[e].copy_(meg_w2.t())

    def set_input_tensor(self, input_tensor):
        """Required by Megatron's pipeline parallel (no-op for non-PP)."""
        pass

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Required by Megatron's checkpointing. Returns plain state_dict
        (no sharding metadata) since FluidMoE manages its own parameter layout."""
        return self.state_dict(prefix=prefix)

__all__ = ["FluidMoEModel"]
