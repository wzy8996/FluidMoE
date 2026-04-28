"""
Megatron-compatible adapter for FluidMoE's TransformerLayer.

Lets Megatron's GPTModel instantiate FluidMoE's TransformerLayer via ModuleSpec,
so embedding / final_layernorm / output_layer / loss / optimizer all stay with
Megatron and only the per-layer block scheduling differs.
"""

import os
from typing import Optional

import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import BaseTransformerLayer

from fluid.layer.transformer import TransformerLayer as _FluidTransformerLayer


def _env_or(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


class FluidTransformerLayer(MegatronModule, BaseTransformerLayer):
    """Megatron-compatible wrapper around FluidMoE's TransformerLayer.

    Signature matches what megatron.core.transformer_block._build_layers passes
    to build_module():  (config, layer_number, pg_collection, vp_stage).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules=None,
        layer_number: int = 1,
        pg_collection=None,
        vp_stage=None,
    ):
        super().__init__(config=config)

        cp_group = pg_collection.cp if pg_collection is not None else None
        ep_group = pg_collection.ep if pg_collection is not None else None
        assert cp_group is not None, "FluidTransformerLayer requires pg_collection.cp"
        assert ep_group is not None, "FluidTransformerLayer requires pg_collection.ep"

        num_kv_heads = getattr(config, 'num_query_groups', None) or config.num_attention_heads

        activation_func = getattr(config, 'activation_func', None)
        if activation_func is None:
            from fluid.core.te_ops import te_gelu
            activation_func = te_gelu

        capacity_factor = getattr(config, 'moe_expert_capacity_factor', None)
        if capacity_factor is None:
            capacity_factor = getattr(config, 'moe_capacity_factor', 1.0) or 1.0

        init_std = getattr(config, 'init_method_std', 0.02)
        num_layers = max(config.num_layers, 1)
        output_init_std = init_std / (2.0 * num_layers) ** 0.5

        dtype = config.params_dtype

        # MoE stability options from Megatron config.
        router_dtype_str = getattr(config, 'moe_router_dtype', None)
        if router_dtype_str == 'fp32':
            router_dtype = torch.float32
        elif router_dtype_str == 'fp64':
            router_dtype = torch.float64
        else:
            router_dtype = dtype
        aux_loss_coeff = getattr(config, 'moe_aux_loss_coeff', 0.0) or 0.0
        z_loss_coeff = getattr(config, 'moe_z_loss_coeff', 0.0) or 0.0

        self.layer = _FluidTransformerLayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            ffn_hidden_size=config.ffn_hidden_size,
            num_experts=config.num_moe_experts,
            top_k=config.moe_router_topk,
            cp_group=cp_group,
            ep_group=ep_group,
            layer_id=layer_number - 1,
            moe_combine_chunks=_env_or('FLUIDMOE_MOE_COMBINE_CHUNKS', 1),
            moe_dispatch_chunks=_env_or('FLUIDMOE_MOE_DISPATCH_CHUNKS', 1),
            attn_proj_chunks=_env_or('FLUIDMOE_ATTN_PROJ_CHUNKS', 1),
            attn_qkv_chunks=_env_or('FLUIDMOE_ATTN_QKV_CHUNKS', 1),
            activation_func=activation_func,
            capacity_factor=capacity_factor,
            init_std=init_std,
            output_init_std=output_init_std,
            dtype=dtype,
            device=torch.cuda.current_device(),
            router_dtype=router_dtype,
            aux_loss_coeff=aux_loss_coeff,
            z_loss_coeff=z_loss_coeff,
        )

        # Flags used by Megatron's TransformerBlock for recompute / offload logic.
        self.is_moe_layer = True
        self.submodules_config = submodules

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        inference_context=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        output = self.layer(hidden_states, rotary_pos_emb=rotary_pos_emb)
        return output, context


def get_fluid_gpt_layer_spec() -> ModuleSpec:
    """ModuleSpec suitable for GPTModel(transformer_layer_spec=...).

    Passed as a single spec (not TransformerBlockSubmodules); TransformerBlock
    replicates it num_layers times.
    """
    return ModuleSpec(module=FluidTransformerLayer)
