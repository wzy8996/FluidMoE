"""
Megatron-Core baseline transformer for benchmark comparison.

This wrapper keeps benchmark I/O shape as [seq_local, batch, hidden] and uses
Megatron-Core TransformerLayer (TE spec) + MoE alltoall dispatcher.
Both Megatron and FluidMoE use TransformerEngine compute primitives
(TE LayerNorm, TE FMHA, CUTLASS GroupGEMM), isolating the scheduling diff.
"""

import os
import sys
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


def _ensure_megatron_importable() -> None:
    """Make local Megatron-LM source importable if it's not installed in site-packages."""
    try:
        import megatron  # noqa: F401

        return
    except ImportError:
        pass

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_parent = os.path.dirname(root_dir)
    megatron_repo = os.path.join(repo_parent, "Megatron-LM")
    if os.path.isdir(megatron_repo):
        sys.path.insert(0, megatron_repo)
    else:
        raise ImportError(
            "Cannot import megatron and local Megatron-LM repo not found at "
            f"{megatron_repo}"
        )


_ensure_megatron_importable()

from megatron.core.models.gpt.gpt_layer_specs import (  # noqa: E402
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.transformer.spec_utils import build_module  # noqa: E402
from megatron.core.transformer.transformer_config import TransformerConfig  # noqa: E402


def _build_singleton_group(rank: int, world_size: int):
    """Create a per-rank singleton process group (all ranks must participate in creation)."""
    single_group = None
    for r in range(world_size):
        g = dist.new_group(ranks=[r])
        if r == rank:
            single_group = g
    return single_group


class MegatronBaselineTransformerModel(nn.Module):
    """
    Baseline model backed by Megatron-Core transformer layers.

    Notes:
    - Uses TP=1 in this benchmark wrapper.
    - Uses provided CP/EP groups to match current benchmark parallel setting.
    """

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
        *,
        shared_dp_group: Optional[dist.ProcessGroup] = None,
        expert_dp_group: Optional[dist.ProcessGroup] = None,
        capacity_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device(f"cuda:{dist.get_rank()}")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        tp_group = _build_singleton_group(rank, world_size)

        # Minimal process-group set required by Megatron attention + MoE modules.
        pg_collection = ProcessGroupCollection(
            tp=tp_group,
            pp=None,
            cp=cp_group,
            ep=ep_group,
            expt_tp=tp_group,
            expt_dp=expert_dp_group if expert_dp_group is not None else tp_group,
            tp_ep=ep_group,
            tp_cp=cp_group,
            tp_dp_cp=shared_dp_group if shared_dp_group is not None else cp_group,
        )

        moe_capacity = capacity_factor if capacity_factor > 0 else None
        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_query_groups=num_kv_heads,
            ffn_hidden_size=ffn_hidden_size,
            moe_ffn_hidden_size=ffn_hidden_size,
            num_moe_experts=num_experts,
            moe_router_topk=top_k,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="none",
            moe_aux_loss_coeff=0.0,
            moe_expert_capacity_factor=moe_capacity,
            moe_pad_expert_input_to_capacity=False,
            moe_grouped_gemm=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_group.size(),
            expert_model_parallel_size=ep_group.size(),
            transformer_impl="transformer_engine",
            hidden_dropout=0.0,
            attention_dropout=0.0,
            cp_comm_type="a2a",
            bf16=(dtype == torch.bfloat16),
            fp16=(dtype == torch.float16),
            params_dtype=dtype,
            pipeline_dtype=dtype,
        )

        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=num_experts,
            moe_grouped_gemm=True,
        )

        self.layers = nn.ModuleList(
            [
                build_module(
                    layer_spec,
                    config=config,
                    layer_number=i + 1,
                    pg_collection=pg_collection,
                )
                for i in range(num_layers)
            ]
        )
        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in self.layers:
            hidden, _ = layer(hidden, attention_mask=None)
        return hidden
