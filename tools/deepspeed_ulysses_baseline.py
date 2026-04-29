"""DeepSpeed block-level baseline backed by Ulysses attention + official DeepSpeed MoE.

This module keeps the same block-benchmark contract as FluidMoE/Megatron:

- input/output shape: [seq_local, batch, hidden]
- context parallel attention via DeepSpeed DistributedAttention (Ulysses-style)
- expert path implemented with the official ``deepspeed.moe.layer.MoE``
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from fluid.attention.forward import scaled_dot_product_attention_forward


def _require_deepspeed_block_deps():
    try:
        import deepspeed
        import deepspeed.comm as ds_comm
        from deepspeed.moe.layer import MoE
        from deepspeed.sequence.layer import DistributedAttention
    except ImportError as exc:
        raise ImportError(
            "DeepSpeed block baseline requires `deepspeed`, "
            "`deepspeed.sequence.layer.DistributedAttention`, and "
            "`deepspeed.moe.layer.MoE`."
        ) from exc

    if dist.is_initialized():
        is_init = False
        if hasattr(ds_comm, "is_initialized"):
            try:
                is_init = bool(ds_comm.is_initialized())
            except Exception:
                is_init = False
        if not is_init:
            deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

    return {"DistributedAttention": DistributedAttention, "MoE": MoE}


class _UlyssesLocalAttention(nn.Module):
    """Local attention kernel used inside DeepSpeed DistributedAttention.

    Expects batch-first input: [batch, seq, heads, head_dim].
    """

    def __init__(self, scale: float):
        super().__init__()
        self.scale = float(scale)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args, **kwargs):
        del args, kwargs
        q_bf = query.permute(0, 2, 1, 3)
        k_bf = key.permute(0, 2, 1, 3)
        v_bf = value.permute(0, 2, 1, 3)
        enable_gqa = q_bf.shape[1] != k_bf.shape[1]
        out_bf = scaled_dot_product_attention_forward(
            q_bf,
            k_bf,
            v_bf,
            scale=self.scale,
            is_causal=True,
            enable_gqa=enable_gqa,
        )
        return out_bf.permute(0, 2, 1, 3).contiguous()


class _DeepSpeedExpertMLP(nn.Module):
    """Per-expert MLP used by the official DeepSpeed MoE layer."""

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        activation_func: Callable,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False, dtype=dtype, device=device)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False, dtype=dtype, device=device)
        self.activation_func = activation_func
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation_func(self.fc1(x)))


class DeepSpeedBlockBaselineLayer(nn.Module):
    """Transformer block with DeepSpeed Ulysses attention and official DeepSpeed MoE."""

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
        activation_func: Optional[Callable] = None,
        capacity_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        wire_ep_group: bool = True,
    ):
        super().__init__()

        deps = _require_deepspeed_block_deps()
        distributed_attention_cls = deps["DistributedAttention"]
        ds_moe_cls = deps["MoE"]

        if device is None:
            device = torch.device(f"cuda:{dist.get_rank()}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.cp_group = cp_group
        self.ep_group = ep_group

        activation_func = activation_func or F.gelu
        head_dim = hidden_size // num_heads
        q_per_kv = num_heads // num_kv_heads
        qkv_size = num_kv_heads * (q_per_kv + 2) * head_dim

        self.ln1_weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.ln1_bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))
        self.ln2_weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.ln2_bias = nn.Parameter(torch.zeros(hidden_size, dtype=dtype, device=device))

        self.qkv_weight = nn.Parameter(torch.empty(qkv_size, hidden_size, dtype=dtype, device=device))
        self.proj_weight = nn.Parameter(torch.empty(hidden_size, num_heads * head_dim, dtype=dtype, device=device))

        self.ulysses_attn = distributed_attention_cls(
            local_attention=_UlyssesLocalAttention(scale=1.0 / math.sqrt(head_dim)),
            sequence_process_group=cp_group,
            scatter_idx=2,
            gather_idx=0,
        )

        expert = _DeepSpeedExpertMLP(
            hidden_size=hidden_size,
            ffn_hidden_size=ffn_hidden_size,
            activation_func=activation_func,
            dtype=dtype,
            device=device,
        )
        self.moe = ds_moe_cls(
            hidden_size=hidden_size,
            expert=expert,
            num_experts=num_experts,
            ep_size=ep_group.size(),
            k=top_k,
            capacity_factor=capacity_factor,
            eval_capacity_factor=capacity_factor,
            min_capacity=4,
            use_residual=False,
            noisy_gate_policy=None,
            drop_tokens=True,
            use_rts=True,
            use_tutel=False,
            enable_expert_tensor_parallelism=False,
            top2_2nd_expert_sampling=True,
        )
        self.moe.to(device=device)
        # In block-test mode (wire_ep_group=True) we wire the benchmark's
        # expert-parallel group manually since there is no deepspeed.initialize()
        # step that would do it. In E2E mode (wire_ep_group=False), deepspeed
        # engine's set_deepspeed_parallelism() handles ep_group wiring; setting
        # it now would later trip the "override ep_group" assert in
        # deepspeed.moe.sharded_moe._set_ep_group.
        if wire_ep_group:
            self.moe.deepspeed_moe._set_ep_group(ep_group)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.xavier_uniform_(self.proj_weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len, batch_size, hidden_size = hidden_states.shape
        head_dim = hidden_size // self.num_heads

        ln1_out = F.layer_norm(hidden_states, (hidden_size,), self.ln1_weight, self.ln1_bias)
        qkv = F.linear(ln1_out, self.qkv_weight)

        q_per_kv = self.num_heads // self.num_kv_heads
        group_size = (q_per_kv + 2) * head_dim
        qkv = qkv.view(seq_len, batch_size, self.num_kv_heads, group_size)

        q_dim = q_per_kv * head_dim
        q_sp = qkv[:, :, :, :q_dim].reshape(seq_len, batch_size, self.num_heads, head_dim)
        k_sp = qkv[:, :, :, q_dim:q_dim + head_dim]
        v_sp = qkv[:, :, :, q_dim + head_dim:]

        # Use batch_dim_idx=0 to match the existing benchmark contract and avoid
        # the reverse-AllToAll shape issue in DistributedAttention.
        q_bf = q_sp.permute(1, 0, 2, 3).contiguous()
        k_bf = k_sp.permute(1, 0, 2, 3).contiguous()
        v_bf = v_sp.permute(1, 0, 2, 3).contiguous()
        attn_out_bf = self.ulysses_attn(q_bf, k_bf, v_bf, batch_dim_idx=0)
        attn_out_sp = attn_out_bf.permute(1, 0, 2, 3).contiguous()

        proj_out = F.linear(attn_out_sp.view(seq_len, batch_size, -1), self.proj_weight)
        hidden_after_attn = hidden_states + proj_out

        ln2_out = F.layer_norm(hidden_after_attn, (hidden_size,), self.ln2_weight, self.ln2_bias)
        moe_out, l_aux, _ = self.moe(ln2_out)
        # Captured for E2E aux-loss aggregation in pretrain_deepspeed.py;
        # block-test callers ignore this attribute.
        self._last_l_aux = l_aux
        return hidden_after_attn + moe_out


class DeepSpeedBlockBaselineTransformerModel(nn.Module):
    """Multi-layer block-level DeepSpeed baseline for the current benchmark."""

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
        activation_func: Optional[Callable] = None,
        capacity_factor: float = 1.0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        wire_ep_group: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DeepSpeedBlockBaselineLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    num_experts=num_experts,
                    top_k=top_k,
                    cp_group=cp_group,
                    ep_group=ep_group,
                    layer_id=i,
                    activation_func=activation_func,
                    capacity_factor=capacity_factor,
                    dtype=dtype,
                    device=device,
                    wire_ep_group=wire_ep_group,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
