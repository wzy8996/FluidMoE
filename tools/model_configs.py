"""Paper-oriented MoE benchmark presets.

These presets target the long-sequence Ulysses CP + EP setting used by the
FluidMoE paper: ``dp=2, cp=4, ep=4`` and 32K-token blocks.

For public MoE models, the per-layer dimensions follow the released configs as
closely as this block benchmark supports; ``num_layers`` is reduced to keep the
8-GPU experiments practical. Entries ending in ``_proxy`` use the official MoE
dimensions but map non-standard attention variants (MLA, separated head_dim, or
sliding attention) onto FluidMoE's current MHA/GQA block.
"""

from __future__ import annotations

from typing import Dict, Any


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Long-sequence Ulysses CP + EP presets (default: dp=2, cp=4, ep=4)
    #
    # ``seq_len`` is the benchmark workload length. For models whose original
    # context is shorter, this is a long-sequence block stress setting rather
    # than a checkpoint-compatible full-model setting.
    # ------------------------------------------------------------------
    # 1) Databricks - DBRX Base.
    # Official per-layer MoE dimensions; layers reduced from 40.
    "dbrx_base": {
        "hf_model_id": "databricks/dbrx-base",
        "hidden_size": 6144,
        "num_heads": 48,
        "num_kv_heads": 8,
        "ffn_hidden": 10752,
        "num_experts": 16,
        "top_k": 4,
        "num_layers": 4,
        "seq_len": 4096,
        "batch_size": 2,
        "capacity_factor": 1.0,
    },

    # 2) DeepSeek AI - DeepSeek-V3.
    # Official MoE dimensions: hidden=7168, routed experts=256, top-k=8,
    # moe_intermediate=2048, layers=61. DeepSeek-V3 uses MLA; this proxy maps
    # it to an MHA block while keeping the MoE sizes. Layers are capped at 2
    # because the true expert count is very memory-heavy even on 8x H100.
    "deepseek_v3_mha_proxy": {
        "hf_model_id": "deepseek-ai/DeepSeek-V3",
        "hidden_size": 7168,
        "num_heads": 128,
        "num_kv_heads": 128,
        "ffn_hidden": 2048,
        "num_experts": 256,
        "top_k": 8,
        "num_layers": 2,
        "seq_len": 4096,
        "batch_size": 2,
        "capacity_factor": 1.0,
    },

    # 3) Z.ai - GLM-4.5-Air.
    # Modern GLM MoE with 128 routed experts, top-k=8, 131K context, and 46
    # layers. Official GLM attention has a separate head_dim with 96 attention
    # heads; this proxy uses 32 heads so hidden=4096 maps cleanly to head_dim=128.
    "glm4_5_air_mha_proxy": {
        "hf_model_id": "zai-org/GLM-4.5-Air",
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "ffn_hidden": 1408,
        "num_experts": 128,
        "top_k": 8,
        "num_layers": 4,
        "seq_len": 4096,
        "batch_size": 2,
        "capacity_factor": 1.0,
    },

    # 4) Qwen - Qwen3-30B-A3B.
    # Official per-layer MoE dimensions; layers reduced from 48.
    "qwen3_30b_a3b": {
        "hf_model_id": "Qwen/Qwen3-30B-A3B",
        "hidden_size": 2048,
        "num_heads": 32,
        "num_kv_heads": 4,
        "ffn_hidden": 768,
        "num_experts": 128,
        "top_k": 8,
        "num_layers": 4,
        "seq_len": 2048,
        "batch_size": 2,
        "capacity_factor": 1.0,
    },
}


def list_model_names():
    return list(MODEL_CONFIGS.keys())


def get_model_config(model_name: str) -> Dict[str, Any]:
    if model_name not in MODEL_CONFIGS:
        names = ", ".join(list_model_names())
        raise KeyError(f"Unknown model '{model_name}'. Available: {names}")
    return dict(MODEL_CONFIGS[model_name])
