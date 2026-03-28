"""Paper-oriented MoE benchmark presets.

These presets are tailored for the NeurIPS-facing FluidMoE paper:

- Main paper, default setting: ``dp=2, cp=4, ep=4``
- Scaling setting: ``dp=2, cp=8, ep=8``

Model choice is driven by the paper's core problem setting rather than by
"newest model wins". The final paper plan uses:

- public and recognizable MoE families
- a main-result set that stays close to real public models
- a scaling set that stresses higher-CP / higher-expert-count behavior
- complementary roles across models:
  - canonical balanced case
  - newer many-expert public case
  - expert-heavier / higher-top-k case
  - modern long-context case
  - higher-expert-count scaling case

For public frontier models, we keep the per-layer architecture faithful to the
official configs while using reduced depth and conservative sequence/batch
settings so the benchmarks remain practical on 8/16-GPU nodes.
"""

from __future__ import annotations

from typing import Dict, Any


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Main-paper presets (default setting: dp=2, cp=4, ep=4)
    # These are the headline models used in the main-result tables.
    # ------------------------------------------------------------------
    # 1) Mistral AI - Mixtral-8x7B-v0.1
    # Canonical GQA + top-k=2 MoE model.
    # Also the cleanest "all baselines comparable" case.
    "mixtral_8x7b": {
        "hf_model_id": "mistralai/Mixtral-8x7B-v0.1",
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "ffn_hidden": 14336,
        "num_experts": 8,
        "top_k": 2,
        "num_layers": 2,
        "seq_len": 4096,
        "batch_size": 4,
        "capacity_factor": 1.0,
    },

    # 2) Qwen - Qwen3-30B-A3B
    # Newer many-expert public MoE model used in the main-result set.
    # It is intentionally kept in the main cp=4 configuration only because
    # num_kv_heads=4 does not support cp=8 in the current benchmark.
    "qwen3_30b_a3b": {
        "hf_model_id": "Qwen/Qwen3-30B-A3B",
        "hidden_size": 2048,
        "num_heads": 32,
        "num_kv_heads": 4,
        "ffn_hidden": 768,
        "num_experts": 128,
        "top_k": 8,
        "num_layers": 2,
        "seq_len": 2048,
        "batch_size": 2,
        "capacity_factor": 1.0,
    },

    # 3) Databricks - DBRX
    # Expert-heavier public MoE model:
    # kv_n_heads=8, top-k=4, 16 experts, 32K context.
    # This preset stresses expert-path communication while remaining compatible
    # with both cp=4 and cp=8.
    "dbrx": {
        "hf_model_id": "databricks/dbrx-base",
        "hidden_size": 6144,
        "num_heads": 48,
        "num_kv_heads": 8,
        "ffn_hidden": 10752,
        "num_experts": 16,
        "top_k": 4,
        "num_layers": 6,
        "seq_len": 8192,
        "batch_size": 2,
        "capacity_factor": 1.0,
    },

    # 4) Microsoft - Phi-3.5-MoE
    # Modern long-context GQA + top-k=2 model that remains close to a standard
    # Transformer+MoE block and is compatible with both cp=4 and cp=8.
    "phi_3_5_moe": {
        "hf_model_id": "microsoft/Phi-3.5-MoE-instruct",
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "ffn_hidden": 6400,
        "num_experts": 16,
        "top_k": 2,
        "num_layers": 8,
        "seq_len": 8192,
        "batch_size": 2,
        "capacity_factor": 1.0,
    },

    # ------------------------------------------------------------------
    # Scaling / extension preset
    # Used in higher-CP and higher-expert-count stress experiments.
    # ------------------------------------------------------------------
    # 5) AllenAI - OLMoE-1B-7B-0924
    # Higher-expert-count public model (64 experts, top-k=8, MHA) that is
    # well-suited to cp=8 / ep=8 scaling experiments.
    "olmoe_1b_7b": {
        "hf_model_id": "allenai/OLMoE-1B-7B-0924",
        "hidden_size": 2048,
        "num_heads": 16,
        "num_kv_heads": 16,
        "ffn_hidden": 1024,
        "num_experts": 64,
        "top_k": 8,
        "num_layers": 8,
        "seq_len": 4096,
        "batch_size": 4,
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
