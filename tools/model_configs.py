"""Native MoE model presets, ordered roughly by model scale."""

from __future__ import annotations

from typing import Dict, Any


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 1) Alibaba Cloud - Qwen1.5-MoE-A2.7B (~14.3B total, 2.7B active)
    "qwen_moe_a2_7b": {
        "hf_model_id": "Qwen/Qwen1.5-MoE-A2.7B",
        "hidden_size": 2048,
        "num_heads": 16,
        "num_kv_heads": 16,
        "ffn_hidden": 1408,
        "num_experts": 60,
        "top_k": 4,
        "num_layers": 8,
        "seq_len": 4096,
        "batch_size": 4,
        "capacity_factor": 1.0,
    },
    # 2) OpenAI - gpt-oss-20b (~21B total, 3.6B active)
    "openai_gpt_oss_20b": {
        "hidden_size": 2880,
        "num_heads": 64,
        "num_kv_heads": 8,
        "ffn_hidden": 2880,
        "num_experts": 32,
        "top_k": 4,
        "num_layers": 4,
        "seq_len": 4096,
        "batch_size": 2,
        "capacity_factor": 1.0,
    },
    # 3) Mistral AI - Mixtral-8x7B-v0.1 (~46.7B total, ~13B active)
    "mixtral_8x7b": {
        "hf_model_id": "mistralai/Mixtral-8x7B-v0.1",
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "ffn_hidden": 14336,
        "num_experts": 16,
        "top_k": 2,
        "num_layers": 2,
        "seq_len": 4096,
        "batch_size": 4,
        "capacity_factor": 1.0,
    },
    # 4) DeepSeek - DeepSeek-V3 (native MoE, 671B total)
    "deepseek_v3": {
        "hf_model_id": "deepseek-ai/DeepSeek-V3",
        "hidden_size": 7168,
        "num_heads": 128,
        "num_kv_heads": 128,
        "ffn_hidden": 2048,
        "num_experts": 256,
        "top_k": 8,
        "num_layers": 4,
        "seq_len": 4096,
        "batch_size": 1,
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
