"""Shared experiment defaults for benchmark and tuning scripts."""

from __future__ import annotations

from pathlib import Path
from pprint import pformat
from typing import Any, Dict


BLOCK_BENCHMARK_DEFAULTS: Dict[str, Any] = {'dp_size': 1,
 'cp_size': 2,
 'ep_size': 2,
 'moe_combine_chunks': 8,
 'moe_dispatch_chunks': 1,
 'attn_proj_chunks': 2,
 'attn_qkv_chunks': 2,
 'warmup': 10,
 'iters': 50,
 'gap_budgets': {'moe_combine': 1.0195,
                 'moe_dispatch': 0.6267,
                 'attn_proj': 0.3954,
                 'attn_qkv': 0.3041},
 'shared_ar_bw': 5334635.7,
 'expert_ar_bw': 0.0}

TUNE_DEFAULTS: Dict[str, Any] = {
    "dp_size": 1,
    "cp_size": 2,
    "ep_size": 2,
    "chunk_search_iters": 10,
    "chunk_search_max_c": 8,
    "chunk_stop_min_saving_ms": 0.05,
    "ar_warmup": 5,
    "ar_iters": 10,
}


DEEPSPEED_ULYSSES_DEFAULTS: Dict[str, Any] = {
    "zero_stage": 0,
    "learning_rate": 1e-4,
    "warmup": 3,
    "iters": 10,
}


def get_block_benchmark_defaults() -> Dict[str, Any]:
    return dict(BLOCK_BENCHMARK_DEFAULTS)


def get_tune_defaults() -> Dict[str, Any]:
    return dict(TUNE_DEFAULTS)


def get_deepspeed_ulysses_defaults() -> Dict[str, Any]:
    return dict(DEEPSPEED_ULYSSES_DEFAULTS)


def persist_block_benchmark_defaults(updates: Dict[str, Any]) -> None:
    current = dict(BLOCK_BENCHMARK_DEFAULTS)
    current.update(updates)

    rendered = pformat(current, width=100, sort_dicts=False)
    replacement = f"BLOCK_BENCHMARK_DEFAULTS: Dict[str, Any] = {rendered}"

    path = Path(__file__)
    text = path.read_text(encoding="utf-8")
    start_marker = "BLOCK_BENCHMARK_DEFAULTS: Dict[str, Any] ="
    end_marker = "\n\nTUNE_DEFAULTS: Dict[str, Any] ="
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start == -1 or end == -1:
        raise RuntimeError("Failed to update BLOCK_BENCHMARK_DEFAULTS in experiment_configs.py")
    updated_text = text[:start] + replacement + text[end:]

    path.write_text(updated_text, encoding="utf-8")
    BLOCK_BENCHMARK_DEFAULTS.clear()
    BLOCK_BENCHMARK_DEFAULTS.update(current)
