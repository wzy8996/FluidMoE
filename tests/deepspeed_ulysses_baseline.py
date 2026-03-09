"""Official-style DeepSpeed Ulysses baseline helpers for block benchmark orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import torch
import torch.distributed as dist


def _require_hf_ulysses_deps():
    try:
        import deepspeed  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "DeepSpeed Ulysses baseline requires both `deepspeed` and `transformers`."
        ) from exc

    from deepspeed.runtime.sequence_parallel.ulysses_sp import (
        UlyssesSPAttentionHF,
        UlyssesSPDataLoaderAdapter,
    )
    from deepspeed.utils import groups
    from transformers import AutoConfig, AutoModelForCausalLM

    return {
        "AutoConfig": AutoConfig,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "UlyssesSPAttentionHF": UlyssesSPAttentionHF,
        "UlyssesSPDataLoaderAdapter": UlyssesSPDataLoaderAdapter,
        "groups": groups,
    }


@dataclass
class DeepSpeedUlyssesBaselineSpec:
    model_name_or_path: str
    sequence_parallel_size: int
    micro_batch_size: int
    seq_length: int
    warmup: int
    iters: int
    bf16: bool = True
    zero_stage: int = 0
    learning_rate: float = 1e-4
    gradient_checkpointing: bool = False
    load_pretrained: bool = True
    trust_remote_code: bool = True
    core_attn_implementation: str = "sdpa"
    seq_length_is_variable: bool = False
    weight_decay: float = 0.0


class SyntheticCausalLMDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size: int, seq_length: int, micro_batch_size: int, total_steps: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.seq_length = int(seq_length)
        self.micro_batch_size = int(micro_batch_size)
        self.total_steps = int(total_steps)

    def __len__(self) -> int:
        return self.total_steps * self.micro_batch_size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        del index
        tokens = torch.randint(0, self.vocab_size, (self.seq_length,), dtype=torch.long)
        return {
            "input_ids": tokens,
            "attention_mask": torch.ones(self.seq_length, dtype=torch.long),
            "labels": tokens.clone(),
        }


def make_deepspeed_config(spec: DeepSpeedUlyssesBaselineSpec) -> Dict[str, Any]:
    return {
        "train_micro_batch_size_per_gpu": spec.micro_batch_size,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 0,
        "wall_clock_breakdown": False,
        "bf16": {"enabled": bool(spec.bf16)},
        "fp16": {"enabled": False if spec.bf16 else True},
        "zero_optimization": {"stage": int(spec.zero_stage)},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": spec.learning_rate,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": spec.weight_decay,
            },
        },
    }


def register_ulysses_with_transformers(spec: DeepSpeedUlyssesBaselineSpec):
    deps = _require_hf_ulysses_deps()
    kwargs = {
        "model_name_or_path": spec.model_name_or_path,
        "core_attn_implementation": spec.core_attn_implementation,
        "sequence_parallel_size": spec.sequence_parallel_size,
        "micro_batch_size": spec.micro_batch_size,
        "seq_length_is_variable": spec.seq_length_is_variable,
    }
    if not spec.seq_length_is_variable:
        kwargs["seq_length"] = spec.seq_length
    return deps["UlyssesSPAttentionHF"].register_with_transformers(**kwargs)


def build_hf_model(spec: DeepSpeedUlyssesBaselineSpec):
    deps = _require_hf_ulysses_deps()
    auto_config = deps["AutoConfig"]
    auto_model = deps["AutoModelForCausalLM"]

    if spec.load_pretrained:
        model = auto_model.from_pretrained(
            spec.model_name_or_path,
            torch_dtype=torch.bfloat16 if spec.bf16 else torch.float16,
            trust_remote_code=spec.trust_remote_code,
        )
    else:
        config = auto_config.from_pretrained(
            spec.model_name_or_path,
            trust_remote_code=spec.trust_remote_code,
        )
        model = auto_model.from_config(config, trust_remote_code=spec.trust_remote_code)

    if spec.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def initialize_deepspeed_engine(spec: DeepSpeedUlyssesBaselineSpec):
    import deepspeed

    mpu = register_ulysses_with_transformers(spec)
    model = build_hf_model(spec)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=make_deepspeed_config(spec),
        mpu=mpu,
    )
    return engine


def get_sp_runtime_handles():
    deps = _require_hf_ulysses_deps()
    groups = deps["groups"]
    return (
        groups._get_sequence_parallel_group(),
        groups._get_sequence_parallel_world_size(),
        groups._get_sequence_parallel_rank(),
    )


def adapt_dataloader_for_ulysses(
    dataloader: Iterable[Dict[str, torch.Tensor]],
    model_device: torch.device,
):
    deps = _require_hf_ulysses_deps()
    sp_group, sp_world_size, sp_rank = get_sp_runtime_handles()
    return deps["UlyssesSPDataLoaderAdapter"](
        dataloader,
        sp_rank=sp_rank,
        sp_group=sp_group,
        sp_world_size=sp_world_size,
        device=model_device,
    )


def reduce_sequence_parallel_loss(
    loss: torch.Tensor,
    shift_labels: torch.Tensor,
    sp_group,
    sp_world_size: int,
) -> torch.Tensor:
    losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
    good_tokens = (shift_labels != -100).view(-1).sum()
    good_tokens_per_rank = torch.distributed.nn.functional.all_gather(
        good_tokens, group=sp_group
    )
    total_loss = sum(
        losses_per_rank[i] * good_tokens_per_rank[i] for i in range(sp_world_size)
    )
    total_good_tokens = sum(good_tokens_per_rank)
    return total_loss / total_good_tokens


def benchmark_deepspeed_ulysses(spec: DeepSpeedUlyssesBaselineSpec, device: torch.device):
    from torch.utils.data import DataLoader

    engine = initialize_deepspeed_engine(spec)
    vocab_size = int(getattr(engine.module.config, "vocab_size"))
    dataset = SyntheticCausalLMDataset(
        vocab_size=vocab_size,
        seq_length=spec.seq_length,
        micro_batch_size=spec.micro_batch_size,
        total_steps=spec.warmup + spec.iters + 4,
    )
    dataloader = DataLoader(dataset, batch_size=spec.micro_batch_size, drop_last=True)
    ulysses_loader = adapt_dataloader_for_ulysses(dataloader, device)
    loader_iter = iter(ulysses_loader)

    def one_step():
        batch = next(loader_iter)
        outputs = engine(**batch)
        loss = outputs.loss
        if loss.dim() != 0:
            loss = loss.mean()

        if spec.sequence_parallel_size > 1:
            labels = batch["labels"]
            shift_labels = labels[..., 1:].contiguous() if labels.size(-1) > 1 else labels
            sp_group, sp_world_size, _ = get_sp_runtime_handles()
            loss = reduce_sequence_parallel_loss(loss, shift_labels, sp_group, sp_world_size)

        engine.backward(loss)
        engine.step()
        return loss.detach()

    for _ in range(spec.warmup):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()

    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    for _ in range(spec.iters):
        one_step()
    ev_e.record()
    torch.cuda.synchronize()
    dist.barrier()

    iter_ms = ev_s.elapsed_time(ev_e) / spec.iters
    tokens_per_s = spec.micro_batch_size * spec.seq_length / (iter_ms / 1000.0)
    return {
        "iter_ms": iter_ms,
        "tokens_per_s": tokens_per_s,
        "model_name_or_path": spec.model_name_or_path,
    }
