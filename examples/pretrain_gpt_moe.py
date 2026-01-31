"""
FluidMoE GPT-MoE Training Script

Uses Megatron's parallel_state for distributed setup and MockGPTDataset for data,
but with FluidMoE's TransformerModel as the core transformer layers.

Usage:
    torchrun --nproc_per_node=2 examples/pretrain_gpt_moe.py
"""

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from typing import Iterator, Dict

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
megatron_path = os.environ.get('MEGATRON_PATH', '/home/zju/wzy/Megatron-LM')
if megatron_path not in sys.path:
    sys.path.insert(0, megatron_path)

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.datasets.utils import compile_helpers
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.tokenizer.tokenizer import _NullTokenizer

from fluid.model import FluidMoEGPTModel
from fluid.core.scheduler import get_backward_scheduler
from megatron.core.transformer.transformer_config import TransformerConfig


def parse_args():
    parser = argparse.ArgumentParser(description='FluidMoE GPT-MoE Training')
    # Model
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=1024)
    parser.add_argument('--ffn-hidden-size', type=int, default=4096)
    parser.add_argument('--num-attention-heads', type=int, default=8)
    parser.add_argument('--seq-length', type=int, default=2048)
    parser.add_argument('--vocab-size', type=int, default=8192)
    # MoE
    parser.add_argument('--num-experts', type=int, default=2)
    parser.add_argument('--moe-router-topk', type=int, default=2)
    # Parallelism
    parser.add_argument('--context-parallel-size', type=int, default=None)
    parser.add_argument('--expert-model-parallel-size', type=int, default=None)
    # Training
    parser.add_argument('--micro-batch-size', type=int, default=1)
    parser.add_argument('--train-iters', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=1)
    return parser.parse_args()


def initialize_distributed(cp_size, ep_size):
    """Initialize torch.distributed and Megatron parallel groups."""
    parallel_state.destroy_model_parallel()

    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(world_size=world_size, rank=rank)

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
    )
    return rank, world_size


def get_train_data_iterator(seq_length, vocab_size, batch_size):
    """Create mock dataset and return data iterator."""
    if torch.distributed.get_rank() == 0:
        compile_helpers()
    torch.distributed.barrier()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=seq_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=vocab_size),
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(
        datasets[0], batch_size=batch_size, shuffle=True
    )
    return iter(train_dataloader)


def main():
    args = parse_args()
    world_size = torch.cuda.device_count()

    # Default: CP=world_size, EP=world_size (same group)
    cp_size = args.context_parallel_size or world_size
    ep_size = args.expert_model_parallel_size or world_size

    rank, world_size = initialize_distributed(cp_size, ep_size)
    model_parallel_cuda_manual_seed(args.seed)
    device = torch.device("cuda")

    if rank == 0:
        print("=" * 60)
        print("FluidMoE GPT-MoE Training")
        print("=" * 60)
        print(f"  GPUs: {world_size}")
        print(f"  Layers: {args.num_layers}")
        print(f"  Hidden: {args.hidden_size}, FFN: {args.ffn_hidden_size}")
        print(f"  Heads: {args.num_attention_heads}")
        print(f"  Seq length: {args.seq_length}")
        print(f"  Experts: {args.num_experts}, Top-K: {args.moe_router_topk}")
        print(f"  CP: {cp_size}, EP: {ep_size}")
        print(f"  Batch size: {args.micro_batch_size}")
        print(f"  LR: {args.lr}")
        print("=" * 60)

    # Build model config
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        num_moe_experts=args.num_experts,
        moe_router_topk=args.moe_router_topk,
        use_cpu_initialization=False,
        pipeline_dtype=torch.bfloat16,
        params_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
    )

    # Vocab size: NullTokenizer uses vocab_size as eod_id, so need +1
    model_vocab_size = args.vocab_size + 1

    model = FluidMoEGPTModel(
        config=config,
        vocab_size=model_vocab_size,
        max_sequence_length=args.seq_length,
    )
    model = model.to(device).to(torch.bfloat16)

    # Enable FluidMoE scheduler
    scheduler = get_backward_scheduler()
    scheduler.enable()

    # Configure AllReduce for DP gradient sync
    dp_group = parallel_state.get_data_parallel_group()
    dp_world_size = parallel_state.get_data_parallel_world_size()
    if dp_world_size > 1:
        scheduler.configure_allreduce(enabled=True, dp_group=dp_group)
    else:
        scheduler.configure_allreduce(enabled=False, dp_group=dp_group)

    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"\n  Parameters: {num_params:,}")

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Data
    train_iterator = get_train_data_iterator(
        args.seq_length, args.vocab_size, args.micro_batch_size
    )

    # Training loop
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)

    model.train()
    for iteration in range(1, args.train_iters + 1):
        t_start = time.perf_counter()

        # Get batch
        try:
            data = next(train_iterator)
        except StopIteration:
            train_iterator = get_train_data_iterator(
                args.seq_length, args.vocab_size, args.micro_batch_size
            )
            data = next(train_iterator)

        tokens = data["tokens"].to(device)
        labels = data["labels"].to(device)
        loss_mask = data["loss_mask"].to(device).float()
        position_ids = data["position_ids"].to(device)
        attention_mask = None  # FluidMoE uses causal mask internally

        # Forward
        output_tensor = model(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Loss: output_tensor is [batch*seq] per-token losses (float32 from cross_entropy)
        # Compute scalar loss in float32 for numerical precision
        losses_f32 = output_tensor.float()
        loss_mask_flat = loss_mask.view(-1)
        loss = torch.sum(losses_f32 * loss_mask_flat) / loss_mask_flat.sum()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # FluidMoE: finish deferred dW tasks + AR
        scheduler.finish_batch()

        # Step
        optimizer.step()
        scheduler.clear_iteration()

        torch.cuda.synchronize()
        t_elapsed = (time.perf_counter() - t_start) * 1000

        if rank == 0 and iteration % args.log_interval == 0:
            print(f"  iter {iteration:>4d}/{args.train_iters} | "
                  f"loss: {loss.item():.4f} | "
                  f"time: {t_elapsed:.1f}ms")

    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

    parallel_state.destroy_model_parallel()


if __name__ == "__main__":
    main()
