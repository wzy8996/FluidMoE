#!/usr/bin/env python3
"""
FlashOverlap-style Offline Tuning for FC2+AllToAll Overlap.

This script profiles different parameter combinations and saves optimal
configurations for runtime use.

Usage:
    # Single GPU profiling (for GEMM only)
    python flash_overlap_tuner.py --mode gemm

    # Distributed profiling (for full overlap)
    torchrun --nproc_per_node=2 flash_overlap_tuner.py --mode overlap
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed as dist

# Add FluidMoE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def get_config_key(num_experts: int, tokens_per_expert: int,
                   hidden_size: int, intermediate_size: int) -> str:
    """Generate unique key for a model configuration."""
    return f"E{num_experts}_T{tokens_per_expert}_H{hidden_size}_I{intermediate_size}"


def setup_distributed():
    """Initialize distributed environment."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    return rank, world_size, local_rank


def profile_gemm_only(
    fluid_kernels,
    device: torch.device,
    num_experts: int,
    tokens_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    tiles_per_wave_options: List[int],
    num_warmup: int = 5,
    num_iters: int = 20
) -> Dict[int, float]:
    """Profile GEMM with different tiles_per_wave settings."""

    total_tokens = num_experts * tokens_per_expert

    # Create tensors
    fc2_input = torch.randn(total_tokens, intermediate_size,
                            device=device, dtype=torch.float16)
    fc2_weight = torch.randn(num_experts, intermediate_size, hidden_size,
                             device=device, dtype=torch.float16)
    fc2_output = torch.zeros(total_tokens, hidden_size,
                             device=device, dtype=torch.float16)
    tokens_tensor = torch.full((num_experts,), tokens_per_expert,
                               device=device, dtype=torch.int32)

    results = {}

    for tpw in tiles_per_wave_options:
        # Warmup
        for _ in range(num_warmup):
            fc2_output.zero_()
            fluid_kernels.grouped_gemm_epilogue_signal(
                fc2_input, fc2_weight, fc2_output, tokens_tensor, tpw
            )
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(num_iters):
            fc2_output.zero_()
            fluid_kernels.grouped_gemm_epilogue_signal(
                fc2_input, fc2_weight, fc2_output, tokens_tensor, tpw
            )
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / num_iters * 1000  # ms

        results[tpw] = elapsed

    return results


def profile_overlap(
    fluid_kernels,
    rank: int,
    world_size: int,
    device: torch.device,
    num_experts: int,
    tokens_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    tiles_per_wave_options: List[int],
    waves_per_comm_options: List[int],
    num_warmup: int = 5,
    num_iters: int = 20
) -> Tuple[Dict, float]:
    """Profile overlap with different parameter combinations."""

    total_tokens = num_experts * tokens_per_expert

    # Create tensors
    fc2_input = torch.randn(total_tokens, intermediate_size,
                            device=device, dtype=torch.float16)
    fc2_weight = torch.randn(num_experts, intermediate_size, hidden_size,
                             device=device, dtype=torch.float16)
    fc2_output = torch.zeros(total_tokens, hidden_size,
                             device=device, dtype=torch.float16)
    tokens_tensor = torch.full((num_experts,), tokens_per_expert,
                               device=device, dtype=torch.int32)
    send_buf = torch.zeros_like(fc2_output)
    recv_buf = torch.zeros_like(fc2_output)

    # Initialize with first tiles_per_wave to setup NCCL
    fluid_kernels.grouped_gemm_epilogue_signal(
        fc2_input, fc2_weight, fc2_output, tokens_tensor, tiles_per_wave_options[0]
    )

    # Initialize NCCL
    if rank == 0:
        nccl_id = fluid_kernels.get_nccl_unique_id()
        nccl_id_tensor = torch.tensor(nccl_id, dtype=torch.int64, device=device)
    else:
        nccl_id_tensor = torch.zeros(17, dtype=torch.int64, device=device)
    dist.broadcast(nccl_id_tensor, src=0)
    fluid_kernels.init_nccl_comm_with_id(nccl_id_tensor.tolist(), rank, world_size)

    # Measure sequential baseline (best case without overlap)
    best_tpw_for_seq = tiles_per_wave_options[0]
    best_seq_time = float('inf')

    for tpw in tiles_per_wave_options:
        # Warmup
        for _ in range(num_warmup):
            fc2_output.zero_()
            fluid_kernels.grouped_gemm_epilogue_signal(
                fc2_input, fc2_weight, fc2_output, tokens_tensor, tpw
            )
            torch.cuda.synchronize()
            send_buf.copy_(fc2_output)
            dist.all_to_all_single(recv_buf, send_buf)
            torch.cuda.synchronize()

        # Measure
        torch.cuda.synchronize()
        dist.barrier()
        start = time.time()
        for _ in range(num_iters):
            fc2_output.zero_()
            fluid_kernels.grouped_gemm_epilogue_signal(
                fc2_input, fc2_weight, fc2_output, tokens_tensor, tpw
            )
            torch.cuda.synchronize()
            send_buf.copy_(fc2_output)
            dist.all_to_all_single(recv_buf, send_buf)
            torch.cuda.synchronize()
        seq_time = (time.time() - start) / num_iters * 1000

        if seq_time < best_seq_time:
            best_seq_time = seq_time
            best_tpw_for_seq = tpw

    # Profile overlap combinations
    results = {}

    for tpw in tiles_per_wave_options:
        for wpc in waves_per_comm_options:
            key = f"tpw{tpw}_wpc{wpc}"

            try:
                # Reinit context with new tiles_per_wave
                fluid_kernels.grouped_gemm_epilogue_signal(
                    fc2_input, fc2_weight, fc2_output, tokens_tensor, tpw
                )

                # Warmup
                for _ in range(num_warmup):
                    fc2_output.zero_()
                    fluid_kernels.grouped_gemm_epilogue_signal(
                        fc2_input, fc2_weight, fc2_output, tokens_tensor, tpw
                    )
                    fluid_kernels.queue_flash_overlap_alltoall_single(
                        fc2_output.data_ptr(), recv_buf.data_ptr(), wpc
                    )
                    fluid_kernels.sync_comm_to_compute()
                    torch.cuda.synchronize()

                # Measure
                torch.cuda.synchronize()
                dist.barrier()
                start = time.time()
                for _ in range(num_iters):
                    fc2_output.zero_()
                    fluid_kernels.grouped_gemm_epilogue_signal(
                        fc2_input, fc2_weight, fc2_output, tokens_tensor, tpw
                    )
                    fluid_kernels.queue_flash_overlap_alltoall_single(
                        fc2_output.data_ptr(), recv_buf.data_ptr(), wpc
                    )
                    fluid_kernels.sync_comm_to_compute()
                    torch.cuda.synchronize()
                overlap_time = (time.time() - start) / num_iters * 1000

                speedup = best_seq_time / overlap_time
                results[key] = {
                    'tiles_per_wave': tpw,
                    'waves_per_comm': wpc,
                    'overlap_time_ms': overlap_time,
                    'speedup': speedup
                }

            except Exception as e:
                if rank == 0:
                    print(f"  {key}: ERROR - {e}")

    return results, best_seq_time


def find_best_config(results: Dict) -> Tuple[int, int, float]:
    """Find the best configuration from profiling results."""
    best_key = max(results.keys(), key=lambda k: results[k]['speedup'])
    best = results[best_key]
    return best['tiles_per_wave'], best['waves_per_comm'], best['speedup']


def save_tuning_results(
    results: Dict,
    config_key: str,
    output_dir: str = None
):
    """Save tuning results to JSON file."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "configs"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing configs or create new
    config_file = output_dir / "flash_overlap_configs.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            all_configs = json.load(f)
    else:
        all_configs = {}

    # Find best config
    best_tpw, best_wpc, best_speedup = find_best_config(results)

    all_configs[config_key] = {
        'tiles_per_wave': best_tpw,
        'waves_per_comm': best_wpc,
        'speedup': best_speedup,
        'all_results': results
    }

    with open(config_file, 'w') as f:
        json.dump(all_configs, f, indent=2)

    return best_tpw, best_wpc, best_speedup


def load_tuning_config(
    num_experts: int,
    tokens_per_expert: int,
    hidden_size: int,
    intermediate_size: int,
    config_dir: str = None
) -> Optional[Tuple[int, int]]:
    """Load pre-tuned configuration for given model parameters."""
    if config_dir is None:
        config_dir = Path(__file__).parent / "configs"
    else:
        config_dir = Path(config_dir)

    config_file = config_dir / "flash_overlap_configs.json"
    if not config_file.exists():
        return None

    with open(config_file, 'r') as f:
        all_configs = json.load(f)

    config_key = get_config_key(num_experts, tokens_per_expert,
                                hidden_size, intermediate_size)

    if config_key in all_configs:
        cfg = all_configs[config_key]
        return cfg['tiles_per_wave'], cfg['waves_per_comm']

    # Try to find a similar config
    # (same hidden/intermediate, different tokens)
    for key, cfg in all_configs.items():
        if f"H{hidden_size}_I{intermediate_size}" in key:
            return cfg['tiles_per_wave'], cfg['waves_per_comm']

    return None


# ============================================================================
# Predefined Model Configurations
# ============================================================================

MODEL_CONFIGS = {
    'small': {
        'num_experts': 8,
        'tokens_per_expert': 256,
        'hidden_size': 1024,
        'intermediate_size': 4096,
    },
    'medium': {
        'num_experts': 8,
        'tokens_per_expert': 512,
        'hidden_size': 2048,
        'intermediate_size': 8192,
    },
    'large': {
        'num_experts': 8,
        'tokens_per_expert': 256,
        'hidden_size': 4096,
        'intermediate_size': 14336,
    },
    'mixtral': {
        'num_experts': 8,
        'tokens_per_expert': 512,
        'hidden_size': 4096,
        'intermediate_size': 14336,
    },
}


def main():
    parser = argparse.ArgumentParser(description='FlashOverlap Offline Tuner')
    parser.add_argument('--mode', choices=['gemm', 'overlap'], default='overlap',
                        help='Profiling mode')
    parser.add_argument('--models', nargs='+', default=['small', 'medium', 'large'],
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model configurations to tune')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for config files')
    parser.add_argument('--num-warmup', type=int, default=5,
                        help='Number of warmup iterations')
    parser.add_argument('--num-iters', type=int, default=20,
                        help='Number of profiling iterations')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # Import fluid_kernels
    sys.path.insert(0, str(Path(__file__).parent.parent / "ops"))
    import fluid_kernels

    # Parameter search space
    tiles_per_wave_options = [16, 32, 48, 64, 96, 128]
    waves_per_comm_options = [1, 2, 3, 4]

    if rank == 0:
        print("=" * 70)
        print("FlashOverlap Offline Tuner")
        print(f"Mode: {args.mode}")
        print(f"World size: {world_size}")
        print(f"Models: {args.models}")
        print("=" * 70)

    for model_name in args.models:
        cfg = MODEL_CONFIGS[model_name]
        config_key = get_config_key(
            cfg['num_experts'], cfg['tokens_per_expert'],
            cfg['hidden_size'], cfg['intermediate_size']
        )

        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Tuning: {model_name} ({config_key})")
            print(f"  Experts: {cfg['num_experts']}")
            print(f"  Tokens/Expert: {cfg['tokens_per_expert']}")
            print(f"  Hidden: {cfg['hidden_size']}")
            print(f"  Intermediate: {cfg['intermediate_size']}")
            print("-" * 70)

        if args.mode == 'gemm':
            results = profile_gemm_only(
                fluid_kernels, device,
                cfg['num_experts'], cfg['tokens_per_expert'],
                cfg['hidden_size'], cfg['intermediate_size'],
                tiles_per_wave_options,
                args.num_warmup, args.num_iters
            )

            if rank == 0:
                print("GEMM-only profiling results:")
                for tpw, time_ms in sorted(results.items()):
                    print(f"  tiles_per_wave={tpw}: {time_ms:.3f}ms")
                best_tpw = min(results.keys(), key=lambda k: results[k])
                print(f"Best: tiles_per_wave={best_tpw} ({results[best_tpw]:.3f}ms)")

        else:  # overlap mode
            if world_size < 2:
                if rank == 0:
                    print("ERROR: Overlap mode requires at least 2 GPUs")
                    print("Run with: torchrun --nproc_per_node=2 flash_overlap_tuner.py")
                continue

            results, seq_time = profile_overlap(
                fluid_kernels, rank, world_size, device,
                cfg['num_experts'], cfg['tokens_per_expert'],
                cfg['hidden_size'], cfg['intermediate_size'],
                tiles_per_wave_options, waves_per_comm_options,
                args.num_warmup, args.num_iters
            )

            if rank == 0:
                print(f"Sequential baseline: {seq_time:.3f}ms")
                print("\nOverlap profiling results:")

                # Sort by speedup
                sorted_results = sorted(
                    results.items(),
                    key=lambda x: x[1]['speedup'],
                    reverse=True
                )

                for key, r in sorted_results[:10]:  # Top 10
                    mark = "+" if r['speedup'] > 1.0 else "-"
                    print(f"  {key}: {r['overlap_time_ms']:.3f}ms "
                          f"({mark}{(r['speedup']-1)*100:.1f}%)")

                # Save best config
                best_tpw, best_wpc, best_speedup = save_tuning_results(
                    results, config_key, args.output_dir
                )

                print(f"\nBest config saved:")
                print(f"  tiles_per_wave={best_tpw}, waves_per_comm={best_wpc}")
                print(f"  Speedup: {best_speedup:.3f}x ({(best_speedup-1)*100:.1f}%)")

    if world_size > 1:
        fluid_kernels.destroy_epilogue_signal_context()
        dist.destroy_process_group()

    if rank == 0:
        print("\n" + "=" * 70)
        print("Tuning complete!")
        print("=" * 70)


if __name__ == "__main__":
    main()
