"""
FlashOverlap-style Overlap Tuning Module for FluidMoE

Implements advanced tuning features:
1. Hint计算 - Run monitoring mode to collect tile completion order
2. Tile重排序 - Compute reorder array for contiguous wave output
3. SM竞争估算 - Estimate NCCL SM usage
4. 搜索策略 - Exhaustive + Predictive search
5. 带宽曲线 - Profile NCCL bandwidth at different data sizes
"""

import os
import json
import time
import torch
import torch.distributed as dist
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import numpy as np


@dataclass
class TuningConfig:
    """Configuration for a specific problem size."""
    tiles_per_wave: int
    waves_per_comm: int
    reorder_array: Optional[List[int]] = None
    predicted_time_us: float = 0.0
    measured_time_us: float = 0.0


@dataclass
class BandwidthPoint:
    """Single point in bandwidth curve."""
    data_size_bytes: int
    bandwidth_gbps: float
    latency_us: float


class OverlapTuner:
    """
    FlashOverlap-style tuner for GroupedGEMM + AllToAll overlap.

    Usage:
        tuner = OverlapTuner(rank, world_size)

        # Profile bandwidth curve (once per cluster)
        tuner.profile_bandwidth()

        # Collect hints for a specific workload
        hints = tuner.collect_hints(input, weight, tokens_per_expert)

        # Compute reorder array
        reorder_array = tuner.compute_reorder_array(hints)

        # Search for optimal configuration
        config = tuner.search(total_tiles, elements_per_tile)
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        sm_count: Optional[int] = None,
        nccl_sms: int = 2,  # FlashOverlap default
        config_dir: Optional[str] = None
    ):
        self.rank = rank
        self.world_size = world_size
        self.nccl_sms = nccl_sms

        # Get SM count
        if sm_count is None:
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            self.sm_count = props.multi_processor_count
        else:
            self.sm_count = sm_count

        self.compute_sms = max(1, self.sm_count - self.nccl_sms)

        # Config directory
        if config_dir is None:
            config_dir = Path(__file__).parent / "configs"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Bandwidth curve (profiled or loaded)
        self.bandwidth_curve: List[BandwidthPoint] = []

        # Cached configurations
        self.configs: Dict[str, TuningConfig] = {}

        # Try to load existing configs
        self._load_configs()

    # ========================================
    # 1. Hint Calculation (Monitoring Mode)
    # ========================================

    def collect_hints(
        self,
        input_tensor: torch.Tensor,
        weight_tensor: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_runs: int = 3
    ) -> torch.Tensor:
        """
        Run GEMM in monitoring mode to collect tile completion order.

        This is the "hint" that FlashOverlap uses to determine optimal
        tile reordering for communication overlap.

        Args:
            input_tensor: [total_tokens, K]
            weight_tensor: [num_experts, K, N]
            tokens_per_expert: [num_experts]
            num_runs: Number of runs to average

        Returns:
            hint_buffer: [total_tiles] - completion order of each tile
        """
        try:
            import fluid_kernels
        except ImportError:
            print("Warning: fluid_kernels not available, using simulated hints")
            return self._simulate_hints(tokens_per_expert, weight_tensor.size(2))

        total_tokens = input_tensor.size(0)
        K = input_tensor.size(1)
        N = weight_tensor.size(2)
        num_experts = tokens_per_expert.size(0)

        # Calculate total tiles
        TILE_M, TILE_N = 128, 128
        total_tiles = 0
        for e in range(num_experts):
            tokens = tokens_per_expert[e].item()
            tiles_m = (tokens + TILE_M - 1) // TILE_M
            tiles_n = (N + TILE_N - 1) // TILE_N
            total_tiles += tiles_m * tiles_n

        # Allocate hint buffer
        hint_buffer = torch.zeros(total_tiles, dtype=torch.int32, device='cuda')

        # Allocate output
        output = torch.zeros(total_tokens, N, dtype=torch.float16, device='cuda')

        # Run multiple times and aggregate
        all_hints = []
        for _ in range(num_runs):
            hint_buffer.zero_()

            # Call kernel in monitoring mode
            # Note: This requires the kernel to support monitor_mode parameter
            try:
                fluid_kernels.grouped_gemm_with_monitoring(
                    input_tensor,
                    weight_tensor,
                    output,
                    tokens_per_expert,
                    hint_buffer
                )
                torch.cuda.synchronize()
                all_hints.append(hint_buffer.clone())
            except AttributeError:
                # Fallback if monitoring not implemented
                return self._simulate_hints(tokens_per_expert, N)

        # Average the hints (use most common order)
        if len(all_hints) > 1:
            stacked = torch.stack(all_hints, dim=0)
            hint_buffer = stacked.float().mean(dim=0).int()

        return hint_buffer

    def _simulate_hints(
        self,
        tokens_per_expert: torch.Tensor,
        N: int
    ) -> torch.Tensor:
        """Simulate hint buffer when kernel monitoring is not available."""
        TILE_M, TILE_N = 128, 128
        num_experts = tokens_per_expert.size(0)

        # Calculate tiles per expert
        tiles_list = []
        for e in range(num_experts):
            tokens = tokens_per_expert[e].item()
            tiles_m = (tokens + TILE_M - 1) // TILE_M
            tiles_n = (N + TILE_N - 1) // TILE_N
            tiles_list.append(tiles_m * tiles_n)

        total_tiles = sum(tiles_list)

        # Simulate: tiles complete in roughly round-robin order across experts
        # This is a simplified model - real order depends on GPU scheduler
        hint_buffer = torch.zeros(total_tiles, dtype=torch.int32, device='cuda')

        order = 0
        tile_offsets = [0]
        for t in tiles_list:
            tile_offsets.append(tile_offsets[-1] + t)

        # Simulate round-robin completion
        max_tiles = max(tiles_list) if tiles_list else 0
        for tile_idx in range(max_tiles):
            for e in range(num_experts):
                if tile_idx < tiles_list[e]:
                    global_idx = tile_offsets[e] + tile_idx
                    hint_buffer[global_idx] = order
                    order += 1

        return hint_buffer

    # ========================================
    # 2. Tile Reordering
    # ========================================

    def compute_reorder_array(
        self,
        hints: torch.Tensor,
        tiles_per_wave: int
    ) -> torch.Tensor:
        """
        Compute reorder array from hint buffer.

        FlashOverlap's approach:
        - Sort tiles by completion order
        - Group tiles completing together into waves
        - Return reorder_array[new_idx] = old_tile_idx

        This makes communication more efficient because each wave's
        output is contiguous after reordering.
        """
        hints_cpu = hints.cpu().numpy()
        total_tiles = len(hints_cpu)

        # Create (completion_order, tile_idx) pairs
        order_tile_pairs = [(hints_cpu[i], i) for i in range(total_tiles)]

        # Sort by completion order
        order_tile_pairs.sort(key=lambda x: x[0])

        # Create reorder array
        reorder_array = torch.tensor(
            [pair[1] for pair in order_tile_pairs],
            dtype=torch.int32,
            device='cuda'
        )

        return reorder_array

    def compute_inverse_reorder(
        self,
        reorder_array: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute inverse reorder array.
        inverse_ra[old_tile_idx] = new_position
        """
        ra_cpu = reorder_array.cpu().numpy()
        total_tiles = len(ra_cpu)

        inverse_ra = np.zeros(total_tiles, dtype=np.int32)
        for new_idx in range(total_tiles):
            old_idx = ra_cpu[new_idx]
            inverse_ra[old_idx] = new_idx

        return torch.tensor(inverse_ra, dtype=torch.int32, device='cuda')

    # ========================================
    # 3. SM Competition Estimation
    # ========================================

    def estimate_compute_efficiency(self) -> float:
        """
        Estimate compute efficiency considering NCCL SM usage.

        FlashOverlap insight: NCCL uses ~2 SMs for collective operations.
        So effective compute throughput = (total_SMs - nccl_SMs) / total_SMs
        """
        return self.compute_sms / self.sm_count

    def get_adjusted_compute_time(
        self,
        base_compute_time_us: float
    ) -> float:
        """Adjust compute time for SM competition."""
        efficiency = self.estimate_compute_efficiency()
        return base_compute_time_us / efficiency

    # ========================================
    # 4. Bandwidth Curve Profiling
    # ========================================

    def profile_bandwidth(
        self,
        test_sizes: Optional[List[int]] = None,
        num_iters: int = 10,
        warmup_iters: int = 3
    ) -> List[BandwidthPoint]:
        """
        Profile NCCL AllToAll bandwidth at different data sizes.

        This helps predict communication time for different wave sizes.
        """
        if test_sizes is None:
            # Default test sizes: 1KB to 16MB
            test_sizes = [
                1024, 4096, 16384, 65536, 262144,
                1048576, 4194304, 16777216
            ]

        if self.world_size <= 1:
            # Single GPU: use theoretical bandwidth
            self.bandwidth_curve = [
                BandwidthPoint(size, 200.0, size / (200e9 / 8) * 1e6)
                for size in test_sizes
            ]
            return self.bandwidth_curve

        results = []

        # Allocate buffers
        max_size = max(test_sizes)
        send_buf = torch.zeros(max_size // 2, dtype=torch.float16, device='cuda')
        recv_buf = torch.zeros(max_size // 2, dtype=torch.float16, device='cuda')

        # Warmup
        for _ in range(warmup_iters):
            dist.all_to_all_single(recv_buf, send_buf)
        torch.cuda.synchronize()

        # Profile each size
        for size in test_sizes:
            count = size // 2 // self.world_size  # Elements per rank
            send_slice = send_buf[:count * self.world_size]
            recv_slice = recv_buf[:count * self.world_size]

            # Warmup this size
            dist.all_to_all_single(recv_slice, send_slice)
            torch.cuda.synchronize()

            # Time it
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_iters):
                dist.all_to_all_single(recv_slice, send_slice)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_ms = start_event.elapsed_time(end_event)
            avg_time_us = elapsed_ms * 1000 / num_iters
            bandwidth_gbps = (size * 8 / 1e9) / (avg_time_us / 1e6)

            results.append(BandwidthPoint(size, bandwidth_gbps, avg_time_us))

        self.bandwidth_curve = results
        self._save_bandwidth_curve()

        return results

    def interpolate_bandwidth(self, data_size_bytes: int) -> float:
        """Interpolate bandwidth for a given data size."""
        if not self.bandwidth_curve:
            return 100.0  # Default 100 Gbps

        # Find surrounding points
        for i in range(len(self.bandwidth_curve) - 1):
            if (self.bandwidth_curve[i].data_size_bytes <= data_size_bytes <=
                self.bandwidth_curve[i + 1].data_size_bytes):
                # Linear interpolation
                p1, p2 = self.bandwidth_curve[i], self.bandwidth_curve[i + 1]
                t = (data_size_bytes - p1.data_size_bytes) / \
                    (p2.data_size_bytes - p1.data_size_bytes)
                return p1.bandwidth_gbps + t * (p2.bandwidth_gbps - p1.bandwidth_gbps)

        # Extrapolate from last point
        return self.bandwidth_curve[-1].bandwidth_gbps

    # ========================================
    # 5. Search Strategies
    # ========================================

    def exhaustive_search(
        self,
        total_tiles: int,
        elements_per_tile: int,
        compute_time_per_tile_us: float = 10.0
    ) -> TuningConfig:
        """
        Exhaustive search: Try all valid (tiles_per_wave, waves_per_comm) combinations.

        FlashOverlap's exhaustive_search from search.py
        """
        best_config = TuningConfig(tiles_per_wave=1, waves_per_comm=1, predicted_time_us=1e9)

        sm_efficiency = self.estimate_compute_efficiency()

        # Try different tiles_per_wave
        for tpw in range(1, total_tiles + 1):
            num_waves = (total_tiles + tpw - 1) // tpw
            if num_waves < 2:
                continue  # Need at least 2 waves for overlap

            # Try different waves_per_comm
            for wpc in range(1, num_waves + 1):
                num_comms = (num_waves + wpc - 1) // wpc

                # Compute time for all tiles
                total_compute_time = total_tiles * compute_time_per_tile_us / sm_efficiency

                # Communication time estimation
                total_comm_time = 0
                for c in range(num_comms):
                    waves_in_comm = min(wpc, num_waves - c * wpc)
                    tiles_in_comm = min(waves_in_comm * tpw, total_tiles - c * wpc * tpw)

                    data_bytes = tiles_in_comm * elements_per_tile * 2  # fp16
                    bw = self.interpolate_bandwidth(data_bytes)
                    comm_time = (data_bytes * 8 / 1e9) / (bw / 1e6)  # us
                    total_comm_time += comm_time

                # Overlap model
                overlap_benefit = min(total_compute_time, total_comm_time) * 0.8
                estimated_time = total_compute_time + total_comm_time - overlap_benefit

                if estimated_time < best_config.predicted_time_us:
                    best_config = TuningConfig(
                        tiles_per_wave=tpw,
                        waves_per_comm=wpc,
                        predicted_time_us=estimated_time
                    )

        return best_config

    def predictive_search(
        self,
        total_tiles: int,
        elements_per_tile: int,
        compute_time_per_tile_us: float = 10.0
    ) -> TuningConfig:
        """
        Predictive search: Use bandwidth curve to predict optimal configuration.

        FlashOverlap's fast_search from search.py
        """
        sm_efficiency = self.estimate_compute_efficiency()

        # Find bandwidth saturation point
        if self.bandwidth_curve:
            max_bw = max(pt.bandwidth_gbps for pt in self.bandwidth_curve)
            saturation_size = next(
                (pt.data_size_bytes for pt in self.bandwidth_curve
                 if pt.bandwidth_gbps >= 0.9 * max_bw),
                self.bandwidth_curve[-1].data_size_bytes
            )
        else:
            max_bw = 200.0
            saturation_size = 1048576  # 1MB

        # Target: tiles_per_wave such that compute_time ≈ comm_time
        best_tpw = self.sm_count  # Default to 1 wave per SM

        for tpw in range(1, total_tiles // 2 + 1):
            data_size = tpw * elements_per_tile * 2  # fp16
            bw = self.interpolate_bandwidth(data_size)
            comm_time = (data_size * 8) / bw  # ns
            compute_time = (tpw * compute_time_per_tile_us * 1000) / sm_efficiency  # ns

            if compute_time >= comm_time:
                best_tpw = tpw
                break

        # Determine waves_per_comm based on bandwidth efficiency
        num_waves = (total_tiles + best_tpw - 1) // best_tpw
        best_wpc = 1
        min_overhead = 1e9

        for wpc in range(1, num_waves + 1):
            data_per_comm = wpc * best_tpw * elements_per_tile * 2
            bw = self.interpolate_bandwidth(data_per_comm)

            num_comms = (num_waves + wpc - 1) // wpc
            startup_overhead = num_comms * 5.0  # ~5us per NCCL call
            bw_overhead = (1.0 / bw) * 1e6

            total_overhead = startup_overhead + bw_overhead
            if total_overhead < min_overhead:
                min_overhead = total_overhead
                best_wpc = wpc

        return TuningConfig(
            tiles_per_wave=best_tpw,
            waves_per_comm=best_wpc,
            predicted_time_us=0.0  # Not computed in predictive mode
        )

    def search(
        self,
        total_tiles: int,
        elements_per_tile: int,
        compute_time_per_tile_us: float = 10.0,
        use_predictive: bool = True
    ) -> TuningConfig:
        """
        Search for optimal configuration.

        Args:
            total_tiles: Total number of tiles in GEMM
            elements_per_tile: Number of elements per tile
            compute_time_per_tile_us: Estimated compute time per tile
            use_predictive: Use fast predictive search (vs exhaustive)
        """
        if use_predictive:
            return self.predictive_search(
                total_tiles, elements_per_tile, compute_time_per_tile_us
            )
        else:
            return self.exhaustive_search(
                total_tiles, elements_per_tile, compute_time_per_tile_us
            )

    # ========================================
    # Config Persistence
    # ========================================

    def _get_config_key(
        self,
        total_tiles: int,
        elements_per_tile: int
    ) -> str:
        """Generate config key for caching."""
        return f"tiles{total_tiles}_elem{elements_per_tile}_ws{self.world_size}"

    def save_config(
        self,
        total_tiles: int,
        elements_per_tile: int,
        config: TuningConfig
    ):
        """Save configuration for a specific problem size."""
        key = self._get_config_key(total_tiles, elements_per_tile)
        self.configs[key] = config
        self._save_configs()

    def load_config(
        self,
        total_tiles: int,
        elements_per_tile: int
    ) -> Optional[TuningConfig]:
        """Load configuration for a specific problem size."""
        key = self._get_config_key(total_tiles, elements_per_tile)
        return self.configs.get(key)

    def _save_configs(self):
        """Save all configurations to disk."""
        config_file = self.config_dir / "overlap_configs.json"
        data = {k: asdict(v) for k, v in self.configs.items()}
        with open(config_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_configs(self):
        """Load configurations from disk."""
        config_file = self.config_dir / "overlap_configs.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                self.configs = {
                    k: TuningConfig(**v) for k, v in data.items()
                }
            except Exception as e:
                print(f"Warning: Failed to load configs: {e}")

    def _save_bandwidth_curve(self):
        """Save bandwidth curve to disk."""
        bw_file = self.config_dir / "bandwidth_curve.json"
        data = [asdict(pt) for pt in self.bandwidth_curve]
        with open(bw_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_bandwidth_curve(self) -> bool:
        """Load bandwidth curve from disk."""
        bw_file = self.config_dir / "bandwidth_curve.json"
        if bw_file.exists():
            try:
                with open(bw_file, 'r') as f:
                    data = json.load(f)
                self.bandwidth_curve = [BandwidthPoint(**pt) for pt in data]
                return True
            except Exception as e:
                print(f"Warning: Failed to load bandwidth curve: {e}")
        return False


def tune_overlap_config(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    rank: int = 0,
    world_size: int = 1,
    use_hints: bool = True,
    use_predictive: bool = True
) -> Tuple[TuningConfig, Optional[torch.Tensor]]:
    """
    Convenience function to tune overlap configuration.

    Returns:
        config: Optimal TuningConfig
        reorder_array: Tile reorder array (if use_hints=True)
    """
    tuner = OverlapTuner(rank=rank, world_size=world_size)

    # Try to load bandwidth curve
    if not tuner.load_bandwidth_curve():
        # Profile if not available
        print("Profiling bandwidth curve...")
        tuner.profile_bandwidth()

    # Calculate problem dimensions
    total_tokens = input_tensor.size(0)
    K = input_tensor.size(1)
    N = weight_tensor.size(2)
    num_experts = tokens_per_expert.size(0)

    TILE_M, TILE_N = 128, 128
    total_tiles = 0
    for e in range(num_experts):
        tokens = tokens_per_expert[e].item()
        tiles_m = (tokens + TILE_M - 1) // TILE_M
        tiles_n = (N + TILE_N - 1) // TILE_N
        total_tiles += tiles_m * tiles_n

    elements_per_tile = TILE_M * TILE_N

    # Collect hints and compute reorder array
    reorder_array = None
    if use_hints:
        hints = tuner.collect_hints(input_tensor, weight_tensor, tokens_per_expert)
        config = tuner.search(total_tiles, elements_per_tile, use_predictive=use_predictive)
        reorder_array = tuner.compute_reorder_array(hints, config.tiles_per_wave)
    else:
        config = tuner.search(total_tiles, elements_per_tile, use_predictive=use_predictive)

    # Save config
    tuner.save_config(total_tiles, elements_per_tile, config)

    return config, reorder_array
