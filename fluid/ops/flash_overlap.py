"""
FlashOverlap API - High-level interface for FC2+AllToAll overlap.

This module provides an easy-to-use API that automatically loads
pre-tuned configurations for optimal performance.

Usage:
    from fluid.ops.flash_overlap import FlashOverlapFC2

    # Create instance (auto-loads tuned config if available)
    flash_fc2 = FlashOverlapFC2(
        num_experts=8,
        hidden_size=4096,
        intermediate_size=14336,
        rank=rank,
        world_size=world_size
    )

    # Execute overlapped FC2 + AllToAll
    output = flash_fc2(fc2_input, fc2_weight, tokens_per_expert)
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.distributed as dist

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from tuning import load_tuning_config
except ImportError:
    load_tuning_config = None


class FlashOverlapFC2:
    """
    High-level API for FlashOverlap FC2+AllToAll.

    Automatically loads pre-tuned configurations when available,
    otherwise uses reasonable defaults.
    """

    # Default configurations based on model size
    DEFAULT_CONFIGS = {
        'small': {'tiles_per_wave': 32, 'waves_per_comm': 1},   # hidden <= 2048
        'medium': {'tiles_per_wave': 64, 'waves_per_comm': 1},  # hidden <= 4096
        'large': {'tiles_per_wave': 64, 'waves_per_comm': 2},   # hidden > 4096
    }

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        world_size: int,
        tiles_per_wave: Optional[int] = None,
        waves_per_comm: Optional[int] = None,
        config_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize FlashOverlap FC2.

        Args:
            num_experts: Number of experts
            hidden_size: Hidden dimension (output of FC2)
            intermediate_size: Intermediate dimension (input of FC2)
            rank: Current process rank
            world_size: Total number of processes
            tiles_per_wave: Override tiles per wave (auto if None)
            waves_per_comm: Override waves per comm (auto if None)
            config_dir: Directory to load tuned configs from
            verbose: Print configuration info
        """
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose

        # Import fluid_kernels
        sys.path.insert(0, str(Path(__file__).parent))
        import fluid_kernels
        self.fluid_kernels = fluid_kernels

        # Determine tiles_per_wave and waves_per_comm
        self.tiles_per_wave, self.waves_per_comm = self._get_config(
            tiles_per_wave, waves_per_comm, config_dir
        )

        # NCCL initialization state
        self._nccl_initialized = False
        self._context_initialized = False

        if verbose and rank == 0:
            print(f"[FlashOverlapFC2] Initialized:")
            print(f"  tiles_per_wave={self.tiles_per_wave}")
            print(f"  waves_per_comm={self.waves_per_comm}")

    def _get_config(
        self,
        tiles_per_wave: Optional[int],
        waves_per_comm: Optional[int],
        config_dir: Optional[str]
    ) -> Tuple[int, int]:
        """Get configuration, trying tuned config first."""

        # If both specified, use them
        if tiles_per_wave is not None and waves_per_comm is not None:
            return tiles_per_wave, waves_per_comm

        # Try to load tuned config
        if load_tuning_config is not None:
            # We don't know tokens_per_expert yet, use 256 as default for lookup
            tuned = load_tuning_config(
                self.num_experts, 256,
                self.hidden_size, self.intermediate_size,
                config_dir
            )
            if tuned is not None:
                tpw, wpc = tuned
                if self.verbose and self.rank == 0:
                    print(f"[FlashOverlapFC2] Loaded tuned config: tpw={tpw}, wpc={wpc}")
                return (
                    tiles_per_wave if tiles_per_wave is not None else tpw,
                    waves_per_comm if waves_per_comm is not None else wpc
                )

        # Fall back to defaults based on model size
        if self.hidden_size <= 2048:
            cfg = self.DEFAULT_CONFIGS['small']
        elif self.hidden_size <= 4096:
            cfg = self.DEFAULT_CONFIGS['medium']
        else:
            cfg = self.DEFAULT_CONFIGS['large']

        return (
            tiles_per_wave if tiles_per_wave is not None else cfg['tiles_per_wave'],
            waves_per_comm if waves_per_comm is not None else cfg['waves_per_comm']
        )

    def _ensure_nccl_initialized(self, device: torch.device):
        """Initialize NCCL communicator if not already done."""
        if self._nccl_initialized:
            return

        if self.rank == 0:
            nccl_id = self.fluid_kernels.get_nccl_unique_id()
            nccl_id_tensor = torch.tensor(nccl_id, dtype=torch.int64, device=device)
        else:
            nccl_id_tensor = torch.zeros(17, dtype=torch.int64, device=device)

        dist.broadcast(nccl_id_tensor, src=0)
        self.fluid_kernels.init_nccl_comm_with_id(
            nccl_id_tensor.tolist(), self.rank, self.world_size
        )
        self._nccl_initialized = True

    def forward(
        self,
        fc2_input: torch.Tensor,
        fc2_weight: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        recv_buffer: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Execute FC2 + AllToAll with overlap.

        Args:
            fc2_input: Input tensor [total_tokens, intermediate_size]
            fc2_weight: Weight tensor [num_experts, intermediate_size, hidden_size]
            tokens_per_expert: Token counts per expert [num_experts]
            recv_buffer: Optional pre-allocated receive buffer

        Returns:
            Output tensor after AllToAll [total_tokens, hidden_size]
        """
        device = fc2_input.device
        total_tokens = fc2_input.size(0)

        # Allocate output
        fc2_output = torch.zeros(
            total_tokens, self.hidden_size,
            device=device, dtype=fc2_input.dtype
        )

        # Allocate recv buffer if not provided
        if recv_buffer is None:
            recv_buffer = torch.zeros_like(fc2_output)

        # Initialize context if needed
        self.fluid_kernels.grouped_gemm_epilogue_signal(
            fc2_input, fc2_weight, fc2_output,
            tokens_per_expert, self.tiles_per_wave
        )

        # Initialize NCCL if needed
        self._ensure_nccl_initialized(device)

        # Queue overlap
        self.fluid_kernels.queue_flash_overlap_alltoall_single(
            fc2_output.data_ptr(),
            recv_buffer.data_ptr(),
            self.waves_per_comm
        )

        # Wait for completion
        self.fluid_kernels.sync_comm_to_compute()

        return recv_buffer

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def cleanup(self):
        """Clean up resources."""
        self.fluid_kernels.destroy_epilogue_signal_context()

