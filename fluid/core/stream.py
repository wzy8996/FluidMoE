"""
Unified CUDA Stream Management for FluidMoE

This module provides a global StreamManager singleton that manages all CUDA streams
and events used by both forward (P2P overlap) and backward (dW-AllToAll overlap) passes.

Design Principles:
1. Single source of truth for all CUDA streams
2. Unified synchronization point at iteration boundaries
3. Clear lifecycle management for streams and events

Usage:
    from fluid.core.stream import get_stream_manager

    manager = get_stream_manager()
    comm_stream = manager.comm_stream

    # At end of iteration:
    manager.sync_all()
"""

import torch
import threading
from typing import Optional, Dict


class StreamManager:
    """
    Global CUDA stream manager singleton.

    Manages:
    - comm_stream: For communication operations (P2P, AllToAll)
    - Reusable CUDA events to avoid allocation overhead

    Thread-safe singleton pattern ensures only one instance per process.
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if StreamManager._initialized:
            return

        with StreamManager._lock:
            if StreamManager._initialized:
                return

            self._device = None
            self._comm_stream: Optional[torch.cuda.Stream] = None

            # Data ready event (for signaling data preparation complete in forward)
            self._data_ready_event = None

            # AllToAll completion event (for backward)
            self._alltoall_end_event = None

            # Compute sync event (for backward chunked operations)
            self._compute_sync_event = None

            # Track if streams have been lazily initialized
            self._streams_initialized = False

            StreamManager._initialized = True

    def _ensure_initialized(self, device: Optional[torch.device] = None):
        """Lazily initialize streams on first use."""
        if self._streams_initialized:
            return

        with self._lock:
            if self._streams_initialized:
                return

            # Determine device
            if device is not None:
                self._device = device
            elif torch.cuda.is_available():
                self._device = torch.device(f'cuda:{torch.cuda.current_device()}')
            else:
                raise RuntimeError("CUDA is not available")

            # Create communication stream
            self._comm_stream = torch.cuda.Stream(device=self._device)

            # Create reusable events
            self._data_ready_event = torch.cuda.Event()
            self._alltoall_end_event = torch.cuda.Event()
            self._compute_sync_event = torch.cuda.Event()

            self._streams_initialized = True

    @property
    def device(self) -> torch.device:
        """Get the CUDA device."""
        self._ensure_initialized()
        return self._device

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        """Get the communication stream for P2P and AllToAll operations."""
        self._ensure_initialized()
        return self._comm_stream

    @property
    def data_ready_event(self) -> torch.cuda.Event:
        """Get the data ready event (for forward compute-comm sync)."""
        self._ensure_initialized()
        return self._data_ready_event

    @property
    def alltoall_end_event(self) -> torch.cuda.Event:
        """Get the AllToAll completion event (for backward)."""
        self._ensure_initialized()
        return self._alltoall_end_event

    @property
    def compute_sync_event(self) -> torch.cuda.Event:
        """Get the compute sync event (for backward chunked operations)."""
        self._ensure_initialized()
        return self._compute_sync_event

    def record_alltoall_end(self, stream: Optional[torch.cuda.Stream] = None) -> torch.cuda.Event:
        """
        Record AllToAll completion event.

        Args:
            stream: Stream to record on (default: comm_stream)

        Returns:
            The recorded event
        """
        self._ensure_initialized()
        if stream is None:
            stream = self._comm_stream
        self._alltoall_end_event.record(stream)
        return self._alltoall_end_event

    def sync_all(self):
        """
        Synchronize the communication stream.

        This should be called at iteration boundaries (after backward, before next forward)
        to ensure all communication work is complete.
        """
        if not self._streams_initialized:
            return

        self._comm_stream.synchronize()

    def sync_to_current(self, device: Optional[torch.device] = None):
        """
        Make current stream wait for comm stream.

        This is a lighter-weight alternative to sync_all() that doesn't block CPU.
        """
        if not self._streams_initialized:
            return

        if device is None:
            device = self._device

        current = torch.cuda.current_stream(device)
        current.wait_stream(self._comm_stream)

    def initialize(self, device: torch.device):
        """
        Explicitly initialize streams on a specific device.

        Args:
            device: CUDA device to create streams on
        """
        if self._streams_initialized:
            # Already initialized, check if device matches
            if self._device != device:
                raise RuntimeError(
                    f"StreamManager already initialized on {self._device}, "
                    f"cannot reinitialize on {device}"
                )
            return

        self._ensure_initialized(device)

    @classmethod
    def reset(cls):
        """
        Reset the singleton instance.

        WARNING: This should only be used in testing or when completely
        reinitializing the system.
        """
        with cls._lock:
            if cls._instance is not None:
                # Sync streams before destroying
                if cls._instance._streams_initialized:
                    cls._instance._comm_stream.synchronize()

                cls._instance._comm_stream = None
                cls._instance._data_ready_event = None
                cls._instance._alltoall_end_event = None
                cls._instance._compute_sync_event = None
                cls._instance._streams_initialized = False
                cls._instance._device = None

            cls._instance = None
            cls._initialized = False


# Global accessor function
def get_stream_manager() -> StreamManager:
    """Get the global StreamManager singleton instance."""
    return StreamManager()


__all__ = [
    'StreamManager',
    'get_stream_manager',
]
