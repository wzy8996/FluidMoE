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
    compute_stream = manager.compute_stream
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

    Manages two primary streams:
    - compute_stream: For compute operations (FC1, FC2, dW, etc.)
    - comm_stream: For communication operations (P2P, AllToAll)

    Also manages reusable CUDA events to avoid allocation overhead.

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
            self._compute_stream: Optional[torch.cuda.Stream] = None
            self._comm_stream: Optional[torch.cuda.Stream] = None

            # Reusable ping-pong events for P2P synchronization
            self._p2p_events = [None, None]

            # Reusable events for compute-comm synchronization
            self._compute_events = [None, None]

            # Data ready event (for signaling data preparation complete)
            self._data_ready_event = None

            # AllToAll completion event
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

            # Create streams
            self._compute_stream = torch.cuda.Stream(device=self._device)
            self._comm_stream = torch.cuda.Stream(device=self._device)

            # Create reusable events
            self._p2p_events = [
                torch.cuda.Event(),
                torch.cuda.Event()
            ]
            self._compute_events = [
                torch.cuda.Event(),
                torch.cuda.Event()
            ]
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
    def compute_stream(self) -> torch.cuda.Stream:
        """Get the compute stream for FC1, FC2, dW operations."""
        self._ensure_initialized()
        return self._compute_stream

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        """Get the communication stream for P2P and AllToAll operations."""
        self._ensure_initialized()
        return self._comm_stream

    @property
    def p2p_events(self):
        """Get ping-pong events for P2P synchronization."""
        self._ensure_initialized()
        return self._p2p_events

    @property
    def compute_events(self):
        """Get ping-pong events for compute synchronization."""
        self._ensure_initialized()
        return self._compute_events

    @property
    def data_ready_event(self) -> torch.cuda.Event:
        """Get the data ready event."""
        self._ensure_initialized()
        return self._data_ready_event

    @property
    def alltoall_end_event(self) -> torch.cuda.Event:
        """Get the AllToAll completion event."""
        self._ensure_initialized()
        return self._alltoall_end_event

    @property
    def compute_sync_event(self) -> torch.cuda.Event:
        """Get the compute sync event for chunked backward."""
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
        Synchronize all managed streams.

        This should be called at iteration boundaries (after backward, before next forward)
        to ensure all work is complete.
        """
        if not self._streams_initialized:
            return

        self._compute_stream.synchronize()
        self._comm_stream.synchronize()

    def sync_to_current(self, device: Optional[torch.device] = None):
        """
        Make current stream wait for both compute and comm streams.

        This is a lighter-weight alternative to sync_all() that doesn't block CPU.
        """
        if not self._streams_initialized:
            return

        if device is None:
            device = self._device

        current = torch.cuda.current_stream(device)
        current.wait_stream(self._compute_stream)
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
                    cls._instance._compute_stream.synchronize()
                    cls._instance._comm_stream.synchronize()

                cls._instance._compute_stream = None
                cls._instance._comm_stream = None
                cls._instance._p2p_events = [None, None]
                cls._instance._compute_events = [None, None]
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
