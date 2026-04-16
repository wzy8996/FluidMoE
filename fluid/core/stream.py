"""
Unified CUDA Stream Management for FluidMoE

Provides a global StreamManager singleton that manages CUDA streams
and events used by both forward (P2P overlap) and backward (dW-A2A overlap).

Usage:
    from fluid.core.stream import get_stream_manager
    manager = get_stream_manager()
    comm_stream = manager.comm_stream
"""

import torch
import threading
from typing import Optional


class StreamManager:
    """Global CUDA stream manager singleton.

    Manages:
    - comm_stream: For communication operations (P2P, AllToAll, AllReduce)
    - data_ready_event: For forward compute-comm synchronization
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
        if StreamManager._initialized:
            return
        with StreamManager._lock:
            if StreamManager._initialized:
                return
            self._device = None
            self._comm_stream: Optional[torch.cuda.Stream] = None
            self._ar_stream: Optional[torch.cuda.Stream] = None
            self._data_ready_event = None
            self._sync_event_pool = {}
            self._streams_initialized = False
            self._nvshmem_initialized = False
            StreamManager._initialized = True

    def _ensure_initialized(self, device: Optional[torch.device] = None):
        """Lazily initialize streams on first use."""
        if self._streams_initialized:
            return
        with self._lock:
            if self._streams_initialized:
                return
            if device is not None:
                self._device = device
            elif torch.cuda.is_available():
                self._device = torch.device(f'cuda:{torch.cuda.current_device()}')
            else:
                raise RuntimeError("CUDA is not available")
            self._comm_stream = torch.cuda.Stream(device=self._device)
            self._ar_stream = torch.cuda.Stream(device=self._device)
            self._data_ready_event = torch.cuda.Event()
            self._streams_initialized = True

    @property
    def device(self) -> torch.device:
        self._ensure_initialized()
        return self._device

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        self._ensure_initialized()
        return self._comm_stream

    @property
    def ar_stream(self) -> torch.cuda.Stream:
        self._ensure_initialized()
        return self._ar_stream

    @property
    def data_ready_event(self) -> torch.cuda.Event:
        self._ensure_initialized()
        return self._data_ready_event

    def get_sync_event(self, key) -> torch.cuda.Event:
        """Get a reusable non-timing CUDA event keyed by a stable identifier."""
        self._ensure_initialized()
        evt = self._sync_event_pool.get(key)
        if evt is None:
            evt = torch.cuda.Event()
            self._sync_event_pool[key] = evt
        return evt

    def initialize(self, device: torch.device):
        """Explicitly initialize streams on a specific device."""
        if self._streams_initialized:
            if self._device != device:
                raise RuntimeError(
                    f"StreamManager already initialized on {self._device}, "
                    f"cannot reinitialize on {device}"
                )
            return
        self._ensure_initialized(device)

    def init_nvshmem(self):
        """Initialize NVSHMEM (called once by NVSHMEMBackend)."""
        if self._nvshmem_initialized:
            return
        with self._lock:
            if self._nvshmem_initialized:
                return
            import os
            if "NVSHMEM_BOOTSTRAP" not in os.environ:
                os.environ["NVSHMEM_BOOTSTRAP"] = "pmi"
            from fluid.csrc import nvshmem_init
            nvshmem_init()
            self._nvshmem_initialized = True
            import atexit
            atexit.register(self._finalize_nvshmem)

    def _finalize_nvshmem(self):
        if self._nvshmem_initialized:
            try:
                from fluid.csrc import nvshmem_finalize
                nvshmem_finalize()
            except Exception:
                pass
            self._nvshmem_initialized = False

    @property
    def nvshmem_initialized(self) -> bool:
        return self._nvshmem_initialized


def get_stream_manager() -> StreamManager:
    """Get the global StreamManager singleton instance."""
    return StreamManager()


__all__ = ['StreamManager', 'get_stream_manager']
