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
        """Initialize NVSHMEM (called once by NVSHMEMBackend).

        Bootstrap selection:
          * If ``NVSHMEM_BOOTSTRAP`` is set in the env, honor it and use the
            env-driven init path (PMI/MPI/SHMEM). Required for srun --mpi=pmix
            and similar PMI-aware launchers.
          * Otherwise, if ``torch.distributed`` is initialized (the typical
            torchrun deployment), use **UID bootstrap**: rank 0 mints a
            ``nvshmemx_uniqueid_t``, broadcasts it via the WORLD process
            group, and every PE inits with ``NVSHMEMX_INIT_WITH_UNIQUEID``.
            This avoids the silent NVSHMEM init failure that previously
            forced FluidMoE back to NCCL on every torchrun launch.
          * As a last resort, fall back to the env-driven path with no
            bootstrap explicitly chosen — this will fail loudly if no
            launcher provided one, which is what we want.
        """
        if self._nvshmem_initialized:
            return
        with self._lock:
            if self._nvshmem_initialized:
                return
            import os
            import torch
            import torch.distributed as dist

            env_bootstrap = os.environ.get("NVSHMEM_BOOTSTRAP")
            from fluid.csrc import (
                nvshmem_init,
                nvshmem_init_with_uniqueid,
                nvshmem_get_uniqueid,
                nvshmem_uniqueid_size,
            )

            if env_bootstrap is None and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                uid_size = int(nvshmem_uniqueid_size())

                # Broadcast the UID over the existing torch.distributed WORLD
                # group as a uint8 byte tensor. Use a CUDA tensor when the
                # default backend is NCCL; gloo handles CPU tensors natively.
                # We pick the device by checking the current CUDA device,
                # which both NCCL and gloo paths can produce a tensor for.
                pg = dist.group.WORLD
                backend = dist.get_backend(pg)
                if backend == "nccl":
                    uid_dev = torch.device(f"cuda:{torch.cuda.current_device()}")
                else:
                    uid_dev = torch.device("cpu")
                uid_buf = torch.zeros(uid_size, dtype=torch.uint8, device=uid_dev)

                if rank == 0:
                    uid_bytes = nvshmem_get_uniqueid()
                    assert len(uid_bytes) == uid_size, \
                        f"uniqueid size mismatch: {len(uid_bytes)} vs {uid_size}"
                    uid_buf.copy_(torch.frombuffer(
                        bytearray(uid_bytes), dtype=torch.uint8).to(uid_dev))
                dist.broadcast(uid_buf, src=0)
                uid_bytes = bytes(uid_buf.cpu().numpy().tobytes())

                nvshmem_init_with_uniqueid(rank, world_size, uid_bytes)
            else:
                # Env-driven path. If env_bootstrap is None we leave it unset
                # and let NVSHMEM's default selection logic surface a clear
                # error rather than guessing a wrong default.
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
