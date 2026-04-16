"""
P2P Communication Backend Abstraction

Provides a pluggable backend for forward P2P communication:
- NCCLBackend: wraps existing dist.batch_isend_irecv (fallback, always available)
- NVSHMEMBackend: one-sided writes via NVSHMEM (stream-ordered, events work)

Selection via FLUIDMOE_P2P_BACKEND env var: "nvshmem" | "nccl" | "auto" (default).

Backward AllToAll/AllReduce stays on NCCL regardless of this setting.
"""

import os
import threading
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Dict

import torch
import torch.distributed as dist


class P2PBackend(ABC):
    """Abstract base for forward P2P communication."""

    @abstractmethod
    def exchange(
        self,
        send_buf: Optional[torch.Tensor],
        recv_buf: Optional[torch.Tensor],
        partner_global_rank: int,
        partner_local_rank: int,
        group: dist.ProcessGroup,
        comm_stream: torch.cuda.Stream,
        event: Optional[torch.cuda.Event],
    ) -> None:
        """Stream-ordered send+recv with partner, optionally record event."""

    @abstractmethod
    def alloc_recv_buffer(
        self, tag: str, numel: int, dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor:
        """Allocate a recv buffer (symmetric heap for NVSHMEM, regular for NCCL)."""

    @abstractmethod
    def needs_final_wait(self) -> bool:
        """Whether the caller must call final_wait() after all rounds."""

    @abstractmethod
    def final_wait(self) -> None:
        """Flush pending NCCL requests for allocator safety (NCCL only)."""

    def clear_iteration(self) -> None:
        """Reset per-iteration state (called between forward passes)."""

    def finalize(self) -> None:
        """Release resources."""


# =========================================================================
# NCCL Backend (wraps existing code, zero behavior change)
# =========================================================================

_NCCL_RECV_BUF_CACHE: Dict[tuple, torch.Tensor] = {}


class NCCLBackend(P2PBackend):
    """NCCL P2P backend — identical behavior to the original inline code."""

    def __init__(self):
        self._all_reqs = []

    def exchange(self, send_buf, recv_buf, partner_global_rank, partner_local_rank,
                 group, comm_stream, event):
        p2p_ops = []
        if recv_buf is not None:
            p2p_ops.append(
                dist.P2POp(dist.irecv, recv_buf, partner_global_rank, group=group))
        if send_buf is not None:
            send_buf.record_stream(comm_stream)
            p2p_ops.append(
                dist.P2POp(dist.isend, send_buf, partner_global_rank, group=group))
        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            self._all_reqs.extend(reqs)
            if reqs:
                reqs[-1].wait()
        if event is not None:
            event.record(comm_stream)

    def alloc_recv_buffer(self, tag, numel, dtype, device):
        key = (tag, str(dtype), device.type,
               device.index if device.index is not None else -1)
        buf = _NCCL_RECV_BUF_CACHE.get(key)
        if buf is None or buf.numel() < numel:
            buf = torch.empty(max(numel, 1), dtype=dtype, device=device)
            _NCCL_RECV_BUF_CACHE[key] = buf
        return buf[:numel]

    def needs_final_wait(self) -> bool:
        return True

    def final_wait(self):
        if self._all_reqs:
            self._all_reqs[-1].wait()

    def clear_iteration(self):
        self._all_reqs.clear()


# =========================================================================
# NVSHMEM Backend
# =========================================================================

class NVSHMEMBackend(P2PBackend):
    """NVSHMEM one-sided P2P backend.

    Uses nvshmemx_putmem_signal_on_stream (stream-ordered host-side API)
    so CUDA events correctly capture completion — no blocking wait needed.

    Recv buffers MUST be allocated from NVSHMEM symmetric heap (remote PE
    writes directly into them via NVLink). Send buffers stay in regular
    CUDA memory. Use alloc_recv_buffer() for all recv buffers.
    """

    def __init__(self):
        from fluid.core.stream import get_stream_manager
        sm = get_stream_manager()
        sm.init_nvshmem()

        import fluid.csrc as _ops
        self._ops = _ops

        self._my_pe = _ops.nvshmem_my_pe()
        self._n_pes = _ops.nvshmem_n_pes()
        self._SIGNAL_SET = _ops.SIGNAL_SET()
        self._CMP_GE = _ops.CMP_GE()
        self._INT64_SIZE = 8

        # Per-group PE mapping: group_id -> {group_rank: nvshmem_pe}
        self._pe_map: Dict[int, dict] = {}

        # Signal array: ONE shared symmetric tensor (uint64[n_pes]) across all groups.
        # signal_array[pe_i] = last signal value written by PE_i to this PE.
        # Using one array avoids cross-group counter collision — each
        # (sender_pe, receiver_pe) pair has exactly one signal slot regardless
        # of which group the exchange belongs to.
        self._signal_array: Optional[torch.Tensor] = None

        # Cached remote signal base pointers: {pe: int64 addr}
        self._remote_signal_ptrs: Dict[int, int] = {}

        # Per-partner monotonic signal counter (keyed by partner PE only,
        # shared across groups — the signal slot is the same physical location)
        self._signal_counter: Dict[int, int] = {}

        # Symmetric recv buffer cache: tag -> tensor
        self._sym_buf_cache: Dict[str, torch.Tensor] = {}

    def _ensure_init(self):
        """Lazily allocate the global signal array (once)."""
        if self._signal_array is not None:
            return
        self._signal_array = self._ops.nvshmem_malloc_tensor(
            self._n_pes, torch.int64)

    def _ensure_group(self, group: dist.ProcessGroup):
        """Lazily build PE mapping for a process group."""
        gid = id(group)
        if gid in self._pe_map:
            return
        pe_list = [None] * group.size()
        dist.all_gather_object(pe_list, self._my_pe, group=group)
        self._pe_map[gid] = {rank: pe for rank, pe in enumerate(pe_list)}

    def _get_remote_signal_base(self, partner_pe: int) -> int:
        """Get base address of signal_array on partner PE (cached)."""
        ptr = self._remote_signal_ptrs.get(partner_pe)
        if ptr is None:
            ptr = self._ops.nvshmem_ptr(self._signal_array, partner_pe)
            if ptr == 0:
                raise RuntimeError(
                    f"nvshmem_ptr returned NULL for PE {partner_pe}. "
                    f"Peer may not be on the same node or NVLink unavailable. "
                    f"Fall back to NCCL with FLUIDMOE_P2P_BACKEND=nccl.")
            self._remote_signal_ptrs[partner_pe] = ptr
        return ptr

    def exchange(self, send_buf, recv_buf, partner_global_rank, partner_local_rank,
                 group, comm_stream, event):
        self._ensure_init()
        self._ensure_group(group)

        gid = id(group)
        partner_pe = self._pe_map[gid][partner_local_rank]
        my_pe = self._my_pe

        # Monotonic signal counter per partner PE (shared across groups —
        # one physical signal slot per PE pair)
        self._signal_counter[partner_pe] = self._signal_counter.get(partner_pe, 0) + 1
        sig_val = self._signal_counter[partner_pe]

        stream_ptr = comm_stream.cuda_stream

        if send_buf is not None:
            send_buf.record_stream(comm_stream)
            nbytes = send_buf.numel() * send_buf.element_size()

            # Remote signal address: partner's signal_array[my_pe]
            remote_sig_base = self._get_remote_signal_base(partner_pe)
            remote_sig_addr = remote_sig_base + my_pe * self._INT64_SIZE

            # Remote recv buffer: partner's recv_buf at same symmetric offset
            remote_recv_ptr = self._ops.nvshmem_ptr(recv_buf, partner_pe)
            if remote_recv_ptr == 0:
                raise RuntimeError(
                    f"nvshmem_ptr returned NULL for recv_buf on PE {partner_pe}. "
                    f"recv_buf must be allocated from symmetric heap via "
                    f"alloc_recv_buffer().")

            self._ops.putmem_signal_on_stream(
                remote_recv_ptr,
                send_buf,
                nbytes,
                remote_sig_addr,
                sig_val,
                self._SIGNAL_SET,
                partner_pe,
                stream_ptr,
            )

        if recv_buf is not None:
            # Wait for partner's write: partner sets our signal_array[partner_pe]
            local_sig_addr = (self._signal_array.data_ptr()
                              + partner_pe * self._INT64_SIZE)

            self._ops.signal_wait_until_on_stream(
                local_sig_addr,
                self._CMP_GE,
                sig_val,
                stream_ptr,
            )

        if event is not None:
            event.record(comm_stream)

    def alloc_recv_buffer(self, tag, numel, dtype, device):
        """Allocate recv buffer from symmetric heap (grow-only cache).

        Returned tensor is in symmetric heap — nvshmem_ptr() works on it.
        Slices/views of this tensor are also valid symmetric addresses.
        """
        if tag in self._sym_buf_cache:
            buf = self._sym_buf_cache[tag]
            if buf.numel() >= numel:
                return buf[:numel]
        buf = self._ops.nvshmem_malloc_tensor(max(numel, 1), dtype)
        self._sym_buf_cache[tag] = buf
        return buf[:numel]

    def needs_final_wait(self) -> bool:
        return False

    def final_wait(self):
        pass

    def clear_iteration(self):
        pass

    def finalize(self):
        for buf in self._sym_buf_cache.values():
            self._ops.nvshmem_free_tensor(buf)
        self._sym_buf_cache.clear()
        if self._signal_array is not None:
            self._ops.nvshmem_free_tensor(self._signal_array)
            self._signal_array = None


# =========================================================================
# Singleton factory
# =========================================================================

_backend_instance: Optional[P2PBackend] = None
_backend_lock = threading.Lock()


def _has_nvlink() -> bool:
    """Detect whether any GPU pair has NVLink P2P access."""
    if not torch.cuda.is_available():
        return False
    n = torch.cuda.device_count()
    if n < 2:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if torch.cuda.can_device_access_peer(i, j):
                return True
    return False


def get_p2p_backend() -> P2PBackend:
    """Get the process-wide P2P backend singleton.

    Controlled by FLUIDMOE_P2P_BACKEND env var:
      "nccl"    — always use NCCL
      "nvshmem" — use NVSHMEM, error if unavailable
      "auto"    — (default) check NVLink + NVSHMEM, fall back to NCCL
    """
    global _backend_instance
    if _backend_instance is not None:
        return _backend_instance
    with _backend_lock:
        if _backend_instance is not None:
            return _backend_instance

        choice = os.environ.get("FLUIDMOE_P2P_BACKEND", "auto").lower()
        rank = dist.get_rank() if dist.is_initialized() else 0

        if choice == "nccl":
            _backend_instance = NCCLBackend()
        elif choice == "nvshmem":
            _backend_instance = NVSHMEMBackend()
        elif choice == "auto":
            nvlink = _has_nvlink()
            if nvlink:
                try:
                    _backend_instance = NVSHMEMBackend()
                    if rank == 0:
                        print("[FluidMoE] P2P backend: NVSHMEM "
                              "(NVLink detected, NVSHMEM available)")
                except (ImportError, RuntimeError, NotImplementedError):
                    _backend_instance = NCCLBackend()
                    if rank == 0:
                        warnings.warn(
                            "[FluidMoE] NVLink detected but NVSHMEM unavailable, "
                            "using NCCL P2P. Install NVSHMEM for optimal bandwidth.")
            else:
                _backend_instance = NCCLBackend()
                if rank == 0:
                    print("[FluidMoE] P2P backend: NCCL (no NVLink detected)")
        else:
            raise ValueError(
                f"Unknown FLUIDMOE_P2P_BACKEND={choice!r}. "
                f"Expected: nvshmem, nccl, or auto")

        return _backend_instance


def reset_p2p_backend():
    """Reset the singleton (for testing)."""
    global _backend_instance
    with _backend_lock:
        if _backend_instance is not None:
            _backend_instance.finalize()
        _backend_instance = None


__all__ = ['P2PBackend', 'NCCLBackend', 'NVSHMEMBackend',
           'get_p2p_backend', 'reset_p2p_backend']
