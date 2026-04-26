"""
P2P Communication Backend Abstraction
=====================================

Pluggable backend for forward P2P communication:
- ``NCCLBackend``: wraps ``dist.batch_isend_irecv`` (always available).
- ``NVSHMEMBackend``: one-sided writes via NVSHMEM ``putmem_signal_on_stream``
  (stream-ordered; CUDA events correctly capture completion; no host blocking
  wait).

Selection via ``FLUIDMOE_P2P_BACKEND`` env var: ``nvshmem`` | ``nccl`` | ``auto``
(default). Backward AllToAll/AllReduce stays on NCCL regardless of this setting.

NVSHMEM signal protocol
-----------------------
Each ``ProcessGroup`` instance gets its OWN symmetric ``int64[n_pes]`` signal
array, allocated lazily on first use. The slot at index ``sender_pe`` on the
receiver PE is written by the sender to a strictly increasing per-(group,
partner) counter. This is critical: with a single global signal array shared
across groups, an exchange in ``cp_group`` could be falsely satisfied by a
signal write from an exchange in ``ep_group`` whose data has not yet been
delivered to the cp_group recv buffer, since both groups' (sender, receiver)
pairs would collide on the same signal slot.

Sender-side completion
----------------------
``putmem_signal_on_stream`` returns when the put has been **submitted** on the
local stream. For NVLink-only paths the submission already encompasses the
copy; for RDMA transports (cross-node IB) the local stream completes while
the NIC may still be draining. To guarantee that a persistent ``send_buf``
slot can be safely reused on the next iteration, we call
``nvshmemx_quiet_on_stream`` in ``clear_iteration()`` and from ``final_wait()``.
On NVLink this is essentially free; on IB it is required for correctness.

Symmetric heap lifetime
-----------------------
Tensors returned by ``nvshmem_malloc_tensor`` are non-owning (``from_blob``
without a deleter — see ``fluid/csrc/nvshmem_ops.cpp``). On grow we explicitly
call ``nvshmem_free_tensor`` on the master tensor before reallocating. Both
free and malloc are collectives in NVSHMEM_TEAM_WORLD, so all PEs must invoke
``alloc_recv_buffer(tag, numel, ...)`` with identical ``numel`` — which holds
in production because the size is determined by globally consistent metadata.
"""

import os
import threading
import warnings
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple

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
        """Whether the caller must call ``final_wait()`` after all rounds."""

    @abstractmethod
    def final_wait(self) -> None:
        """Drain pending comm at end of forward (NCCL: wait reqs; NVSHMEM: quiet)."""

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

    Per-group signal array isolation, monotonic per-(group, partner) counters,
    and stream-ordered ``quiet`` on iteration boundary.

    Recv buffers MUST be allocated from the NVSHMEM symmetric heap via
    ``alloc_recv_buffer`` (the producer PE writes directly into them via NVLink
    or RDMA). Send buffers stay in regular CUDA memory.

    Cross-node correctness
    ----------------------
    All addresses passed to ``putmem_signal_on_stream`` (``dest`` and
    ``sig_addr``) are **LOCAL symmetric heap pointers**. The NVSHMEM runtime
    translates them to the partner PE based on the symmetric heap layout and
    routes via NVLink or RDMA depending on reachability. The previous version
    of this backend resolved the partner pointer with ``nvshmem_ptr(buf, pe)``,
    which only returns non-NULL when the partner is directly LD/ST-addressable
    (NVLink P2P on the same node) — that path raised on every cross-node
    exchange. Using local symmetric pointers is the documented contract and
    works on both NVLink and IB/RoCE.
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

        # Per-group PE mapping: group_id -> [partner_pe for partner_local_rank]
        self._pe_map: Dict[int, list] = {}

        # Per-group symmetric signal arrays. ``_group_signal[gid]`` is the
        # master 1-D ``int64[n_pes]`` symmetric tensor for ProcessGroup
        # identified by ``gid``. The slot at index ``sender_pe`` on the
        # receiver is written by the sender. Allocating per group eliminates
        # cross-group signal collision (the only sound choice when the same
        # PE pair belongs to multiple groups, e.g. cp_group ∩ ep_group).
        self._group_signal: Dict[int, torch.Tensor] = {}
        # Per-group cached "put-side sig_addr": local symmetric address of
        # ``sig.data_ptr() + my_pe * 8``. Same for every partner in this
        # group (NVSHMEM translates via the ``pe`` argument to put_signal),
        # so we compute it once when the group is first seen.
        self._sig_put_addr: Dict[int, int] = {}
        # Per-(group, partner) monotonic signal counter. Distinct signal
        # slots per group, so cp_group counter cannot poison ep_group wait.
        self._signal_counter: Dict[Tuple[int, int], int] = {}

        # Symmetric recv buffer cache: tag -> master tensor (1-D, full numel).
        # Slices/views handed to callers stay valid only until the next grow.
        self._sym_buf_cache: Dict[str, torch.Tensor] = {}

        # Track the last comm_stream we issued work on so ``final_wait`` /
        # ``clear_iteration`` can drain it without callers having to thread it
        # through. Stream pointer (cudaStream_t cast to int64).
        self._last_comm_stream_ptr: int = 0

    # --- Lazy collective allocations ---------------------------------------

    def _ensure_group(self, group: dist.ProcessGroup):
        """Build PE mapping for a group AND allocate that group's signal array.

        Both ``all_gather_object`` (collective on the supplied group) and
        ``nvshmem_malloc`` (collective on NVSHMEM_TEAM_WORLD) must be entered
        in lockstep across ranks. Production code does so naturally because
        every layer of every iter calls ``exchange`` in the same order.
        """
        gid = id(group)
        if gid in self._pe_map:
            return
        size = group.size()
        pe_list = [None] * size
        dist.all_gather_object(pe_list, self._my_pe, group=group)
        # ``pe_map[gid][local_rank]`` -> nvshmem PE
        self._pe_map[gid] = list(pe_list)

        # Allocate this group's signal array. Collective on NVSHMEM_TEAM_WORLD,
        # which is a superset of ``group``, so all PEs must reach this point —
        # which they do because every PE eventually exchanges in this group.
        # If a PE doesn't participate in the group, it still owns a signal
        # slot it just never reads, but the allocation must succeed globally.
        sig = self._ops.nvshmem_malloc_tensor(self._n_pes, torch.int64)
        self._group_signal[gid] = sig
        # Cache the LOCAL symmetric address we hand to put_signal as
        # ``sig_addr``. NVSHMEM translates this to the partner PE's
        # ``sig[my_pe]`` slot, which is what we want updated.
        self._sig_put_addr[gid] = sig.data_ptr() + self._my_pe * self._INT64_SIZE

    def _get_sig_put_addr(self, gid: int) -> int:
        """Local symmetric address passed to put_signal as ``sig_addr``.

        Identical for every partner in this group — NVSHMEM uses the ``pe``
        argument to route to the partner's same-offset signal slot.
        """
        return self._sig_put_addr[gid]

    def _get_local_signal_addr(self, gid: int, partner_pe: int) -> int:
        """Address of ``signal_array[partner_pe]`` on local PE for this group.

        This is the slot the partner writes to (from the partner's side, with
        ``sig_addr = partner_sig.data_ptr() + partner_pe * 8`` translating to
        our ``sig[partner_pe]``). We wait on it.
        """
        sig = self._group_signal[gid]
        return sig.data_ptr() + partner_pe * self._INT64_SIZE

    # --- Exchange ----------------------------------------------------------

    def exchange(self, send_buf, recv_buf, partner_global_rank, partner_local_rank,
                 group, comm_stream, event):
        self._ensure_group(group)
        gid = id(group)
        partner_pe = self._pe_map[gid][partner_local_rank]

        # Monotonic per-(group, partner) counter. Both peers compute the same
        # value because they each increment LOCAL counter[(gid, other_pe)] in
        # lockstep across the round. The counter is only used to derive
        # ``sig_val``; CPU-side increment ordering vs GPU put ordering is fine
        # because production uses a single ``comm_stream`` per group and
        # exchanges to the same partner are FIFO-serialized on it.
        key = (gid, partner_pe)
        sig_val = self._signal_counter.get(key, 0) + 1
        self._signal_counter[key] = sig_val

        stream_ptr = comm_stream.cuda_stream
        self._last_comm_stream_ptr = stream_ptr

        if send_buf is not None:
            send_buf.record_stream(comm_stream)
            nbytes = send_buf.numel() * send_buf.element_size()

            if recv_buf is None:
                raise RuntimeError(
                    "exchange() with send_buf but no recv_buf is unsupported "
                    "in NVSHMEM mode — both peers must hand each other a "
                    "symmetric recv buffer for put_signal to land in.")

            # Both ``dest`` and ``sig_addr`` are LOCAL symmetric heap
            # pointers; NVSHMEM translates to ``partner_pe`` via the ``pe``
            # arg. Works on NVLink (intra-node) and IB/RoCE (cross-node).
            #
            # ``recv_buf`` came from ``alloc_recv_buffer`` so its
            # ``data_ptr()`` lives in the symmetric heap (collectively
            # allocated, identical offset on every PE). Slices preserve the
            # symmetric-heap-ness of the address.
            local_recv_sym_ptr = recv_buf.data_ptr()
            sig_put_addr = self._get_sig_put_addr(gid)

            self._ops.putmem_signal_on_stream(
                local_recv_sym_ptr,
                send_buf,
                nbytes,
                sig_put_addr,
                sig_val,
                self._SIGNAL_SET,
                partner_pe,
                stream_ptr,
            )

        if recv_buf is not None:
            local_sig_addr = self._get_local_signal_addr(gid, partner_pe)
            self._ops.signal_wait_until_on_stream(
                local_sig_addr,
                self._CMP_GE,
                sig_val,
                stream_ptr,
            )

        if event is not None:
            event.record(comm_stream)

    # --- Symmetric heap recv buffer cache ----------------------------------

    def alloc_recv_buffer(self, tag, numel, dtype, device):
        """Allocate / grow a symmetric recv buffer keyed by ``tag``.

        Grow path is COLLECTIVE: all PEs must call with identical (tag, dtype,
        numel) on the same iteration boundary. The previous master is freed
        via ``nvshmem_free_tensor`` before the new one is allocated, both of
        which are collectives in NVSHMEM_TEAM_WORLD. This eliminates the
        symmetric-heap leak that the previous "drop and replace" pattern
        produced (the drop did nothing — ``from_blob`` has no deleter).
        """
        if numel <= 0:
            numel = 1
        master = self._sym_buf_cache.get(tag)
        if master is not None and master.numel() >= numel and master.dtype == dtype:
            return master[:numel]

        # Grow (or first-time alloc with mismatched dtype). Free the old
        # master collectively, then allocate the new one collectively.
        if master is not None:
            self._ops.nvshmem_free_tensor(master)
        new_master = self._ops.nvshmem_malloc_tensor(numel, dtype)
        self._sym_buf_cache[tag] = new_master
        # Existing remote pointers may have been to slots inside the now-freed
        # master — we cannot rely on the new master being at the same symmetric
        # offset. Invalidate the cached remote pointers (signal pointers are
        # on a separate per-group symmetric tensor, untouched here).
        return new_master[:numel]

    # --- Iteration boundary draining ---------------------------------------

    def needs_final_wait(self) -> bool:
        # Production callers honor this flag to drain at end of forward.
        # NVSHMEM needs a stream-ordered ``quiet`` on RDMA transports to
        # guarantee sender-side completion before send_buf reuse.
        return True

    def final_wait(self):
        """Drain outstanding RDMA puts on the most recently used comm_stream."""
        if self._last_comm_stream_ptr != 0:
            self._ops.quiet_on_stream(self._last_comm_stream_ptr)

    def clear_iteration(self):
        # Defensive drain at iter boundary. ``final_wait`` already ran above
        # if the caller respected ``needs_final_wait`` — this is a belt-and-
        # suspenders second drain, costless on NVLink.
        if self._last_comm_stream_ptr != 0:
            self._ops.quiet_on_stream(self._last_comm_stream_ptr)

    def finalize(self):
        """Release all symmetric heap allocations. Collective on all PEs."""
        for buf in list(self._sym_buf_cache.values()):
            self._ops.nvshmem_free_tensor(buf)
        self._sym_buf_cache.clear()
        for sig in list(self._group_signal.values()):
            self._ops.nvshmem_free_tensor(sig)
        self._group_signal.clear()
        self._sig_put_addr.clear()
        self._signal_counter.clear()


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
    """Reset the singleton (for testing).

    NOTE: ``finalize()`` on NVSHMEMBackend is a collective; the caller must
    invoke ``reset_p2p_backend`` from all ranks at the same logical time.
    """
    global _backend_instance
    with _backend_lock:
        if _backend_instance is not None:
            _backend_instance.finalize()
        _backend_instance = None


__all__ = ['P2PBackend', 'NCCLBackend', 'NVSHMEMBackend',
           'get_p2p_backend', 'reset_p2p_backend']
