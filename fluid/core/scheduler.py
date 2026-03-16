"""
FluidMoE Backward Scheduler

Design (single-thread, dual-stream):
- Main thread controls everything — no background comm thread.
- default_stream: compute (dW, autograd ops)
- comm_stream: A2A + AR (serialized on same stream → no NCCL ordering issues)
- submit_alltoall(): executes comm_fn on comm_stream (non-blocking CPU).
- wait_alltoall(): default_stream.wait_event(end_evt), returns result.
- AR trickle: only when no un-waited A2A pending, budget-controlled per region.
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Callable, Any
from dataclasses import dataclass
import time
from collections import deque
from datetime import timedelta


@dataclass
class DWTask:
    """Deferred weight gradient computation task."""
    layer_name: str
    layer_id: int
    compute_fn: Callable
    weight_param: Optional[torch.nn.Parameter] = None
    needs_ar: bool = True
    target_buffer: Optional[Any] = None  # pre-bound _FlatARBuffer or None


class _FlatARBuffer:
    """Flat contiguous AR buffer with bf16/fp32 sub-buffers and read/write cursors."""

    def __init__(self):
        self.param_map = {}  # param -> (buf, offset, numel)
        self.bf16 = None
        self.fp32 = None
        self.read_bf16 = 0
        self.write_bf16 = 0
        self.read_fp32 = 0
        self.write_fp32 = 0

    def setup(self, params):
        """Allocate buffer from param list, grouped by dtype."""
        self.param_map = {}
        if not params:
            self.bf16 = self.fp32 = None
            self.reset_cursors()
            return

        device = params[0].device
        for dtype_val, attr in [(torch.bfloat16, 'bf16'), (torch.float32, 'fp32')]:
            typed = [(p, p.numel()) for p in params if p.dtype == dtype_val]
            if typed:
                total = sum(n for _, n in typed)
                buf = torch.zeros(total, dtype=dtype_val, device=device)
                offset = 0
                for p, n in typed:
                    self.param_map[p] = (buf, offset, n)
                    offset += n
                setattr(self, attr, buf)
            else:
                setattr(self, attr, None)
        self.reset_cursors()

    def reset_cursors(self):
        self.read_bf16 = self.write_bf16 = 0
        self.read_fp32 = self.write_fp32 = 0

    def has_pending(self):
        """True if dW has written grads past the AR read cursor."""
        return ((self.bf16 is not None and self.read_bf16 < self.write_bf16) or
                (self.fp32 is not None and self.read_fp32 < self.write_fp32))

    def has_remainder(self):
        """True if there are un-AR'd elements in the full buffer."""
        return ((self.bf16 is not None and self.read_bf16 < self.bf16.numel()) or
                (self.fp32 is not None and self.read_fp32 < self.fp32.numel()))

    def trickle(self, group, budget, min_numel):
        """Submit budgeted AR, return (ops_submitted, bytes_submitted)."""
        ops = 0
        bytes_sub = 0
        if self.bf16 is not None and self.read_bf16 < self.write_bf16:
            start = self.read_bf16
            end = min(start + budget // 2, self.write_bf16) if budget > 0 else self.write_bf16
            if (end - start) >= min_numel:
                dist.all_reduce(self.bf16[start:end], group=group)
                self.read_bf16 = end
                bytes_sub += (end - start) * 2
                ops += 1
        if self.fp32 is not None and self.read_fp32 < self.write_fp32:
            start = self.read_fp32
            if budget > 0:
                remaining = budget - bytes_sub
                if remaining <= 0:
                    return ops, bytes_sub
                end = min(start + remaining // 4, self.write_fp32)
            else:
                end = self.write_fp32
            if (end - start) >= min_numel:
                dist.all_reduce(self.fp32[start:end], group=group)
                self.read_fp32 = end
                bytes_sub += (end - start) * 4
                ops += 1
        return ops, bytes_sub

    def flush(self, group):
        """Submit AR for all remaining buffer elements, return number of ops."""
        ops = 0
        if self.bf16 is not None and self.read_bf16 < self.bf16.numel():
            dist.all_reduce(self.bf16[self.read_bf16:], group=group)
            self.read_bf16 = self.bf16.numel()
            ops += 1
        if self.fp32 is not None and self.read_fp32 < self.fp32.numel():
            dist.all_reduce(self.fp32[self.read_fp32:], group=group)
            self.read_fp32 = self.fp32.numel()
            ops += 1
        return ops

    def sync_allreduce(self, group):
        """Synchronous AR of full buffers (used when ar_enabled=False)."""
        if self.bf16 is not None:
            dist.all_reduce(self.bf16, group=group)
        if self.fp32 is not None:
            dist.all_reduce(self.fp32, group=group)

    def write_grad(self, param, grad_weight):
        """Write grad into buffer, advance write cursor. Returns True if handled."""
        if param not in self.param_map:
            return False
        buf, offset, numel = self.param_map[param]
        flat_grad = grad_weight.view(-1)
        if param.grad is None:
            buf[offset:offset+numel].copy_(flat_grad)
        else:
            buf[offset:offset+numel].add_(flat_grad)
        param.grad = buf[offset:offset+numel].view(param.shape)
        end = offset + numel
        if buf.dtype == torch.bfloat16:
            if end > self.write_bf16:
                self.write_bf16 = end
        else:
            if end > self.write_fp32:
                self.write_fp32 = end
        return True


class BackwardScheduler:
    """Singleton scheduler for backward pass compute-communication overlap."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.enabled = False
        self.default_stream = None
        self.comm_stream = None
        self._device = None

        # dW queue
        self._dw_queue = deque()
        self._ar_params_for_sync = []

        # AllToAll tracking
        self._alltoall_task_id = 0
        self._alltoall_results = {}  # task_id -> (result, end_evt, start_evt)

        # AR tracking
        self._pending_ar_tensors = deque()
        self._ar_task_count = 0
        self._last_ar_cuda_event = None
        self._ar_done_event = None

        # Flat AR buffers (shared + expert)
        self._shared_ar = _FlatARBuffer()
        self._expert_ar = _FlatARBuffer()

        # AR config
        self.ar_enabled = False
        self.ar_safe_mode = False
        self.ar_lockstep = False
        self.shared_dp_group = None
        self.ar_group = None
        self.ar_ctrl_group = None
        self.shared_dp_world_size = 1
        self.expert_dp_group = None
        self.expert_ar_group = None
        self.expert_dp_world_size = 1
        self.ar_window_mode = "strict_window"

        # Stats
        self.total_dw = 0
        self.completed_dw = 0
        self.total_ar = 0
        self.completed_ar = 0
        self.ar_during_overlap = 0
        self.ar_submitted_during_bwd = 0
        self.ar_submitted_during_finish = 0
        self._in_finish_batch = False

        # BDP: per-region gap on comm_stream
        self._last_a2a_end_event = None
        self._last_a2a_end_region = None
        self._first_a2a_in_region = True
        self._gap_event_pairs = []
        self._region_gaps = {}
        self.ar_trickle_sizes = {}

        # Profiling
        self.profiling = False
        self._region_name = None
        self._region_profiles = {}
        self._region_a2a_times = []
        self._region_dw_time = 0.0

        # Communication visibility metrics
        # - a2a_total_ms: sum of A2A kernel durations on comm_stream
        # - a2a_visible_ms: sum of default_stream stall windows waiting on A2A
        self.comm_metrics_enabled = False
        self._a2a_event_pairs = []         # [(start_evt, end_evt), ...]
        self._visible_wait_pairs = []      # [(wait_start_evt, wait_end_evt), ...]
        # AR visibility metrics (same pattern as A2A)
        self._ar_event_pairs = []          # [(start_evt, end_evt), ...] on comm_stream
        self._ar_visible_wait_pairs = []   # [(wait_start, wait_end), ...] on default_stream

        # Event pool (initialized in _init_cuda, defaults here for safety)
        self._event_pool = []
        self._event_pool_size = 0

        self._init_cuda()

    def _init_cuda(self, force_reinit=False):
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            if not force_reinit and self._device == current_device and self.comm_stream is not None:
                return
            self._device = current_device
            self.default_stream = torch.cuda.default_stream(current_device)
            from fluid.core.stream import get_stream_manager
            self.comm_stream = get_stream_manager().comm_stream
            self._ar_done_event = torch.cuda.Event()
            self._dw_sync_event = torch.cuda.Event()
            # Pre-allocate event pool (non-timing events for sync)
            _POOL_SIZE = 64
            self._event_pool = [torch.cuda.Event() for _ in range(_POOL_SIZE)]
            self._event_pool_size = _POOL_SIZE

    def enable(self):
        self.enabled = True
        self._init_cuda(force_reinit=True)

    def is_enabled(self):
        return self.enabled

    def configure_allreduce(self, enabled=True, shared_dp_group=None, expert_dp_group=None, **kwargs):
        """Configure AllReduce for shared and expert parameters."""
        self.ar_enabled = enabled
        self.shared_dp_group = shared_dp_group
        self.shared_dp_world_size = dist.get_world_size(shared_dp_group) if shared_dp_group else (
            dist.get_world_size() if dist.is_initialized() else 1)
        self.expert_dp_group = expert_dp_group
        self.expert_dp_world_size = dist.get_world_size(expert_dp_group) if expert_dp_group else 1

        mode_env = os.environ.get("FLUID_AR_WINDOW_MODE", "strict_window").lower()
        self.ar_window_mode = "relaxed_window" if mode_env in ("relaxed", "relaxed_window", "relaxed-window") else "strict_window"

        safe_env = os.environ.get("FLUID_AR_SAFE_MODE", "auto").lower()
        self.ar_safe_mode = safe_env in ("1", "true", "yes", "on")

        lockstep_env = os.environ.get("FLUID_AR_LOCKSTEP", "auto").lower()
        if lockstep_env in ("1", "true", "yes", "on"):
            self.ar_lockstep = True
        elif lockstep_env in ("0", "false", "no", "off"):
            self.ar_lockstep = False
        else:
            self.ar_lockstep = (self.expert_dp_world_size > 1)
        if self.ar_safe_mode:
            self.ar_lockstep = False

        # Create independent NCCL communicators
        if shared_dp_group is not None and dist.is_initialized():
            ranks = dist.get_process_group_ranks(shared_dp_group)
            self.ar_group = dist.new_group(ranks)
            self.ar_ctrl_group = dist.new_group(ranks=ranks, backend="gloo") if (self.shared_dp_world_size > 1 and self.ar_lockstep) else None
        else:
            self.ar_group = self.ar_ctrl_group = None

        if expert_dp_group is not None and dist.is_initialized() and self.expert_dp_world_size > 1:
            self.expert_ar_group = dist.new_group(dist.get_process_group_ranks(expert_dp_group))
        else:
            self.expert_ar_group = None

        self._init_cuda(force_reinit=True)

    def setup_ar_buffer(self, params):
        """Set up flat AR buffer for shared parameters in backward execution order."""
        self._shared_ar.setup(params)

    def setup_expert_ar_buffer(self, params):
        """Set up flat AR buffer for expert parameters."""
        self._expert_ar.setup(params)

    def _use_interleaved_ar(self) -> bool:
        return self.ar_enabled and (not self.ar_safe_mode)

    # ========================================
    # dW Tasks
    # ========================================
    def register_dw_task(self, layer_name, layer_id, compute_fn, weight_param=None, needs_ar=True, **kwargs):
        if not self.enabled:
            return
        # Pre-bind target buffer to avoid dict lookup in execute_dw_tasks
        target_buffer = None
        if weight_param is not None:
            if weight_param in self._shared_ar.param_map:
                target_buffer = self._shared_ar
            elif weight_param in self._expert_ar.param_map:
                target_buffer = self._expert_ar
        self._dw_queue.append(DWTask(layer_name, layer_id, compute_fn, weight_param, needs_ar, target_buffer))
        self.total_dw += 1

    def execute_dw_tasks(self) -> bool:
        if self.profiling and self._region_name and self._dw_queue:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        else:
            t0 = None

        while self._dw_queue:
            task = self._dw_queue.popleft()
            grad_weight = task.compute_fn()

            if task.weight_param is not None and grad_weight is not None:
                if grad_weight.dtype != task.weight_param.dtype:
                    grad_weight = grad_weight.to(task.weight_param.dtype)

                # Use pre-bound target buffer (avoids dict lookup)
                if task.target_buffer is not None:
                    task.target_buffer.write_grad(task.weight_param, grad_weight)
                else:
                    # Fallback for unregistered params
                    if task.weight_param.grad is None:
                        task.weight_param.grad = grad_weight
                    else:
                        task.weight_param.grad.add_(grad_weight)
                    if task.needs_ar and self.shared_dp_world_size > 1:
                        if self._use_interleaved_ar():
                            self._pending_ar_tensors.append(task.weight_param.grad)
                        else:
                            self._ar_params_for_sync.append(task.weight_param)

            self.completed_dw += 1

        if t0 is not None:
            torch.cuda.synchronize()
            self._region_dw_time += (time.perf_counter() - t0) * 1000

        # Record event after dW writes on default_stream.
        # _trickle_ar uses wait_event to ensure data visibility on comm_stream.
        self._dw_sync_event.record(self.default_stream)
        return False

    # ========================================
    # AllToAll (inline on comm_stream)
    # ========================================
    def _get_pooled_event(self, idx: int) -> torch.cuda.Event:
        """Get a pre-allocated event from pool by index (grow on demand)."""
        if idx < self._event_pool_size:
            return self._event_pool[idx]
        # Grow pool to cover idx (no modular wrap → no cross-task collision)
        while self._event_pool_size <= idx:
            self._event_pool.append(torch.cuda.Event())
            self._event_pool_size += 1
        return self._event_pool[idx]

    def submit_alltoall(self, comm_fn: Callable) -> int:
        """Execute comm_fn on comm_stream. NCCL kernel launch is non-blocking."""
        if not self.enabled:
            return comm_fn()

        task_id = self._alltoall_task_id
        self._alltoall_task_id += 1

        # Use pooled event for input sync (no timing needed)
        input_event = self._get_pooled_event(task_id * 2)
        input_event.record(self.default_stream)

        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(input_event)
            need_timing = self.profiling or self.comm_metrics_enabled
            if need_timing:
                start_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record(self.comm_stream)
            else:
                start_evt = None
            result = comm_fn()
            if need_timing:
                end_evt = torch.cuda.Event(enable_timing=True)
                end_evt.record(self.comm_stream)
            else:
                # Use pooled event for sync-only (no timing overhead)
                end_evt = self._get_pooled_event(task_id * 2 + 1)
                end_evt.record(self.comm_stream)

        self._alltoall_results[task_id] = (result, end_evt, start_evt)
        return task_id

    def submit_alltoall_batch(self, comm_fns: list) -> list:
        """Submit multiple AllToAll ops in one stream switch (for comm-first pipelines)."""
        if not self.enabled:
            return [fn() for fn in comm_fns]

        # Single input event for the batch
        input_event = self._get_pooled_event(self._alltoall_task_id * 2)
        input_event.record(self.default_stream)

        task_ids = []
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(input_event)
            for fn in comm_fns:
                task_id = self._alltoall_task_id
                self._alltoall_task_id += 1

                need_timing = self.profiling or self.comm_metrics_enabled
                if need_timing:
                    start_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record(self.comm_stream)
                else:
                    start_evt = None

                result = fn()

                if need_timing:
                    end_evt = torch.cuda.Event(enable_timing=True)
                    end_evt.record(self.comm_stream)
                else:
                    end_evt = self._get_pooled_event(task_id * 2 + 1)
                    end_evt.record(self.comm_stream)

                self._alltoall_results[task_id] = (result, end_evt, start_evt)
                task_ids.append(task_id)

        return task_ids

    def wait_alltoall(self, task_id: int, num_tasks: int = 1, try_trickle: bool = True) -> Any:
        if not self.enabled:
            return None
        task_data = self._alltoall_results.get(task_id)
        if task_data is None:
            return None

        result, end_evt, start_evt = task_data

        # BDP gap tracking (only when profiling creates timing events)
        if self.profiling and self._first_a2a_in_region and start_evt is not None:
            if self._last_a2a_end_event is not None:
                self._gap_event_pairs.append((
                    self._last_a2a_end_region, self._last_a2a_end_event, start_evt))
                self._last_a2a_end_event = None
                self._last_a2a_end_region = None
            self._first_a2a_in_region = False

        if self.profiling and start_evt is not None:
            # Defer elapsed_time to process_gap_events (no sync here to avoid
            # stalling CPU and artificially inflating comm_stream gaps).
            if self._region_name:
                self._region_a2a_times.append((start_evt, end_evt))
        # For batch waits (num_tasks > 1), record a single A2A pair
        # spanning first_chunk_start → last_chunk_end to keep 1:1 with visible_wait.
        if self.comm_metrics_enabled and start_evt is not None and end_evt is not None:
            if num_tasks > 1:
                first_tid = task_id - num_tasks + 1
                first_td = self._alltoall_results.get(first_tid)
                batch_start = first_td[2] if first_td is not None else start_evt
                self._a2a_event_pairs.append((batch_start, end_evt))
            else:
                self._a2a_event_pairs.append((start_evt, end_evt))

        if end_evt is not None:
            if self.comm_metrics_enabled:
                wait_start_evt = torch.cuda.Event(enable_timing=True)
                wait_end_evt = torch.cuda.Event(enable_timing=True)
                wait_start_evt.record(self.default_stream)
                self.default_stream.wait_event(end_evt)
                wait_end_evt.record(self.default_stream)
                self._visible_wait_pairs.append((wait_start_evt, wait_end_evt))
            else:
                self.default_stream.wait_event(end_evt)

        # Clean up waited tasks
        for tid in range(task_id - num_tasks + 1, task_id + 1):
            td = self._alltoall_results.pop(tid, None)

        if not self._alltoall_results:
            self._last_a2a_end_event = end_evt
            self._last_a2a_end_region = self._region_name

        # Only trickle AR when caller allows AND no pending A2A
        if try_trickle and not self._alltoall_results:
            self._trickle_ar()
        return result

    # ========================================
    # AllReduce (inline on comm_stream)
    # ========================================
    def submit_pending_ar(self):
        """Submit all pending AR to comm_stream for overlap with subsequent compute.

        Call between regions when comm_stream is idle and default_stream has
        significant compute ahead (e.g. SDPA backward).  The AR runs on
        comm_stream in parallel with whatever default_stream does next.
        """
        if not self._use_interleaved_ar():
            return
        has_shared = self.shared_dp_world_size > 1
        has_expert = self.expert_dp_world_size > 1
        if not has_shared and not has_expert:
            return
        if not (has_shared and self._shared_ar.has_pending()) and \
           not (has_expert and self._expert_ar.has_pending()) and \
           not self._pending_ar_tensors:
            return
        if self._alltoall_results:
            return

        ar_group = self.ar_group or self.shared_dp_group
        expert_group = self.expert_ar_group or self.expert_dp_group
        _MIN = 512 * 1024

        need_ar_timing = self.comm_metrics_enabled

        with torch.cuda.stream(self.comm_stream):
            # Wait only for dW writes, not subsequent compute
            self.comm_stream.wait_event(self._dw_sync_event)
            if need_ar_timing:
                ar_s = torch.cuda.Event(enable_timing=True)
                ar_s.record(self.comm_stream)
            ops = 0
            if has_shared:
                # Unlimited budget: flush everything written so far
                s_ops, _ = self._shared_ar.trickle(ar_group, 0, _MIN)
                ops += s_ops
            if has_expert:
                e_ops, _ = self._expert_ar.trickle(expert_group, 0, _MIN)
                ops += e_ops
            while self._pending_ar_tensors:
                grad = self._pending_ar_tensors.popleft()
                if grad is not None:
                    dist.all_reduce(grad, group=ar_group)
                    ops += 1
            if need_ar_timing and ops > 0:
                ar_e = torch.cuda.Event(enable_timing=True)
                ar_e.record(self.comm_stream)
                self._ar_event_pairs.append((ar_s, ar_e))

        self._ar_task_count += ops
        self.total_ar += ops
        self.ar_during_overlap += ops
        self.ar_submitted_during_bwd += ops

    def _trickle_ar(self):
        """Trickle budgeted AR onto comm_stream when no A2A is pending."""
        if not self._use_interleaved_ar():
            return
        has_shared = self.shared_dp_world_size > 1
        has_expert = self.expert_dp_world_size > 1
        if not has_shared and not has_expert:
            return
        if not (has_shared and self._shared_ar.has_pending()) and \
           not (has_expert and self._expert_ar.has_pending()) and \
           not self._pending_ar_tensors:
            return
        if self._alltoall_results:
            return

        region = self._region_name
        budget = self.ar_trickle_sizes.get(region, 0) if region else 0
        _MIN = 512 * 1024

        ar_group = self.ar_group or self.shared_dp_group
        expert_group = self.expert_ar_group or self.expert_dp_group

        need_ar_timing = self.comm_metrics_enabled

        with torch.cuda.stream(self.comm_stream):
            # Wait only for dW writes, not subsequent chunk compute
            self.comm_stream.wait_event(self._dw_sync_event)
            if need_ar_timing:
                ar_s = torch.cuda.Event(enable_timing=True)
                ar_s.record(self.comm_stream)
            ops = 0
            bytes_used = 0
            if has_shared:
                s_ops, s_bytes = self._shared_ar.trickle(ar_group, budget, _MIN)
                ops += s_ops
                bytes_used += s_bytes
            if has_expert:
                remaining = max(budget - bytes_used, 0) if budget > 0 else 0
                e_ops, e_bytes = self._expert_ar.trickle(expert_group, remaining, _MIN)
                ops += e_ops
                bytes_used += e_bytes
            # Fallback: pending tensors
            while self._pending_ar_tensors:
                if budget > 0 and bytes_used >= budget:
                    break
                grad = self._pending_ar_tensors.popleft()
                if grad is None:
                    continue
                dist.all_reduce(grad, group=ar_group)
                bytes_used += grad.numel() * grad.element_size()
                ops += 1
            if need_ar_timing and ops > 0:
                ar_e = torch.cuda.Event(enable_timing=True)
                ar_e.record(self.comm_stream)
                self._ar_event_pairs.append((ar_s, ar_e))

        self._ar_task_count += ops
        self.total_ar += ops
        self.ar_during_overlap += ops
        if self._in_finish_batch:
            self.ar_submitted_during_finish += ops
        else:
            self.ar_submitted_during_bwd += ops

    def _flush_ar_pending_strict(self, final: bool = False):
        """Flush all remaining AR buffer and pending tensors."""
        has_remaining = self._shared_ar.has_remainder() or self._expert_ar.has_remainder()
        if not self._pending_ar_tensors and not has_remaining:
            if self.ar_lockstep and self.ar_ctrl_group is not None:
                dist.monitored_barrier(group=self.ar_ctrl_group,
                    timeout=timedelta(seconds=int(os.environ.get("FLUID_AR_LOCKSTEP_TIMEOUT_SEC", "120"))))
            return

        ar_group = self.ar_group or self.shared_dp_group
        expert_group = self.expert_ar_group or self.expert_dp_group

        need_ar_timing = self.comm_metrics_enabled

        with torch.cuda.stream(self.comm_stream):
            # Wait for latest dW writes before flushing entire buffer
            self.comm_stream.wait_event(self._dw_sync_event)
            if need_ar_timing:
                ar_s = torch.cuda.Event(enable_timing=True)
                ar_s.record(self.comm_stream)
            ops = self._shared_ar.flush(ar_group)
            if expert_group is not None:
                ops += self._expert_ar.flush(expert_group)
            while self._pending_ar_tensors:
                grad = self._pending_ar_tensors.popleft()
                if grad is not None:
                    dist.all_reduce(grad, group=ar_group)
                    ops += 1
            self._ar_done_event.record(self.comm_stream)
            self._last_ar_cuda_event = self._ar_done_event
            if need_ar_timing and ops > 0:
                ar_e = torch.cuda.Event(enable_timing=True)
                ar_e.record(self.comm_stream)
                self._ar_event_pairs.append((ar_s, ar_e))

        self._ar_task_count += ops
        self.total_ar += ops
        self.ar_during_overlap += ops
        self.ar_submitted_during_finish += ops

        if self.ar_lockstep and self.ar_ctrl_group is not None:
            dist.monitored_barrier(group=self.ar_ctrl_group,
                timeout=timedelta(seconds=int(os.environ.get("FLUID_AR_LOCKSTEP_TIMEOUT_SEC", "120"))))

    def flush_ar_pending(self, final: bool = False):
        if not self._use_interleaved_ar() or (self.shared_dp_world_size <= 1 and self.expert_dp_world_size <= 1):
            self._pending_ar_tensors.clear()
            return
        self._flush_ar_pending_strict(final=final)

    def _wait_all_ar(self):
        if self._last_ar_cuda_event is None:
            return
        if self.comm_metrics_enabled:
            w_s = torch.cuda.Event(enable_timing=True)
            w_e = torch.cuda.Event(enable_timing=True)
            w_s.record(self.default_stream)
            self.default_stream.wait_event(self._last_ar_cuda_event)
            w_e.record(self.default_stream)
            self._ar_visible_wait_pairs.append((w_s, w_e))
        else:
            self.default_stream.wait_event(self._last_ar_cuda_event)
        self.completed_ar = self._ar_task_count

    def _sync_allreduce_all_params(self):
        """Synchronous AR (used when ar_enabled=False)."""
        ar_group = self.ar_group or self.shared_dp_group
        expert_group = self.expert_ar_group or self.expert_dp_group
        if self.shared_dp_world_size > 1:
            self._shared_ar.sync_allreduce(ar_group)
        if self.expert_dp_world_size > 1 and expert_group is not None:
            self._expert_ar.sync_allreduce(expert_group)
        for param in self._ar_params_for_sync:
            if param.grad is not None:
                dist.all_reduce(param.grad, group=ar_group)
        self._ar_params_for_sync.clear()

    # ========================================
    # Iteration management
    # ========================================
    def clear_iteration(self):
        """Clear state for new iteration."""
        for result, end_evt, start_evt in list(self._alltoall_results.values()):
            if end_evt is not None:
                self.default_stream.wait_event(end_evt)
        self._wait_all_ar()
        # Stream-event waits above guarantee GPU ordering; host-side sync
        # is only needed when there are truly un-waited tasks (not after
        # finish_batch which already waited for everything).
        if self._alltoall_results and torch.cuda.is_available():
            torch.cuda.synchronize()

        self._dw_queue.clear()
        self._alltoall_results.clear()
        self._pending_ar_tensors.clear()
        self._ar_params_for_sync.clear()
        self._last_ar_cuda_event = None
        self._alltoall_task_id = 0
        self._ar_task_count = 0
        self._in_finish_batch = False
        self._last_a2a_end_event = None
        self._last_a2a_end_region = None
        self._first_a2a_in_region = True
        self._shared_ar.reset_cursors()
        self._expert_ar.reset_cursors()

        self.total_dw = self.completed_dw = 0
        self.total_ar = self.completed_ar = 0
        self.ar_during_overlap = self.ar_submitted_during_bwd = self.ar_submitted_during_finish = 0

    def finish_batch(self):
        """Complete all dW + AR before optimizer.step()."""
        if not self.enabled:
            return
        self._in_finish_batch = True

        needs_ar = self.shared_dp_world_size > 1 or self.expert_dp_world_size > 1
        if self._use_interleaved_ar() and needs_ar:
            # Submit AR for gradients already in buffer BEFORE dW tasks.
            # Only processes up to write_cursor (safe — unwritten positions untouched).
            # This AR runs on comm_stream while dW compute runs on default_stream.
            self.submit_pending_ar()

        self.execute_dw_tasks()

        if self._use_interleaved_ar() and needs_ar:
            # Flush remaining AR (newly written by dW + tail of buffer).
            self.flush_ar_pending(final=True)
            self._wait_all_ar()
        self._last_a2a_end_event = None
        self._last_a2a_end_region = None
        if (not self._use_interleaved_ar()) and needs_ar:
            self._sync_allreduce_all_params()
        self._in_finish_batch = False

    # ========================================
    # Profiling
    # ========================================
    def begin_region(self, name: str):
        self._region_name = name
        self._first_a2a_in_region = True
        if not self.profiling:
            return
        torch.cuda.synchronize()
        self._region_a2a_times = []
        self._region_dw_time = 0.0
        self._region_wall_start = time.perf_counter()

    def end_region(self):
        if not self.profiling or self._region_name is None:
            self._region_name = None
            return
        torch.cuda.synchronize()
        T_region = (time.perf_counter() - self._region_wall_start) * 1000
        name = self._region_name
        # Resolve deferred event pairs to elapsed times
        T_comm = sum(s.elapsed_time(e) for s, e in self._region_a2a_times)
        T_dW = self._region_dw_time
        if name not in self._region_profiles:
            self._region_profiles[name] = {'T_region': 0.0, 'T_comm': 0.0, 'T_dW': 0.0, 'n_a2a': 0, 'count': 0}
        p = self._region_profiles[name]
        p['T_region'] += T_region
        p['T_comm'] += T_comm
        p['T_dW'] += T_dW
        p['n_a2a'] += len(self._region_a2a_times)
        p['count'] += 1
        self._region_name = None

    def get_region_profiles(self):
        result = {}
        for name, p in self._region_profiles.items():
            n = max(p['count'], 1)
            T_region, T_comm, T_dW = p['T_region']/n, p['T_comm']/n, p['T_dW']/n
            result[name] = {'T_region': T_region, 'T_comm': T_comm,
                            'T_comp': max(0, T_region - max(T_comm, T_dW)),
                            'T_dW': T_dW, 'n_a2a': p['n_a2a'] // n}
        return result

    # ========================================
    # BDP trickle size computation
    # ========================================
    def reset_gap_times(self):
        self._gap_event_pairs = []
        self._region_gaps = {}
        self._last_a2a_end_event = None
        self._last_a2a_end_region = None
        self._first_a2a_in_region = True

    # ========================================
    # Communication visibility metrics
    # ========================================
    def set_comm_metrics_enabled(self, enabled: bool):
        self.comm_metrics_enabled = bool(enabled)

    def reset_comm_metrics(self):
        self._a2a_event_pairs = []
        self._visible_wait_pairs = []
        self._ar_event_pairs = []
        self._ar_visible_wait_pairs = []

    def get_comm_metrics(self):
        """Return A2A and AR total/visible communication time in ms."""
        torch.cuda.synchronize()

        # A2A metrics
        a2a_total = 0.0
        a2a_visible = 0.0
        for (a2a_s, a2a_e), (w_s, w_e) in zip(self._a2a_event_pairs, self._visible_wait_pairs):
            a2a_ms = a2a_s.elapsed_time(a2a_e)
            wait_ms = w_s.elapsed_time(w_e)
            a2a_total += a2a_ms
            a2a_visible += min(wait_ms, a2a_ms)
        a2a_overlap = 0.0 if a2a_total <= 1e-9 else max(0.0, min(1.0, 1.0 - a2a_visible / a2a_total))

        # AR metrics
        ar_total = 0.0
        for s, e in self._ar_event_pairs:
            ar_total += s.elapsed_time(e)
        ar_visible = 0.0
        for s, e in self._ar_visible_wait_pairs:
            ar_visible += s.elapsed_time(e)
        ar_overlap = 0.0 if ar_total <= 1e-9 else max(0.0, min(1.0, 1.0 - ar_visible / ar_total))

        return {
            'a2a_total_ms': float(a2a_total),
            'a2a_visible_ms': float(a2a_visible),
            'a2a_overlap_ratio': float(a2a_overlap),
            'n_a2a': len(self._a2a_event_pairs),
            'n_waits': len(self._visible_wait_pairs),
            'ar_total_ms': float(ar_total),
            'ar_visible_ms': float(ar_visible),
            'ar_overlap_ratio': float(ar_overlap),
            'n_ar': len(self._ar_event_pairs),
        }

    def process_gap_events(self):
        for region, end_evt, start_evt in self._gap_event_pairs:
            gap_ms = end_evt.elapsed_time(start_evt)
            if region not in self._region_gaps:
                self._region_gaps[region] = []
            self._region_gaps[region].append(gap_ms)
        self._gap_event_pairs = []

    @staticmethod
    def measure_ar_bandwidth(ar_group=None, sizes_mb=(1,2,4,8,16,32,64,96,128),
                             warmup=5, repeat=20, dtype=torch.bfloat16) -> dict:
        device = torch.cuda.current_device()
        results = {'sizes_mb': [], 'bw_GBps': [], 'latency_ms': []}
        elem_size = 2 if dtype == torch.bfloat16 else 4
        for size_mb in sizes_mb:
            numel = int(size_mb * 1024 * 1024 / elem_size)
            tensor = torch.randn(numel, dtype=dtype, device=device)
            for _ in range(warmup):
                dist.all_reduce(tensor, group=ar_group)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                dist.all_reduce(tensor, group=ar_group)
            end.record()
            torch.cuda.synchronize()
            lat = start.elapsed_time(end) / repeat
            bw = (numel * elem_size / 1e9) / (lat / 1e3) if lat > 0 else 0
            results['sizes_mb'].append(size_mb)
            results['bw_GBps'].append(bw)
            results['latency_ms'].append(lat)
            del tensor
        results['peak_bw_GBps'] = max(results['bw_GBps']) if results['bw_GBps'] else 0
        return results

    def compute_bdp_trickle_size(self, bw_GBps=0.0, percentile=10.0,
                                  safety_factor=0.9, bw_profile=None) -> dict:
        if not self._region_gaps:
            raise RuntimeError("No region gaps recorded. Run profiling iterations first.")

        def _pct(vals, p):
            n = len(vals)
            idx = p / 100.0 * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            return vals[lo] * (1 - (idx - lo)) + vals[hi] * (idx - lo)

        def _bw_at(mb):
            if bw_profile is None:
                return bw_GBps
            sizes, bws = bw_profile['sizes_mb'], bw_profile['bw_GBps']
            if mb <= sizes[0]: return bws[0]
            if mb >= sizes[-1]: return bws[-1]
            for i in range(len(sizes) - 1):
                if sizes[i] <= mb <= sizes[i+1]:
                    f = (mb - sizes[i]) / (sizes[i+1] - sizes[i])
                    return bws[i] + f * (bws[i+1] - bws[i])
            return bws[-1]

        eff_bw = bw_GBps
        per_region = {}
        for region, windows in self._region_gaps.items():
            ws = sorted(windows)
            T_pct = _pct(ws, percentile)
            T_mean = sum(ws) / len(ws)
            if bw_profile is not None:
                bw = _bw_at(64)
                for _ in range(5):
                    bdp = (T_pct / 1000) * (bw * 1e9) * safety_factor
                    bw = _bw_at(bdp / (1024*1024))
                eff_bw = bw
            else:
                bw = bw_GBps
                bdp = (T_pct / 1000) * (bw * 1e9) * safety_factor
            mb = int(bdp / (1024*1024))
            per_region[region] = {
                'trickle_size_bytes': mb * 1024 * 1024,
                'trickle_size_MB': mb,
                'trickle_size_bytes_exact': bdp,
                'T_window_min_ms': ws[0],
                'T_window_percentile_ms': T_pct,
                'T_window_mean_ms': T_mean,
                'n_windows': len(ws),
                'bw_GBps_used': bw,
            }
        return {'per_region': per_region, 'bw_GBps': eff_bw,
                'safety_factor': safety_factor, 'percentile': percentile}

    def get_stats(self):
        return {
            'total_dw_tasks': self.total_dw, 'completed_dw_tasks': self.completed_dw,
            'total_ar_tasks': self.total_ar, 'completed_ar_tasks': self.completed_ar,
            'ar_during_gap': self.ar_during_overlap,
            'ar_submitted_during_bwd': self.ar_submitted_during_bwd,
            'ar_submitted_during_finish': self.ar_submitted_during_finish,
            'pending_dw_at_finish': len(self._dw_queue),
            'region_gaps': dict(self._region_gaps),
            'ar_window_mode': self.ar_window_mode,
        }

    @classmethod
    def reset(cls):
        cls._instance = None


def get_backward_scheduler() -> BackwardScheduler:
    return BackwardScheduler()
