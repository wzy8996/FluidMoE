"""
FluidMoE Backward Scheduler

Design (inline AR on comm_stream, no background thread):
- Main thread: default_stream (compute/dW), comm_stream (A2A + AR)
- AR submitted inline at deterministic code points:
    1. wait_alltoall(): after all A2A done (gap filling)
    2. finish_batch(): after each dW task (AR-dW overlap via streams)
- Expert-first priority: expert AR fills gap first, shared gets remainder
- No dist.new_group() needed — AR and A2A never concurrent on comm_stream
- GPU overlap: comm_stream (AR) runs in parallel with default_stream (dW)
"""

import torch
import torch.distributed as dist
from typing import Optional, Callable, Any
from dataclasses import dataclass
from collections import deque


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
    """Flat contiguous AR buffer with bf16/fp32 sub-buffers.

    Cursors:
      write     - high-water mark of dW writes (GPU not yet guaranteed visible)
      committed - dW writes guaranteed visible on default_stream up to here
                  (advanced only AFTER _dw_sync_event.record())
      read      - AR has been submitted up to here
    """

    def __init__(self):
        self.param_map = {}
        self.bf16 = None
        self.fp32 = None
        self.read_bf16 = 0
        self.write_bf16 = 0
        self.committed_bf16 = 0
        self.read_fp32 = 0
        self.write_fp32 = 0
        self.committed_fp32 = 0

    def setup(self, params, accumulate_in_fp32=True):
        """Set up flat buffer for params.

        Args:
            params: list of parameters to register.
            accumulate_in_fp32: if True, always use fp32 buffer (matching Megatron's
                accumulate_allreduce_grads_in_fp32 default). Gradients are cast to fp32
                on write and the allreduce runs in fp32 for numerical stability.
        """
        self.param_map = {}
        if not params:
            self.bf16 = self.fp32 = None
            self.reset_cursors()
            return
        device = params[0].device
        if accumulate_in_fp32:
            # All params share a single fp32 buffer (Megatron convention)
            total = sum(p.numel() for p in params)
            buf = torch.zeros(total, dtype=torch.float32, device=device)
            offset = 0
            for p in params:
                n = p.numel()
                self.param_map[p] = (buf, offset, n)
                offset += n
            self.fp32 = buf
            self.bf16 = None
        else:
            # Group by dtype (legacy path)
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
        self.read_bf16 = self.write_bf16 = self.committed_bf16 = 0
        self.read_fp32 = self.write_fp32 = self.committed_fp32 = 0

    def commit(self):
        """Advance committed cursor to current write position.
        Must be called AFTER _dw_sync_event.record() on default_stream."""
        self.committed_bf16 = self.write_bf16
        self.committed_fp32 = self.write_fp32

    def has_pending(self):
        """True if committed but un-ARed data exists."""
        return ((self.bf16 is not None and self.read_bf16 < self.committed_bf16) or
                (self.fp32 is not None and self.read_fp32 < self.committed_fp32))

    def has_remainder(self):
        """True if un-ARed data exists (including uncommitted)."""
        return ((self.bf16 is not None and self.read_bf16 < self.write_bf16) or
                (self.fp32 is not None and self.read_fp32 < self.write_fp32))

    def pending_bytes(self):
        """Committed but un-ARed bytes."""
        b = 0
        if self.bf16 is not None:
            b += max(0, self.committed_bf16 - self.read_bf16) * 2
        if self.fp32 is not None:
            b += max(0, self.committed_fp32 - self.read_fp32) * 4
        return b

    def submit_block(self, group, max_bytes=0, pre_scale: float = 1.0):
        """Submit all_reduce for committed data, up to max_bytes (0=unlimited).
        Returns (ops, bytes_submitted)."""
        ops = 0
        submitted = 0
        if self.bf16 is not None and self.read_bf16 < self.committed_bf16:
            avail = self.committed_bf16 - self.read_bf16
            numel = min(avail, max_bytes // 2) if max_bytes > 0 else avail
            if numel > 0:
                ar_slice = self.bf16[self.read_bf16:self.read_bf16 + numel]
                if pre_scale != 1.0:
                    ar_slice.mul_(pre_scale)
                dist.all_reduce(ar_slice, group=group)
                self.read_bf16 += numel
                submitted += numel * 2
                ops += 1
        if self.fp32 is not None and self.read_fp32 < self.committed_fp32:
            avail = self.committed_fp32 - self.read_fp32
            if max_bytes > 0:
                remaining = max_bytes - submitted
                if remaining <= 0:
                    return ops, submitted
                numel = min(avail, remaining // 4)
            else:
                numel = avail
            if numel > 0:
                ar_slice_fp32 = self.fp32[self.read_fp32:self.read_fp32 + numel]
                if pre_scale != 1.0:
                    ar_slice_fp32.mul_(pre_scale)
                dist.all_reduce(ar_slice_fp32, group=group)
                self.read_fp32 += numel
                submitted += numel * 4
                ops += 1
        return ops, submitted

    def sync_allreduce(self, group, pre_scale: float = 1.0):
        """Synchronous AR of full buffers (used when ar_enabled=False)."""
        if self.bf16 is not None:
            if pre_scale != 1.0:
                self.bf16.mul_(pre_scale)
            dist.all_reduce(self.bf16, group=group)
        if self.fp32 is not None:
            if pre_scale != 1.0:
                self.fp32.mul_(pre_scale)
            dist.all_reduce(self.fp32, group=group)

    def scale_pending(self, scale: float):
        """Scale all committed-but-not-yet-read data (for expert_dp=1 scaling)."""
        if self.bf16 is not None and self.read_bf16 < self.committed_bf16:
            self.bf16[self.read_bf16:self.committed_bf16].mul_(scale)
            self.read_bf16 = self.committed_bf16
        if self.fp32 is not None and self.read_fp32 < self.committed_fp32:
            self.fp32[self.read_fp32:self.committed_fp32].mul_(scale)
            self.read_fp32 = self.committed_fp32

    def write_grad(self, param, grad_weight):
        """Write grad into buffer, advance write cursor. Returns True if handled."""
        if param not in self.param_map:
            return False
        buf, offset, numel = self.param_map[param]
        flat_grad = grad_weight.view(-1)
        grad_view = buf[offset:offset + numel]
        if not hasattr(param, '_ar_buf_written') or not param._ar_buf_written:
            grad_view.copy_(flat_grad)
            param._ar_buf_written = True
        else:
            grad_view.add_(flat_grad)
        # If buffer dtype matches param dtype, use param.grad directly.
        # Otherwise use main_grad (for fp32 buffer + bf16 param, matching Megatron).
        shaped_view = grad_view.view(param.shape)
        if buf.dtype == param.dtype:
            param.grad = shaped_view
        else:
            param.main_grad = shaped_view
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
        self.shared_dp_group = None
        self.shared_dp_world_size = 1
        self.expert_dp_group = None
        self.expert_dp_world_size = 1
        # Per-region gap budgets (ms) and AR bandwidth (bytes/ms). Set by tune.py.
        self.gap_budgets: dict = {}    # region -> gap duration in ms (from tuning)
        self.shared_ar_bw: float = 0.0  # shared AR bandwidth in bytes/ms (from tuning)
        self.expert_ar_bw: float = 0.0  # expert AR bandwidth in bytes/ms (from tuning)

        # Stats
        self.total_dw = 0
        self.completed_dw = 0
        self.total_ar = 0
        self.completed_ar = 0
        self.ar_during_overlap = 0
        self.ar_submitted_during_bwd = 0
        self.ar_submitted_during_finish = 0
        self._in_finish_batch = False

        # BDP profiling (gap measurement for tune.py)
        self._last_a2a_end_event = None
        self._last_a2a_end_region = None
        self._first_a2a_in_region = True
        self._gap_event_pairs = []
        self._region_gaps = {}

        # Lightweight profiling
        self.profiling = False
        self._region_name = None
        self._region_profiles = {}
        self._region_a2a_times = []
        self._region_dw_pairs = []
        self._region_start_evt = None

        # Communication visibility metrics
        self.comm_metrics_enabled = False
        self._a2a_event_pairs = []
        self._visible_wait_pairs = []
        self._ar_event_pairs = []
        self._ar_visible_wait_pairs = []

        # Event pool
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
            sm = get_stream_manager()
            self._stream_manager = sm
            self.comm_stream = sm.comm_stream
            self._ar_done_event = sm.get_sync_event(("scheduler", "ar_done"))
            self._dw_sync_event = sm.get_sync_event(("scheduler", "dw_sync"))
            _POOL_SIZE = 64
            self._event_pool = [
                sm.get_sync_event(("scheduler", "pool", idx))
                for idx in range(_POOL_SIZE)
            ]
            self._event_pool_size = _POOL_SIZE

    def enable(self):
        self.enabled = True
        self._init_cuda(force_reinit=True)

    def is_enabled(self):
        return self.enabled

    def configure_allreduce(self, enabled=True, shared_dp_group=None,
                            expert_dp_group=None, **kwargs):
        """Configure AllReduce (inline mode, no background thread).

        No dist.new_group() needed — AR is submitted inline by the main thread
        at deterministic code points, so NCCL call ordering is consistent
        across all ranks.
        """
        self.ar_enabled = enabled
        self.shared_dp_group = shared_dp_group
        self.shared_dp_world_size = dist.get_world_size(shared_dp_group) if shared_dp_group else (
            dist.get_world_size() if dist.is_initialized() else 1)
        self.expert_dp_group = expert_dp_group
        self.expert_dp_world_size = dist.get_world_size(expert_dp_group) if expert_dp_group else 1

        # Accept runtime AR parameters from tuning
        if 'gap_budgets' in kwargs:
            self.gap_budgets = kwargs['gap_budgets'] or {}
        if 'shared_ar_bw' in kwargs:
            self.shared_ar_bw = float(kwargs['shared_ar_bw'] or 0.0)
        if 'expert_ar_bw' in kwargs:
            self.expert_ar_bw = float(kwargs['expert_ar_bw'] or 0.0)

        self._init_cuda(force_reinit=True)

    def setup_ar_buffer(self, params):
        self._shared_ar.setup(params)

    def setup_expert_ar_buffer(self, params):
        self._expert_ar.setup(params)

    def _use_interleaved_ar(self) -> bool:
        return self.ar_enabled and (self.shared_dp_world_size > 1 or self.expert_dp_world_size > 1)

    # ========================================
    # dW Tasks
    # ========================================
    def register_dw_task(self, layer_name, layer_id, compute_fn,
                         weight_param=None, needs_ar=True, **kwargs):
        if not self.enabled:
            return
        target_buffer = None
        if weight_param is not None:
            if weight_param in self._shared_ar.param_map:
                target_buffer = self._shared_ar
            elif weight_param in self._expert_ar.param_map:
                target_buffer = self._expert_ar
        self._dw_queue.append(DWTask(layer_name, layer_id, compute_fn,
                                     weight_param, needs_ar, target_buffer))
        self.total_dw += 1

    def _execute_one_dw_task(self, task: DWTask):
        """Execute a single dW task and write result to AR buffer."""
        grad = task.compute_fn()
        if task.weight_param is not None and grad is not None:
            if grad.dtype != task.weight_param.dtype:
                grad = grad.to(task.weight_param.dtype)
            if task.target_buffer is not None:
                task.target_buffer.write_grad(task.weight_param, grad)
            else:
                if task.weight_param.grad is None:
                    task.weight_param.grad = grad
                else:
                    task.weight_param.grad.add_(grad)
                if task.needs_ar:
                    if self._use_interleaved_ar():
                        self._pending_ar_tensors.append(task.weight_param.grad)
                    elif self.shared_dp_world_size > 1:
                        self._ar_params_for_sync.append(task.weight_param)
        self.completed_dw += 1

    def _commit_and_submit_ar(self):
        """Record sync event, commit cursors, submit AR inline on comm_stream.
        Called after each dW task in finish_batch (commit_per_task=True)."""
        self._dw_sync_event.record(self.default_stream)
        self._shared_ar.commit()
        self._expert_ar.commit()
        self._submit_ar_inline(budgeted=False)

    def execute_dw_tasks(self, commit_per_task: bool = False) -> bool:
        """Execute all queued dW tasks.

        commit_per_task=True: commit + submit AR after each task (finish_batch,
            enables AR-dW overlap via comm_stream vs default_stream).
        commit_per_task=False: batch commit at end (backward regions
            where A2A will follow — AR submitted later in wait_alltoall).
        """
        dw_start_evt = None
        if self.profiling and self._region_name and self._dw_queue:
            dw_start_evt = torch.cuda.Event(enable_timing=True)
            dw_start_evt.record(self.default_stream)

        while self._dw_queue:
            task = self._dw_queue.popleft()
            self._execute_one_dw_task(task)
            if commit_per_task:
                self._commit_and_submit_ar()

        if dw_start_evt is not None:
            dw_end_evt = torch.cuda.Event(enable_timing=True)
            dw_end_evt.record(self.default_stream)
            self._region_dw_pairs.append((dw_start_evt, dw_end_evt))

        if not commit_per_task:
            # Batch commit: record once after all tasks
            self._dw_sync_event.record(self.default_stream)
            self._shared_ar.commit()
            self._expert_ar.commit()
            # AR not submitted here — will be submitted in wait_alltoall
            # after A2A completes (gap filling).
        return False

    # ========================================
    # AR: inline submission on comm_stream
    # ========================================
    def _has_ar_pending(self) -> bool:
        return (self._shared_ar.has_pending() or
                self._expert_ar.has_pending() or
                bool(self._pending_ar_tensors))

    def _compute_ar_budget(self):
        """Compute expert/shared byte budgets for current gap.

        Uses gap_budgets (ms) + AR bandwidth (bytes/ms) to dynamically
        compute how much AR data to submit, with expert-first splitting.
        Falls back to 0 (unlimited) if not configured.

        Returns (expert_budget, shared_budget) in bytes.  0 = unlimited.
        """
        region = self._region_name
        has_expert = self.expert_dp_world_size > 1 and self.expert_dp_group is not None
        has_shared = self.shared_dp_world_size > 1

        gap_ms = self.gap_budgets.get(region, 0) if region else 0
        if gap_ms > 0 and (self.expert_ar_bw > 0 or self.shared_ar_bw > 0):
            # Expert-first: use expert AR bandwidth first, remainder for shared
            expert_avail = self._expert_ar.pending_bytes() if has_expert else 0
            if has_expert and expert_avail > 0 and self.expert_ar_bw > 0:
                expert_time = expert_avail / self.expert_ar_bw
                if expert_time >= gap_ms:
                    return (int(self.expert_ar_bw * gap_ms), 0)
                else:
                    remaining_ms = gap_ms - expert_time
                    shared_budget = int(self.shared_ar_bw * remaining_ms) if self.shared_ar_bw > 0 else 0
                    return (expert_avail, shared_budget)
            else:
                shared_budget = int(self.shared_ar_bw * gap_ms) if has_shared and self.shared_ar_bw > 0 else 0
                return (0, shared_budget)

        # Not configured — no budgeted AR
        return (0, 0)

    def _submit_ar_inline(self, budgeted=True):
        """Submit AR on comm_stream. Called by main thread at deterministic points.

        budgeted=True:  use _compute_ar_budget (gap filling during backward)
        budgeted=False: submit all pending data (finish_batch drain)

        CPU cost: microseconds (just queuing NCCL kernels).
        GPU: comm_stream runs AR concurrently with default_stream compute.
        """
        has_expert_ar = self.expert_dp_world_size > 1 and self.expert_dp_group is not None
        has_shared = self.shared_dp_world_size > 1
        has_expert_buf = self._expert_ar is not None and self._expert_ar.has_pending()

        if not has_expert_ar and not has_shared and not has_expert_buf:
            return

        if budgeted:
            expert_budget, shared_budget = self._compute_ar_budget()
        else:
            expert_budget, shared_budget = 0, 0  # 0 = unlimited

        pre_scale = 1.0 / self.shared_dp_world_size if self.shared_dp_world_size > 1 else 1.0

        ops = 0
        with torch.cuda.stream(self.comm_stream):
            self.comm_stream.wait_event(self._dw_sync_event)

            need_timing = self.comm_metrics_enabled
            ar_s = torch.cuda.Event(enable_timing=True) if need_timing else None
            if ar_s:
                ar_s.record(self.comm_stream)

            # Expert: always pre_scale, only allreduce when expert_dp > 1
            if has_expert_buf:
                if has_expert_ar:
                    e_ops, _ = self._expert_ar.submit_block(
                        self.expert_dp_group, expert_budget,
                        pre_scale=pre_scale)
                    ops += e_ops
                elif pre_scale != 1.0:
                    # expert_dp=1: no allreduce, just scale for AVG convention
                    self._expert_ar.scale_pending(pre_scale)
            # Shared
            if has_shared and self._shared_ar.has_pending():
                s_ops, _ = self._shared_ar.submit_block(
                    self.shared_dp_group, shared_budget,
                    pre_scale=pre_scale)
                ops += s_ops
            # Fallback: params not in flat buffer
            if not budgeted:
                while self._pending_ar_tensors:
                    grad = self._pending_ar_tensors.popleft()
                    if grad is not None:
                        # Direct-grad fallback tensors may be dropped from
                        # model_param.grad as soon as optimizer preprocessing
                        # starts. Record the comm stream so the caching
                        # allocator does not recycle their storage before this
                        # async all-reduce completes.
                        grad.record_stream(self.comm_stream)
                        grad.div_(self.shared_dp_world_size)
                        dist.all_reduce(grad, group=self.shared_dp_group)
                        ops += 1

            if ops > 0:
                self._ar_done_event.record(self.comm_stream)
                self._last_ar_cuda_event = self._ar_done_event

            if need_timing and ops > 0:
                ar_e = torch.cuda.Event(enable_timing=True)
                ar_e.record(self.comm_stream)
                self._ar_event_pairs.append((ar_s, ar_e))

        if ops > 0:
            self._ar_task_count += ops
            self.total_ar += ops
            self.ar_during_overlap += ops
            if self._in_finish_batch:
                self.ar_submitted_during_finish += ops
            else:
                self.ar_submitted_during_bwd += ops

    # ========================================
    # AllToAll (inline on comm_stream)
    # ========================================
    def _get_pooled_event(self, idx: int) -> torch.cuda.Event:
        if idx < self._event_pool_size:
            return self._event_pool[idx]
        while self._event_pool_size <= idx:
            self._event_pool.append(
                self._stream_manager.get_sync_event(("scheduler", "pool", self._event_pool_size))
            )
            self._event_pool_size += 1
        return self._event_pool[idx]

    def submit_alltoall(self, comm_fn: Callable) -> int:
        """Execute comm_fn on comm_stream. NCCL kernel launch is non-blocking."""
        if not self.enabled:
            return comm_fn()

        task_id = self._alltoall_task_id
        self._alltoall_task_id += 1

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
                end_evt = self._get_pooled_event(task_id * 2 + 1)
                end_evt.record(self.comm_stream)

        self._alltoall_results[task_id] = (result, end_evt, start_evt)
        return task_id

    def submit_alltoall_batch(self, comm_fns: list) -> list:
        """Submit multiple AllToAll ops in one stream switch."""
        if not self.enabled:
            return [fn() for fn in comm_fns]

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

    def wait_alltoall(self, task_id: int, num_tasks: int = 1,
                      try_trickle: bool = True) -> Any:
        if not self.enabled:
            return None
        task_data = self._alltoall_results.get(task_id)
        if task_data is None:
            return None

        result, end_evt, start_evt = task_data

        # BDP gap tracking
        if self.profiling and self._first_a2a_in_region and start_evt is not None:
            if self._last_a2a_end_event is not None:
                self._gap_event_pairs.append((
                    self._last_a2a_end_region, self._last_a2a_end_event, start_evt))
                self._last_a2a_end_event = None
                self._last_a2a_end_region = None
            self._first_a2a_in_region = False

        if self.profiling and start_evt is not None and self._region_name:
            self._region_a2a_times.append((start_evt, end_evt))

        # A2A total/visible metrics
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
            self._alltoall_results.pop(tid, None)

        if not self._alltoall_results:
            self._last_a2a_end_event = end_evt
            self._last_a2a_end_region = self._region_name
            # Gap filling: submit AR inline now that all A2A are done.
            # All ranks reach this point at the same logical time,
            # so NCCL call ordering is consistent across ranks.
            if self._use_interleaved_ar():
                self._submit_ar_inline(budgeted=True)

        return result

    # ========================================
    # Synchronous AR (ar_enabled=False path)
    # ========================================
    def _sync_allreduce_all_params(self):
        if self.shared_dp_world_size > 1:
            self._shared_ar.sync_allreduce(
                self.shared_dp_group,
                pre_scale=1.0 / self.shared_dp_world_size,
            )
        if self.expert_dp_world_size > 1 and self.expert_dp_group is not None:
            self._expert_ar.sync_allreduce(
                self.expert_dp_group,
                pre_scale=1.0 / self.shared_dp_world_size,
            )
        elif self.shared_dp_world_size > 1 and self._expert_ar.bf16 is not None:
            # Expert params don't need AR (expert_dp=1) but still need scaling
            # to match shared params' AVG convention. Expert grads come from
            # all CP ranks via AlltoAll dispatch, so divide by dp_cp_size.
            self._expert_ar.bf16.mul_(1.0 / self.shared_dp_world_size)
        elif self.shared_dp_world_size > 1 and self._expert_ar.fp32 is not None:
            self._expert_ar.fp32.mul_(1.0 / self.shared_dp_world_size)
        for param in self._ar_params_for_sync:
            if param.grad is not None:
                param.grad.div_(self.shared_dp_world_size)
                dist.all_reduce(param.grad, group=self.shared_dp_group)
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
            # Execute dW with per-task commits + inline AR submission:
            # GPU overlap via comm_stream (AR) vs default_stream (dw)
            self.execute_dw_tasks(commit_per_task=True)
            # Flush any remaining AR (includes expert pre_scale even when expert_dp=1)
            self._submit_ar_inline(budgeted=False)
            # Final GPU sync: default_stream waits for comm_stream completion
            self._wait_all_ar()
        else:
            self.execute_dw_tasks()
            if needs_ar:
                self._sync_allreduce_all_params()

        self._last_a2a_end_event = None
        self._last_a2a_end_region = None
        self._in_finish_batch = False

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

    # ========================================
    # Profiling
    # ========================================
    def begin_region(self, name: str):
        self._region_name = name
        self._first_a2a_in_region = True
        if not self.profiling:
            return
        self._region_a2a_times = []
        self._region_dw_pairs = []
        self._region_start_evt = torch.cuda.Event(enable_timing=True)
        self._region_start_evt.record(self.default_stream)

    def end_region(self):
        if not self.profiling or self._region_name is None:
            self._region_name = None
            return
        name = self._region_name
        region_end_evt = torch.cuda.Event(enable_timing=True)
        region_end_evt.record(self.default_stream)
        if name not in self._region_profiles:
            self._region_profiles[name] = {'records': []}
        p = self._region_profiles[name]
        p['records'].append({
            'region_pair': (self._region_start_evt, region_end_evt),
            'a2a_pairs': list(self._region_a2a_times),
            'dw_pairs': list(self._region_dw_pairs),
        })
        self._region_start_evt = None
        self._region_name = None

    def get_region_profiles(self):
        torch.cuda.synchronize()
        result = {}
        for name, p in self._region_profiles.items():
            records = p.get('records', [])
            if not records:
                continue
            region_ms, comm_ms, dw_ms = [], [], []
            n_a2a = 0
            for rec in records:
                rs, re = rec['region_pair']
                region_ms.append(rs.elapsed_time(re))
                comm_ms.append(sum(s.elapsed_time(e) for s, e in rec['a2a_pairs']))
                dw_ms.append(sum(s.elapsed_time(e) for s, e in rec['dw_pairs']))
                n_a2a += len(rec['a2a_pairs'])
            T_region = sum(region_ms) / len(region_ms)
            T_comm = sum(comm_ms) / len(comm_ms)
            T_dW = sum(dw_ms) / len(dw_ms)
            result[name] = {
                'T_region': T_region,
                'T_comm': T_comm,
                'T_comp': max(0, T_region - max(T_comm, T_dW)),
                'T_dW': T_dW,
                'n_a2a': n_a2a // len(records),
            }
        return result

    # ========================================
    # Gap profiling (for tune.py chunk search)
    # ========================================
    def reset_gap_times(self):
        self._gap_event_pairs = []
        self._region_gaps = {}
        self._last_a2a_end_event = None
        self._last_a2a_end_region = None
        self._first_a2a_in_region = True

    def process_gap_events(self):
        torch.cuda.synchronize()
        for region, end_evt, start_evt in self._gap_event_pairs:
            gap_ms = end_evt.elapsed_time(start_evt)
            if region not in self._region_gaps:
                self._region_gaps[region] = []
            self._region_gaps[region].append(gap_ms)
        self._gap_event_pairs = []

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
        a2a_total = 0.0
        a2a_visible = 0.0
        for (a2a_s, a2a_e), (w_s, w_e) in zip(self._a2a_event_pairs, self._visible_wait_pairs):
            a2a_ms = a2a_s.elapsed_time(a2a_e)
            wait_ms = w_s.elapsed_time(w_e)
            a2a_total += a2a_ms
            a2a_visible += min(wait_ms, a2a_ms)
        a2a_overlap = 0.0 if a2a_total <= 1e-9 else max(0.0, min(1.0, 1.0 - a2a_visible / a2a_total))

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

    @staticmethod
    def measure_ar_bandwidth(ar_group=None, sizes_mb=(1, 2, 4, 8, 16, 32, 64, 96, 128),
                             warmup=5, repeat=20, dtype=torch.float32) -> dict:
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

    def get_stats(self):
        return {
            'total_dw_tasks': self.total_dw,
            'completed_dw_tasks': self.completed_dw,
            'total_ar_tasks': self.total_ar,
            'completed_ar_tasks': self.completed_ar,
            'ar_during_gap': self.ar_during_overlap,
            'ar_submitted_during_bwd': self.ar_submitted_during_bwd,
            'ar_submitted_during_finish': self.ar_submitted_during_finish,
            'pending_dw_at_finish': len(self._dw_queue),
            'region_gaps': dict(self._region_gaps),
            'shared_ar_bw': self.shared_ar_bw,
            'expert_ar_bw': self.expert_ar_bw,
        }

    @classmethod
    def reset(cls):
        cls._instance = None


def get_backward_scheduler() -> BackwardScheduler:
    return BackwardScheduler()
