"""
FluidMoE Backward Scheduler

Design:
- dW overlaps with AllToAll (dW on default_stream, AllToAll on comm_stream)
- comm_thread handles AllToAll dispatch
- AR is enqueued inline by the main thread on comm_stream at flush points
- Both A2A and AR share comm_stream → CUDA stream ordering prevents
  cross-communicator NCCL deadlocks (no concurrent GPU kernels from
  different PGs)
- Deterministic ordering: main thread controls when AR goes onto comm_stream,
  and the backward execution order is the same on all ranks
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Callable, Any
from dataclasses import dataclass
import threading
import queue
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


@dataclass
class AllToAllTask:
    """AllToAll communication task."""
    task_id: int
    comm_fn: Callable
    input_event: Any  # CUDA event from default_stream (sync point)
    done_event: threading.Event
    result_holder: list  # [result, end_event, start_event]


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
        self._ar_params_for_sync = []  # params needing AR when ar_enabled=False

        # AllToAll: comm_thread + comm_stream
        self._alltoall_queue = queue.Queue()
        self._comm_thread = None
        self._comm_thread_stop = False

        # AllToAll tracking
        self._alltoall_task_id = 0
        self._alltoall_results = {}
        self._a2a_submitted = 0   # Total AllToAll tasks submitted
        self._a2a_completed = 0   # Total AllToAll tasks completed by comm_thread
        self._a2a_lock = threading.Lock()
        self._a2a_idle = threading.Event()  # Set when no AllToAll in flight
        self._a2a_idle.set()
        self._last_a2a_event = None  # Last AllToAll cuda_event

        # AR tracking (inline execution from main thread on comm_stream)
        self._pending_ar_tensors = deque()  # grads collected during dW (fallback)
        self._ar_task_count = 0       # total AR ops executed this iteration
        self._last_ar_cuda_event = None  # last AR completion event on comm_stream
        # Pre-allocated CUDA event for AR completion tracking (avoids cudaEventCreate per flush)
        self._ar_done_event = None

        # Flat AR buffer (DDP-style zero-copy trickle) — shared params
        self._ar_param_map = {}       # param → (buffer, offset, numel)
        self._ar_buffer_bf16 = None   # flat contiguous buffer for bf16 grads
        self._ar_buffer_fp32 = None   # flat contiguous buffer for fp32 grads
        self._ar_read_cursor_bf16 = 0   # how far AR has been submitted
        self._ar_write_cursor_bf16 = 0  # how far dW has written grads
        self._ar_read_cursor_fp32 = 0
        self._ar_write_cursor_fp32 = 0

        # Flat AR buffer — expert params (separate group, separate buffer)
        self._expert_ar_param_map = {}
        self._expert_ar_buffer_bf16 = None
        self._expert_ar_buffer_fp32 = None
        self._expert_ar_read_cursor_bf16 = 0
        self._expert_ar_write_cursor_bf16 = 0
        self._expert_ar_read_cursor_fp32 = 0
        self._expert_ar_write_cursor_fp32 = 0

        # AR config
        self.ar_enabled = False
        self.ar_safe_mode = False
        self.ar_lockstep = False
        self.shared_dp_group = None
        self.ar_group = None  # Independent NCCL communicator for shared param AR
        self.ar_ctrl_group = None  # Optional Gloo control group for lockstep gating
        self.shared_dp_world_size = 1
        self.expert_dp_group = None
        self.expert_ar_group = None  # Independent NCCL communicator for expert param AR
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

        # BDP: per-region gap on comm_stream (via CUDA events)
        # T_gap[R] = comm_stream time from last A2A end (region R) to first A2A start (region R+1)
        # This is the exact window on comm_stream available for AR.
        self._last_a2a_end_event = None     # comm_stream event: last A2A completion in region
        self._last_a2a_end_region = None    # region name of that last A2A
        self._first_a2a_in_region = True    # flag: next A2A is the first in current region
        self._gap_event_pairs = []          # [(region, end_event, start_event), ...]
        self._region_gaps = {}              # region_name -> [gap_ms, ...] (filled after sync)
        # Per-region trickle sizes (bytes). Set by tune.py after BDP profiling.
        # Budget is per-REGION (accumulated across all trickle calls within a region).
        self.ar_trickle_sizes = {}  # per-region override

        # Profiling: per-region timing
        self.profiling = False
        self._region_name = None
        self._region_profiles = {}  # region -> {T_comm, T_comp, T_dW, count}
        self._region_a2a_times = []  # comm times collected in current region
        self._region_dw_time = 0.0   # dW time accumulated in current region

        self._init_cuda()

    def _init_cuda(self, force_reinit=False):
        """Initialize CUDA resources on current device."""
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()

            if not force_reinit and self._device == current_device and self.comm_stream is not None:
                return

            self._device = current_device
            self.default_stream = torch.cuda.default_stream(current_device)
            # Use StreamManager's comm_stream to share with forward P2P operations
            from fluid.core.stream import get_stream_manager
            self.comm_stream = get_stream_manager().comm_stream
            # Pre-allocate reusable CUDA event (avoids cudaEventCreate per flush)
            self._ar_done_event = torch.cuda.Event()

    def enable(self):
        self.enabled = True
        self._init_cuda(force_reinit=True)

        # Start comm thread (AllToAll)
        if self._comm_thread is None:
            self._start_comm_thread()

    def is_enabled(self):
        return self.enabled

    def configure_allreduce(self, enabled=True, shared_dp_group=None, expert_dp_group=None, **kwargs):
        """Configure AllReduce settings for shared and expert parameters.

        Args:
            shared_dp_group: Process group for shared parameter gradient AR.
                This should include ALL ranks that hold identical shared params
                (= all ranks across both dp and cp dimensions).
            expert_dp_group: Process group for expert parameter gradient AR.
                This should include only ranks holding the same expert partition
                (= dp subgroup). None means no expert AR (dp=1).

        Creates independent NCCL communicators for AllReduce.
        AR runs on comm_stream (same as A2A) to prevent cross-communicator
        NCCL deadlocks via CUDA stream-level serialization.
        """
        self.ar_enabled = enabled
        self.shared_dp_group = shared_dp_group
        self.shared_dp_world_size = dist.get_world_size(shared_dp_group) if shared_dp_group else (
            dist.get_world_size() if dist.is_initialized() else 1)
        self.expert_dp_group = expert_dp_group
        self.expert_dp_world_size = dist.get_world_size(expert_dp_group) if expert_dp_group else 1

        mode_env = os.environ.get("FLUID_AR_WINDOW_MODE", "strict_window").lower()
        if mode_env in ("strict", "strict_window", "strict-window"):
            self.ar_window_mode = "strict_window"
        elif mode_env in ("relaxed", "relaxed_window", "relaxed-window"):
            self.ar_window_mode = "relaxed_window"
        else:
            self.ar_window_mode = "strict_window"

        # Safe mode policy
        safe_env = os.environ.get("FLUID_AR_SAFE_MODE", "auto").lower()
        if safe_env in ("1", "true", "yes", "on"):
            self.ar_safe_mode = True
        elif safe_env in ("0", "false", "no", "off"):
            self.ar_safe_mode = False
        else:
            self.ar_safe_mode = False

        # Lockstep: Gloo barrier at flush points for cross-rank synchronization
        lockstep_env = os.environ.get("FLUID_AR_LOCKSTEP", "auto").lower()
        if lockstep_env in ("1", "true", "yes", "on"):
            self.ar_lockstep = True
        elif lockstep_env in ("0", "false", "no", "off"):
            self.ar_lockstep = False
        else:
            self.ar_lockstep = self.shared_dp_world_size > 1
        if self.ar_safe_mode:
            self.ar_lockstep = False

        # Create independent communicator for shared param AR
        if shared_dp_group is not None and dist.is_initialized():
            ranks = dist.get_process_group_ranks(shared_dp_group)
            self.ar_group = dist.new_group(ranks)
            if self.shared_dp_world_size > 1 and self.ar_lockstep:
                self.ar_ctrl_group = dist.new_group(ranks=ranks, backend="gloo")
            else:
                self.ar_ctrl_group = None
        else:
            self.ar_group = None
            self.ar_ctrl_group = None

        # Create independent communicator for expert param AR
        if expert_dp_group is not None and dist.is_initialized() and self.expert_dp_world_size > 1:
            expert_ranks = dist.get_process_group_ranks(expert_dp_group)
            self.expert_ar_group = dist.new_group(expert_ranks)
        else:
            self.expert_ar_group = None

        self._init_cuda(force_reinit=True)

    def setup_ar_buffer(self, params):
        """Set up flat AR buffer from parameters in backward execution order.

        Allocates contiguous buffers (one per dtype) and maps each parameter
        to a segment.  During backward, execute_dw_tasks() writes grads
        directly into the buffer; _trickle_ar() advances a cursor with a
        single dist.all_reduce() call per trickle — zero copy, zero alloc.

        Args:
            params: list of nn.Parameter in the order dW tasks will execute
                    (i.e. backward execution order: last layer first).
        """
        self._ar_param_map = {}

        # Group by dtype
        bf16_params = [(p, p.numel()) for p in params if p.dtype == torch.bfloat16]
        fp32_params = [(p, p.numel()) for p in params if p.dtype == torch.float32]

        device = params[0].device if params else torch.cuda.current_device()

        # Allocate bf16 buffer
        if bf16_params:
            total = sum(n for _, n in bf16_params)
            self._ar_buffer_bf16 = torch.zeros(total, dtype=torch.bfloat16, device=device)
            offset = 0
            for p, n in bf16_params:
                self._ar_param_map[p] = (self._ar_buffer_bf16, offset, n)
                offset += n
        else:
            self._ar_buffer_bf16 = None

        # Allocate fp32 buffer
        if fp32_params:
            total = sum(n for _, n in fp32_params)
            self._ar_buffer_fp32 = torch.zeros(total, dtype=torch.float32, device=device)
            offset = 0
            for p, n in fp32_params:
                self._ar_param_map[p] = (self._ar_buffer_fp32, offset, n)
                offset += n
        else:
            self._ar_buffer_fp32 = None

        self._ar_read_cursor_bf16 = 0
        self._ar_write_cursor_bf16 = 0
        self._ar_read_cursor_fp32 = 0
        self._ar_write_cursor_fp32 = 0

    def setup_expert_ar_buffer(self, params):
        """Set up flat AR buffer for expert parameters (uses expert_dp_group).

        Same mechanism as setup_ar_buffer but uses a separate buffer and
        AR group for expert params that are partitioned by EP.
        """
        self._expert_ar_param_map = {}
        if not params:
            self._expert_ar_buffer_bf16 = None
            self._expert_ar_buffer_fp32 = None
            return

        bf16_params = [(p, p.numel()) for p in params if p.dtype == torch.bfloat16]
        fp32_params = [(p, p.numel()) for p in params if p.dtype == torch.float32]
        device = params[0].device

        if bf16_params:
            total = sum(n for _, n in bf16_params)
            self._expert_ar_buffer_bf16 = torch.zeros(total, dtype=torch.bfloat16, device=device)
            offset = 0
            for p, n in bf16_params:
                self._expert_ar_param_map[p] = (self._expert_ar_buffer_bf16, offset, n)
                offset += n
        else:
            self._expert_ar_buffer_bf16 = None

        if fp32_params:
            total = sum(n for _, n in fp32_params)
            self._expert_ar_buffer_fp32 = torch.zeros(total, dtype=torch.float32, device=device)
            offset = 0
            for p, n in fp32_params:
                self._expert_ar_param_map[p] = (self._expert_ar_buffer_fp32, offset, n)
                offset += n
        else:
            self._expert_ar_buffer_fp32 = None

        self._expert_ar_read_cursor_bf16 = 0
        self._expert_ar_write_cursor_bf16 = 0
        self._expert_ar_read_cursor_fp32 = 0
        self._expert_ar_write_cursor_fp32 = 0

    def _use_interleaved_ar(self) -> bool:
        """Whether interleaved AR submission is allowed in current mode."""
        return self.ar_enabled and (not self.ar_safe_mode)

    def _start_comm_thread(self):
        """Start AllToAll communication thread."""
        self._comm_thread_stop = False
        self._comm_thread = threading.Thread(
            target=self._comm_thread_worker,
            daemon=True
        )
        self._comm_thread.start()

    def _comm_thread_worker(self):
        """Comm thread: handles AllToAll only (on comm_stream).

        IMPORTANT: _a2a_idle and _last_a2a_event are updated BEFORE
        done_event.set() so that the main thread sees _a2a_idle==True
        immediately when it wakes from wait_alltoall().  This enables
        safe AR trickling from the main thread.
        """
        if self._device is not None:
            torch.cuda.set_device(self._device)

        while not self._comm_thread_stop:
            try:
                task = self._alltoall_queue.get(timeout=0.001)
                if task is None:
                    break
                self._execute_alltoall(task)
                with self._a2a_lock:
                    self._last_a2a_event = task.result_holder[1]
                    self._a2a_completed += 1
                    if self._a2a_completed >= self._a2a_submitted:
                        self._a2a_idle.set()
                # Signal AFTER _a2a_idle update
                task.done_event.set()
            except queue.Empty:
                pass

    def _execute_alltoall(self, task: AllToAllTask):
        """Execute AllToAll task on comm_stream.

        NOTE: done_event is NOT set here.  The caller (_comm_thread_worker)
        sets it AFTER updating _a2a_idle so that the main thread can see
        _a2a_idle==True immediately when done_event unblocks wait_alltoall().

        Records start_event on comm_stream AFTER wait_event(input_event) resolves
        but BEFORE the actual A2A runs.  This gives the exact A2A start time on
        comm_stream for BDP gap measurement.
        """
        with torch.cuda.stream(self.comm_stream):
            # Wait for default_stream to reach the submit point
            self.comm_stream.wait_event(task.input_event)
            # Record A2A start on comm_stream (after wait_event, before A2A)
            start_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record(self.comm_stream)
            # Execute the actual AllToAll
            result = task.comm_fn()
            # Record A2A end on comm_stream
            end_evt = torch.cuda.Event(enable_timing=True)
            end_evt.record(self.comm_stream)

        task.result_holder[0] = result
        task.result_holder[1] = end_evt    # [1] = end event (used by wait_alltoall)
        task.result_holder.append(start_evt)  # [2] = start event (used by BDP gap)

    # ========================================
    # dW Tasks
    # ========================================
    def register_dw_task(self, layer_name, layer_id, compute_fn, weight_param=None, needs_ar=True, **kwargs):
        """Register a deferred dW task."""
        if not self.enabled:
            return
        self._dw_queue.append(DWTask(layer_name, layer_id, compute_fn, weight_param, needs_ar))
        self.total_dw += 1

    def execute_dw_tasks(self, check_alltoall_id: Optional[int] = None) -> bool:
        """Execute deferred dW tasks, optionally yielding when AllToAll completes."""
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

                if task.weight_param in self._ar_param_map:
                    # Shared flat buffer path
                    buf, offset, numel = self._ar_param_map[task.weight_param]
                    flat_grad = grad_weight.view(-1)
                    if task.weight_param.grad is None:
                        buf[offset:offset+numel].copy_(flat_grad)
                    else:
                        buf[offset:offset+numel].add_(flat_grad)
                    task.weight_param.grad = buf[offset:offset+numel].view(task.weight_param.shape)
                    # Advance write cursor
                    end = offset + numel
                    if buf.dtype == torch.bfloat16:
                        if end > self._ar_write_cursor_bf16:
                            self._ar_write_cursor_bf16 = end
                    else:
                        if end > self._ar_write_cursor_fp32:
                            self._ar_write_cursor_fp32 = end
                elif task.weight_param in self._expert_ar_param_map:
                    # Expert flat buffer path
                    buf, offset, numel = self._expert_ar_param_map[task.weight_param]
                    flat_grad = grad_weight.view(-1)
                    if task.weight_param.grad is None:
                        buf[offset:offset+numel].copy_(flat_grad)
                    else:
                        buf[offset:offset+numel].add_(flat_grad)
                    task.weight_param.grad = buf[offset:offset+numel].view(task.weight_param.shape)
                    end = offset + numel
                    if buf.dtype == torch.bfloat16:
                        if end > self._expert_ar_write_cursor_bf16:
                            self._expert_ar_write_cursor_bf16 = end
                    else:
                        if end > self._expert_ar_write_cursor_fp32:
                            self._expert_ar_write_cursor_fp32 = end
                else:
                    # Original path for unregistered params
                    if task.weight_param.grad is None:
                        task.weight_param.grad = grad_weight
                    else:
                        task.weight_param.grad.add_(grad_weight)

                    # AR handling (fallback)
                    if task.needs_ar and self.shared_dp_world_size > 1:
                        if self._use_interleaved_ar():
                            self._pending_ar_tensors.append(task.weight_param.grad)
                        else:
                            self._ar_params_for_sync.append(task.weight_param)

            self.completed_dw += 1

            # Check if the monitored AllToAll has completed
            if check_alltoall_id is not None:
                task_data = self._alltoall_results.get(check_alltoall_id)
                if task_data is not None and task_data[0].is_set():
                    if t0 is not None:
                        torch.cuda.synchronize()
                        self._region_dw_time += (time.perf_counter() - t0) * 1000
                    return True

        if t0 is not None:
            torch.cuda.synchronize()
            self._region_dw_time += (time.perf_counter() - t0) * 1000
        return False

    # ========================================
    # AllToAll
    # ========================================
    def submit_alltoall(self, comm_fn: Callable) -> int:
        """Submit AllToAll to comm thread.

        Since AR is now enqueued inline on comm_stream by the main thread at
        flush points, and A2A is enqueued by comm_thread on comm_stream, the
        CUDA stream ordering guarantees A2A kernels only run after any
        preceding AR kernels complete on comm_stream.  No explicit lock or
        wait is needed.
        """
        if not self.enabled:
            return comm_fn()

        task_id = self._alltoall_task_id
        self._alltoall_task_id += 1

        # Record event on default_stream (sync point for comm_stream)
        input_event = torch.cuda.Event()
        input_event.record(self.default_stream)

        done_event = threading.Event()
        result_holder = [None, None]

        task = AllToAllTask(
            task_id=task_id,
            comm_fn=comm_fn,
            input_event=input_event,
            done_event=done_event,
            result_holder=result_holder,
        )

        # Mark AllToAll submitted
        with self._a2a_lock:
            self._a2a_submitted += 1
            self._a2a_idle.clear()

        self._alltoall_queue.put(task)
        self._alltoall_results[task_id] = (done_event, result_holder)

        return task_id

    def wait_alltoall(self, task_id: int, num_tasks: int = 1) -> Any:
        """Wait for AllToAll to complete."""
        if not self.enabled:
            return None

        task_data = self._alltoall_results.get(task_id)
        if task_data is None:
            return None

        done_event, result_holder = task_data

        done_event.wait()

        result = result_holder[0]
        end_evt = result_holder[1]   # A2A end on comm_stream
        start_evt = result_holder[2] if len(result_holder) > 2 else None  # A2A start on comm_stream

        # BDP: pair first A2A's start_event with previous region's last A2A end_event
        if self._first_a2a_in_region and start_evt is not None:
            if self._last_a2a_end_event is not None:
                self._gap_event_pairs.append((
                    self._last_a2a_end_region,
                    self._last_a2a_end_event,  # comm_stream: last A2A end
                    start_evt,                  # comm_stream: first A2A start
                ))
                self._last_a2a_end_event = None
                self._last_a2a_end_region = None
            self._first_a2a_in_region = False

        # Collect per-AllToAll comm time for profiling
        if self.profiling and start_evt is not None:
            torch.cuda.synchronize()
            comm_ms = start_evt.elapsed_time(end_evt)
            if self._region_name:
                self._region_a2a_times.append(comm_ms)

        # Wait for comm_stream completion on default_stream
        if end_evt is not None:
            self.default_stream.wait_event(end_evt)

        # Clean up this task and all earlier tasks in the batch
        first_task_id = task_id - num_tasks + 1
        for tid in range(first_task_id, task_id + 1):
            self._alltoall_results.pop(tid, None)

        # BDP: if this was the last A2A in the region, save end_event
        if not self._alltoall_results:
            self._last_a2a_end_event = end_evt
            self._last_a2a_end_region = self._region_name

        # Trickle one AR chunk onto comm_stream while it's idle
        self._trickle_ar()

        return result

    # ========================================
    # AllReduce (inline on comm_stream)
    # ========================================
    def _trickle_ar(self):
        """Trickle AR onto comm_stream using flat buffer cursor or fallback queue.

        Handles both shared param buffer (ar_group) and expert param buffer
        (expert_ar_group) in a single comm_stream submission.

        Determinism guarantee:
        1. execute_dw_tasks() always drains the full dW queue, so buffer
           write cursors and _pending_ar_tensors are identical on all ranks.
        2. We only trickle when _alltoall_results is empty (all submitted
           A2As have been waited for).  This is a deterministic condition
           that does NOT depend on timing.
        """
        if not self._use_interleaved_ar():
            return

        has_shared = self.shared_dp_world_size > 1
        has_expert = self.expert_dp_world_size > 1

        if not has_shared and not has_expert:
            return

        # Check if there's anything to submit (shared)
        has_shared_buffer = has_shared and (
            (self._ar_buffer_bf16 is not None and self._ar_read_cursor_bf16 < self._ar_write_cursor_bf16) or
            (self._ar_buffer_fp32 is not None and self._ar_read_cursor_fp32 < self._ar_write_cursor_fp32)
        )
        # Check if there's anything to submit (expert)
        has_expert_buffer = has_expert and (
            (self._expert_ar_buffer_bf16 is not None and self._expert_ar_read_cursor_bf16 < self._expert_ar_write_cursor_bf16) or
            (self._expert_ar_buffer_fp32 is not None and self._expert_ar_read_cursor_fp32 < self._expert_ar_write_cursor_fp32)
        )
        if not has_shared_buffer and not has_expert_buffer and not self._pending_ar_tensors:
            return

        # Deterministic check: only trickle when ALL submitted A2As have
        # been waited for.
        if self._alltoall_results:
            return

        region = self._region_name
        budget = self.ar_trickle_sizes.get(region, 0) if region else 0
        _MIN_TRICKLE_NUMEL = 512 * 1024  # ~1 MB for bf16

        # --- Pre-compute shared buffer ranges ---
        bf16_start = bf16_end = 0
        if has_shared and self._ar_buffer_bf16 is not None:
            cursor = self._ar_read_cursor_bf16
            write_end = self._ar_write_cursor_bf16
            if cursor < write_end:
                if budget > 0:
                    budget_numel = budget // self._ar_buffer_bf16.element_size()
                    bf16_end = min(cursor + budget_numel, write_end)
                else:
                    bf16_end = write_end
                bf16_start = cursor

        fp32_start = fp32_end = 0
        if has_shared and self._ar_buffer_fp32 is not None:
            cursor32 = self._ar_read_cursor_fp32
            write_end32 = self._ar_write_cursor_fp32
            if cursor32 < write_end32:
                fp32_start = cursor32
                fp32_end = write_end32

        # --- Pre-compute expert buffer ranges ---
        exp_bf16_start = exp_bf16_end = 0
        if has_expert and self._expert_ar_buffer_bf16 is not None:
            cursor = self._expert_ar_read_cursor_bf16
            write_end = self._expert_ar_write_cursor_bf16
            if cursor < write_end:
                if budget > 0:
                    budget_numel = budget // self._expert_ar_buffer_bf16.element_size()
                    exp_bf16_end = min(cursor + budget_numel, write_end)
                else:
                    exp_bf16_end = write_end
                exp_bf16_start = cursor

        exp_fp32_start = exp_fp32_end = 0
        if has_expert and self._expert_ar_buffer_fp32 is not None:
            cursor32 = self._expert_ar_read_cursor_fp32
            write_end32 = self._expert_ar_write_cursor_fp32
            if cursor32 < write_end32:
                exp_fp32_start = cursor32
                exp_fp32_end = write_end32

        # Check minimum size thresholds
        bf16_ok = (bf16_end - bf16_start) >= _MIN_TRICKLE_NUMEL
        fp32_ok = (fp32_end - fp32_start) >= _MIN_TRICKLE_NUMEL
        exp_bf16_ok = (exp_bf16_end - exp_bf16_start) >= _MIN_TRICKLE_NUMEL
        exp_fp32_ok = (exp_fp32_end - exp_fp32_start) >= _MIN_TRICKLE_NUMEL
        has_work = bf16_ok or fp32_ok or exp_bf16_ok or exp_fp32_ok or bool(self._pending_ar_tensors)
        if not has_work:
            return

        # Only now touch comm_stream
        self._a2a_idle.wait()
        ar_group = self.ar_group if self.ar_group is not None else self.shared_dp_group
        expert_ar_group = self.expert_ar_group if self.expert_ar_group is not None else self.expert_dp_group

        with torch.cuda.stream(self.comm_stream):
            # Ensure comm_stream sees all gradients written on default_stream
            self.comm_stream.wait_stream(self.default_stream)

            # --- Shared flat buffer: bf16 ---
            if bf16_ok:
                dist.all_reduce(self._ar_buffer_bf16[bf16_start:bf16_end], group=ar_group)
                self._ar_read_cursor_bf16 = bf16_end
                self._ar_task_count += 1
                self.total_ar += 1
                self.ar_during_overlap += 1
                if self._in_finish_batch:
                    self.ar_submitted_during_finish += 1
                else:
                    self.ar_submitted_during_bwd += 1

            # --- Shared flat buffer: fp32 ---
            if fp32_ok:
                dist.all_reduce(self._ar_buffer_fp32[fp32_start:fp32_end], group=ar_group)
                self._ar_read_cursor_fp32 = fp32_end
                self._ar_task_count += 1
                self.total_ar += 1
                self.ar_during_overlap += 1
                if self._in_finish_batch:
                    self.ar_submitted_during_finish += 1
                else:
                    self.ar_submitted_during_bwd += 1

            # --- Expert flat buffer: bf16 ---
            if exp_bf16_ok:
                dist.all_reduce(self._expert_ar_buffer_bf16[exp_bf16_start:exp_bf16_end], group=expert_ar_group)
                self._expert_ar_read_cursor_bf16 = exp_bf16_end
                self._ar_task_count += 1
                self.total_ar += 1
                self.ar_during_overlap += 1
                if self._in_finish_batch:
                    self.ar_submitted_during_finish += 1
                else:
                    self.ar_submitted_during_bwd += 1

            # --- Expert flat buffer: fp32 ---
            if exp_fp32_ok:
                dist.all_reduce(self._expert_ar_buffer_fp32[exp_fp32_start:exp_fp32_end], group=expert_ar_group)
                self._expert_ar_read_cursor_fp32 = exp_fp32_end
                self._ar_task_count += 1
                self.total_ar += 1
                self.ar_during_overlap += 1
                if self._in_finish_batch:
                    self.ar_submitted_during_finish += 1
                else:
                    self.ar_submitted_during_bwd += 1

            # --- Fallback: pending tensors (shared group) ---
            bytes_submitted = 0
            while self._pending_ar_tensors:
                if budget > 0 and bytes_submitted >= budget:
                    break
                grad = self._pending_ar_tensors.popleft()
                if grad is None:
                    continue
                dist.all_reduce(grad, group=ar_group)
                bytes_submitted += grad.numel() * grad.element_size()
                self._ar_task_count += 1
                self.total_ar += 1
                self.ar_during_overlap += 1
                if self._in_finish_batch:
                    self.ar_submitted_during_finish += 1
                else:
                    self.ar_submitted_during_bwd += 1

            # No need to record a done event here — _flush_ar_pending_strict()
            # will record the final event before _wait_all_ar() reads it.

    def _flush_ar_pending_strict(self, final: bool = False):
        """Flush AR: drain all remaining buffer and pending tensors.

        Used by finish_batch() to ensure all AR completes before
        optimizer.step().  No budget limit — submits everything remaining.
        Handles both shared and expert buffers.
        """
        has_shared_remaining = (
            (self._ar_buffer_bf16 is not None and self._ar_read_cursor_bf16 < self._ar_buffer_bf16.numel()) or
            (self._ar_buffer_fp32 is not None and self._ar_read_cursor_fp32 < self._ar_buffer_fp32.numel())
        )
        has_expert_remaining = (
            (self._expert_ar_buffer_bf16 is not None and self._expert_ar_read_cursor_bf16 < self._expert_ar_buffer_bf16.numel()) or
            (self._expert_ar_buffer_fp32 is not None and self._expert_ar_read_cursor_fp32 < self._expert_ar_buffer_fp32.numel())
        )
        if not self._pending_ar_tensors and not has_shared_remaining and not has_expert_remaining:
            if self.ar_lockstep and self.ar_ctrl_group is not None:
                timeout_s = int(os.environ.get("FLUID_AR_LOCKSTEP_TIMEOUT_SEC", "120"))
                dist.monitored_barrier(
                    group=self.ar_ctrl_group,
                    timeout=timedelta(seconds=timeout_s),
                )
            return

        self._a2a_idle.wait()

        ar_group = self.ar_group if self.ar_group is not None else self.shared_dp_group
        expert_ar_group = self.expert_ar_group if self.expert_ar_group is not None else self.expert_dp_group

        with torch.cuda.stream(self.comm_stream):
            # Ensure comm_stream sees all gradients written on default_stream
            self.comm_stream.wait_stream(self.default_stream)

            # --- Shared: flush bf16 ---
            if self._ar_buffer_bf16 is not None:
                buf = self._ar_buffer_bf16
                cursor = self._ar_read_cursor_bf16
                total = buf.numel()
                if cursor < total:
                    dist.all_reduce(buf[cursor:total], group=ar_group)
                    self._ar_read_cursor_bf16 = total
                    self._ar_task_count += 1
                    self.total_ar += 1
                    self.ar_during_overlap += 1
                    self.ar_submitted_during_finish += 1

            # --- Shared: flush fp32 ---
            if self._ar_buffer_fp32 is not None:
                buf32 = self._ar_buffer_fp32
                cursor32 = self._ar_read_cursor_fp32
                total32 = buf32.numel()
                if cursor32 < total32:
                    dist.all_reduce(buf32[cursor32:total32], group=ar_group)
                    self._ar_read_cursor_fp32 = total32
                    self._ar_task_count += 1
                    self.total_ar += 1
                    self.ar_during_overlap += 1
                    self.ar_submitted_during_finish += 1

            # --- Expert: flush bf16 ---
            if self._expert_ar_buffer_bf16 is not None and expert_ar_group is not None:
                buf = self._expert_ar_buffer_bf16
                cursor = self._expert_ar_read_cursor_bf16
                total = buf.numel()
                if cursor < total:
                    dist.all_reduce(buf[cursor:total], group=expert_ar_group)
                    self._expert_ar_read_cursor_bf16 = total
                    self._ar_task_count += 1
                    self.total_ar += 1
                    self.ar_during_overlap += 1
                    self.ar_submitted_during_finish += 1

            # --- Expert: flush fp32 ---
            if self._expert_ar_buffer_fp32 is not None and expert_ar_group is not None:
                buf32 = self._expert_ar_buffer_fp32
                cursor32 = self._expert_ar_read_cursor_fp32
                total32 = buf32.numel()
                if cursor32 < total32:
                    dist.all_reduce(buf32[cursor32:total32], group=expert_ar_group)
                    self._expert_ar_read_cursor_fp32 = total32
                    self._ar_task_count += 1
                    self.total_ar += 1
                    self.ar_during_overlap += 1
                    self.ar_submitted_during_finish += 1

            # Flush pending tensors (unregistered params, shared group)
            while self._pending_ar_tensors:
                grad = self._pending_ar_tensors.popleft()
                if grad is None:
                    continue
                dist.all_reduce(grad, group=ar_group)
                self._ar_task_count += 1
                self.total_ar += 1
                self.ar_during_overlap += 1
                self.ar_submitted_during_finish += 1

            self._ar_done_event.record(self.comm_stream)
            self._last_ar_cuda_event = self._ar_done_event

        if self.ar_lockstep and self.ar_ctrl_group is not None:
            timeout_s = int(os.environ.get("FLUID_AR_LOCKSTEP_TIMEOUT_SEC", "120"))
            dist.monitored_barrier(
                group=self.ar_ctrl_group,
                timeout=timedelta(seconds=timeout_s),
            )

    def flush_ar_pending(self, final: bool = False):
        """Submit all deferred AR tensors in deterministic order."""
        if not self._use_interleaved_ar() or (self.shared_dp_world_size <= 1 and self.expert_dp_world_size <= 1):
            self._pending_ar_tensors.clear()
            return

        self._flush_ar_pending_strict(final=final)

    def _wait_all_ar(self):
        """Wait for all enqueued AR to complete on comm_stream."""
        if self._last_ar_cuda_event is None:
            return
        # All AR chunks were enqueued on comm_stream in FIFO order.
        # Waiting for the last event implies all are done.
        self.default_stream.wait_event(self._last_ar_cuda_event)
        self.completed_ar = self._ar_task_count

    def _sync_allreduce_all_params(self):
        """Synchronously AllReduce all collected params and flat buffers."""
        # Shared flat buffer AR
        if self._ar_buffer_bf16 is not None and self.shared_dp_world_size > 1:
            dist.all_reduce(self._ar_buffer_bf16, group=self.shared_dp_group)
        if self._ar_buffer_fp32 is not None and self.shared_dp_world_size > 1:
            dist.all_reduce(self._ar_buffer_fp32, group=self.shared_dp_group)
        # Expert flat buffer AR
        expert_ar_group = self.expert_ar_group if self.expert_ar_group is not None else self.expert_dp_group
        if self._expert_ar_buffer_bf16 is not None and self.expert_dp_world_size > 1 and expert_ar_group is not None:
            dist.all_reduce(self._expert_ar_buffer_bf16, group=expert_ar_group)
        if self._expert_ar_buffer_fp32 is not None and self.expert_dp_world_size > 1 and expert_ar_group is not None:
            dist.all_reduce(self._expert_ar_buffer_fp32, group=expert_ar_group)
        # Fallback: individual params (unregistered)
        for param in self._ar_params_for_sync:
            if param.grad is not None:
                dist.all_reduce(param.grad, group=self.shared_dp_group)
        self._ar_params_for_sync.clear()

    # ========================================
    # Iteration management
    # ========================================
    def clear_iteration(self):
        """Clear state for new iteration, defensively draining all in-flight tasks."""
        # 1. Wait for all AllToAll tasks that were submitted but not yet waited.
        for done_event, result_holder in list(self._alltoall_results.values()):
            done_event.wait()
            cuda_event = result_holder[1]
            if cuda_event is not None:
                self.default_stream.wait_event(cuda_event)

        # 2. Wait for all AR on comm_stream
        self._wait_all_ar()

        # 3. GPU synchronize only when pending work existed.
        had_pending = (
            bool(self._alltoall_results)
            or self._last_ar_cuda_event is not None
            or (not self._alltoall_queue.empty())
        )
        if had_pending and torch.cuda.is_available():
            torch.cuda.synchronize()

        # 4. Drain leftover items from alltoall queue.
        while True:
            try:
                self._alltoall_queue.get_nowait()
            except queue.Empty:
                break

        # 5. Clear accounting state.
        self._dw_queue.clear()
        self._alltoall_results.clear()
        self._pending_ar_tensors.clear()
        self._ar_params_for_sync.clear()
        self._last_a2a_event = None
        self._last_ar_cuda_event = None
        self._alltoall_task_id = 0
        self._ar_task_count = 0
        self._a2a_submitted = 0
        self._a2a_completed = 0
        self._a2a_idle.set()
        self._in_finish_batch = False
        # Reset flat buffer cursors (buffer contents will be overwritten by next dW)
        self._ar_read_cursor_bf16 = 0
        self._ar_write_cursor_bf16 = 0
        self._ar_read_cursor_fp32 = 0
        self._ar_write_cursor_fp32 = 0
        self._expert_ar_read_cursor_bf16 = 0
        self._expert_ar_write_cursor_bf16 = 0
        self._expert_ar_read_cursor_fp32 = 0
        self._expert_ar_write_cursor_fp32 = 0

        # 6. Reset per-iteration stats.
        self.total_dw = 0
        self.completed_dw = 0
        self.total_ar = 0
        self.completed_ar = 0
        self.ar_during_overlap = 0
        self.ar_submitted_during_bwd = 0
        self.ar_submitted_during_finish = 0

    def finish_batch(self):
        """Finish batch: complete all dW tasks and wait for all AR.

        Must complete before optimizer.step() so that param.grad is fully
        allreduced and ready for use.
        """
        if not self.enabled:
            return

        self._in_finish_batch = True

        # 1. Execute any remaining dW tasks (uses flat buffer path if configured).
        self.execute_dw_tasks()

        # 2. Flush + wait for ALL AR before returning.
        needs_ar = self.shared_dp_world_size > 1 or self.expert_dp_world_size > 1
        if self._use_interleaved_ar() and needs_ar:
            self.flush_ar_pending(final=True)
            self._wait_all_ar()

        # Prevent cross-iteration gap measurement
        # (last R4 end → next iteration's first R1 start is not a valid gap)
        self._last_a2a_end_event = None
        self._last_a2a_end_region = None

        # 3. In sync mode (ar_enabled=False) OR safe mode, do sync AR.
        if (not self._use_interleaved_ar()) and needs_ar:
            self._sync_allreduce_all_params()

        self._in_finish_batch = False

    # ========================================
    # Profiling: region-level timing
    # ========================================
    def begin_region(self, name: str):
        """Mark start of a backward region for profiling."""
        # Always track region name (used by BDP per-region trickle)
        self._region_name = name
        self._first_a2a_in_region = True  # next A2A is the first in this region
        if not self.profiling:
            return
        torch.cuda.synchronize()
        self._region_a2a_times = []
        self._region_dw_time = 0.0
        self._region_wall_start = time.perf_counter()

    def end_region(self):
        """Mark end of a backward region."""
        if not self.profiling or self._region_name is None:
            self._region_name = None
            return
        torch.cuda.synchronize()
        T_region = (time.perf_counter() - self._region_wall_start) * 1000

        name = self._region_name
        T_comm = sum(self._region_a2a_times)
        T_dW = self._region_dw_time

        if name not in self._region_profiles:
            self._region_profiles[name] = {
                'T_region': 0.0, 'T_comm': 0.0, 'T_dW': 0.0,
                'n_a2a': 0, 'count': 0,
            }
        p = self._region_profiles[name]
        p['T_region'] += T_region
        p['T_comm'] += T_comm
        p['T_dW'] += T_dW
        p['n_a2a'] += len(self._region_a2a_times)
        p['count'] += 1
        self._region_name = None

    def get_region_profiles(self):
        """Get averaged per-region profiles."""
        result = {}
        for name, p in self._region_profiles.items():
            n = max(p['count'], 1)
            T_region = p['T_region'] / n
            T_comm = p['T_comm'] / n
            T_dW = p['T_dW'] / n
            T_comp = max(0, T_region - max(T_comm, T_dW))
            result[name] = {
                'T_region': T_region,
                'T_comm': T_comm,
                'T_comp': T_comp,
                'T_dW': T_dW,
                'n_a2a': p['n_a2a'] // n,
            }
        return result

    # ========================================
    # BDP-based AR trickle size computation
    # ========================================
    def reset_gap_times(self):
        """Clear accumulated gap/window times for fresh profiling."""
        self._gap_event_pairs = []
        self._region_gaps = {}
        self._last_a2a_end_event = None
        self._last_a2a_end_region = None
        self._first_a2a_in_region = True

    def process_gap_events(self):
        """Process recorded CUDA event pairs into gap times.

        Must be called after torch.cuda.synchronize() to ensure all events
        have been recorded on the GPU.
        """
        for region, end_evt, submit_evt in self._gap_event_pairs:
            gap_ms = end_evt.elapsed_time(submit_evt)
            if region not in self._region_gaps:
                self._region_gaps[region] = []
            self._region_gaps[region].append(gap_ms)
        self._gap_event_pairs = []

    @staticmethod
    def measure_ar_bandwidth(
        ar_group=None,
        sizes_mb=(1, 2, 4, 8, 16, 32, 64, 96, 128),
        warmup=5,
        repeat=20,
        dtype=torch.bfloat16,
    ) -> dict:
        """Measure AllReduce bandwidth at various message sizes.

        Returns dict with sizes_mb, bw_GBps, latency_ms, peak_bw_GBps.
        """
        device = torch.cuda.current_device()
        results = {'sizes_mb': [], 'bw_GBps': [], 'latency_ms': []}

        for size_mb in sizes_mb:
            elem_size = 2 if dtype == torch.bfloat16 else 4
            numel = int(size_mb * 1024 * 1024 / elem_size)
            tensor = torch.randn(numel, dtype=dtype, device=device)

            for _ in range(warmup):
                dist.all_reduce(tensor, group=ar_group)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                dist.all_reduce(tensor, group=ar_group)
            end.record()
            torch.cuda.synchronize()

            latency_ms = start.elapsed_time(end) / repeat
            size_bytes = numel * elem_size
            bw_GBps = (size_bytes / 1e9) / (latency_ms / 1e3) if latency_ms > 0 else 0

            results['sizes_mb'].append(size_mb)
            results['bw_GBps'].append(bw_GBps)
            results['latency_ms'].append(latency_ms)

            del tensor

        results['peak_bw_GBps'] = max(results['bw_GBps']) if results['bw_GBps'] else 0
        return results

    def compute_bdp_trickle_size(
        self,
        bw_GBps: float = 0.0,
        percentile: float = 10.0,
        safety_factor: float = 0.9,
        bw_profile: dict = None,
    ) -> dict:
        """Compute per-region optimal AR trickle size using Bandwidth-Delay Product.

        For each region R, the gap from R's trickle to the next region's first
        submit_alltoall() determines how much AR can run without delaying A2A.

        optimal_trickle_size[R] = T_gap[R] * BW_AR(size) * safety_factor

        Args:
            bw_GBps: Fixed bandwidth (used if bw_profile is None).
            percentile: Gap time percentile (conservative = lower).
            safety_factor: Multiply BDP by this factor (< 1.0 for margin).
            bw_profile: Output of measure_ar_bandwidth(). When provided,
                        uses size-dependent bandwidth: iteratively estimates
                        the trickle size, looks up BW at that size, and
                        recomputes until convergence.

        Returns dict with per-region results and overall summary.
        """
        if not self._region_gaps:
            raise RuntimeError(
                "No region gaps recorded. Run profiling iterations first "
                "(with ar_enabled=True)."
            )

        def _calc_percentile(sorted_vals, pct):
            n = len(sorted_vals)
            idx = pct / 100.0 * (n - 1)
            lo = int(idx)
            hi = min(lo + 1, n - 1)
            frac = idx - lo
            return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

        def _bw_at_size_mb(size_mb):
            """Interpolate BW from profile at given message size."""
            if bw_profile is None:
                return bw_GBps
            sizes = bw_profile['sizes_mb']
            bws = bw_profile['bw_GBps']
            if size_mb <= sizes[0]:
                return bws[0]
            if size_mb >= sizes[-1]:
                return bws[-1]
            for i in range(len(sizes) - 1):
                if sizes[i] <= size_mb <= sizes[i + 1]:
                    frac = (size_mb - sizes[i]) / (sizes[i + 1] - sizes[i])
                    return bws[i] + frac * (bws[i + 1] - bws[i])
            return bws[-1]

        effective_bw = bw_GBps  # for summary output

        per_region = {}
        for region, windows in self._region_gaps.items():
            ws = sorted(windows)
            n = len(ws)
            T_min = ws[0]
            T_mean = sum(ws) / n
            T_pct = _calc_percentile(ws, percentile)

            if bw_profile is not None:
                # Iterative: estimate size → look up BW → recompute size
                bw = _bw_at_size_mb(64)  # initial guess
                for _ in range(5):
                    bdp_bytes = (T_pct / 1000.0) * (bw * 1e9) * safety_factor
                    size_mb = bdp_bytes / (1024 * 1024)
                    bw = _bw_at_size_mb(size_mb)
                effective_bw = bw
            else:
                bw = bw_GBps
                bdp_bytes = (T_pct / 1000.0) * (bw * 1e9) * safety_factor

            bdp_MB = int(bdp_bytes / (1024 * 1024))

            per_region[region] = {
                'trickle_size_bytes': bdp_MB * 1024 * 1024,
                'trickle_size_MB': bdp_MB,
                'trickle_size_bytes_exact': bdp_bytes,
                'T_window_min_ms': T_min,
                'T_window_percentile_ms': T_pct,
                'T_window_mean_ms': T_mean,
                'n_windows': n,
                'bw_GBps_used': bw,
            }

        return {
            'per_region': per_region,
            'bw_GBps': effective_bw,
            'safety_factor': safety_factor,
            'percentile': percentile,
        }

    def get_stats(self):
        """Get scheduler statistics."""
        return {
            'total_dw_tasks': self.total_dw,
            'completed_dw_tasks': self.completed_dw,
            'total_ar_tasks': self.total_ar,
            'completed_ar_tasks': self.completed_ar,
            'ar_during_gap': self.ar_during_overlap,
            'ar_submitted_during_bwd': self.ar_submitted_during_bwd,
            'ar_submitted_during_finish': self.ar_submitted_during_finish,
            'pending_dw_at_finish': len(self._dw_queue),
            'wait_prev_ar_ms': getattr(self, '_wait_prev_ar_time', 0) * 1000,
            'region_gaps': dict(self._region_gaps),
            'ar_window_mode': self.ar_window_mode,
        }

    @classmethod
    def reset(cls):
        """Reset singleton instance."""
        cls._instance = None


def get_backward_scheduler() -> BackwardScheduler:
    """Get the singleton scheduler instance."""
    return BackwardScheduler()
