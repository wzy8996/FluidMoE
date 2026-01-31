"""
FluidMoE Backward Scheduler

Design:
- dW overlaps with AllToAll (dW on default_stream, AllToAll on comm_stream)
- comm_thread handles both AllToAll and AR tasks (all NCCL in one thread)
- AR is interleaved into AllToAll idle gaps (AllToAll has absolute priority)
- AR tensors are chunked (default 2MB) to avoid blocking AllToAll
- AR is submitted at synchronization points to guarantee consistent order across ranks
"""

import os
import torch
import torch.distributed as dist
from typing import Optional, Callable, Any, List, Tuple
from dataclasses import dataclass, field
import threading
import queue
import time



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
    done_event: threading.Event
    result_holder: list  # [result, cuda_event]


@dataclass
class ARTask:
    """AllReduce communication task."""
    task_id: int
    tensor: torch.Tensor
    group: Any
    done_event: threading.Event
    input_event: Optional[Any] = None   # CUDA event: wait for grad data on default_stream
    cuda_event: Optional[Any] = None    # CUDA event: signal AR completion on comm_stream
    a2a_event_snapshot: Optional[Any] = None  # Snapshot of _last_a2a_event at dispatch time


@dataclass(order=True)
class PrioritizedTask:
    """Wrapper for priority queue ordering."""
    priority: int
    seq_num: int = field(compare=True)  # For FIFO within same priority
    task: Any = field(compare=False)


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
        self._dw_queue = []
        self._debug_dw = False
        self._ar_params_for_sync = []  # params needing AR when ar_enabled=False

        # AllToAll: comm_thread + comm_stream
        self._alltoall_queue = queue.Queue()
        self._comm_thread = None
        self._comm_thread_stop = False

        # AR: ar_thread + ar_stream (independent from AllToAll)
        self._ar_queue = queue.Queue()
        self._ar_thread = None
        self._ar_thread_stop = False
        self.ar_stream = None

        # AllToAll tracking
        self._alltoall_task_id = 0
        self._alltoall_results = {}
        self._a2a_submitted = 0   # Total AllToAll tasks submitted
        self._a2a_completed = 0   # Total AllToAll tasks completed by comm_thread
        self._a2a_lock = threading.Lock()
        self._a2a_idle = threading.Event()  # Set when no AllToAll in flight
        self._a2a_idle.set()
        self._last_a2a_event = None  # Last AllToAll cuda_event for GPU-side AR dependency

        # AR tracking
        self._ar_task_id = 0
        self._ar_results = {}  # task_id -> ARTask
        self._ar_pending_params = []  # [(layer_name, param)] - params awaiting AR submission

        # AR config
        self.ar_enabled = False
        self.dp_group = None
        self.ar_group = None  # Independent communicator for AR (avoids NCCL deadlock with AllToAll)
        self.dp_world_size = 1
        self.ar_chunk_size = int(os.environ.get('FLUID_AR_CHUNK_SIZE', 16 * 1024 * 1024))  # 16MB default

        # Stats
        self.total_dw = 0
        self.completed_dw = 0
        self.total_ar = 0
        self.completed_ar = 0
        self.ar_during_overlap = 0
        self.ar_submitted_during_bwd = 0
        self.ar_submitted_during_finish = 0
        self._in_finish_batch = False
        self._gap_times = []  # Track AllToAll gap durations
        self._gap_start_time = 0

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
            self.comm_stream = torch.cuda.Stream(device=current_device)
            self.ar_stream = torch.cuda.Stream(device=current_device)

    def enable(self):
        self.enabled = True
        self._init_cuda(force_reinit=True)

        # Start comm thread (AllToAll)
        if self._comm_thread is None:
            self._start_comm_thread()

        # Start AR thread
        if self._ar_thread is None:
            self._start_ar_thread()

    def disable(self):
        self.enabled = False

    def is_enabled(self):
        return self.enabled

    def configure_allreduce(self, enabled=True, dp_group=None, **kwargs):
        """Configure AllReduce settings.

        Creates an independent NCCL communicator (ar_group) for AllReduce,
        so AR and AllToAll never conflict even when dp_group == ep_group.
        """
        self.ar_enabled = enabled
        self.dp_group = dp_group
        self.dp_world_size = dist.get_world_size(dp_group) if dp_group else (
            dist.get_world_size() if dist.is_initialized() else 1)

        # Create independent communicator for AR
        if dp_group is not None and dist.is_initialized():
            ranks = dist.get_process_group_ranks(dp_group)
            self.ar_group = dist.new_group(ranks)
        else:
            self.ar_group = None

        self._init_cuda(force_reinit=True)

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

        After completing an AllToAll, signals _a2a_idle so the AR thread
        can start AR tasks immediately without waiting for the main thread.
        """
        if self._device is not None:
            torch.cuda.set_device(self._device)

        while not self._comm_thread_stop:
            try:
                task = self._alltoall_queue.get(timeout=0.001)
                if task is None:
                    break
                self._execute_alltoall(task)
                # Signal idle immediately after AllToAll completes on comm_stream
                with self._a2a_lock:
                    self._a2a_completed += 1
                    if self._a2a_completed >= self._a2a_submitted:
                        self._a2a_idle.set()
                        self._gap_start_time = time.perf_counter()
            except queue.Empty:
                pass

    def _start_ar_thread(self):
        """Start AR communication thread (independent from AllToAll)."""
        self._ar_thread_stop = False
        self._ar_thread = threading.Thread(
            target=self._ar_thread_worker,
            daemon=True
        )
        self._ar_thread.start()

    def _ar_thread_worker(self):
        """AR thread: handles AllReduce only (on ar_stream, ar_group).

        Waits for _a2a_idle (set by comm_thread after AllToAll completion)
        before executing AR tasks. This ensures AR only runs during AllToAll
        idle gaps, avoiding NVLink bandwidth contention.
        """
        if self._device is not None:
            torch.cuda.set_device(self._device)

        while not self._ar_thread_stop:
            try:
                task = self._ar_queue.get(timeout=0.001)
                if task is None:
                    break
                # Wait until no AllToAll is in flight
                self._a2a_idle.wait()
                task.a2a_event_snapshot = self._last_a2a_event
                self._execute_ar(task)
            except queue.Empty:
                pass

    def _execute_alltoall(self, task: AllToAllTask):
        """Execute AllToAll task on comm_stream."""
        with torch.cuda.stream(self.comm_stream):
            if self.profiling:
                start_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record(self.comm_stream)
            result = task.comm_fn()
            cuda_event = torch.cuda.Event(enable_timing=True) if self.profiling else torch.cuda.Event()
            cuda_event.record(self.comm_stream)

        task.result_holder[0] = result
        task.result_holder[1] = cuda_event
        if self.profiling:
            task.result_holder.append(start_evt)  # [2] = start event for timing
        self._last_a2a_event = cuda_event
        task.done_event.set()

    def _execute_ar(self, task: ARTask):
        """Execute AllReduce task on ar_stream (independent from comm_stream)."""
        with torch.cuda.stream(self.ar_stream):
            # Wait for gradient data to be ready on default_stream
            if task.input_event is not None:
                self.ar_stream.wait_event(task.input_event)
            # Wait for last AllToAll to finish on GPU
            last_event = task.a2a_event_snapshot
            if last_event is not None:
                self.ar_stream.wait_event(last_event)
            dist.all_reduce(task.tensor, group=task.group)
            cuda_event = torch.cuda.Event()
            cuda_event.record(self.ar_stream)

        task.cuda_event = cuda_event
        task.done_event.set()
        self.ar_during_overlap += 1

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
            task = self._dw_queue.pop(0)
            grad_weight = task.compute_fn()

            if task.weight_param is not None and grad_weight is not None:
                # Ensure grad dtype matches param dtype (e.g. LayerNorm produces float32 grads for bf16 params)
                if grad_weight.dtype != task.weight_param.dtype:
                    grad_weight = grad_weight.to(task.weight_param.dtype)
                if task.weight_param.grad is None:
                    task.weight_param.grad = grad_weight.clone()
                else:
                    task.weight_param.grad.add_(grad_weight)

                # AR handling
                if task.needs_ar and self.dp_world_size > 1:
                    if self.ar_enabled:
                        self._submit_ar_chunked(task.weight_param)
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

    def flush_pending_ar(self):
        """Submit all pending AR tasks to comm_thread."""
        if not self._ar_pending_params:
            return

        # Sort by layer_name to guarantee identical order across all ranks
        self._ar_pending_params.sort(key=lambda x: x[0])

        for _, param in self._ar_pending_params:
            if param.grad is not None:
                self._submit_ar_chunked(param)

        self._ar_pending_params.clear()

    # ========================================
    # AllToAll
    # ========================================
    def submit_alltoall(self, comm_fn: Callable) -> int:
        """Submit AllToAll to comm thread."""
        if not self.enabled:
            return comm_fn()

        task_id = self._alltoall_task_id
        self._alltoall_task_id += 1

        # Record event on default_stream
        input_event = torch.cuda.Event()
        input_event.record(self.default_stream)

        done_event = threading.Event()
        result_holder = [None, None]

        def wrapped_fn():
            # Wait for data to be ready
            self.comm_stream.wait_event(input_event)
            return comm_fn()

        task = AllToAllTask(
            task_id=task_id,
            comm_fn=wrapped_fn,
            done_event=done_event,
            result_holder=result_holder,
        )

        # Mark AllToAll submitted (AR thread will yield)
        with self._a2a_lock:
            if self._a2a_submitted == self._a2a_completed and self._gap_start_time > 0:
                gap = time.perf_counter() - self._gap_start_time
                self._gap_times.append(gap * 1000)  # ms
            self._a2a_submitted += 1
            self._a2a_idle.clear()

        self._alltoall_queue.put(task)
        self._alltoall_results[task_id] = (done_event, result_holder)

        return task_id

    def wait_alltoall(self, task_id: int, num_tasks: int = 1) -> Any:
        """Wait for AllToAll to complete.

        For chunked AllToAll, only need to wait for the last one since NCCL ops
        on the same stream are serialized. Use num_tasks to clean up earlier tasks.
        """
        if not self.enabled:
            return None

        task_data = self._alltoall_results.get(task_id)
        if task_data is None:
            return None

        done_event, result_holder = task_data
        done_event.wait()

        result = result_holder[0]
        cuda_event = result_holder[1]

        # Collect per-AllToAll comm time for profiling
        if self.profiling and len(result_holder) > 2:
            start_evt = result_holder[2]
            torch.cuda.synchronize()
            comm_ms = start_evt.elapsed_time(cuda_event)
            if self._region_name:
                self._region_a2a_times.append(comm_ms)

        # Wait for comm_stream completion on default_stream
        if cuda_event is not None:
            self.default_stream.wait_event(cuda_event)

        # Clean up this task and all earlier tasks in the batch
        first_task_id = task_id - num_tasks + 1
        for tid in range(first_task_id, task_id + 1):
            self._alltoall_results.pop(tid, None)

        # Note: _a2a_idle is now set by comm_thread after AllToAll completion,
        # removing the CPU latency between GPU completion and AR thread unblocking.

        return result

    # ========================================
    # AllReduce
    # ========================================
    def _submit_ar(self, tensor: torch.Tensor) -> int:
        """Submit a single AllReduce chunk to comm thread."""
        task_id = self._ar_task_id
        self._ar_task_id += 1

        # Record event on default_stream so comm_stream waits for grad data
        input_event = torch.cuda.Event()
        input_event.record(self.default_stream)

        done_event = threading.Event()

        task = ARTask(
            task_id=task_id,
            tensor=tensor,
            group=self.ar_group if self.ar_group is not None else self.dp_group,
            done_event=done_event,
            input_event=input_event,
        )

        self._ar_queue.put(task)
        self._ar_results[task_id] = task
        self.total_ar += 1
        if self._in_finish_batch:
            self.ar_submitted_during_finish += 1
        else:
            self.ar_submitted_during_bwd += 1

        return task_id

    def _submit_ar_chunked(self, param: torch.nn.Parameter):
        """Submit AR for a parameter, chunked into small pieces.

        Small chunks allow AR to fit into narrow AllToAll idle gaps without
        blocking subsequent AllToAll operations.
        """
        grad = param.grad
        if grad is None:
            return

        chunk_bytes = self.ar_chunk_size
        num_elements_per_chunk = chunk_bytes // grad.element_size()

        if num_elements_per_chunk <= 0 or grad.numel() <= num_elements_per_chunk:
            # Small enough: submit as single AR
            self._submit_ar(grad)
            return

        # Split into chunks and submit each
        flat = grad.view(-1)
        for start in range(0, flat.numel(), num_elements_per_chunk):
            end = min(start + num_elements_per_chunk, flat.numel())
            chunk = flat[start:end]
            self._submit_ar(chunk)

    def _wait_all_ar(self):
        """Wait for all pending AR tasks to complete."""
        if not self._ar_results:
            return
        last_event = None
        for task_id, task in list(self._ar_results.items()):
            task.done_event.wait()
            if task.cuda_event is not None:
                last_event = task.cuda_event
            self.completed_ar += 1
        # Only need one wait_event for the last cuda_event (comm_stream is serialized)
        if last_event is not None:
            self.default_stream.wait_event(last_event)
        self._ar_results.clear()

    def _sync_allreduce_all_params(self):
        """Synchronously AllReduce all collected params on default_stream."""
        for param in self._ar_params_for_sync:
            if param.grad is not None:
                dist.all_reduce(param.grad, group=self.dp_group)
        self._ar_params_for_sync.clear()

    # ========================================
    # Iteration management
    # ========================================
    def clear_iteration(self):
        """Clear state for new iteration."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._dw_queue.clear()
        self._alltoall_results.clear()
        self._ar_results.clear()
        self._ar_pending_params.clear()
        self._ar_params_for_sync.clear()
        self._alltoall_task_id = 0
        self._ar_task_id = 0
        self._a2a_submitted = 0
        self._a2a_completed = 0
        self._a2a_idle.set()

        self.total_dw = 0
        self.completed_dw = 0
        self.total_ar = 0
        self.completed_ar = 0
        self.ar_during_overlap = 0
        self.ar_submitted_during_bwd = 0
        self.ar_submitted_during_finish = 0

    def finish_batch(self):
        """Finish batch: wait for previous AR, execute remaining dW, flush new AR.

        AR from the current batch is submitted to comm_thread but NOT waited on.
        It will be waited on at the start of the next finish_batch(), overlapping
        with the next iteration's forward + backward pass.
        """
        if not self.enabled:
            return

        self._in_finish_batch = True
        self._pending_dw_at_finish_start = len(self._dw_queue)
        self._pending_ar_params_at_finish_start = len(self._ar_pending_params)

        # 1. Wait for AR from PREVIOUS iteration (submitted last finish_batch)
        t0 = time.perf_counter()
        n_prev_ar = len(self._ar_results)
        self._wait_all_ar()
        self._wait_prev_ar_time = time.perf_counter() - t0
        self._prev_ar_count = n_prev_ar

        # 2. Execute any remaining dW tasks (AR still chunked during dW phase)
        finish_ar_params = []  # collect params for unchunked AR after all dW done
        while self._dw_queue:
            task = self._dw_queue.pop(0)
            grad_weight = task.compute_fn()

            if task.weight_param is not None and grad_weight is not None:
                if grad_weight.dtype != task.weight_param.dtype:
                    grad_weight = grad_weight.to(task.weight_param.dtype)
                if task.weight_param.grad is None:
                    task.weight_param.grad = grad_weight.clone()
                else:
                    task.weight_param.grad.add_(grad_weight)

                if task.needs_ar and self.dp_world_size > 1:
                    if self.ar_enabled:
                        finish_ar_params.append(task.weight_param)
                    else:
                        self._ar_params_for_sync.append(task.weight_param)

            self.completed_dw += 1

        # 3. Flush any remaining pending AR (from backward's flush_pending_ar)
        self.flush_pending_ar()

        # 4. All dW done, no more AllToAll will come — submit remaining AR unchunked
        if self.ar_enabled and finish_ar_params:
            for param in finish_ar_params:
                if param.grad is not None:
                    self._submit_ar(param.grad)

        # 5. If ar_enabled=False, do sync AR for all params that need it
        if not self.ar_enabled and self.dp_world_size > 1:
            self._sync_allreduce_all_params()

        self._in_finish_batch = False
        # NOTE: When ar_enabled=True, no _wait_all_ar() here!
        # AR continues executing on comm_thread while next iteration runs

    # ========================================
    # Profiling: region-level timing
    # ========================================
    def begin_region(self, name: str):
        """Mark start of a backward region for profiling."""
        if not self.profiling:
            return
        # Sync to get clean boundary
        torch.cuda.synchronize()
        self._region_name = name
        self._region_a2a_times = []
        self._region_dw_time = 0.0
        self._region_wall_start = time.perf_counter()

    def end_region(self):
        """Mark end of a backward region."""
        if not self.profiling or self._region_name is None:
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
            # T_comp = T_region - max(T_comm, T_dW) for pipelined execution
            # This works because: T_region ≈ max(T_comm, T_dW) + T_comp (pipeline)
            T_comp = max(0, T_region - max(T_comm, T_dW))
            result[name] = {
                'T_region': T_region,
                'T_comm': T_comm,
                'T_comp': T_comp,
                'T_dW': T_dW,
                'n_a2a': p['n_a2a'] // n,
            }
        return result

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
            'prev_ar_count': getattr(self, '_prev_ar_count', 0),
            'pending_dw_at_finish_start': getattr(self, '_pending_dw_at_finish_start', 0),
            'pending_ar_params_at_finish_start': getattr(self, '_pending_ar_params_at_finish_start', 0),
            'gap_times': self._gap_times[-20:] if self._gap_times else [],
            'gap_count': len(self._gap_times),
            'gap_total_ms': sum(self._gap_times) if self._gap_times else 0,
        }

    @classmethod
    def reset(cls):
        """Reset singleton instance."""
        cls._instance = None


def get_backward_scheduler() -> BackwardScheduler:
    """Get the singleton scheduler instance."""
    return BackwardScheduler()
