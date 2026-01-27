"""
FluidMoE Backward Scheduler

Design:
- dW overlaps with AllToAll (dW on default_stream, AllToAll on comm_stream)
- AR submitted to unified comm queue, executed on comm_stream when idle
- AllToAll has higher priority than AR in the queue

Timeline:
1. MoE/Attention backward: dX computed, dW registered (deferred)
2. AllToAll submitted to comm_stream (high priority)
3. dW executed on default_stream (overlaps with AllToAll)
4. dW completion submits AR to comm queue (low priority)
5. Comm thread processes queue: AllToAll first, then AR when idle
6. At batch end: wait for all tasks to complete
"""

import torch
import torch.distributed as dist
from typing import Optional, Callable, Any
from dataclasses import dataclass
import threading
import queue
from enum import IntEnum


class CommTaskType(IntEnum):
    """Communication task priority (lower = higher priority)."""
    ALLTOALL = 0  # Highest priority
    ALLREDUCE = 1  # Lower priority


@dataclass
class CommTask:
    """A communication task in the queue."""
    task_type: CommTaskType
    task_id: int
    # For AllToAll
    comm_fn: Optional[Callable] = None
    done_event: Optional[threading.Event] = None
    result_holder: Optional[list] = None  # [result, cuda_event]
    # For AR
    param: Optional[Any] = None

    def __lt__(self, other):
        """For priority queue ordering: by task_type first, then by task_id (FIFO)."""
        if self.task_type != other.task_type:
            return self.task_type < other.task_type
        return self.task_id < other.task_id


@dataclass
class DWTask:
    """Deferred weight gradient computation task."""
    layer_name: str
    layer_id: int
    compute_fn: Callable
    weight_param: Optional[torch.nn.Parameter] = None
    needs_ar: bool = True


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

        # dW queue
        self._dw_queue = []

        # Unified communication queue (priority queue: AllToAll > AR)
        self._comm_queue = queue.PriorityQueue()
        self._comm_thread = None
        self._comm_thread_stop = False
        self._device = None

        # AR state
        self._ar_completed = set()
        self._ar_completed_lock = threading.Lock()
        self._ar_pending_count = 0
        self._ar_done_event = threading.Event()
        self._in_finish_batch = False
        self._ar_task_id = 0

        # AllToAll state
        self._alltoall_task_id = 0
        self._alltoall_in_progress = 0
        self._alltoall_results = {}

        # AR config
        self.ar_enabled = False
        self.dp_group = None
        self.dp_world_size = 1

        # Stats
        self.total_dw = 0
        self.completed_dw = 0
        self.total_ar = 0
        self.completed_ar = 0
        self.ar_during_gap = 0

        self._init_cuda()

    def _init_cuda(self):
        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()
            self.default_stream = torch.cuda.default_stream()
            self.comm_stream = torch.cuda.Stream()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def is_enabled(self):
        return self.enabled

    def configure_allreduce(self, enabled=True, dp_group=None, **kwargs):
        """Configure AllReduce settings."""
        self.ar_enabled = enabled
        self.dp_group = dp_group
        self.dp_world_size = dist.get_world_size(dp_group) if dp_group else (
            dist.get_world_size() if dist.is_initialized() else 1)

        if torch.cuda.is_available():
            self._device = torch.cuda.current_device()

        if enabled and self._comm_thread is None:
            self._start_comm_thread()

    def _start_comm_thread(self):
        """Start unified communication thread."""
        self._comm_thread_stop = False
        self._comm_thread = threading.Thread(target=self._comm_thread_worker, daemon=True)
        self._comm_thread.start()

    def _stop_comm_thread(self):
        """Stop communication thread."""
        if self._comm_thread is not None:
            self._comm_thread_stop = True
            self._comm_queue.put(CommTask(CommTaskType.ALLTOALL, -1))
            self._comm_thread.join(timeout=5.0)
            self._comm_thread = None

    def _comm_thread_worker(self):
        """Unified communication thread: process AllToAll and AR by priority."""
        if self._device is not None:
            torch.cuda.set_device(self._device)

        while not self._comm_thread_stop:
            try:
                task = self._comm_queue.get(timeout=0.001)
            except queue.Empty:
                continue

            if task.task_id == -1:  # Sentinel
                break

            if task.task_type == CommTaskType.ALLTOALL:
                with torch.cuda.stream(self.comm_stream):
                    result = task.comm_fn()
                    cuda_event = torch.cuda.Event()
                    cuda_event.record(self.comm_stream)

                task.result_holder[0] = result
                task.result_holder[1] = cuda_event
                task.done_event.set()
                self._alltoall_in_progress -= 1

            elif task.task_type == CommTaskType.ALLREDUCE:
                param = task.param
                param_id = id(param)

                with self._ar_completed_lock:
                    if param_id in self._ar_completed or param.grad is None:
                        self._ar_pending_count -= 1
                        if self._ar_pending_count == 0:
                            self._ar_done_event.set()
                        continue

                with torch.cuda.stream(self.comm_stream):
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.dp_group)
                    param.grad.div_(self.dp_world_size)

                with self._ar_completed_lock:
                    self._ar_completed.add(param_id)
                    self.total_ar += 1
                    self.completed_ar += 1
                    if not self._in_finish_batch:
                        self.ar_during_gap += 1
                    self._ar_pending_count -= 1
                    if self._ar_pending_count == 0:
                        self._ar_done_event.set()

    # ========================================
    # dW Tasks
    # ========================================
    def register_dw_task(self, layer_name, layer_id, compute_fn, weight_param=None, needs_ar=True, **kwargs):
        """Register a deferred dW task."""
        if not self.enabled:
            return
        self._dw_queue.append(DWTask(layer_name, layer_id, compute_fn, weight_param, needs_ar))
        self.total_dw += 1

    def execute_dw_tasks(self):
        """Execute dW tasks. After dW completes, submit AR to comm queue."""
        while self._dw_queue:
            task = self._dw_queue.pop(0)
            grad_weight = task.compute_fn()

            if task.weight_param is not None and grad_weight is not None:
                if task.weight_param.grad is None:
                    task.weight_param.grad = grad_weight.clone()
                else:
                    task.weight_param.grad.add_(grad_weight)

                if task.needs_ar and self.ar_enabled:
                    param = task.weight_param
                    param_id = id(param)

                    with self._ar_completed_lock:
                        if param_id not in self._ar_completed:
                            self._ar_pending_count += 1
                            self._ar_done_event.clear()

                    ar_task = CommTask(
                        task_type=CommTaskType.ALLREDUCE,
                        task_id=self._ar_task_id,
                        param=param,
                    )
                    self._ar_task_id += 1
                    self._comm_queue.put(ar_task)

            self.completed_dw += 1

    # ========================================
    # AllToAll
    # ========================================
    def submit_alltoall(self, comm_fn: Callable) -> int:
        """Submit AllToAll to comm queue (high priority)."""
        if not self.enabled:
            return comm_fn()

        task_id = self._alltoall_task_id
        self._alltoall_task_id += 1
        self._alltoall_in_progress += 1

        done_event = threading.Event()
        result_holder = [None, None]

        input_event = torch.cuda.Event()
        input_event.record(self.default_stream)

        def wrapped_fn():
            self.comm_stream.wait_event(input_event)
            return comm_fn()

        task = CommTask(
            task_type=CommTaskType.ALLTOALL,
            task_id=task_id,
            comm_fn=wrapped_fn,
            done_event=done_event,
            result_holder=result_holder,
        )
        self._comm_queue.put(task)
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
        cuda_event = result_holder[1]
        if cuda_event is not None:
            self.default_stream.wait_event(cuda_event)

        del self._alltoall_results[task_id]
        return result

    # ========================================
    # Iteration management
    # ========================================
    def clear_iteration(self):
        """Clear state for new iteration."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._dw_queue.clear()
        while not self._comm_queue.empty():
            try:
                self._comm_queue.get_nowait()
            except queue.Empty:
                break

        with self._ar_completed_lock:
            self._ar_completed.clear()
            self._ar_pending_count = 0
        self._ar_done_event.set()
        self._alltoall_results.clear()
        self._alltoall_in_progress = 0
        self._alltoall_task_id = 0
        self._ar_task_id = 0
        self._in_finish_batch = False

        self.total_dw = 0
        self.completed_dw = 0
        self.total_ar = 0
        self.completed_ar = 0
        self.ar_during_gap = 0

    def finish_batch(self):
        """Finish batch: execute remaining dW, wait for comm thread to complete, sync."""
        if not self.enabled:
            return

        self._in_finish_batch = True
        self.execute_dw_tasks()

        if self._comm_thread is not None and self.ar_enabled:
            self._ar_done_event.wait(timeout=10.0)

        if torch.cuda.is_available():
            self.comm_stream.synchronize()
            torch.cuda.synchronize()

        with self._ar_completed_lock:
            self._ar_completed.clear()
            self._ar_pending_count = 0
        self._ar_done_event.set()
        self._in_finish_batch = False

    def get_stats(self):
        """Get scheduler statistics."""
        return {
            'total_dw_tasks': self.total_dw,
            'completed_dw_tasks': self.completed_dw,
            'total_ar_tasks': self.total_ar,
            'completed_ar_tasks': self.completed_ar,
            'ar_during_gap': self.ar_during_gap,
            'ar_at_end': self.completed_ar - self.ar_during_gap,
        }

    @classmethod
    def reset(cls):
        """Reset singleton instance."""
        cls._instance = None


def get_backward_scheduler() -> BackwardScheduler:
    """Get the singleton scheduler instance."""
    return BackwardScheduler()
