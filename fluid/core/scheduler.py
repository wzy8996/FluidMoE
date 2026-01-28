"""
FluidMoE Backward Scheduler

Design:
- dW overlaps with AllToAll (dW on default_stream, AllToAll on comm_stream)
- comm_thread handles both AllToAll and AR tasks (all NCCL in one thread)
- AR overlaps with dW computation in finish_batch
"""

import torch
import torch.distributed as dist
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
import threading
import queue



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

        # Separate queues for AllToAll and AR
        self._alltoall_queue = queue.Queue()  # High priority
        self._ar_queue = queue.Queue()        # Low priority, processed when AllToAll is idle
        self._comm_thread = None
        self._comm_thread_stop = False
        self._alltoall_idle_count = 0  # Count how many cycles AllToAll queue has been empty

        # AllToAll tracking
        self._alltoall_task_id = 0
        self._alltoall_results = {}

        # AR tracking
        self._ar_task_id = 0
        self._ar_results = {}
        self._ar_pending_params = []  # [(layer_name, param)] - params that need AR

        # AR config
        self.ar_enabled = False
        self.dp_group = None
        self.dp_world_size = 1

        # Stats
        self.total_dw = 0
        self.completed_dw = 0
        self.total_ar = 0
        self.completed_ar = 0
        self.ar_during_overlap = 0

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

    def enable(self):
        self.enabled = True
        self._init_cuda(force_reinit=True)

        # Start comm thread
        if self._comm_thread is None:
            self._start_comm_thread()

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
        self._init_cuda(force_reinit=True)

    def _start_comm_thread(self):
        """Start unified communication thread."""
        self._comm_thread_stop = False
        self._comm_thread = threading.Thread(
            target=self._comm_thread_worker,
            daemon=True
        )
        self._comm_thread.start()

    def _comm_thread_worker(self):
        """Comm thread: handles AllToAll tasks only.

        AR is executed synchronously in finish_batch to avoid deadlock
        when dp_group == ep_group.
        """
        if self._device is not None:
            torch.cuda.set_device(self._device)

        while not self._comm_thread_stop:
            try:
                task = self._alltoall_queue.get(timeout=0.01)  # 10ms timeout
                if task is None:  # Sentinel to stop
                    break
                self._execute_alltoall(task)
            except queue.Empty:
                pass

    def _execute_alltoall(self, task: AllToAllTask):
        """Execute AllToAll task on comm_stream.

        Note: No synchronize here - NCCL ops on same stream are naturally serialized.
        Main thread waits via cuda_event in wait_alltoall().
        """
        with torch.cuda.stream(self.comm_stream):
            result = task.comm_fn()
            cuda_event = torch.cuda.Event()
            cuda_event.record(self.comm_stream)

        # No synchronize - let comm_thread continue to next task
        # NCCL ops will still execute in order on the stream
        task.result_holder[0] = result
        task.result_holder[1] = cuda_event
        task.done_event.set()  # Signal task submitted (not completed)

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
        """Execute all deferred dW tasks.

        dW computation overlaps with AllToAll on comm_thread.
        AR is deferred to finish_batch to avoid deadlock.
        """
        while self._dw_queue:
            task = self._dw_queue.pop(0)
            grad_weight = task.compute_fn()

            if task.weight_param is not None and grad_weight is not None:
                if task.weight_param.grad is None:
                    task.weight_param.grad = grad_weight.clone()
                else:
                    task.weight_param.grad.add_(grad_weight)

                # Record params that need AR (will be done in finish_batch)
                if task.needs_ar and self.ar_enabled and self.dp_world_size > 1:
                    self._ar_pending_params.append((task.layer_name, task.weight_param))

            self.completed_dw += 1

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

        # Wait for comm_stream completion on default_stream
        # This ensures all prior AllToAll ops are also complete (same stream)
        if cuda_event is not None:
            self.default_stream.wait_event(cuda_event)

        # Clean up this task and all earlier tasks in the batch
        first_task_id = task_id - num_tasks + 1
        for tid in range(first_task_id, task_id + 1):
            self._alltoall_results.pop(tid, None)

        return result

    # ========================================
    # AllReduce
    # ========================================
    def _submit_ar(self, tensor: torch.Tensor) -> int:
        """Submit AllReduce to comm thread."""
        task_id = self._ar_task_id
        self._ar_task_id += 1

        done_event = threading.Event()

        task = ARTask(
            task_id=task_id,
            tensor=tensor,
            group=self.dp_group,
            done_event=done_event,
        )

        self._ar_queue.put(task)
        self._ar_results[task_id] = done_event
        self.total_ar += 1

        return task_id

    def _wait_ar(self, task_id: int):
        """Wait for AllReduce to complete."""
        done_event = self._ar_results.get(task_id)
        if done_event is None:
            return

        done_event.wait()
        del self._ar_results[task_id]
        self.completed_ar += 1

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
        self._alltoall_task_id = 0
        self._ar_task_id = 0
        self._task_seq_num = 0

        self.total_dw = 0
        self.completed_dw = 0
        self.total_ar = 0
        self.completed_ar = 0
        self.ar_during_overlap = 0

    def finish_batch(self):
        """Finish batch: wait for all pending tasks to complete."""
        if not self.enabled:
            return

        # 1. Execute any remaining dW tasks
        while self._dw_queue:
            task = self._dw_queue.pop(0)
            grad_weight = task.compute_fn()

            if task.weight_param is not None and grad_weight is not None:
                if task.weight_param.grad is None:
                    task.weight_param.grad = grad_weight.clone()
                else:
                    task.weight_param.grad.add_(grad_weight)

                if task.needs_ar and self.ar_enabled and self.dp_world_size > 1:
                    self._ar_pending_params.append((task.layer_name, task.weight_param))

            self.completed_dw += 1

        # 2. Execute all pending AR synchronously
        for _, param in self._ar_pending_params:
            if param.grad is not None:
                dist.all_reduce(param.grad, group=self.dp_group)
                self.completed_ar += 1

        self._ar_pending_params.clear()

        # 3. Final sync
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_stats(self):
        """Get scheduler statistics."""
        return {
            'total_dw_tasks': self.total_dw,
            'completed_dw_tasks': self.completed_dw,
            'overlap_completed_dw_tasks': 0,
            'finish_batch_completed_dw_tasks': self.completed_dw,
            'total_ar_tasks': self.total_ar,
            'ar_during_gap': self.ar_during_overlap,
        }

    @classmethod
    def reset(cls):
        """Reset singleton instance."""
        cls._instance = None


def get_backward_scheduler() -> BackwardScheduler:
    """Get the singleton scheduler instance."""
    return BackwardScheduler()
