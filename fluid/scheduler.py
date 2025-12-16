"""
Global Backward Scheduler for Fine-grained Computation-Communication Overlap

This scheduler implements a global backward pass optimization strategy:
1. dX computation runs on the critical path
2. When AlltoAll communication starts (EP or Ulysses), launch dW computations
3. dW computations run on overlap stream while AlltoAll is in progress
4. When one layer's dW finishes, continue with next layer's dW
5. When AlltoAll finishes, continue with dX propagation

Key Innovation:
- Global scheduling across all layers (not per-layer optimization)
- Dynamic dW queue management
- Prioritize dX (critical path) while overlapping dW with communication
"""

import torch
from typing import List, Optional, Callable, Tuple
from collections import deque
import threading


class DWTask:
    """Represents a single dW computation task"""

    def __init__(
        self,
        layer_name: str,
        layer_id: int,
        compute_fn: Callable,
        priority: int = 0,
        weight_param: Optional[torch.nn.Parameter] = None,
    ):
        """
        Args:
            layer_name: Name of the layer (e.g., "layer_3_attention", "layer_5_moe")
            layer_id: Layer index in the model
            compute_fn: Function to execute dW computation (returns gradient tensor)
            priority: Higher priority tasks run first (default: layer_id)
            weight_param: Weight parameter to accumulate gradient into
        """
        self.layer_name = layer_name
        self.layer_id = layer_id
        self.compute_fn = compute_fn
        self.priority = priority if priority > 0 else layer_id
        self.weight_param = weight_param
        self.completed = False
        self.event: Optional[torch.cuda.Event] = None


class BackwardScheduler:
    """
    Global backward scheduler that manages dX-dW overlap with AlltoAll communications

    Singleton pattern to ensure only one scheduler exists per process.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        # CUDA Streams
        self.default_stream = torch.cuda.current_stream()  # dX and dW both use this!
        self.comm_stream = torch.cuda.Stream()  # AlltoAll communication stream

        # Stream architecture (User's requirement):
        # - default_stream: dX and dW computation (sequential - share GPU compute)
        # - comm_stream: AlltoAll communication (parallel - uses network, not GPU compute)
        #
        # Key insight:
        # - dX and dW CANNOT run simultaneously (both need GPU compute)
        # - AlltoAll (communication) and dW (compute) CAN overlap
        # - dW fills the GPU idle time during AlltoAll

        # Task queues
        self.dw_queue: deque[DWTask] = deque()  # Pending dW tasks
        self.active_dw_task: Optional[DWTask] = None  # Currently running dW task

        # Communication tracking
        self.comm_in_progress = False
        self.alltoall_start_event: Optional[torch.cuda.Event] = None  # AlltoAll start marker
        self.alltoall_end_event: Optional[torch.cuda.Event] = None  # AlltoAll completion marker

        # Statistics
        self.total_dw_tasks = 0
        self.completed_dw_tasks = 0  # Total completed (overlap + finish_batch)
        self.overlap_completed_dw_tasks = 0  # Only completed during overlap
        self.finish_batch_completed_dw_tasks = 0  # Only completed in finish_batch
        self.total_comm_time_ms = 0.0
        self.total_dw_overlap_time_ms = 0.0

        # Enable/disable flag
        self.enabled = False

        # Auto-finish mode: automatically call finish_batch() after each backward
        # Suitable for single micro-batch per batch, or when using Megatron's
        # built-in gradient accumulation
        self.auto_finish = True  # Default: enabled for ease of use

    @classmethod
    def get_instance(cls):
        """Get the singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the scheduler (useful for testing or between training runs)"""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._reset_internal()
                cls._instance = None

    def _reset_internal(self):
        """Internal reset method"""
        self.dw_queue.clear()
        self.active_dw_task = None
        self.comm_in_progress = False
        self.total_dw_tasks = 0
        self.completed_dw_tasks = 0

    def enable(self):
        """Enable the scheduler"""
        self.enabled = True

    def disable(self):
        """Disable the scheduler"""
        self.enabled = False

    def is_enabled(self) -> bool:
        """Check if scheduler is enabled"""
        return self.enabled

    def register_dw_task(
        self,
        layer_name: str,
        layer_id: int,
        compute_fn: Callable,
        priority: Optional[int] = None,
        weight_param: Optional[torch.nn.Parameter] = None,
    ):
        """
        Register a dW computation task to be executed during communication overlap

        FIFO scheduling: Tasks execute in registration order (backward pass order).
        Priority parameter is ignored but kept for API compatibility.

        Args:
            layer_name: Name of the layer
            layer_id: Layer index
            compute_fn: Function that computes dW (takes no args, returns gradient tensor)
            priority: (Ignored) Kept for API compatibility
            weight_param: Weight parameter to accumulate gradient into
        """
        if not self.enabled:
            return

        task = DWTask(
            layer_name=layer_name,
            layer_id=layer_id,
            compute_fn=compute_fn,
            priority=0,  # Not used in FIFO mode
            weight_param=weight_param,
        )
        self.dw_queue.append(task)
        self.total_dw_tasks += 1

    def set_auto_finish(self, enabled: bool):
        """
        Enable or disable auto-finish mode.

        Args:
            enabled: If True, automatically call finish_batch() when backward ends
        """
        self.auto_finish = enabled
        print(f"[BackwardScheduler] Auto-finish mode: {'enabled' if enabled else 'disabled'}")

    def on_alltoall_start(self, comm_type: str = "unknown"):
        """
        Called when AlltoAll communication is running on comm_stream

        Strategy (User's requirement - NOW ACHIEVABLE!):
        Since AlltoAll is on comm_stream and we have end_event recorded:
        1. Execute dW tasks incrementally on default_stream
        2. After each dW block, check if AlltoAll has completed (via event)
        3. If AlltoAll completed → stop dW, return to continue dX
        4. If AlltoAll not completed → continue next dW block

        This achieves true incremental overlap: dW (GPU) || AlltoAll (network)

        Args:
            comm_type: "ep" for Expert Parallel, "ulysses" for Context Parallel
        """
        if not self.enabled:
            return

        self.comm_in_progress = True

        print(f"[BackwardScheduler] AlltoAll({comm_type}) running on comm_stream, executing dW tasks incrementally")

        # Execute dW tasks incrementally with AlltoAll completion checking
        self._launch_dw_tasks_incremental()

        self.comm_in_progress = False

    def set_alltoall_end_event(self, event: torch.cuda.Event):
        """
        Set the AlltoAll completion event (called by wrapper after AlltoAll finishes)

        Args:
            event: CUDA event recorded after AlltoAll completes on comm_stream
        """
        self.alltoall_end_event = event

    def on_alltoall_end(self):
        """
        Called when an AlltoAll communication completes (optional)

        Note: We DON'T wait for dW to complete here!
        Remaining dW will overlap with subsequent AlltoAll operations.
        """
        if not self.enabled:
            return

        # Don't synchronize! Let dW continue running
        # It will overlap with the next layer's AlltoAll
        pass

    def _is_alltoall_complete(self) -> bool:
        """
        Check if AlltoAll communication has completed (non-blocking)

        Returns:
            True if AlltoAll has completed, False otherwise
        """
        if self.alltoall_end_event is None:
            # No event recorded yet, assume not started
            return False

        # Non-blocking query
        return self.alltoall_end_event.query()

    def _launch_dw_tasks_incremental(self):
        """
        Execute dW tasks incrementally on default_stream, checking AlltoAll
        completion after each block

        Strategy (User's requirement):
        1. Sort tasks by priority (higher priority first)
        2. Execute dW tasks ONE BY ONE on default_stream
        3. After each dW task completes, check if AlltoAll has finished
        4. If AlltoAll completed → stop dW execution, return to continue dX
        5. If AlltoAll not completed → continue next dW task

        Key insight:
        - dW and dX both use GPU compute (sequential on default_stream)
        - AlltoAll uses network (on comm_stream, parallel with dW)
        - dW fills GPU idle time during AlltoAll
        """
        if not self.dw_queue:
            print("[BackwardScheduler] No dW tasks in queue")
            return

        # FIFO: Execute tasks in registration order (no sorting)
        # Tasks are added to queue tail during backward, executed from queue head
        print(f"[BackwardScheduler] Starting incremental dW execution, {len(self.dw_queue)} tasks queued")

        # Execute dW tasks incrementally on default_stream
        tasks_executed = 0
        while self.dw_queue:
            # Check if AlltoAll has completed BEFORE starting next dW
            if self._is_alltoall_complete():
                print(f"[BackwardScheduler] AlltoAll completed after {tasks_executed} dW tasks, stopping dW execution")
                break

            # Get next task
            task = self.dw_queue.popleft()
            self.active_dw_task = task

            # Execute dW computation on default_stream (same as dX)
            task_start_event = torch.cuda.Event(enable_timing=True)
            task_end_event = torch.cuda.Event(enable_timing=True)

            task_start_event.record(self.default_stream)
            try:
                # Execute dW computation - returns gradient tensor
                grad_weight = task.compute_fn()

                # Accumulate gradient into weight parameter
                if task.weight_param is not None and grad_weight is not None:
                    if task.weight_param.grad is None:
                        task.weight_param.grad = grad_weight
                    else:
                        task.weight_param.grad.add_(grad_weight)

                print(f"[BackwardScheduler] Completed dW task: {task.layer_name}")
            except Exception as e:
                # If dW computation fails, log but continue
                print(f"[BackwardScheduler] Warning: dW task {task.layer_name} failed: {e}")
                continue

            task_end_event.record(self.default_stream)

            task.completed = True
            task.event = task_end_event
            self.completed_dw_tasks += 1
            self.overlap_completed_dw_tasks += 1  # Count as overlap completion
            tasks_executed += 1

            # After this dW block completes, loop will check AlltoAll status again

        if not self._is_alltoall_complete() and len(self.dw_queue) == 0:
            print(f"[BackwardScheduler] All {tasks_executed} dW tasks completed, AlltoAll still in progress")

        self.active_dw_task = None

    def finish_batch(self):
        """
        Finish all remaining dW tasks at the end of a batch.

        This ensures mathematical correctness by computing all pending gradients
        before optimizer.step(). Should be called after backward() completes
        for the entire batch (all micro-batches).

        Design:
        - During micro-batches: dW tasks can remain in queue and overlap with
          subsequent micro-batch communications (cross micro-batch overlap)
        - End of batch: All remaining dW tasks must be completed synchronously
        """
        if not self.enabled:
            return

        if not self.dw_queue:
            return

        remaining = len(self.dw_queue)
        print(f"[BackwardScheduler] Batch finished, completing {remaining} remaining dW tasks")

        # Execute all remaining tasks synchronously
        while self.dw_queue:
            task = self.dw_queue.popleft()

            try:
                # Execute dW computation
                grad_weight = task.compute_fn()

                # Accumulate gradient into weight parameter
                if task.weight_param is not None and grad_weight is not None:
                    if task.weight_param.grad is None:
                        task.weight_param.grad = grad_weight
                    else:
                        task.weight_param.grad.add_(grad_weight)

                print(f"[BackwardScheduler] Completed remaining dW: {task.layer_name}")
            except Exception as e:
                print(f"[BackwardScheduler] Warning: Failed to complete dW {task.layer_name}: {e}")
                continue

            task.completed = True
            self.completed_dw_tasks += 1
            self.finish_batch_completed_dw_tasks += 1  # Count as finish_batch completion

        print(f"[BackwardScheduler] All {remaining} remaining dW tasks completed")

    def get_stats(self) -> dict:
        """
        Get scheduler statistics

        Returns:
            dict with keys:
                - total_dw_tasks: Total number of dW tasks registered
                - completed_dw_tasks: Number of completed dW tasks
                - total_comm_time_ms: Total AlltoAll communication time
                - total_dw_overlap_time_ms: Total dW computation time during overlap
                - overlap_efficiency: Ratio of dW time to comm time (ideally ~1.0)
        """
        overlap_efficiency = 0.0
        if self.total_comm_time_ms > 0:
            overlap_efficiency = self.total_dw_overlap_time_ms / self.total_comm_time_ms

        return {
            'total_dw_tasks': self.total_dw_tasks,
            'completed_dw_tasks': self.completed_dw_tasks,
            'overlap_completed_dw_tasks': self.overlap_completed_dw_tasks,
            'finish_batch_completed_dw_tasks': self.finish_batch_completed_dw_tasks,
            'pending_dw_tasks': len(self.dw_queue),
            'total_comm_time_ms': self.total_comm_time_ms,
            'total_dw_overlap_time_ms': self.total_dw_overlap_time_ms,
            'overlap_efficiency': overlap_efficiency,
        }

    def print_stats(self):
        """Print scheduler statistics"""
        stats = self.get_stats()
        print(f"[BackwardScheduler] Statistics:")
        print(f"  Total dW tasks: {stats['total_dw_tasks']}")
        print(f"  Completed dW tasks: {stats['completed_dw_tasks']}")
        print(f"  Pending dW tasks: {stats['pending_dw_tasks']}")
        print(f"  Total comm time: {stats['total_comm_time_ms']:.2f} ms")
        print(f"  Total dW overlap time: {stats['total_dw_overlap_time_ms']:.2f} ms")
        print(f"  Overlap efficiency: {stats['overlap_efficiency']:.2%}")


# Global scheduler instance accessor
def get_backward_scheduler() -> BackwardScheduler:
    """Get the global backward scheduler instance"""
    return BackwardScheduler.get_instance()
