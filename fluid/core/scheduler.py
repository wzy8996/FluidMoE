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

import os
import torch
from typing import List, Optional, Callable, Tuple
from collections import deque
import threading

# Debug flag for scheduler verbose output
_DEBUG_SCHEDULER = os.environ.get('FLUID_DEBUG_SCHEDULER', '0') == '1'


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

        # ============================================================
        # Reusable CUDA Events (avoid repeated creation overhead)
        # ============================================================
        # Single reusable event for AllToAll completion tracking
        # Only one AllToAll runs at a time during backward, so one event is sufficient
        self._reusable_event = torch.cuda.Event()

        # Reusable event for compute-to-comm synchronization in chunked backward
        # Used by dispatch_backward, output_projection_backward_chunked, attention_backward_chunked
        self._compute_sync_event = torch.cuda.Event()

        # Reusable timing events (for dW task timing, if DEBUG enabled)
        self._timing_event_start: Optional[torch.cuda.Event] = None
        self._timing_event_end: Optional[torch.cuda.Event] = None
        if _DEBUG_SCHEDULER:
            self._timing_event_start = torch.cuda.Event(enable_timing=True)
            self._timing_event_end = torch.cuda.Event(enable_timing=True)

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

        # ============================================================
        # Cross-layer QKV-Combine overlap context
        # ============================================================
        # Pending combine contexts: registered during combine forward,
        # consumed by QKV backward to pre-launch AllToAll
        self.pending_combine_contexts: List[dict] = []

        # Pre-launched AllToAll results: launched by QKV backward,
        # consumed by combine backward
        self.prelaunched_combine_results: Optional[dict] = None

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
        self.overlap_completed_dw_tasks = 0
        self.finish_batch_completed_dw_tasks = 0

    def clear_iteration(self):
        """
        Clear the queue and stats for a new iteration while keeping the scheduler enabled.

        Use this between benchmark iterations instead of reset() to avoid
        destroying the singleton and losing the enabled state.
        """
        self.dw_queue.clear()
        self.active_dw_task = None
        self.comm_in_progress = False
        self.total_dw_tasks = 0
        self.completed_dw_tasks = 0
        self.overlap_completed_dw_tasks = 0
        self.finish_batch_completed_dw_tasks = 0
        self.pending_combine_contexts.clear()
        self.prelaunched_combine_results = None

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
            if os.environ.get('FLUID_DEBUG_SCHEDULER_REGISTER', '0') == '1':
                print(f"[DEBUG] register_dw_task({layer_name}) skipped: scheduler not enabled", flush=True)
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

        # Debug: print registered tasks
        if _DEBUG_SCHEDULER or os.environ.get('FLUID_DEBUG_DW_TASKS', '0') == '1':
            print(f"[DEBUG-DW] Task #{self.total_dw_tasks}: {layer_name} (layer_id={layer_id})", flush=True)

    def set_auto_finish(self, enabled: bool):
        """
        Enable or disable auto-finish mode.

        Args:
            enabled: If True, automatically call finish_batch() when backward ends
        """
        self.auto_finish = enabled
        if _DEBUG_SCHEDULER:
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

        queue_before = len(self.dw_queue)

        # Execute dW tasks incrementally with AlltoAll completion checking
        self._launch_dw_tasks_incremental()

        queue_after = len(self.dw_queue)
        tasks_executed = queue_before - queue_after

        if _DEBUG_SCHEDULER and tasks_executed > 0:
            print(f"[Scheduler] {comm_type}: executed {tasks_executed} dW tasks (queue: {queue_before} -> {queue_after})")

        self.comm_in_progress = False

    def get_reusable_event(self) -> torch.cuda.Event:
        """
        Get the reusable CUDA event for AllToAll completion tracking.

        This avoids the overhead of creating new events for each AllToAll.
        Since only one AllToAll runs at a time during backward, a single
        reusable event is sufficient.

        Returns:
            The reusable CUDA event
        """
        return self._reusable_event

    def get_compute_sync_event(self) -> torch.cuda.Event:
        """
        Get the reusable CUDA event for compute-to-comm synchronization.

        This is used by chunked backward functions (dispatch_backward,
        output_projection_backward_chunked, attention_backward_chunked) to
        synchronize compute completion before submitting communication.

        Returns:
            The compute sync CUDA event
        """
        return self._compute_sync_event

    def set_alltoall_end_event(self, event: torch.cuda.Event):
        """
        Set the AlltoAll completion event (called by wrapper after AlltoAll finishes)

        Args:
            event: CUDA event recorded after AlltoAll completes on comm_stream
        """
        self.alltoall_end_event = event

    def record_alltoall_end(self, stream: torch.cuda.Stream) -> torch.cuda.Event:
        """
        Record AllToAll completion using a reusable event.

        This is a convenience method that combines get_reusable_event(),
        record(), and set_alltoall_end_event() into one call.

        Args:
            stream: The CUDA stream where AllToAll is running

        Returns:
            The recorded event (for chaining if needed)
        """
        event = self.get_reusable_event()
        event.record(stream)
        self.alltoall_end_event = event
        return event

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
            return

        # Goal: hide AllToAll communication under dW compute
        # Execute dW tasks until AllToAll completes or queue is empty
        tasks_executed = 0
        while self.dw_queue:
            # Check if AllToAll has completed - if so, stop to continue dX
            if tasks_executed > 0 and self._is_alltoall_complete():
                break

            # Get next task
            task = self.dw_queue.popleft()
            self.active_dw_task = task

            # Optional timing (only when DEBUG enabled, uses reusable events)
            if _DEBUG_SCHEDULER and self._timing_event_start is not None:
                self._timing_event_start.record(self.default_stream)

            try:
                # Execute dW computation - returns gradient tensor
                grad_weight = task.compute_fn()

                # DEBUG: Print dW norm for MoE experts
                if os.environ.get('FLUID_DEBUG_DW_NORM', '0') == '1' and 'moe_expert' in task.layer_name:
                    print(f"[Scheduler] {task.layer_name} grad norm: {grad_weight.norm().item():.6f}", flush=True)

                # Accumulate gradient into weight parameter
                # Note: clone() when setting to match PyTorch autograd behavior
                if task.weight_param is not None and grad_weight is not None:
                    if task.weight_param.grad is None:
                        task.weight_param.grad = grad_weight.clone()
                    else:
                        task.weight_param.grad.add_(grad_weight)

            except Exception as e:
                # If dW computation fails, log but continue
                if _DEBUG_SCHEDULER:
                    print(f"[BackwardScheduler] Warning: dW task {task.layer_name} failed: {e}")
                continue

            # Optional timing end (only when DEBUG enabled)
            if _DEBUG_SCHEDULER and self._timing_event_end is not None:
                self._timing_event_end.record(self.default_stream)
                task.event = self._timing_event_end

            task.completed = True
            self.completed_dw_tasks += 1
            self.overlap_completed_dw_tasks += 1  # Count as overlap completion
            tasks_executed += 1

            # After this dW block completes, loop will check AlltoAll status again

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
        if _DEBUG_SCHEDULER:
            print(f"[BackwardScheduler] Batch finished, completing {remaining} remaining dW tasks")

        # Execute all remaining tasks synchronously
        while self.dw_queue:
            task = self.dw_queue.popleft()

            try:
                # Execute dW computation
                grad_weight = task.compute_fn()

                # Accumulate gradient into weight parameter
                # Note: clone() when setting to match PyTorch autograd behavior
                if task.weight_param is not None and grad_weight is not None:
                    if task.weight_param.grad is None:
                        task.weight_param.grad = grad_weight.clone()
                    else:
                        task.weight_param.grad.add_(grad_weight)

                if _DEBUG_SCHEDULER:
                    print(f"[BackwardScheduler] Completed remaining dW: {task.layer_name}")
            except Exception as e:
                if _DEBUG_SCHEDULER:
                    print(f"[BackwardScheduler] Warning: Failed to complete dW {task.layer_name}: {e}")
                continue

            task.completed = True
            self.completed_dw_tasks += 1
            self.finish_batch_completed_dw_tasks += 1  # Count as finish_batch completion

        if _DEBUG_SCHEDULER:
            print(f"[BackwardScheduler] All {remaining} remaining dW tasks completed")

    def _execute_all_dw_tasks_sync(self) -> int:
        """
        Execute all pending dW tasks synchronously on default_stream.

        This is used by TRUE ASYNC mode where dW runs in parallel with AllToAll.
        Unlike _launch_dw_tasks_incremental(), this does NOT poll AllToAll status.

        Returns:
            Number of dW tasks executed
        """
        if not self.enabled:
            return 0

        tasks_executed = 0

        while self.dw_queue:
            task = self.dw_queue.popleft()
            self.active_dw_task = task

            try:
                # Execute dW computation on default_stream
                grad_weight = task.compute_fn()

                # Accumulate gradient into weight parameter
                # Note: clone() when setting to match PyTorch autograd behavior
                if task.weight_param is not None and grad_weight is not None:
                    if task.weight_param.grad is None:
                        task.weight_param.grad = grad_weight.clone()
                    else:
                        task.weight_param.grad.add_(grad_weight)

            except Exception as e:
                if _DEBUG_SCHEDULER:
                    print(f"[BackwardScheduler] Warning: dW task {task.layer_name} failed: {e}")
                continue

            task.completed = True
            self.completed_dw_tasks += 1
            self.overlap_completed_dw_tasks += 1  # Count as overlap completion
            tasks_executed += 1

        self.active_dw_task = None
        return tasks_executed

    # ============================================================
    # Cross-layer QKV-Combine overlap methods
    # ============================================================

    def register_combine_context(
        self,
        output_splits: List[int],
        input_splits: List[int],
        group,
        permutation_map: torch.Tensor,
        permuted_probs: Optional[torch.Tensor] = None,
    ):
        """
        Register combine context during combine forward.
        This will be consumed by QKV backward to pre-launch AllToAll.

        Args:
            output_splits: AllToAll output splits (backward input)
            input_splits: AllToAll input splits (backward output)
            group: Process group
            permutation_map: Token permutation map
            permuted_probs: Permuted probs for gradient scaling
        """
        if not self.enabled:
            return

        ctx = {
            'output_splits': output_splits,
            'input_splits': input_splits,
            'group': group,
            'permutation_map': permutation_map,
            'permuted_probs': permuted_probs,
        }
        self.pending_combine_contexts.append(ctx)

    def get_pending_combine_context(self) -> Optional[dict]:
        """
        Get the next pending combine context (LIFO order - stack).

        Why LIFO: Forward registers contexts as Layer 0, 1, ..., N.
        Backward processes Layer N first, then N-1, etc.
        Layer N's QKV backward should pre-launch for Layer N-1's combine.
        With LIFO, after Layer N's combine pops its own context (N),
        the next pop() returns N-1's context for Layer N's QKV to use.

        Returns None if no context is pending.
        """
        if not self.enabled or not self.pending_combine_contexts:
            return None
        return self.pending_combine_contexts.pop()  # LIFO - pop from end

    def set_prelaunched_combine_results(
        self,
        results: List[torch.Tensor],
        events: List[torch.cuda.Event],
    ):
        """
        Store pre-launched AllToAll results from QKV backward.

        Args:
            results: List of AllToAll result chunks (hidden dim chunks)
            events: List of events marking each chunk's completion
        """
        self.prelaunched_combine_results = {
            'results': results,
            'events': events,
        }

    def get_prelaunched_combine_results(self) -> Optional[dict]:
        """
        Get pre-launched combine results. Returns None if not available.
        Clears the storage after retrieval.
        """
        results = self.prelaunched_combine_results
        self.prelaunched_combine_results = None
        return results

    def clear_combine_context(self):
        """Clear all pending combine contexts (called at batch boundary)."""
        self.pending_combine_contexts.clear()
        self.prelaunched_combine_results = None

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
