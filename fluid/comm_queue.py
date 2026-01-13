"""
Async Communication Queue for dX + AllToAll Pipeline

Design:
- Separate comm_stream executes AllToAll operations
- Main stream computes dX chunks and submits to queue
- Queue entries are processed sequentially on comm_stream
- True overlap between dX computation and AllToAll communication

Timeline:
  default_stream:  |dX_c0|event0|dX_c1|event1|dX_c2|event2| ... |wait all|
                         ↓           ↓           ↓
  comm_stream:      wait0|A2A_c0|wait1|A2A_c1|wait2|A2A_c2| ...

Usage:
    queue = CommQueue(comm_stream)

    for chunk_idx in range(num_chunks):
        dx_chunk = compute_dx_chunk(...)
        queue.submit_alltoall(dx_chunk, splits, group)

    results = queue.wait_all()
"""

import torch
import torch.distributed as dist
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CommTask:
    """A pending communication task"""
    input_tensor: torch.Tensor
    output_split_sizes: Optional[List[int]]
    input_split_sizes: Optional[List[int]]
    group: any
    compute_done_event: torch.cuda.Event  # Event when input is ready
    output_tensor: Optional[torch.Tensor] = None  # Filled after execution
    comm_done_event: Optional[torch.cuda.Event] = None  # Event when comm is done


class CommQueue:
    """
    Async communication queue for pipelined AllToAll operations.

    Allows submitting AllToAll tasks from compute stream while
    they execute asynchronously on a separate comm stream.
    """

    def __init__(self, comm_stream: torch.cuda.Stream = None):
        """
        Args:
            comm_stream: CUDA stream for communication. If None, creates one.
        """
        self.comm_stream = comm_stream or torch.cuda.Stream()
        self.default_stream = torch.cuda.current_stream()
        self.pending_tasks: List[CommTask] = []
        self._executed = False

    def submit_alltoall(
        self,
        input_tensor: torch.Tensor,
        output_split_sizes: Optional[List[int]],
        input_split_sizes: Optional[List[int]],
        group,
    ) -> int:
        """
        Submit an AllToAll task to the queue.

        The task will execute on comm_stream after the current default stream
        operations complete (tracked via CUDA event).

        Args:
            input_tensor: Input tensor (must be ready on default stream)
            output_split_sizes: Output split sizes for all_to_all_single
            input_split_sizes: Input split sizes for all_to_all_single
            group: Process group

        Returns:
            Task index (for retrieving result later)
        """
        # Record event on default stream to mark input ready
        compute_done_event = torch.cuda.Event()
        compute_done_event.record(self.default_stream)

        task = CommTask(
            input_tensor=input_tensor.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            compute_done_event=compute_done_event,
        )

        task_idx = len(self.pending_tasks)
        self.pending_tasks.append(task)

        # Immediately schedule on comm_stream
        self._schedule_task(task)

        return task_idx

    def _schedule_task(self, task: CommTask):
        """Schedule a single task on comm_stream"""
        with torch.cuda.stream(self.comm_stream):
            # Wait for input to be ready
            self.comm_stream.wait_event(task.compute_done_event)

            # Execute AllToAll
            world_size = task.group.size()
            if world_size == 1:
                task.output_tensor = task.input_tensor.clone()
            else:
                if task.output_split_sizes is None:
                    task.output_tensor = torch.empty_like(task.input_tensor)
                else:
                    task.output_tensor = task.input_tensor.new_empty(
                        size=[sum(task.output_split_sizes)] + list(task.input_tensor.size()[1:]),
                        dtype=task.input_tensor.dtype,
                        device=task.input_tensor.device,
                    )

                dist.all_to_all_single(
                    task.output_tensor,
                    task.input_tensor,
                    output_split_sizes=task.output_split_sizes,
                    input_split_sizes=task.input_split_sizes,
                    group=task.group,
                )

            # Record completion event
            task.comm_done_event = torch.cuda.Event()
            task.comm_done_event.record(self.comm_stream)

    def wait_all(self) -> List[torch.Tensor]:
        """
        Wait for all pending tasks to complete and return results.

        Returns:
            List of output tensors in submission order
        """
        if not self.pending_tasks:
            return []

        # Wait for comm_stream to finish
        self.default_stream.wait_stream(self.comm_stream)

        # Collect results
        results = [task.output_tensor for task in self.pending_tasks]

        # Clear queue
        self.pending_tasks.clear()
        self._executed = True

        return results

    def get_last_event(self) -> Optional[torch.cuda.Event]:
        """Get the completion event of the last submitted task"""
        if not self.pending_tasks:
            return None
        return self.pending_tasks[-1].comm_done_event

    def num_pending(self) -> int:
        """Number of pending tasks"""
        return len(self.pending_tasks)


class ChunkedDxAllToAllPipeline:
    """
    Pipeline for chunked dX computation with AllToAll overlap.

    This class orchestrates:
    1. Splitting input gradient into chunks
    2. Computing dX for each chunk (FC1 backward)
    3. Submitting AllToAll for each chunk to CommQueue
    4. Collecting and reassembling results

    Timeline:
      compute:  |-- dX_c0 --|-- dX_c1 --|-- dX_c2 --|  wait  |
                      ↓           ↓           ↓
      comm:          |A2A_c0|    |A2A_c1|    |A2A_c2|
                          overlap!    overlap!
    """

    def __init__(
        self,
        num_chunks: int,
        comm_stream: torch.cuda.Stream,
        group,
        output_split_sizes: Optional[List[int]] = None,
        input_split_sizes: Optional[List[int]] = None,
    ):
        """
        Args:
            num_chunks: Number of chunks to split dX computation
            comm_stream: CUDA stream for AllToAll communication
            group: Process group for AllToAll
            output_split_sizes: Split sizes for AllToAll output
            input_split_sizes: Split sizes for AllToAll input
        """
        self.num_chunks = num_chunks
        self.comm_queue = CommQueue(comm_stream)
        self.group = group
        self.output_split_sizes = output_split_sizes
        self.input_split_sizes = input_split_sizes
        self.chunk_results: List[Tuple[torch.Tensor, int, int]] = []  # (result, start, end)

    def compute_chunk_bounds(self, total_tokens: int) -> List[Tuple[int, int]]:
        """Compute (start, end) for each chunk"""
        chunk_size = total_tokens // self.num_chunks
        remainder = total_tokens % self.num_chunks

        bounds = []
        offset = 0
        for i in range(self.num_chunks):
            size = chunk_size + (1 if i < remainder else 0)
            bounds.append((offset, offset + size))
            offset += size
        return bounds

    def submit_chunk(
        self,
        dx_chunk: torch.Tensor,
        chunk_start: int,
        chunk_end: int,
        chunk_output_splits: Optional[List[int]] = None,
        chunk_input_splits: Optional[List[int]] = None,
    ):
        """
        Submit a computed dX chunk for AllToAll.

        Args:
            dx_chunk: Computed dX for this chunk
            chunk_start: Start index in original tensor
            chunk_end: End index in original tensor
            chunk_output_splits: Output splits for this chunk (if different from default)
            chunk_input_splits: Input splits for this chunk (if different from default)
        """
        out_splits = chunk_output_splits or self.output_split_sizes
        in_splits = chunk_input_splits or self.input_split_sizes

        self.comm_queue.submit_alltoall(
            dx_chunk,
            out_splits,
            in_splits,
            self.group,
        )

        self.chunk_results.append((None, chunk_start, chunk_end))  # Placeholder

    def finish_and_gather(self) -> torch.Tensor:
        """
        Wait for all AllToAll operations and reassemble results.

        Returns:
            Complete dX tensor (all chunks concatenated)
        """
        # Wait for all communications
        results = self.comm_queue.wait_all()

        # Concatenate results
        if len(results) == 1:
            return results[0]
        return torch.cat(results, dim=0)

    def get_last_comm_event(self) -> Optional[torch.cuda.Event]:
        """Get the completion event of the last AllToAll (for dW overlap)"""
        return self.comm_queue.get_last_event()
