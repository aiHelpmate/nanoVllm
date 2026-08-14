"""Scheduler for continuous batching with advanced scheduling features.

Supports:
- FCFS (First Come First Serve) scheduling (Phase 2)
- Block-based PagedAttention (Phase 3)
- Priority-based scheduling (Phase 4)
- Preemption and swapping (Phase 4)
- Chunked prefill (Phase 4)
"""

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, TYPE_CHECKING

from nano_vllm.core.sequence import Sequence, SequenceStatus
from nano_vllm.core.block import BLOCK_SIZE, compute_num_blocks

if TYPE_CHECKING:
    from nano_vllm.core.block_manager import BlockManager


class SchedulingPolicy(Enum):
    """Scheduling policy for ordering requests."""
    FCFS = "fcfs"  # First Come First Serve (arrival order)
    PRIORITY = "priority"  # Priority-based (higher priority first)


@dataclass
class SchedulerOutputs:
    """Output of a scheduling decision."""

    # Sequences that need prefill (new sequences, process all prompt tokens)
    prefill_sequences: List[Sequence] = field(default_factory=list)

    # Sequences that need decode (continuing sequences, process 1 token)
    decode_sequences: List[Sequence] = field(default_factory=list)

    # Sequences that were preempted this iteration (Phase 4)
    preempted_sequences: List[Sequence] = field(default_factory=list)

    # Sequences doing chunked prefill (partial prefill, Phase 4)
    chunked_prefill_sequences: List[Sequence] = field(default_factory=list)

    # Number of tokens to prefill per chunked sequence (parallel list)
    chunked_prefill_tokens: List[int] = field(default_factory=list)

    @property
    def num_prefill(self) -> int:
        return len(self.prefill_sequences)

    @property
    def num_decode(self) -> int:
        return len(self.decode_sequences)

    @property
    def num_preempted(self) -> int:
        return len(self.preempted_sequences)

    @property
    def num_chunked_prefill(self) -> int:
        return len(self.chunked_prefill_sequences)

    @property
    def total_sequences(self) -> int:
        return self.num_prefill + self.num_decode + self.num_chunked_prefill

    def is_empty(self) -> bool:
        return self.total_sequences == 0


class Scheduler:
    """Advanced scheduler for continuous batching.

    Manages sequences through their lifecycle:
    - WAITING: In queue, not yet started
    - RUNNING: Currently being processed
    - SWAPPED: Preempted, waiting to resume
    - FINISHED: Generation complete

    Each iteration, the scheduler decides which sequences to process.

    Features:
    - FCFS or Priority-based scheduling
    - Preemption of low-priority sequences for high-priority ones
    - Chunked prefill to limit prefill tokens per iteration
    - Coordination with BlockManager for PagedAttention
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        block_manager: Optional["BlockManager"] = None,
        block_size: int = BLOCK_SIZE,
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.PRIORITY,
        enable_preemption: bool = True,
        max_prefill_tokens: int = 512,
    ):
        """Initialize the scheduler.

        Args:
            max_batch_size: Maximum number of sequences to process in one batch
            block_manager: Optional BlockManager for PagedAttention
            block_size: Tokens per block (only used with block_manager)
            scheduling_policy: FCFS or PRIORITY scheduling
            enable_preemption: If True, can preempt low-priority sequences
            max_prefill_tokens: Maximum tokens to prefill per iteration (chunked prefill)
        """
        self.max_batch_size = max_batch_size
        self.block_manager = block_manager
        self.block_size = block_size
        self.scheduling_policy = scheduling_policy
        self.enable_preemption = enable_preemption
        self.max_prefill_tokens = max_prefill_tokens

        # Sequence queues
        # waiting is a heap: (priority_key, sequence) where lower key = higher priority
        self._waiting_heap: List[Tuple[Tuple[int, float, int], Sequence]] = []
        self.running: List[Sequence] = []
        self.finished: List[Sequence] = []

        # Sequence ID counter
        self._next_seq_id = 0

    def _get_priority_key(self, seq: Sequence) -> Tuple[int, float, int]:
        """Get the priority key for heap ordering.

        Returns a tuple that when compared with < gives higher priority first.
        Lower tuple value = higher priority.

        For FCFS policy:
        - Ignores priority, orders purely by arrival time
        - Earlier arrival time comes first
        - Lower seq_id for tie-breaking

        For PRIORITY policy:
        - Higher priority value comes first (negate so lower tuple value wins)
        - Earlier arrival time for tie-breaking among same priority
        - Lower seq_id for determinism
        """
        if self.scheduling_policy == SchedulingPolicy.FCFS:
            # FCFS: ignore priority, use only arrival time
            return (0, seq.arrival_time, seq.seq_id)
        else:
            # Priority-based: higher priority first
            return (-seq.priority, seq.arrival_time, seq.seq_id)

    def _push_waiting(self, seq: Sequence) -> None:
        """Push a sequence onto the waiting heap."""
        key = self._get_priority_key(seq)
        heapq.heappush(self._waiting_heap, (key, seq))

    def _pop_waiting(self) -> Optional[Sequence]:
        """Pop the highest priority sequence from waiting."""
        if not self._waiting_heap:
            return None
        _, seq = heapq.heappop(self._waiting_heap)
        return seq

    def _peek_waiting(self) -> Optional[Sequence]:
        """Peek at the highest priority waiting sequence without removing."""
        if not self._waiting_heap:
            return None
        return self._waiting_heap[0][1]

    @property
    def waiting(self) -> List[Sequence]:
        """Get waiting sequences as a list (for backwards compatibility)."""
        return [seq for _, seq in self._waiting_heap]

    def add_request(
        self,
        prompt_token_ids: List[int],
        max_tokens: int,
        priority: int = 0,
    ) -> Sequence:
        """Add a new request to the waiting queue.

        Args:
            prompt_token_ids: Tokenized prompt
            max_tokens: Maximum tokens to generate
            priority: Request priority (higher = more important)

        Returns:
            The created Sequence object
        """
        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            status=SequenceStatus.WAITING,
            priority=priority,
        )
        self._next_seq_id += 1
        self._push_waiting(seq)
        return seq

    def schedule(self) -> SchedulerOutputs:
        """Decide which sequences to process this iteration.

        Scheduling policy:
        1. Handle preemption if high-priority waiting and not enough resources
        2. Continue chunked prefills that are in progress
        3. Keep all running sequences (they need decode)
        4. Fill remaining slots from waiting queue by priority (if blocks available)
        5. Apply chunked prefill limits to new sequences

        Returns:
            SchedulerOutputs with prefill, decode, chunked_prefill, and preempted sequences
        """
        outputs = SchedulerOutputs()

        # Step 1: Handle preemption if needed
        if self.enable_preemption and self.block_manager is not None:
            self._handle_preemption(outputs)

        # Step 2: Schedule running sequences for decode or chunked prefill continuation
        prefill_budget = self.max_prefill_tokens

        for seq in self.running:
            if seq.is_chunked_prefill():
                # Continue chunked prefill
                remaining = seq.get_remaining_prefill_tokens()
                tokens_to_process = min(prefill_budget, remaining)
                if tokens_to_process > 0:
                    outputs.chunked_prefill_sequences.append(seq)
                    outputs.chunked_prefill_tokens.append(tokens_to_process)
                    prefill_budget -= tokens_to_process
            else:
                # Normal decode
                outputs.decode_sequences.append(seq)

        # Step 3: Calculate remaining batch capacity
        remaining_slots = self.max_batch_size - len(outputs.decode_sequences) - len(outputs.chunked_prefill_sequences)

        # Step 4: Add waiting sequences (they need prefill)
        num_added = 0
        while num_added < remaining_slots and self._waiting_heap and prefill_budget > 0:
            seq = self._peek_waiting()
            if seq is None:
                break

            # Calculate blocks needed for the FULL sequence prompt
            # This ensures that once admitted, the sequence can complete all chunks
            # Without this, a sequence might be admitted but then starve waiting for blocks
            if self.block_manager is not None:
                # Reserve blocks for the full prompt length, not just current chunk
                full_prompt_len = seq.get_prompt_len()
                blocks_needed = compute_num_blocks(full_prompt_len, self.block_size)
                if not self.block_manager.can_allocate(blocks_needed):
                    # Not enough blocks for full prompt, stop admitting new sequences
                    break

            # Admit the sequence
            self._pop_waiting()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

            # Decide if this is a full prefill or chunked prefill
            prompt_len = seq.get_prompt_len()
            if prompt_len <= prefill_budget:
                # Full prefill
                outputs.prefill_sequences.append(seq)
                prefill_budget -= prompt_len
            else:
                # Chunked prefill - only process what fits in budget
                tokens_to_process = prefill_budget
                outputs.chunked_prefill_sequences.append(seq)
                outputs.chunked_prefill_tokens.append(tokens_to_process)
                prefill_budget = 0

            num_added += 1

        return outputs

    def _handle_preemption(self, outputs: SchedulerOutputs) -> None:
        """Handle preemption of low-priority sequences for high-priority waiting.

        Uses recompute-based preemption: preempted sequences have their blocks
        freed and are moved back to waiting to be re-prefilled later.
        """
        if not self._waiting_heap or not self.running:
            return

        highest_waiting = self._peek_waiting()
        if highest_waiting is None:
            return

        # Check if we need to preempt: highest waiting can't get blocks
        blocks_for_waiting = compute_num_blocks(highest_waiting.get_prompt_len(), self.block_size)

        while not self.block_manager.can_allocate(blocks_for_waiting) and self.running:
            # Find lowest priority running sequence
            lowest_running = min(self.running, key=lambda s: (s.priority, -s.arrival_time))

            # Only preempt if waiting has higher priority
            if highest_waiting.priority <= lowest_running.priority:
                # Waiting doesn't have higher priority, can't preempt
                break

            # Preempt the lowest priority running sequence
            self.running.remove(lowest_running)
            outputs.preempted_sequences.append(lowest_running)

            # Free blocks and reset for recompute
            if lowest_running.block_table is not None:
                self.block_manager.free_sequence_blocks(lowest_running.block_table)
            lowest_running.reset_for_recompute()

            # Re-add to waiting queue
            self._push_waiting(lowest_running)

    def update_sequences(self, eos_token_id: int) -> List[Sequence]:
        """Update sequence statuses after a generation step.

        Moves finished sequences from running to finished.

        Args:
            eos_token_id: Token ID that signals end of generation

        Returns:
            List of newly finished sequences
        """
        newly_finished = []

        # Check each running sequence
        still_running = []
        for seq in self.running:
            if seq.is_finished(eos_token_id):
                seq.status = SequenceStatus.FINISHED
                self.finished.append(seq)
                newly_finished.append(seq)
            else:
                still_running.append(seq)

        self.running = still_running
        return newly_finished

    def has_pending_requests(self) -> bool:
        """Check if there are any requests still being processed."""
        return len(self._waiting_heap) > 0 or len(self.running) > 0

    def get_num_waiting(self) -> int:
        return len(self._waiting_heap)

    def get_num_running(self) -> int:
        return len(self.running)

    def get_num_finished(self) -> int:
        return len(self.finished)

    def get_lowest_priority_running(self) -> Optional[Sequence]:
        """Get the lowest priority running sequence (for potential preemption)."""
        if not self.running:
            return None
        return min(self.running, key=lambda s: (s.priority, -s.arrival_time))

    def get_highest_priority_waiting(self) -> Optional[Sequence]:
        """Get the highest priority waiting sequence."""
        return self._peek_waiting()

    def __repr__(self) -> str:
        return (
            f"Scheduler(policy={self.scheduling_policy.value}, "
            f"waiting={self.get_num_waiting()}, "
            f"running={self.get_num_running()}, "
            f"finished={self.get_num_finished()})"
        )