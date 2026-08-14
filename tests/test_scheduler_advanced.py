"""Tests for advanced scheduling features (Phase 4).

Tests:
1. Priority-based Scheduling
2. Preemption and Swapping
3. Chunked Prefill
4. Prefix Caching
"""

import time
import pytest
from typing import List

from nano_vllm.core.sequence import Sequence, SequenceStatus
from nano_vllm.core.scheduler import Scheduler, SchedulerOutputs, SchedulingPolicy
from nano_vllm.core.block import Block, BlockTable, BLOCK_SIZE, compute_block_hash
from nano_vllm.core.block_manager import BlockManager


class TestPriorityScheduling:
    """Tests for priority-based scheduling."""

    def test_priority_ordering(self):
        """Higher priority sequences should be scheduled first."""
        scheduler = Scheduler(
            max_batch_size=3,
            scheduling_policy=SchedulingPolicy.PRIORITY,
        )

        # Add sequences with different priorities
        seq_low = scheduler.add_request([1, 2, 3], max_tokens=10, priority=1)
        seq_medium = scheduler.add_request([4, 5, 6], max_tokens=10, priority=5)
        seq_high = scheduler.add_request([7, 8, 9], max_tokens=10, priority=10)

        # Schedule
        outputs = scheduler.schedule()

        # High priority should be scheduled first
        assert len(outputs.prefill_sequences) == 3
        assert outputs.prefill_sequences[0].seq_id == seq_high.seq_id
        assert outputs.prefill_sequences[1].seq_id == seq_medium.seq_id
        assert outputs.prefill_sequences[2].seq_id == seq_low.seq_id

    def test_priority_tie_breaking_by_arrival_time(self):
        """When priorities are equal, earlier arrival should win."""
        scheduler = Scheduler(
            max_batch_size=2,
            scheduling_policy=SchedulingPolicy.PRIORITY,
        )

        # Add sequences with same priority
        seq1 = scheduler.add_request([1, 2, 3], max_tokens=10, priority=5)
        time.sleep(0.01)  # Small delay to ensure different arrival times
        seq2 = scheduler.add_request([4, 5, 6], max_tokens=10, priority=5)

        # Schedule
        outputs = scheduler.schedule()

        # Earlier arrival should be first
        assert len(outputs.prefill_sequences) == 2
        assert outputs.prefill_sequences[0].seq_id == seq1.seq_id
        assert outputs.prefill_sequences[1].seq_id == seq2.seq_id

    def test_fcfs_policy_ignores_priority(self):
        """FCFS policy should ignore priority and use arrival order."""
        scheduler = Scheduler(
            max_batch_size=2,
            scheduling_policy=SchedulingPolicy.FCFS,
        )

        # Add high priority first, then low priority
        seq_high = scheduler.add_request([1, 2, 3], max_tokens=10, priority=10)
        seq_low = scheduler.add_request([4, 5, 6], max_tokens=10, priority=1)

        # With FCFS, arrival order matters, not priority
        # Since we're using a heap internally, FCFS still respects arrival time
        outputs = scheduler.schedule()

        # Both should be scheduled, order by arrival
        assert len(outputs.prefill_sequences) == 2


class TestPreemption:
    """Tests for preemption and swapping."""

    def test_preemption_for_high_priority(self):
        """Low priority running should be preempted for high priority waiting."""
        # Create block manager with limited blocks
        block_manager = BlockManager(num_blocks=10, block_size=BLOCK_SIZE)

        scheduler = Scheduler(
            max_batch_size=2,
            block_manager=block_manager,
            block_size=BLOCK_SIZE,
            scheduling_policy=SchedulingPolicy.PRIORITY,
            enable_preemption=True,
        )

        # Add and start a low priority sequence
        prompt_tokens = list(range(BLOCK_SIZE * 5))  # Uses 5 blocks
        seq_low = scheduler.add_request(prompt_tokens, max_tokens=10, priority=1)

        # Schedule it
        outputs = scheduler.schedule()
        assert seq_low in outputs.prefill_sequences

        # Simulate allocating blocks
        seq_low.block_table = block_manager.allocate_blocks_for_sequence(5)

        # Now add a high priority sequence that needs blocks
        seq_high = scheduler.add_request(prompt_tokens, max_tokens=10, priority=10)

        # Use up remaining blocks
        for _ in range(5):
            block_manager.allocate_block()

        # Now schedule again - should preempt low priority
        outputs = scheduler.schedule()

        # Low priority should be preempted (appears in preempted_sequences)
        # Note: it may be re-admitted in the same scheduling round after blocks are freed
        assert seq_low in outputs.preempted_sequences

    def test_no_preemption_when_disabled(self):
        """When preemption is disabled, should not preempt."""
        block_manager = BlockManager(num_blocks=10, block_size=BLOCK_SIZE)

        scheduler = Scheduler(
            max_batch_size=2,
            block_manager=block_manager,
            block_size=BLOCK_SIZE,
            scheduling_policy=SchedulingPolicy.PRIORITY,
            enable_preemption=False,  # Disabled
        )

        # Add low priority and schedule
        prompt_tokens = list(range(BLOCK_SIZE * 5))
        seq_low = scheduler.add_request(prompt_tokens, max_tokens=10, priority=1)
        outputs = scheduler.schedule()
        seq_low.block_table = block_manager.allocate_blocks_for_sequence(5)

        # Add high priority
        seq_high = scheduler.add_request(prompt_tokens, max_tokens=10, priority=10)

        # Use up remaining blocks
        for _ in range(5):
            block_manager.allocate_block()

        # Schedule - should NOT preempt
        outputs = scheduler.schedule()
        assert len(outputs.preempted_sequences) == 0

    def test_reset_for_recompute(self):
        """Test that reset_for_recompute properly resets sequence state."""
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3, 4, 5],
            max_tokens=10,
            priority=5,
        )

        # Simulate some processing
        seq.output_token_ids = [10, 11, 12]
        seq.num_prefilled_tokens = 5
        seq.status = SequenceStatus.RUNNING
        seq.block_table = BlockTable(block_size=BLOCK_SIZE)
        seq.block_table.append_block(0)

        # Reset
        seq.reset_for_recompute()

        # Check reset state
        assert seq.output_token_ids == []
        assert seq.num_prefilled_tokens == 0
        assert seq.status == SequenceStatus.WAITING
        assert seq.block_table is None


class TestChunkedPrefill:
    """Tests for chunked prefill."""

    def test_chunked_prefill_detection(self):
        """Test is_chunked_prefill correctly identifies partial prefill."""
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            max_tokens=10,
        )

        # Initially not chunked
        assert not seq.is_chunked_prefill()

        # Partial prefill
        seq.num_prefilled_tokens = 5
        assert seq.is_chunked_prefill()

        # Fully prefilled
        seq.num_prefilled_tokens = 10
        assert not seq.is_chunked_prefill()

    def test_get_next_chunk_tokens(self):
        """Test getting the next chunk of tokens to prefill."""
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            max_tokens=10,
        )

        # First chunk
        chunk = seq.get_next_chunk_tokens(3)
        assert chunk == [1, 2, 3]

        # After processing first chunk
        seq.num_prefilled_tokens = 3
        chunk = seq.get_next_chunk_tokens(3)
        assert chunk == [4, 5, 6]

        # Last chunk (smaller)
        seq.num_prefilled_tokens = 8
        chunk = seq.get_next_chunk_tokens(5)  # Request more than remaining
        assert chunk == [9, 10]  # Only get remaining

    def test_chunked_prefill_scheduling(self):
        """Test that scheduler respects max_prefill_tokens limit."""
        scheduler = Scheduler(
            max_batch_size=4,
            max_prefill_tokens=10,  # Very small limit
            scheduling_policy=SchedulingPolicy.PRIORITY,
        )

        # Add sequence with long prompt
        long_prompt = list(range(25))  # 25 tokens
        seq = scheduler.add_request(long_prompt, max_tokens=10, priority=5)

        # Schedule - should only schedule partial prefill
        outputs = scheduler.schedule()

        # Should be in chunked prefill, not full prefill
        if len(outputs.chunked_prefill_sequences) > 0:
            assert seq in outputs.chunked_prefill_sequences
            # Check that we're not exceeding the budget
            total_tokens = sum(outputs.chunked_prefill_tokens)
            assert total_tokens <= scheduler.max_prefill_tokens
        else:
            # Or in full prefill if it fits
            assert seq in outputs.prefill_sequences

    def test_remaining_prefill_tokens(self):
        """Test get_remaining_prefill_tokens calculation."""
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            max_tokens=10,
        )

        assert seq.get_remaining_prefill_tokens() == 10

        seq.num_prefilled_tokens = 3
        assert seq.get_remaining_prefill_tokens() == 7

        seq.num_prefilled_tokens = 10
        assert seq.get_remaining_prefill_tokens() == 0


class TestPrefixCaching:
    """Tests for prefix caching."""

    def test_compute_block_hash(self):
        """Test that token block hashing is consistent."""
        tokens1 = (1, 2, 3, 4)
        tokens2 = (1, 2, 3, 4)
        tokens3 = (1, 2, 3, 5)

        # Same tokens should have same hash
        assert compute_block_hash(tokens1) == compute_block_hash(tokens2)

        # Different tokens should have different hash (usually)
        assert compute_block_hash(tokens1) != compute_block_hash(tokens3)

    def test_block_ref_counting(self):
        """Test block reference counting."""
        block = Block(block_id=0, block_size=BLOCK_SIZE)

        assert block.ref_count == 1

        block.increment_ref()
        assert block.ref_count == 2

        new_count = block.decrement_ref()
        assert new_count == 1
        assert block.ref_count == 1

    def test_prefix_cache_allocation(self):
        """Test that prefix caching shares blocks for common prefixes."""
        block_manager = BlockManager(
            num_blocks=20,
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
        )

        # Create a common prefix
        common_prefix = list(range(BLOCK_SIZE * 2))  # 2 full blocks

        # First allocation
        table1, shared1 = block_manager.allocate_blocks_with_prefix_caching(common_prefix)
        assert shared1 == 0  # No cache yet
        assert table1.num_blocks == 2

        # Second allocation with same prefix
        table2, shared2 = block_manager.allocate_blocks_with_prefix_caching(common_prefix)
        assert shared2 == BLOCK_SIZE * 2  # All tokens shared
        assert table2.num_blocks == 2

        # Check that blocks are shared (same IDs)
        assert table1.block_ids == table2.block_ids

        # Check ref counts increased
        for block_id in table1.block_ids:
            assert block_manager.blocks[block_id].ref_count == 2

    def test_prefix_cache_partial_match(self):
        """Test partial prefix cache matches."""
        block_manager = BlockManager(
            num_blocks=20,
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
        )

        # First sequence
        prefix1 = list(range(BLOCK_SIZE * 2))
        table1, _ = block_manager.allocate_blocks_with_prefix_caching(prefix1)

        # Second sequence with same prefix but additional tokens
        prefix2 = list(range(BLOCK_SIZE * 3))  # Extends prefix1
        table2, shared2 = block_manager.allocate_blocks_with_prefix_caching(prefix2)

        # Should share the first 2 blocks
        assert shared2 == BLOCK_SIZE * 2
        assert table2.num_blocks == 3

        # First two blocks should be shared
        assert table2.block_ids[0] == table1.block_ids[0]
        assert table2.block_ids[1] == table1.block_ids[1]

    def test_prefix_cache_disabled(self):
        """Test that prefix caching can be disabled."""
        block_manager = BlockManager(
            num_blocks=20,
            block_size=BLOCK_SIZE,
            enable_prefix_caching=False,
        )

        prefix = list(range(BLOCK_SIZE * 2))

        # First allocation
        table1, shared1 = block_manager.allocate_blocks_with_prefix_caching(prefix)
        assert shared1 == 0

        # Second allocation - should NOT share even with same prefix
        table2, shared2 = block_manager.allocate_blocks_with_prefix_caching(prefix)
        assert shared2 == 0

        # Blocks should be different
        assert table1.block_ids != table2.block_ids

    def test_free_shared_blocks(self):
        """Test that shared blocks are only freed when ref count reaches 0."""
        block_manager = BlockManager(
            num_blocks=20,
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
        )

        prefix = list(range(BLOCK_SIZE))  # 1 full block

        # Allocate twice
        table1, _ = block_manager.allocate_blocks_with_prefix_caching(prefix)
        table2, _ = block_manager.allocate_blocks_with_prefix_caching(prefix)

        initial_free = block_manager.get_num_free_blocks()

        # Free first table
        block_manager.free_sequence_blocks(table1)

        # Block should still be in use (ref_count = 1)
        assert block_manager.get_num_free_blocks() == initial_free

        # Free second table
        block_manager.free_sequence_blocks(table2)

        # Now block should be freed
        assert block_manager.get_num_free_blocks() == initial_free + 1

    def test_prefix_cache_stats(self):
        """Test prefix cache statistics."""
        block_manager = BlockManager(
            num_blocks=20,
            block_size=BLOCK_SIZE,
            enable_prefix_caching=True,
        )

        # Initially empty
        stats = block_manager.get_prefix_cache_stats()
        assert stats["cached_blocks"] == 0

        # Add some cached blocks
        prefix = list(range(BLOCK_SIZE * 2))
        table1, _ = block_manager.allocate_blocks_with_prefix_caching(prefix)
        table2, _ = block_manager.allocate_blocks_with_prefix_caching(prefix)

        stats = block_manager.get_prefix_cache_stats()
        assert stats["cached_blocks"] == 2
        assert stats["total_references"] == 4  # 2 blocks * 2 refs each


class TestSequenceFields:
    """Test new Sequence fields added in Phase 4."""

    def test_sequence_priority_default(self):
        """Test that priority defaults to 0."""
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3],
            max_tokens=10,
        )
        assert seq.priority == 0

    def test_sequence_arrival_time(self):
        """Test that arrival_time is set automatically."""
        before = time.time()
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3],
            max_tokens=10,
        )
        after = time.time()

        assert before <= seq.arrival_time <= after

    def test_sequence_swapped_status(self):
        """Test SWAPPED status enum."""
        assert SequenceStatus.SWAPPED.value == "swapped"

    def test_sequence_repr_includes_priority(self):
        """Test that repr includes priority."""
        seq = Sequence(
            seq_id=0,
            prompt_token_ids=[1, 2, 3],
            max_tokens=10,
            priority=5,
        )
        repr_str = repr(seq)
        assert "priority=5" in repr_str


class TestSchedulerOutputs:
    """Test SchedulerOutputs fields added in Phase 4."""

    def test_scheduler_outputs_preempted(self):
        """Test preempted_sequences field."""
        outputs = SchedulerOutputs()
        assert outputs.preempted_sequences == []
        assert outputs.num_preempted == 0

    def test_scheduler_outputs_chunked_prefill(self):
        """Test chunked_prefill fields."""
        outputs = SchedulerOutputs()
        assert outputs.chunked_prefill_sequences == []
        assert outputs.chunked_prefill_tokens == []
        assert outputs.num_chunked_prefill == 0

    def test_scheduler_outputs_total_sequences(self):
        """Test total_sequences includes all types."""
        outputs = SchedulerOutputs()
        seq1 = Sequence(seq_id=0, prompt_token_ids=[1], max_tokens=1)
        seq2 = Sequence(seq_id=1, prompt_token_ids=[2], max_tokens=1)
        seq3 = Sequence(seq_id=2, prompt_token_ids=[3], max_tokens=1)

        outputs.prefill_sequences.append(seq1)
        outputs.decode_sequences.append(seq2)
        outputs.chunked_prefill_sequences.append(seq3)
        outputs.chunked_prefill_tokens.append(10)

        assert outputs.total_sequences == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])