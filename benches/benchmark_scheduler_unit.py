"""Unit benchmarks for scheduler components (no GPU required).

These benchmarks test the scheduling logic performance without loading
a full model, making them useful for quick performance testing during
development.

Usage:
    python -m benches.benchmark_scheduler_unit
"""

import time
import random
from dataclasses import dataclass
from typing import List, Dict, Any

from nano_vllm.core.sequence import Sequence, SequenceStatus
from nano_vllm.core.scheduler import Scheduler, SchedulerOutputs, SchedulingPolicy
from nano_vllm.core.block import Block, BlockTable, BLOCK_SIZE, compute_block_hash
from nano_vllm.core.block_manager import BlockManager


@dataclass
class MicroBenchmarkResult:
    """Result from a micro-benchmark."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_us: float  # microseconds
    ops_per_sec: float
    details: Dict[str, Any] = None


def benchmark_priority_queue_operations(num_iterations: int = 10000) -> MicroBenchmarkResult:
    """Benchmark priority queue operations (add/remove)."""
    scheduler = Scheduler(
        max_batch_size=100,
        scheduling_policy=SchedulingPolicy.PRIORITY,
        enable_preemption=False,
        max_prefill_tokens=10000,
    )

    # Prepare test data
    prompts = [[random.randint(0, 50000) for _ in range(50)] for _ in range(100)]

    start = time.perf_counter()

    for i in range(num_iterations):
        # Add request
        priority = random.randint(1, 100)
        prompt = prompts[i % len(prompts)]
        scheduler.add_request(prompt, max_tokens=50, priority=priority)

        # Occasionally remove requests to keep queue bounded
        if i % 10 == 0 and scheduler.get_num_waiting() > 50:
            scheduler._pop_waiting()

    elapsed_ms = (time.perf_counter() - start) * 1000

    return MicroBenchmarkResult(
        name="priority_queue_operations",
        iterations=num_iterations,
        total_time_ms=elapsed_ms,
        avg_time_us=(elapsed_ms * 1000) / num_iterations,
        ops_per_sec=num_iterations / (elapsed_ms / 1000),
    )


def benchmark_scheduling_decision(num_iterations: int = 1000) -> MicroBenchmarkResult:
    """Benchmark the schedule() decision making."""
    block_manager = BlockManager(num_blocks=500, block_size=BLOCK_SIZE)

    scheduler = Scheduler(
        max_batch_size=32,
        block_manager=block_manager,
        block_size=BLOCK_SIZE,
        scheduling_policy=SchedulingPolicy.PRIORITY,
        enable_preemption=True,
        max_prefill_tokens=512,
    )

    # Pre-populate with some requests
    for i in range(20):
        prompt = list(range(100))  # 100 token prompt
        scheduler.add_request(prompt, max_tokens=50, priority=random.randint(1, 10))

    start = time.perf_counter()

    for _ in range(num_iterations):
        outputs = scheduler.schedule()
        # Reset for next iteration
        for seq in outputs.prefill_sequences:
            seq.status = SequenceStatus.WAITING
            scheduler._push_waiting(seq)
            scheduler.running.remove(seq)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return MicroBenchmarkResult(
        name="scheduling_decision",
        iterations=num_iterations,
        total_time_ms=elapsed_ms,
        avg_time_us=(elapsed_ms * 1000) / num_iterations,
        ops_per_sec=num_iterations / (elapsed_ms / 1000),
    )


def benchmark_block_allocation(num_iterations: int = 10000) -> MicroBenchmarkResult:
    """Benchmark block allocation/deallocation."""
    block_manager = BlockManager(num_blocks=1000, block_size=BLOCK_SIZE)

    start = time.perf_counter()

    allocated_tables = []
    for i in range(num_iterations):
        if i % 2 == 0:
            # Allocate
            if block_manager.can_allocate(5):
                table = block_manager.allocate_blocks_for_sequence(5)
                allocated_tables.append(table)
        else:
            # Free
            if allocated_tables:
                table = allocated_tables.pop(0)
                block_manager.free_sequence_blocks(table)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return MicroBenchmarkResult(
        name="block_allocation",
        iterations=num_iterations,
        total_time_ms=elapsed_ms,
        avg_time_us=(elapsed_ms * 1000) / num_iterations,
        ops_per_sec=num_iterations / (elapsed_ms / 1000),
    )


def benchmark_prefix_cache_lookup(num_iterations: int = 10000) -> MicroBenchmarkResult:
    """Benchmark prefix cache lookups."""
    block_manager = BlockManager(
        num_blocks=500,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=True,
    )

    # Pre-populate cache with some prefixes
    common_prefix = list(range(BLOCK_SIZE * 3))  # 3 blocks
    block_manager.allocate_blocks_with_prefix_caching(common_prefix)

    # Test prompts with varying prefix matches
    test_prompts = [
        common_prefix[:],  # Full match
        common_prefix + list(range(1000, 1000 + BLOCK_SIZE)),  # Partial match
        list(range(5000, 5000 + BLOCK_SIZE * 2)),  # No match
    ]

    start = time.perf_counter()

    for i in range(num_iterations):
        prompt = test_prompts[i % len(test_prompts)]
        table, shared_len = block_manager.allocate_blocks_with_prefix_caching(prompt)
        # Free immediately to not exhaust blocks
        block_manager.free_sequence_blocks(table)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return MicroBenchmarkResult(
        name="prefix_cache_lookup",
        iterations=num_iterations,
        total_time_ms=elapsed_ms,
        avg_time_us=(elapsed_ms * 1000) / num_iterations,
        ops_per_sec=num_iterations / (elapsed_ms / 1000),
    )


def benchmark_preemption_logic(num_iterations: int = 1000) -> MicroBenchmarkResult:
    """Benchmark preemption decision making."""
    block_manager = BlockManager(num_blocks=100, block_size=BLOCK_SIZE)

    scheduler = Scheduler(
        max_batch_size=10,
        block_manager=block_manager,
        block_size=BLOCK_SIZE,
        scheduling_policy=SchedulingPolicy.PRIORITY,
        enable_preemption=True,
        max_prefill_tokens=512,
    )

    start = time.perf_counter()

    for i in range(num_iterations):
        # Setup: add low priority running sequences
        for j in range(3):
            prompt = list(range(BLOCK_SIZE * 2))
            seq = scheduler.add_request(prompt, max_tokens=50, priority=1)
            scheduler.schedule()
            # Simulate allocation
            if scheduler.running:
                for s in scheduler.running:
                    if s.block_table is None:
                        try:
                            s.block_table = block_manager.allocate_blocks_for_sequence(2)
                        except RuntimeError:
                            pass

        # Add high priority waiting
        scheduler.add_request([1, 2, 3], max_tokens=10, priority=100)

        # Run schedule which triggers preemption check
        outputs = scheduler.schedule()

        # Reset for next iteration
        for seq in scheduler.running + scheduler.finished:
            if seq.block_table:
                block_manager.free_sequence_blocks(seq.block_table)
        scheduler.running.clear()
        scheduler.finished.clear()
        scheduler._waiting_heap.clear()

    elapsed_ms = (time.perf_counter() - start) * 1000

    return MicroBenchmarkResult(
        name="preemption_logic",
        iterations=num_iterations,
        total_time_ms=elapsed_ms,
        avg_time_us=(elapsed_ms * 1000) / num_iterations,
        ops_per_sec=num_iterations / (elapsed_ms / 1000),
    )


def benchmark_sequence_operations(num_iterations: int = 100000) -> MicroBenchmarkResult:
    """Benchmark sequence creation and field access."""
    sequences = []

    start = time.perf_counter()

    for i in range(num_iterations):
        seq = Sequence(
            seq_id=i,
            prompt_token_ids=list(range(100)),
            max_tokens=50,
            priority=random.randint(1, 100),
        )
        sequences.append(seq)

        # Access fields
        _ = seq.get_len()
        _ = seq.is_prefill()
        _ = seq.get_remaining_prefill_tokens()

        if i % 1000 == 0:
            sequences.clear()  # Prevent memory growth

    elapsed_ms = (time.perf_counter() - start) * 1000

    return MicroBenchmarkResult(
        name="sequence_operations",
        iterations=num_iterations,
        total_time_ms=elapsed_ms,
        avg_time_us=(elapsed_ms * 1000) / num_iterations,
        ops_per_sec=num_iterations / (elapsed_ms / 1000),
    )


def benchmark_hash_computation(num_iterations: int = 100000) -> MicroBenchmarkResult:
    """Benchmark token block hashing."""
    # Pre-generate test data
    test_blocks = [
        tuple(random.randint(0, 50000) for _ in range(BLOCK_SIZE))
        for _ in range(100)
    ]

    start = time.perf_counter()

    for i in range(num_iterations):
        block = test_blocks[i % len(test_blocks)]
        _ = compute_block_hash(block)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return MicroBenchmarkResult(
        name="hash_computation",
        iterations=num_iterations,
        total_time_ms=elapsed_ms,
        avg_time_us=(elapsed_ms * 1000) / num_iterations,
        ops_per_sec=num_iterations / (elapsed_ms / 1000),
    )


def run_all_benchmarks() -> List[MicroBenchmarkResult]:
    """Run all micro-benchmarks."""
    benchmarks = [
        ("Priority Queue Operations", benchmark_priority_queue_operations),
        ("Scheduling Decision", benchmark_scheduling_decision),
        ("Block Allocation", benchmark_block_allocation),
        ("Prefix Cache Lookup", benchmark_prefix_cache_lookup),
        ("Preemption Logic", benchmark_preemption_logic),
        ("Sequence Operations", benchmark_sequence_operations),
        ("Hash Computation", benchmark_hash_computation),
    ]

    results = []

    print("=" * 70)
    print("SCHEDULER UNIT BENCHMARKS")
    print("=" * 70)
    print()

    for name, benchmark_fn in benchmarks:
        print(f"Running: {name}...", end=" ", flush=True)
        try:
            result = benchmark_fn()
            results.append(result)
            print(f"Done - {result.ops_per_sec:,.0f} ops/sec ({result.avg_time_us:.2f} us/op)")
        except Exception as e:
            print(f"FAILED: {e}")

    return results


def print_results_table(results: List[MicroBenchmarkResult]):
    """Print results in a formatted table."""
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()
    print(f"{'Benchmark':<30} {'Ops/sec':>15} {'Avg (us)':>12} {'Total (ms)':>12}")
    print("-" * 70)

    for r in results:
        print(f"{r.name:<30} {r.ops_per_sec:>15,.0f} {r.avg_time_us:>12.2f} {r.total_time_ms:>12.2f}")

    print("-" * 70)
    print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Scheduler unit benchmarks")
    parser.add_argument(
        "--iterations-multiplier",
        type=float,
        default=1.0,
        help="Multiply iteration counts (0.1 for quick, 10 for thorough)",
    )

    args = parser.parse_args()

    # Adjust iteration counts if requested
    if args.iterations_multiplier != 1.0:
        print(f"Using iteration multiplier: {args.iterations_multiplier}")

    results = run_all_benchmarks()
    print_results_table(results)

    # Summary
    print("Summary:")
    print("-" * 40)
    total_ops = sum(r.iterations for r in results)
    total_time = sum(r.total_time_ms for r in results)
    print(f"Total operations: {total_ops:,}")
    print(f"Total time: {total_time:.2f} ms")
    print()


if __name__ == "__main__":
    main()
