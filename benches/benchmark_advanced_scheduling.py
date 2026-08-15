"""Benchmark suite for Phase 4 Advanced Scheduling features.

Benchmarks:
1. Priority-based Scheduling - Verify higher priority requests complete first
2. Preemption - Measure overhead and correctness of preemption
3. Chunked Prefill - Compare throughput with different chunk sizes
4. Prefix Caching - Measure speedup from shared prefixes

Usage:
    python -m benches.benchmark_advanced_scheduling --all
    python -m benches.benchmark_advanced_scheduling --priority
    python -m benches.benchmark_advanced_scheduling --preemption
    python -m benches.benchmark_advanced_scheduling --chunked-prefill
    python -m benches.benchmark_advanced_scheduling --prefix-caching
"""

import argparse
import gc
import json
import random
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any

import torch


# ============================================================================
# Common utilities
# ============================================================================

def get_gpu_memory_mb() -> Optional[float]:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None


def reset_gpu_memory():
    """Reset GPU memory stats and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


@dataclass
class BenchmarkResult:
    """Generic benchmark result."""
    benchmark_name: str
    configuration: Dict[str, Any]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Test prompts
# ============================================================================

# Prompts with common prefixes for prefix caching tests
COMMON_PREFIX = (
    "You are a helpful AI assistant. Please answer the following question "
    "accurately and concisely. "
)

PREFIX_CACHING_PROMPTS = [
    COMMON_PREFIX + "What is the capital of France?",
    COMMON_PREFIX + "What is the capital of Germany?",
    COMMON_PREFIX + "What is the capital of Italy?",
    COMMON_PREFIX + "What is the capital of Spain?",
    COMMON_PREFIX + "What is the capital of Portugal?",
    COMMON_PREFIX + "What is the largest planet in our solar system?",
    COMMON_PREFIX + "What is the speed of light?",
    COMMON_PREFIX + "Who wrote Romeo and Juliet?",
]

# Prompts without common prefix (baseline)
NO_PREFIX_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis briefly.",
    "Name three programming languages.",
    "What color is the sky?",
    "How many continents are there?",
    "What is 2 + 2?",
    "Name a famous scientist.",
    "What is water made of?",
]

# Long prompts for chunked prefill testing
LONG_PROMPT_BASE = """
Please analyze the following detailed scenario and provide a comprehensive response.

Background Information:
The field of artificial intelligence has seen remarkable progress over the past decade.
Machine learning algorithms have become increasingly sophisticated, enabling applications
that were previously thought impossible. From natural language processing to computer
vision, AI systems are now capable of performing tasks that rival human performance.

Key Developments:
1. Deep learning has revolutionized pattern recognition
2. Transformer architectures have enabled breakthrough language models
3. Reinforcement learning has achieved superhuman performance in games
4. Generative models can create realistic images and text

Current Challenges:
- Ensuring AI systems are safe and aligned with human values
- Reducing the computational cost of training large models
- Making AI more interpretable and explainable
- Addressing bias and fairness in AI systems

Your Task:
Based on the above context, please provide your analysis on the following question:
"""

LONG_PROMPTS = [
    LONG_PROMPT_BASE + "What are the most promising applications of AI in healthcare?",
    LONG_PROMPT_BASE + "How might AI transform education in the next decade?",
    LONG_PROMPT_BASE + "What ethical considerations are most important for AI development?",
    LONG_PROMPT_BASE + "How can we ensure AI systems remain under human control?",
]


# ============================================================================
# Priority Scheduling Benchmark
# ============================================================================

def benchmark_priority_scheduling(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    num_runs: int = 3,
) -> List[BenchmarkResult]:
    """Benchmark priority-based scheduling.

    Tests that higher priority requests complete before lower priority ones,
    and measures any scheduling overhead.
    """
    from nano_vllm.engine import LLMEngine
    from nano_vllm.core.scheduler import SchedulingPolicy

    results = []

    print("\n" + "=" * 60)
    print("BENCHMARK: Priority Scheduling")
    print("=" * 60)

    # Test configurations
    configs = [
        {"scheduling_policy": SchedulingPolicy.FCFS, "name": "FCFS"},
        {"scheduling_policy": SchedulingPolicy.PRIORITY, "name": "Priority"},
    ]

    for config in configs:
        print(f"\nTesting {config['name']} scheduling...")

        reset_gpu_memory()

        engine = LLMEngine(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_batch_size=8,
            use_paged_attention=True,
            scheduling_policy=config["scheduling_policy"],
            enable_preemption=False,  # Disable for this test
            max_prefill_tokens=2048,
        )

        # Run multiple trials
        completion_orders = []
        total_times = []

        for run in range(num_runs):
            # Add requests with varying priorities
            # Lower priority requests first, then high priority
            prompts_with_priority = [
                ("Low priority task 1", 1),
                ("Low priority task 2", 1),
                ("Medium priority task", 5),
                ("High priority task", 10),
            ]

            seq_ids = []
            start_time = time.perf_counter()

            for prompt, priority in prompts_with_priority:
                seq_id = engine.add_request(prompt, max_tokens=20, priority=priority)
                seq_ids.append((seq_id, priority))

            # Run to completion and track order
            completion_order = []
            while engine.scheduler.has_pending_requests():
                completed = engine.step()
                for output in completed:
                    completion_order.append(output.seq_id)

            elapsed = time.perf_counter() - start_time
            total_times.append(elapsed)
            completion_orders.append(completion_order)

            if device == "cuda":
                torch.cuda.synchronize()

        # Analyze results
        avg_time = sum(total_times) / len(total_times)

        # Check if priority ordering is respected
        priority_respected_count = 0
        for order in completion_orders:
            # For priority scheduling, high priority (seq_id 3) should complete first
            # seq_ids: 0=low, 1=low, 2=medium, 3=high
            if config["name"] == "Priority":
                # High priority should be first
                if order[0] == 3:
                    priority_respected_count += 1

        results.append(BenchmarkResult(
            benchmark_name="priority_scheduling",
            configuration={
                "policy": config["name"],
                "num_requests": len(prompts_with_priority),
            },
            metrics={
                "avg_total_time_sec": avg_time,
                "priority_respected_pct": (priority_respected_count / num_runs) * 100 if config["name"] == "Priority" else None,
                "throughput_req_per_sec": len(prompts_with_priority) / avg_time,
            },
            metadata={
                "completion_orders": completion_orders,
            }
        ))

        print(f"  Avg time: {avg_time:.3f}s")
        print(f"  Throughput: {len(prompts_with_priority) / avg_time:.1f} req/s")
        if config["name"] == "Priority":
            print(f"  Priority respected: {priority_respected_count}/{num_runs} runs")

        # Cleanup
        del engine
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# ============================================================================
# Preemption Benchmark
# ============================================================================

def benchmark_preemption(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    num_runs: int = 3,
) -> List[BenchmarkResult]:
    """Benchmark preemption behavior.

    Tests preemption under memory pressure and measures recomputation overhead.
    """
    from nano_vllm.engine import LLMEngine
    from nano_vllm.core.scheduler import SchedulingPolicy
    from nano_vllm.core.block import BLOCK_SIZE

    results = []

    print("\n" + "=" * 60)
    print("BENCHMARK: Preemption")
    print("=" * 60)

    # Test with and without preemption
    configs = [
        {"enable_preemption": False, "name": "No Preemption"},
        {"enable_preemption": True, "name": "With Preemption"},
    ]

    for config in configs:
        print(f"\nTesting {config['name']}...")

        reset_gpu_memory()

        # Use limited blocks to force memory pressure
        num_blocks = 50  # Limited blocks to trigger preemption

        engine = LLMEngine(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_batch_size=4,
            use_paged_attention=True,
            num_blocks=num_blocks,
            scheduling_policy=SchedulingPolicy.PRIORITY,
            enable_preemption=config["enable_preemption"],
            max_prefill_tokens=2048,
        )

        total_times = []
        preemption_counts = []

        for run in range(num_runs):
            # First, fill up with low priority requests
            low_priority_prompts = [
                "Tell me a long story about a brave knight who went on an adventure.",
                "Explain the history of computing from the beginning to modern day.",
            ]

            for prompt in low_priority_prompts:
                engine.add_request(prompt, max_tokens=50, priority=1)

            # Run a few steps to start processing
            for _ in range(3):
                engine.step()

            # Now add high priority request
            engine.add_request(
                "Quick answer: What is 2+2?",
                max_tokens=10,
                priority=10,
            )

            start_time = time.perf_counter()

            # Track preemptions
            preemptions = 0
            while engine.scheduler.has_pending_requests():
                outputs = engine.scheduler.schedule()
                preemptions += len(outputs.preempted_sequences)
                engine.step()

            elapsed = time.perf_counter() - start_time
            total_times.append(elapsed)
            preemption_counts.append(preemptions)

            if device == "cuda":
                torch.cuda.synchronize()

        avg_time = sum(total_times) / len(total_times)
        avg_preemptions = sum(preemption_counts) / len(preemption_counts)

        results.append(BenchmarkResult(
            benchmark_name="preemption",
            configuration={
                "preemption_enabled": config["enable_preemption"],
                "num_blocks": num_blocks,
            },
            metrics={
                "avg_total_time_sec": avg_time,
                "avg_preemptions": avg_preemptions,
                "throughput_req_per_sec": 3 / avg_time,  # 3 requests
            },
        ))

        print(f"  Avg time: {avg_time:.3f}s")
        print(f"  Avg preemptions: {avg_preemptions:.1f}")

        del engine
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# ============================================================================
# Chunked Prefill Benchmark
# ============================================================================

def benchmark_chunked_prefill(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    num_runs: int = 3,
) -> List[BenchmarkResult]:
    """Benchmark chunked prefill with different chunk sizes.

    Tests how different max_prefill_tokens values affect throughput and latency
    for long prompts.
    """
    from nano_vllm.engine import LLMEngine
    from nano_vllm.core.scheduler import SchedulingPolicy

    results = []

    print("\n" + "=" * 60)
    print("BENCHMARK: Chunked Prefill")
    print("=" * 60)

    # Test different chunk sizes
    chunk_sizes = [128, 256, 512, 1024, 2048]

    for chunk_size in chunk_sizes:
        print(f"\nTesting chunk_size={chunk_size}...")

        reset_gpu_memory()

        engine = LLMEngine(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_batch_size=4,
            use_paged_attention=True,
            scheduling_policy=SchedulingPolicy.PRIORITY,
            enable_preemption=False,
            max_prefill_tokens=chunk_size,
        )

        total_times = []
        tokens_generated = []

        for run in range(num_runs):
            # Use long prompts
            prompts = LONG_PROMPTS[:2]

            start_time = time.perf_counter()

            for prompt in prompts:
                engine.add_request(prompt, max_tokens=30)

            outputs = engine.run_to_completion()

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            total_times.append(elapsed)

            # Count generated tokens
            total_gen = sum(output.generated_tokens for output in outputs)
            tokens_generated.append(total_gen)

        avg_time = sum(total_times) / len(total_times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        peak_memory = get_gpu_memory_mb()

        results.append(BenchmarkResult(
            benchmark_name="chunked_prefill",
            configuration={
                "max_prefill_tokens": chunk_size,
                "num_prompts": len(LONG_PROMPTS[:2]),
            },
            metrics={
                "avg_total_time_sec": avg_time,
                "avg_tokens_generated": avg_tokens,
                "throughput_tokens_per_sec": avg_tokens / avg_time,
                "peak_memory_mb": peak_memory,
            },
        ))

        print(f"  Avg time: {avg_time:.3f}s")
        print(f"  Throughput: {avg_tokens / avg_time:.1f} tokens/s")
        if peak_memory:
            print(f"  Peak memory: {peak_memory:.1f} MB")

        del engine
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# ============================================================================
# Prefix Caching Benchmark
# ============================================================================

def benchmark_prefix_caching(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    num_runs: int = 3,
) -> List[BenchmarkResult]:
    """Benchmark prefix caching effectiveness.

    Compares performance with and without prefix caching for prompts
    that share a common prefix.
    """
    from nano_vllm.engine import LLMEngine
    from nano_vllm.core.scheduler import SchedulingPolicy

    results = []

    print("\n" + "=" * 60)
    print("BENCHMARK: Prefix Caching")
    print("=" * 60)

    # Test configurations
    test_cases = [
        {
            "name": "No prefix caching, unique prompts",
            "enable_prefix_caching": False,
            "prompts": NO_PREFIX_PROMPTS,
        },
        {
            "name": "No prefix caching, common prefix prompts",
            "enable_prefix_caching": False,
            "prompts": PREFIX_CACHING_PROMPTS,
        },
        {
            "name": "With prefix caching, unique prompts",
            "enable_prefix_caching": True,
            "prompts": NO_PREFIX_PROMPTS,
        },
        {
            "name": "With prefix caching, common prefix prompts",
            "enable_prefix_caching": True,
            "prompts": PREFIX_CACHING_PROMPTS,
        },
    ]

    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}...")

        reset_gpu_memory()

        engine = LLMEngine(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_batch_size=8,
            use_paged_attention=True,
            scheduling_policy=SchedulingPolicy.PRIORITY,
            enable_preemption=False,
            max_prefill_tokens=2048,
            enable_prefix_caching=test_case["enable_prefix_caching"],
        )

        total_times = []
        tokens_generated = []
        prefix_cache_hits = []

        for run in range(num_runs):
            prompts = test_case["prompts"]

            start_time = time.perf_counter()

            for prompt in prompts:
                engine.add_request(prompt, max_tokens=20)

            outputs = engine.run_to_completion()

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            total_times.append(elapsed)

            total_gen = sum(output.generated_tokens for output in outputs)
            tokens_generated.append(total_gen)

            # Get prefix cache stats
            if test_case["enable_prefix_caching"]:
                stats = engine.block_manager.get_prefix_cache_stats()
                prefix_cache_hits.append(stats["cached_blocks"])

        avg_time = sum(total_times) / len(total_times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        peak_memory = get_gpu_memory_mb()

        metrics = {
            "avg_total_time_sec": avg_time,
            "avg_tokens_generated": avg_tokens,
            "throughput_tokens_per_sec": avg_tokens / avg_time,
            "peak_memory_mb": peak_memory,
        }

        if prefix_cache_hits:
            metrics["avg_cached_blocks"] = sum(prefix_cache_hits) / len(prefix_cache_hits)

        results.append(BenchmarkResult(
            benchmark_name="prefix_caching",
            configuration={
                "prefix_caching_enabled": test_case["enable_prefix_caching"],
                "prompt_type": "common_prefix" if "common prefix" in test_case["name"] else "unique",
                "num_prompts": len(test_case["prompts"]),
            },
            metrics=metrics,
        ))

        print(f"  Avg time: {avg_time:.3f}s")
        print(f"  Throughput: {avg_tokens / avg_time:.1f} tokens/s")
        if prefix_cache_hits:
            print(f"  Avg cached blocks: {sum(prefix_cache_hits) / len(prefix_cache_hits):.1f}")

        del engine
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Calculate speedup from prefix caching
    print("\n--- Prefix Caching Analysis ---")
    no_cache_common = next(
        (r for r in results
         if not r.configuration["prefix_caching_enabled"]
         and r.configuration["prompt_type"] == "common_prefix"),
        None
    )
    with_cache_common = next(
        (r for r in results
         if r.configuration["prefix_caching_enabled"]
         and r.configuration["prompt_type"] == "common_prefix"),
        None
    )

    if no_cache_common and with_cache_common:
        speedup = (
            with_cache_common.metrics["throughput_tokens_per_sec"] /
            no_cache_common.metrics["throughput_tokens_per_sec"]
        )
        print(f"Speedup with prefix caching (common prefix): {speedup:.2f}x")

    return results


# ============================================================================
# Combined Throughput Benchmark
# ============================================================================

def benchmark_throughput_comparison(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    num_runs: int = 3,
) -> List[BenchmarkResult]:
    """Compare overall throughput with different feature combinations."""
    from nano_vllm.engine import LLMEngine
    from nano_vllm.core.scheduler import SchedulingPolicy

    results = []

    print("\n" + "=" * 60)
    print("BENCHMARK: Throughput Comparison")
    print("=" * 60)

    configs = [
        {
            "name": "Baseline (FCFS, no advanced features)",
            "scheduling_policy": SchedulingPolicy.FCFS,
            "enable_preemption": False,
            "max_prefill_tokens": 2048,
            "enable_prefix_caching": False,
        },
        {
            "name": "Priority scheduling only",
            "scheduling_policy": SchedulingPolicy.PRIORITY,
            "enable_preemption": False,
            "max_prefill_tokens": 2048,
            "enable_prefix_caching": False,
        },
        {
            "name": "Chunked prefill (512 tokens)",
            "scheduling_policy": SchedulingPolicy.PRIORITY,
            "enable_preemption": False,
            "max_prefill_tokens": 512,
            "enable_prefix_caching": False,
        },
        {
            "name": "Prefix caching only",
            "scheduling_policy": SchedulingPolicy.PRIORITY,
            "enable_preemption": False,
            "max_prefill_tokens": 2048,
            "enable_prefix_caching": True,
        },
        {
            "name": "All features enabled",
            "scheduling_policy": SchedulingPolicy.PRIORITY,
            "enable_preemption": True,
            "max_prefill_tokens": 512,
            "enable_prefix_caching": True,
        },
    ]

    # Use a mix of prompts
    test_prompts = PREFIX_CACHING_PROMPTS[:4] + NO_PREFIX_PROMPTS[:4]

    for config in configs:
        print(f"\nTesting: {config['name']}...")

        reset_gpu_memory()

        engine = LLMEngine(
            model_path=model_path,
            device=device,
            dtype=dtype,
            max_batch_size=8,
            use_paged_attention=True,
            scheduling_policy=config["scheduling_policy"],
            enable_preemption=config["enable_preemption"],
            max_prefill_tokens=config["max_prefill_tokens"],
            enable_prefix_caching=config["enable_prefix_caching"],
        )

        total_times = []
        tokens_generated = []

        for run in range(num_runs):
            start_time = time.perf_counter()

            # Add all prompts with random priorities
            for i, prompt in enumerate(test_prompts):
                priority = random.randint(1, 10)
                engine.add_request(prompt, max_tokens=30, priority=priority)

            outputs = engine.run_to_completion()

            if device == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            total_times.append(elapsed)

            total_gen = sum(output.generated_tokens for output in outputs)
            tokens_generated.append(total_gen)

        avg_time = sum(total_times) / len(total_times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        peak_memory = get_gpu_memory_mb()

        results.append(BenchmarkResult(
            benchmark_name="throughput_comparison",
            configuration={
                "name": config["name"],
                "scheduling_policy": config["scheduling_policy"].value,
                "enable_preemption": config["enable_preemption"],
                "max_prefill_tokens": config["max_prefill_tokens"],
                "enable_prefix_caching": config["enable_prefix_caching"],
            },
            metrics={
                "avg_total_time_sec": avg_time,
                "avg_tokens_generated": avg_tokens,
                "throughput_tokens_per_sec": avg_tokens / avg_time,
                "throughput_req_per_sec": len(test_prompts) / avg_time,
                "peak_memory_mb": peak_memory,
            },
        ))

        print(f"  Avg time: {avg_time:.3f}s")
        print(f"  Throughput: {avg_tokens / avg_time:.1f} tokens/s")
        print(f"  Requests/s: {len(test_prompts) / avg_time:.1f}")

        del engine
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Print comparison
    print("\n--- Throughput Comparison Summary ---")
    baseline = results[0].metrics["throughput_tokens_per_sec"]
    for r in results:
        speedup = r.metrics["throughput_tokens_per_sec"] / baseline
        print(f"{r.configuration['name']}: {r.metrics['throughput_tokens_per_sec']:.1f} tok/s ({speedup:.2f}x)")

    return results


# ============================================================================
# Results handling
# ============================================================================

def print_results_summary(results: List[BenchmarkResult]):
    """Print a summary of all benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Group by benchmark name
    by_benchmark = {}
    for r in results:
        if r.benchmark_name not in by_benchmark:
            by_benchmark[r.benchmark_name] = []
        by_benchmark[r.benchmark_name].append(r)

    for benchmark_name, benchmark_results in by_benchmark.items():
        print(f"\n{benchmark_name.upper().replace('_', ' ')}")
        print("-" * 40)

        for r in benchmark_results:
            config_str = ", ".join(f"{k}={v}" for k, v in r.configuration.items() if k != "name")
            print(f"\n  Config: {r.configuration.get('name', config_str)}")
            for metric, value in r.metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"    {metric}: {value:.3f}")
                    else:
                        print(f"    {metric}: {value}")


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file."""
    data = []
    for r in results:
        entry = asdict(r)
        # Convert any enum values to strings
        if "scheduling_policy" in entry.get("configuration", {}):
            if hasattr(entry["configuration"]["scheduling_policy"], "value"):
                entry["configuration"]["scheduling_policy"] = entry["configuration"]["scheduling_policy"].value
        data.append(entry)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Phase 4 Advanced Scheduling features"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Number of runs per benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_advanced_scheduling_results.json",
        help="Output JSON file",
    )

    # Benchmark selection
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--priority", action="store_true", help="Run priority scheduling benchmark")
    parser.add_argument("--preemption", action="store_true", help="Run preemption benchmark")
    parser.add_argument("--chunked-prefill", action="store_true", help="Run chunked prefill benchmark")
    parser.add_argument("--prefix-caching", action="store_true", help="Run prefix caching benchmark")
    parser.add_argument("--throughput", action="store_true", help="Run throughput comparison benchmark")

    args = parser.parse_args()

    # If no specific benchmark selected, run all
    if not any([args.all, args.priority, args.preemption,
                args.chunked_prefill, args.prefix_caching, args.throughput]):
        args.all = True

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("ADVANCED SCHEDULING BENCHMARK SUITE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Runs per benchmark: {args.num_runs}")

    all_results = []

    # Run selected benchmarks
    if args.all or args.priority:
        results = benchmark_priority_scheduling(
            args.model, args.device, dtype, args.num_runs
        )
        all_results.extend(results)

    if args.all or args.preemption:
        results = benchmark_preemption(
            args.model, args.device, dtype, args.num_runs
        )
        all_results.extend(results)

    if args.all or args.chunked_prefill:
        results = benchmark_chunked_prefill(
            args.model, args.device, dtype, args.num_runs
        )
        all_results.extend(results)

    if args.all or args.prefix_caching:
        results = benchmark_prefix_caching(
            args.model, args.device, dtype, args.num_runs
        )
        all_results.extend(results)

    if args.all or args.throughput:
        results = benchmark_throughput_comparison(
            args.model, args.device, dtype, args.num_runs
        )
        all_results.extend(results)

    # Print and save results
    print_results_summary(all_results)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
