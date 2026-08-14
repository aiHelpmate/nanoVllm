"""Command-line interface for nano-vllm with continuous batching and PagedAttention support."""

import argparse
import time

import torch

from nano_vllm.engine import LLMEngine
from nano_vllm.core.block import BLOCK_SIZE
from nano_vllm.core.scheduler import SchedulingPolicy


def main():
    parser = argparse.ArgumentParser(description="nano-vllm: Minimalistic LLM inference with continuous batching")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        help="Input prompt(s) for generation. Can be specified multiple times.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate per prompt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Data type for model weights",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for continuous batching",
    )
    # PagedAttention options
    parser.add_argument(
        "--no-paged-attention",
        action="store_true",
        help="Disable PagedAttention (use legacy per-sequence KV caches)",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Number of blocks for PagedAttention (auto-calculated if not specified)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help=f"Tokens per block for PagedAttention (default: {BLOCK_SIZE})",
    )
    parser.add_argument(
        "--show-memory-stats",
        action="store_true",
        help="Show memory statistics after generation",
    )

    # Advanced scheduling options (Phase 4)
    parser.add_argument(
        "--priority",
        type=int,
        action="append",
        help="Priority for each prompt (higher = more important). Can be specified multiple times.",
    )
    parser.add_argument(
        "--scheduling-policy",
        type=str,
        default="priority",
        choices=["fcfs", "priority"],
        help="Scheduling policy: fcfs (first-come-first-serve) or priority (default)",
    )
    parser.add_argument(
        "--no-preemption",
        action="store_true",
        help="Disable preemption of low-priority sequences",
    )
    parser.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=512,
        help="Maximum tokens to prefill per iteration (chunked prefill)",
    )
    parser.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="Disable prefix caching (sharing KV blocks for common prefixes)",
    )

    # Optimization options (Phase 5)
    parser.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable FlashAttention (use standard attention)",
    )

    args = parser.parse_args()

    # Default prompts if none provided
    if not args.prompt:
        args.prompt = ["The capital of France is"]

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Parse scheduling policy
    policy_map = {
        "fcfs": SchedulingPolicy.FCFS,
        "priority": SchedulingPolicy.PRIORITY,
    }
    scheduling_policy = policy_map[args.scheduling_policy]

    # Common engine kwargs
    engine_kwargs = dict(
        model_path=args.model,
        device=args.device,
        dtype=dtype,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        use_paged_attention=not args.no_paged_attention,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        scheduling_policy=scheduling_policy,
        enable_preemption=not args.no_preemption,
        max_prefill_tokens=args.max_prefill_tokens,
        enable_prefix_caching=not args.no_prefix_caching,
        use_flash_attn=not args.no_flash_attn,
    )

    # Initialize engine
    engine = LLMEngine(**engine_kwargs)
    base_engine = engine

    # Parse priorities (default to 0 if not specified)
    priorities = args.priority if args.priority else [0] * len(args.prompt)
    if len(priorities) < len(args.prompt):
        # Extend with default priority
        priorities.extend([0] * (len(args.prompt) - len(priorities)))

    # Print stats
    stats = engine.get_stats()
    print(f"\nEngine configuration:")
    print(f"  Model: {args.model}")
    print(f"  Device: {args.device}, dtype: {args.dtype}")
    print(f"  Max sequence length: {args.max_seq_len}")
    print(f"  Max batch size: {args.max_batch_size}")
    if stats.get("use_paged_attention"):
        print(f"  PagedAttention: enabled")
        print(f"    Block size: {stats['block_size']} tokens")
        print(f"    Total blocks: {stats['total_blocks']}")
        print(f"    KV cache memory: {stats['kv_cache_memory_mb']:.1f}MB")
    else:
        print(f"  PagedAttention: disabled (using per-sequence KV caches)")

    # Advanced scheduling config
    print(f"  Scheduling policy: {stats.get('scheduling_policy', 'fcfs')}")
    print(f"  Preemption: {'enabled' if stats.get('enable_preemption') else 'disabled'}")
    print(f"  Max prefill tokens: {stats.get('max_prefill_tokens', 512)}")
    print(f"  Prefix caching: {'enabled' if stats.get('enable_prefix_caching') else 'disabled'}")

    # Optimization config
    flash_attn_status = "active" if stats.get('flash_attn_available') else ("requested but unavailable" if stats.get('use_flash_attn') else "disabled")
    print(f"  FlashAttention: {flash_attn_status}")

    print(f"\nProcessing {len(args.prompt)} prompt(s):")
    for i, prompt in enumerate(args.prompt):
        priority_str = f" (priority={priorities[i]})" if priorities[i] != 0 else ""
        print(f"  [{i}] {prompt[:50]}{'...' if len(prompt) > 50 else ''}{priority_str}")
    print("-" * 60)

    # Generate
    start_time = time.perf_counter()

    if len(args.prompt) == 1:
        # Single prompt
        engine.add_request(args.prompt[0], max_tokens=args.max_tokens, priority=priorities[0])
        all_outputs = engine.run_to_completion()
        outputs = [all_outputs[0].generated_text] if all_outputs else [""]
    else:
        # Multiple prompts - add each with its priority
        for prompt, priority in zip(args.prompt, priorities):
            engine.add_request(prompt, max_tokens=args.max_tokens, priority=priority)
        all_outputs = engine.run_to_completion()
        outputs = [output.generated_text for output in all_outputs]

    elapsed = time.perf_counter() - start_time

    # Print results
    print("\nGenerated outputs:")
    print("-" * 60)
    for i, output in enumerate(outputs):
        print(f"\n[{i}] {output}")
        print("-" * 60)

    # Stats
    tokenizer = base_engine.tokenizer
    total_tokens = sum(len(tokenizer.encode(o)) for o in outputs)
    print(f"\nGeneration stats:")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Prompts processed: {len(args.prompt)}")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Throughput: {total_tokens / elapsed:.1f} tokens/sec")

    # Show memory stats if requested
    if args.show_memory_stats:
        final_stats = engine.get_stats()
        print(f"\nMemory stats:")
        if final_stats.get("use_paged_attention"):
            print(f"  PagedAttention blocks:")
            print(f"    Total: {final_stats['total_blocks']}")
            print(f"    Free: {final_stats['free_blocks']}")
            print(f"    Used: {final_stats['used_blocks']}")
            print(f"  KV cache memory: {final_stats['kv_cache_memory_mb']:.1f}MB")

            # Prefix cache stats
            if final_stats.get("enable_prefix_caching"):
                print(f"  Prefix cache:")
                print(f"    Cached blocks: {final_stats.get('prefix_cache_blocks', 0)}")
                print(f"    Total references: {final_stats.get('prefix_cache_refs', 0)}")
        else:
            print(f"  Using legacy per-sequence KV caches (no block stats)")


if __name__ == "__main__":
    main()