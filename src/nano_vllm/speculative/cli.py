"""Command-line interface for speculative decoding."""

import argparse
import time

import torch

from nano_vllm.speculative.speculative_decoding import SpeculativeEngine


def main():
    parser = argparse.ArgumentParser(
        description="nano-vllm: Speculative Decoding for faster inference"
    )
    parser.add_argument(
        "--target-model",
        type=str,
        required=True,
        help="Target (main) model path or HuggingFace ID",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        required=True,
        help="Draft (small) model path or HuggingFace ID",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=5,
        help="Number of tokens to speculate per step (K)",
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
        "--no-flash-attn",
        action="store_true",
        help="Disable FlashAttention",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run baseline (target-only) generation for comparison",
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("Speculative Decoding")
    print("=" * 60)
    print(f"Target model: {args.target_model}")
    print(f"Draft model: {args.draft_model}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Speculative tokens (K): {args.num_speculative_tokens}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    # Initialize engine
    print("Initializing speculative engine...")
    engine = SpeculativeEngine(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        device=args.device,
        dtype=dtype,
        num_speculative_tokens=args.num_speculative_tokens,
        use_flash_attn=not args.no_flash_attn,
    )

    print()
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    # Generate with speculative decoding
    print("\nGenerating with speculative decoding...")
    start_time = time.perf_counter()
    output_text, stats = engine.generate(args.prompt, max_new_tokens=args.max_tokens)
    spec_time = time.perf_counter() - start_time

    if args.device == "cuda":
        torch.cuda.synchronize()

    print(f"\nGenerated text:\n{output_text}")
    print()
    print("-" * 60)
    print("Speculative Decoding Stats:")
    print(f"  Time: {spec_time:.2f}s")
    print(f"  Tokens generated: {stats['tokens_generated']}")
    print(f"  Throughput: {stats['tokens_generated'] / spec_time:.1f} tokens/sec")
    print(f"  Draft tokens proposed: {stats['total_draft_tokens']}")
    print(f"  Tokens accepted: {stats['total_accepted_tokens']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.1%}")

    # Optionally compare with baseline
    if args.compare_baseline:
        print()
        print("=" * 60)
        print("Baseline (target-only) comparison")
        print("=" * 60)

        from nano_vllm.engine import LLMEngine

        baseline_engine = LLMEngine(
            model_path=args.target_model,
            device=args.device,
            dtype=dtype,
            use_paged_attention=False,  # Use simpler mode for comparison
            use_flash_attn=not args.no_flash_attn,
        )

        print("\nGenerating with baseline...")
        start_time = time.perf_counter()
        baseline_output = baseline_engine.generate(args.prompt, max_tokens=args.max_tokens)
        baseline_time = time.perf_counter() - start_time

        if args.device == "cuda":
            torch.cuda.synchronize()

        print(f"\nBaseline text:\n{baseline_output}")
        print()
        print("-" * 60)
        print("Baseline Stats:")
        baseline_tokens = len(baseline_engine.tokenizer.encode(baseline_output)) - len(
            baseline_engine.tokenizer.encode(args.prompt)
        )
        print(f"  Time: {baseline_time:.2f}s")
        print(f"  Tokens generated: ~{baseline_tokens}")
        print(f"  Throughput: {baseline_tokens / baseline_time:.1f} tokens/sec")

        print()
        print("=" * 60)
        print("Speedup Summary")
        print("=" * 60)
        speedup = baseline_time / spec_time if spec_time > 0 else 0
        print(f"  Speculative: {spec_time:.2f}s")
        print(f"  Baseline: {baseline_time:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")

    print()


if __name__ == "__main__":
    main()