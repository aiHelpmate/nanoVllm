# nanoVllm

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-lightgrey)

A minimalistic, from-scratch LLM inference engine featuring continuous batching, PagedAttention, priority scheduling, and chunked prefill — built for learning how modern inference servers (vLLM-style) work under the hood.

## Overview

nanoVllm implements the complete inference stack for Llama-family models in pure PyTorch: model architecture, KV cache management, scheduling, and sampling are all written from first principles with no dependency on the HuggingFace modeling code (weights and tokenizers are loaded from HF, but the forward pass is entirely our own).

The engine is organized to mirror a production inference system:

- `core/` — paged KV cache (blocks, block tables) and the continuous-batching scheduler
- `attention/` — paged attention in pure PyTorch plus a unified backend router (flash-attn → PyTorch SDPA)
- `model/` — Llama implementation (RoPE, GQA, RMSNorm, SwiGLU) and weight loader
- `engine.py` — orchestration: prefill/decode phases, chunked prefill, prefix caching
- `cli.py` — command-line entry point

## Features

- **Continuous batching** — new requests join mid-generation, finished requests leave immediately; iteration-level scheduling re-decides the batch every step
- **PagedAttention** — block-based KV cache allocation/deallocation à la virtual memory, with a `BlockTable` page table and O(1) allocation via a free list
- **Prefix caching** — shared KV blocks across sequences with identical prefixes, guarded by cumulative chain hashing and reference counting
- **Priority scheduling & preemption** — request priorities with FCFS or PRIORITY policies; low-priority running requests are preempted (recompute-based) when high-priority ones need blocks
- **Chunked prefill** — long prompts are prefilled in bounded chunks to cap memory per iteration, with the scheduler tracking prefill progress per sequence
- **Two cache modes** — PagedAttention (default) plus a legacy per-sequence KV cache mode
- **FlashAttention with graceful fallback** — uses `flash-attn` when installed, otherwise automatically dispatches to PyTorch's optimized SDPA
- **Correctness verified against HuggingFace** — forward logits match `AutoModelForCausalLM` within 1e-3 (float32), and greedy generation is byte-identical

## Installation

```bash
git clone https://github.com/aiHelpmate/nanoVllm.git
cd nanoVllm

# Create a virtual environment and install dependencies (Python 3.10+)
python -m venv .venv
source .venv/bin/activate
pip install torch transformers safetensors huggingface_hub

# Optional: FlashAttention on CUDA GPUs (Ampere or newer)
pip install flash-attn --no-build-isolation
```

The project uses a `src/` layout and currently has no packaging configuration, so Python needs to find the package by pointing `PYTHONPATH` at `src/`:

```bash
export PYTHONPATH="$PWD/src"
```

(Linux/macOS: add that export to your shell profile, or configure `PYTHONPATH` in your IDE as `<repo>/src`.)

## Usage

Generate text from a prompt (downloads TinyLlama-1.1B-Chat on first run):

```bash
python -m nano_vllm.cli --prompt "The capital of France is"
```

Process several prompts in one request — they are scheduled together with continuous batching:

```bash
python -m nano_vllm.cli \
  --prompt "The capital of France is" \
  --prompt "What is 2 + 2? Answer:" \
  --prompt "Translate to Chinese: hello world"
```

Use the engine from Python instead:

```python
import torch
from nano_vllm.engine import LLMEngine

engine = LLMEngine("TinyLlama/TinyLlama-1.1B-Chat-v1.0", device="cuda", dtype=torch.float16)
outputs = engine.generate_batch(
    ["The capital of France is", "What is 2 + 2? Answer:"],
    max_tokens=32,
)
for text in outputs:
    print(text)
```

## Configuration

All options are exposed as CLI flags; the `--no-*` flags disable a feature (everything else defaults to on).

| Flag | Default | Description |
|---|---|---|
| `--model` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model ID or local path |
| `--prompt` | — | Input prompt(s); repeat the flag for batch generation |
| `--max-tokens` | `50` | Maximum tokens to generate per prompt |
| `--device` | `cuda` if available else `cpu` | Device to run inference on |
| `--dtype` | `float16` | `float16`, `float32`, or `bfloat16` (`float32` recommended on CPU) |
| `--max-seq-len` | `2048` | Maximum sequence length |
| `--max-batch-size` | `8` | Maximum sequences processed in one iteration |
| `--no-paged-attention` | off | Use legacy per-sequence KV caches instead of PagedAttention |
| `--num-blocks` | auto | Number of KV blocks for PagedAttention (auto-calculated if omitted) |
| `--block-size` | `16` | Tokens per KV block |
| `--scheduling-policy` | `priority` | `priority` (higher priority first) or `fcfs` |
| `--priority` | — | Per-prompt priority; repeat to match the order of `--prompt` flags |
| `--no-preemption` | off | Disable preemption of low-priority sequences |
| `--max-prefill-tokens` | `512` | Chunked-prefill budget per iteration |
| `--no-prefix-caching` | off | Disable shared KV blocks for common prefixes |
| `--no-flash-attn` | off | Disable the FlashAttention backend |
| `--show-memory-stats` | off | Print KV-block usage after generation |

Example:

```bash
python -m nano_vllm.cli \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cuda --dtype bfloat16 \
  --max-tokens 100 --max-batch-size 4 \
  --prompt "Tell me a short story about a robot." \
  --prompt "Explain quantum computing in one sentence." \
  --show-memory-stats
```

## Demo

A real generation run on CPU with TinyLlama-1.1B-Chat (float32, 32 tokens):

```bash
python -m nano_vllm.cli --device cpu --dtype float32 --max-tokens 32 \
  --prompt "The capital of France is"
```

```
Generated outputs:
------------------------------------------------------------

[0] The capital of France is Paris.

2. B. The capital of Germany is Berlin.

3. C. The capital of the United States is Washington, D.
------------------------------------------------------------

Generation stats:
  Total time: 2.0s
  Prompts processed: 1
  Total tokens generated: 32
  Throughput: 16.0 tokens/sec
```

## Tests & Benchmarks

```bash
# Correctness (incl. logits parity vs HuggingFace), optimizations, scheduling
PYTHONPATH=src python -m pytest tests/ -q

# Scheduler micro-benchmarks (no model required)
PYTHONPATH=src python -m benches.benchmark_scheduler_unit
```

## License

[MIT](LICENSE)