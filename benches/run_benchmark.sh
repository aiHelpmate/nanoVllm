#!/bin/bash
#
# Benchmark comparison: nano-vllm vs HuggingFace Transformers
#
# Usage:
#   ./run_benchmark.sh                    # Run with TinyLlama (default)
#   ./run_benchmark.sh --model <path>     # Run with specific model
#   ./run_benchmark.sh --quick            # Quick test with fewer configs
#
# Examples:
#   ./run_benchmark.sh --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16
#   ./run_benchmark.sh --model /path/to/local/model --quick
#

set -e

# Default values
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE="float16"
DEVICE="cuda"
QUICK_MODE=""
OUTPUT_DIR="benchmark_results"
SKIP_HF=""
SKIP_PAGED=""
SKIP_LEGACY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE="--quick"
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-hf)
            SKIP_HF="--skip-hf"
            shift
            ;;
        --skip-paged)
            SKIP_PAGED="--skip-paged"
            shift
            ;;
        --skip-legacy)
            SKIP_LEGACY="--skip-legacy"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH      Model path or HuggingFace ID (default: TinyLlama)"
            echo "  --dtype TYPE      Data type: float16, bfloat16, float32 (default: float16)"
            echo "  --device DEVICE   Device: cuda or cpu (default: cuda)"
            echo "  --quick           Quick mode with fewer configurations"
            echo "  --output-dir DIR  Output directory for results (default: benchmark_results)"
            echo "  --skip-hf         Skip HuggingFace benchmarks"
            echo "  --skip-paged      Skip nano-vllm PagedAttention benchmarks"
            echo "  --skip-legacy     Skip nano-vllm legacy benchmarks"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$OUTPUT_DIR/results_${TIMESTAMP}.json"
LOG_FILE="$OUTPUT_DIR/benchmark_${TIMESTAMP}.log"

echo "=============================================================="
echo "  nano-vllm Benchmark Suite"
echo "=============================================================="
echo ""
echo "Configuration:"
echo "  Model:      $MODEL"
echo "  Dtype:      $DTYPE"
echo "  Device:     $DEVICE"
echo "  Quick mode: $([ -n "$QUICK_MODE" ] && echo "Yes" || echo "No")"
echo "  Output:     $RESULTS_FILE"
echo "  Log:        $LOG_FILE"
echo ""
echo "=============================================================="
echo ""

# Check if CUDA is available (if device is cuda)
if [ "$DEVICE" = "cuda" ]; then
    echo "Checking CUDA availability..."
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
        echo "ERROR: CUDA is not available. Use --device cpu or check your installation."
        exit 1
    }
    echo "CUDA is available."
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    echo ""
fi

# Run the benchmark
echo "Starting benchmark..."
echo "This may take a while depending on configurations..."
echo ""

cd "$PROJECT_ROOT"

python -m benches.benchmark_comparison \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --output "$RESULTS_FILE" \
    $QUICK_MODE \
    $SKIP_HF \
    $SKIP_PAGED \
    $SKIP_LEGACY \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================================="
echo "  Benchmark Complete!"
echo "=============================================================="
echo ""
echo "Results saved to:"
echo "  JSON: $RESULTS_FILE"
echo "  Log:  $LOG_FILE"
echo ""

# Print a quick summary from the JSON
echo "Quick Summary (from JSON):"
echo "--------------------------"
python3 << EOF
import json
import sys

try:
    with open("$RESULTS_FILE", "r") as f:
        results = json.load(f)

    # Group by backend
    backends = {}
    for r in results:
        backend = r["backend"]
        if backend not in backends:
            backends[backend] = []
        backends[backend].append(r["throughput_tokens_per_sec"])

    print("")
    for backend, throughputs in sorted(backends.items()):
        avg = sum(throughputs) / len(throughputs)
        min_t = min(throughputs)
        max_t = max(throughputs)
        print(f"{backend}:")
        print(f"  Avg throughput: {avg:.1f} tokens/sec")
        print(f"  Range: {min_t:.1f} - {max_t:.1f} tokens/sec")
        print("")

    # Calculate speedups
    if "huggingface" in backends and "nano_vllm_paged" in backends:
        hf_avg = sum(backends["huggingface"]) / len(backends["huggingface"])
        paged_avg = sum(backends["nano_vllm_paged"]) / len(backends["nano_vllm_paged"])
        print(f"nano-vllm (paged) vs HuggingFace: {paged_avg/hf_avg:.2f}x speedup")

    if "huggingface" in backends and "nano_vllm_legacy" in backends:
        hf_avg = sum(backends["huggingface"]) / len(backends["huggingface"])
        legacy_avg = sum(backends["nano_vllm_legacy"]) / len(backends["nano_vllm_legacy"])
        print(f"nano-vllm (legacy) vs HuggingFace: {legacy_avg/hf_avg:.2f}x speedup")

except Exception as e:
    print(f"Could not parse results: {e}")
    sys.exit(1)
EOF

echo ""
echo "=============================================================="
