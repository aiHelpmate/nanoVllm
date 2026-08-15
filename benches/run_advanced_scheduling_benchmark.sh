#!/bin/bash
#
# Benchmark suite for Phase 4 Advanced Scheduling features
#
# Usage:
#   ./run_advanced_scheduling_benchmark.sh                    # Run all benchmarks
#   ./run_advanced_scheduling_benchmark.sh --priority         # Run specific benchmark
#   ./run_advanced_scheduling_benchmark.sh --quick            # Quick test mode
#
# Examples:
#   ./run_advanced_scheduling_benchmark.sh --model meta-llama/Llama-3.1-8B-Instruct
#   ./run_advanced_scheduling_benchmark.sh --prefix-caching --num-runs 5
#

set -e

# Default values
MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DTYPE="float16"
DEVICE="cuda"
NUM_RUNS=3
OUTPUT_DIR="benchmark_results"
BENCHMARK_ARGS=""

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
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            NUM_RUNS=1
            shift
            ;;
        --all)
            BENCHMARK_ARGS="$BENCHMARK_ARGS --all"
            shift
            ;;
        --priority)
            BENCHMARK_ARGS="$BENCHMARK_ARGS --priority"
            shift
            ;;
        --preemption)
            BENCHMARK_ARGS="$BENCHMARK_ARGS --preemption"
            shift
            ;;
        --chunked-prefill)
            BENCHMARK_ARGS="$BENCHMARK_ARGS --chunked-prefill"
            shift
            ;;
        --prefix-caching)
            BENCHMARK_ARGS="$BENCHMARK_ARGS --prefix-caching"
            shift
            ;;
        --throughput)
            BENCHMARK_ARGS="$BENCHMARK_ARGS --throughput"
            shift
            ;;
        --help)
            echo "Advanced Scheduling Benchmark Suite"
            echo ""
            echo "Usage: $0 [OPTIONS] [BENCHMARKS]"
            echo ""
            echo "Options:"
            echo "  --model PATH        Model path or HuggingFace ID (default: TinyLlama)"
            echo "  --dtype TYPE        Data type: float16, bfloat16, float32 (default: float16)"
            echo "  --device DEVICE     Device: cuda or cpu (default: cuda)"
            echo "  --num-runs N        Number of runs per benchmark (default: 3)"
            echo "  --output-dir DIR    Output directory for results (default: benchmark_results)"
            echo "  --quick             Quick mode: single run per benchmark"
            echo "  --help              Show this help message"
            echo ""
            echo "Benchmarks (if none specified, runs all):"
            echo "  --all               Run all benchmarks"
            echo "  --priority          Priority scheduling benchmark"
            echo "  --preemption        Preemption benchmark"
            echo "  --chunked-prefill   Chunked prefill benchmark"
            echo "  --prefix-caching    Prefix caching benchmark"
            echo "  --throughput        Throughput comparison benchmark"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run all benchmarks"
            echo "  $0 --priority --prefix-caching        # Run specific benchmarks"
            echo "  $0 --quick --throughput               # Quick throughput test"
            echo "  $0 --model meta-llama/Llama-3.1-8B-Instruct --dtype bfloat16"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
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
RESULTS_FILE="$OUTPUT_DIR/advanced_scheduling_${TIMESTAMP}.json"
LOG_FILE="$OUTPUT_DIR/advanced_scheduling_${TIMESTAMP}.log"

echo "=============================================================="
echo "  Advanced Scheduling Benchmark Suite"
echo "=============================================================="
echo ""
echo "Configuration:"
echo "  Model:      $MODEL"
echo "  Dtype:      $DTYPE"
echo "  Device:     $DEVICE"
echo "  Num runs:   $NUM_RUNS"
echo "  Output:     $RESULTS_FILE"
echo "  Log:        $LOG_FILE"
echo ""

# Check if CUDA is available (if device is cuda)
if [ "$DEVICE" = "cuda" ]; then
    echo "Checking CUDA availability..."
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || {
        echo "WARNING: CUDA is not available. Switching to CPU."
        DEVICE="cpu"
    }
    if [ "$DEVICE" = "cuda" ]; then
        echo "CUDA is available."
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    fi
    echo ""
fi

echo "=============================================================="
echo "  Starting Benchmarks..."
echo "=============================================================="
echo ""

cd "$PROJECT_ROOT"

# Run the benchmark
python3 -m benches.benchmark_advanced_scheduling \
    --model "$MODEL" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --num-runs "$NUM_RUNS" \
    --output "$RESULTS_FILE" \
    $BENCHMARK_ARGS \
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

# Print summary from JSON
echo "Quick Summary:"
echo "--------------"
python3 << EOF
import json
import sys

try:
    with open("$RESULTS_FILE", "r") as f:
        results = json.load(f)

    # Group by benchmark name
    by_benchmark = {}
    for r in results:
        name = r["benchmark_name"]
        if name not in by_benchmark:
            by_benchmark[name] = []
        by_benchmark[name].append(r)

    for benchmark, items in by_benchmark.items():
        print(f"\n{benchmark.upper().replace('_', ' ')}:")
        for item in items:
            config_name = item["configuration"].get("name", "")
            if not config_name:
                config_name = ", ".join(f"{k}={v}" for k, v in item["configuration"].items())
            throughput = item["metrics"].get("throughput_tokens_per_sec", 0)
            print(f"  {config_name}: {throughput:.1f} tok/s")

except Exception as e:
    print(f"Could not parse results: {e}")
    sys.exit(1)
EOF

echo ""
echo "=============================================================="
