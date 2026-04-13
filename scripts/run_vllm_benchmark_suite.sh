#!/usr/bin/env bash
# =============================================================================
# vLLM Benchmark Suite Runner
#
# Starts a vLLM Docker container for a specified model, runs the benchmark
# suite, collects GPU metrics, and generates a comparison report.
#
# Usage:
#   ./scripts/run_vllm_benchmark_suite.sh --model deepseek-coder-6.7b --quantization awq
#   ./scripts/run_vllm_benchmark_suite.sh --compare-all
#   ./scripts/run_vllm_benchmark_suite.sh --dry-run
#
# Prerequisites:
#   - Docker with GPU support (nvidia-container-toolkit)
#   - HuggingFace model cache at $HF_HOME (default: ~/.cache/huggingface)
#   - Python 3.10+ with aiohttp, numpy
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_SCRIPT="$SCRIPT_DIR/benchmark_vllm_quantization.py"
MODELS_CONFIG="$PROJECT_ROOT/config/vllm_models.yaml"
REPORTS_DIR="$PROJECT_ROOT/reports"

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
VLLM_PORT="${VLLM_PORT:-8100}"
CONTAINER_NAME="vllm-benchmark"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Timeout for vLLM health check (seconds)
HEALTH_TIMEOUT=300
HEALTH_INTERVAL=5

# Model HuggingFace IDs (matching config/vllm_models.yaml)
declare -A MODEL_HF_IDS=(
    ["deepseek-coder-6.7b"]="deepseek-ai/deepseek-coder-6.7b-instruct"
    ["qwen2-7b"]="Qwen/Qwen2-7B-Instruct"
    ["llama3-8b"]="meta-llama/Meta-Llama-3-8B-Instruct"
    ["deepseek-v2-lite"]="deepseek-ai/DeepSeek-V2-Lite"
)

declare -A MODEL_QUANT_ARGS=(
    ["fp16"]=""
    ["awq"]="--quantization awq"
    ["gptq"]="--quantization gptq"
    ["int8"]="--quantization squeezellm"
)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

log_info()  { echo "[INFO]  $(date +%H:%M:%S) $*"; }
log_warn()  { echo "[WARN]  $(date +%H:%M:%S) $*" >&2; }
log_error() { echo "[ERROR] $(date +%H:%M:%S) $*" >&2; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --model NAME           Model short name (deepseek-coder-6.7b, qwen2-7b, llama3-8b, deepseek-v2-lite)
  --quantization METHOD  Quantization method (fp16, awq, gptq, int8). Default: awq
  --compare-all          Benchmark all candidate models sequentially
  --stress               Include stress test (ramp up concurrency)
  --quantization-sweep   Test all quantization methods for one model
  --dry-run              Run simulated benchmark (no Docker/GPU needed)
  --gpu-id ID            GPU device ID (default: 0)
  --help                 Show this help message

Examples:
  $(basename "$0") --model deepseek-coder-6.7b --quantization awq
  $(basename "$0") --compare-all --stress
  $(basename "$0") --dry-run
EOF
}

cleanup_container() {
    if docker ps -q --filter "name=$CONTAINER_NAME" 2>/dev/null | grep -q .; then
        log_info "Stopping container $CONTAINER_NAME ..."
        docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
}

start_vllm_container() {
    local hf_id="$1"
    local quant_method="$2"
    local gpu_id="$3"

    cleanup_container

    local quant_args="${MODEL_QUANT_ARGS[$quant_method]:-}"
    local gpu_mem="0.90"  # Use 90% of GPU memory

    log_info "Starting vLLM container: model=$hf_id, quantization=$quant_method, gpu=$gpu_id"

    docker run -d \
        --name "$CONTAINER_NAME" \
        --gpus "\"device=$gpu_id\"" \
        --shm-size 8g \
        -v "$HF_HOME:/root/.cache/huggingface" \
        -p "$VLLM_PORT:8000" \
        "$VLLM_IMAGE" \
        --model "$hf_id" \
        --trust-remote-code \
        --gpu-memory-utilization "$gpu_mem" \
        --max-model-len 4096 \
        $quant_args

    log_info "Container started. Waiting for health check ..."
}

wait_for_health() {
    local elapsed=0
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
            log_info "vLLM server is healthy (took ${elapsed}s)"
            return 0
        fi
        sleep $HEALTH_INTERVAL
        elapsed=$((elapsed + HEALTH_INTERVAL))
        if [ $((elapsed % 30)) -eq 0 ]; then
            log_info "Still waiting for vLLM health check ... (${elapsed}s / ${HEALTH_TIMEOUT}s)"
        fi
    done

    log_error "vLLM health check timed out after ${HEALTH_TIMEOUT}s"
    log_info "Container logs:"
    docker logs --tail 50 "$CONTAINER_NAME" 2>&1 || true
    return 1
}

collect_gpu_metrics() {
    local output_file="$1"
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_info "Collecting GPU metrics ..."
        nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
            --format=csv,noheader,nounits > "$output_file" 2>/dev/null || true
    else
        log_warn "nvidia-smi not found; skipping GPU metrics"
        echo "nvidia-smi not available" > "$output_file"
    fi
}

run_benchmark_for_model() {
    local model_name="$1"
    local quant_method="$2"
    local extra_args="$3"
    local hf_id="${MODEL_HF_IDS[$model_name]:-}"

    if [ -z "$hf_id" ]; then
        log_error "Unknown model: $model_name"
        return 1
    fi

    log_info "=========================================="
    log_info "Benchmarking: $model_name ($quant_method)"
    log_info "=========================================="

    # Start container
    start_vllm_container "$hf_id" "$quant_method" "$GPU_ID"

    if ! wait_for_health; then
        cleanup_container
        return 1
    fi

    # Collect pre-benchmark GPU metrics
    local gpu_pre="$REPORTS_DIR/gpu_pre_${model_name}_${quant_method}.csv"
    collect_gpu_metrics "$gpu_pre"

    # Run benchmark
    log_info "Running benchmark ..."
    python3 "$BENCHMARK_SCRIPT" \
        --endpoint "http://localhost:$VLLM_PORT" \
        --model "$hf_id" \
        --quantization "$quant_method" \
        --output-dir "$REPORTS_DIR" \
        $extra_args || true

    # Collect post-benchmark GPU metrics
    local gpu_post="$REPORTS_DIR/gpu_post_${model_name}_${quant_method}.csv"
    collect_gpu_metrics "$gpu_post"

    # Cleanup
    cleanup_container
    log_info "Completed: $model_name ($quant_method)"
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODEL=""
QUANTIZATION="awq"
COMPARE_ALL=false
STRESS=false
QUANT_SWEEP=false
DRY_RUN=false
GPU_ID="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --quantization) QUANTIZATION="$2"; shift 2 ;;
        --compare-all)  COMPARE_ALL=true; shift ;;
        --stress)       STRESS=true; shift ;;
        --quantization-sweep) QUANT_SWEEP=true; shift ;;
        --dry-run)      DRY_RUN=true; shift ;;
        --gpu-id)       GPU_ID="$2"; shift 2 ;;
        --help)         usage; exit 0 ;;
        *)              log_error "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p "$REPORTS_DIR"

# Trap for cleanup on exit
trap cleanup_container EXIT

if $DRY_RUN; then
    log_info "Running in dry-run mode (no Docker/GPU needed)"
    EXTRA=""
    $STRESS && EXTRA="$EXTRA --stress"
    if $COMPARE_ALL; then
        python3 "$BENCHMARK_SCRIPT" --dry-run --compare-all --output-dir "$REPORTS_DIR" $EXTRA
    elif $QUANT_SWEEP; then
        python3 "$BENCHMARK_SCRIPT" --dry-run --model "${MODEL:-deepseek-coder-6.7b}" --quantization-sweep --output-dir "$REPORTS_DIR" $EXTRA
    else
        python3 "$BENCHMARK_SCRIPT" --dry-run --model "${MODEL:-deepseek-coder-6.7b}" --output-dir "$REPORTS_DIR" $EXTRA
    fi
    log_info "Dry-run complete."
    exit 0
fi

# Live mode requires Docker GPU
if ! command -v docker >/dev/null 2>&1; then
    log_error "Docker is required for live benchmarks. Use --dry-run for simulation."
    exit 1
fi

EXTRA_ARGS=""
$STRESS && EXTRA_ARGS="$EXTRA_ARGS --stress"

if $COMPARE_ALL; then
    log_info "Comparing all candidate models ..."
    for model in deepseek-coder-6.7b qwen2-7b llama3-8b deepseek-v2-lite; do
        quant="${QUANTIZATION}"
        # Use fp16 for deepseek-v2-lite (MoE)
        [ "$model" = "deepseek-v2-lite" ] && quant="fp16"
        run_benchmark_for_model "$model" "$quant" "$EXTRA_ARGS" || true
    done
elif $QUANT_SWEEP; then
    if [ -z "$MODEL" ]; then
        log_error "--model is required for quantization sweep"
        exit 1
    fi
    for quant in fp16 awq gptq int8; do
        run_benchmark_for_model "$MODEL" "$quant" "$EXTRA_ARGS" || true
    done
else
    if [ -z "$MODEL" ]; then
        log_error "--model is required (or use --compare-all / --dry-run)"
        usage
        exit 1
    fi
    run_benchmark_for_model "$MODEL" "$QUANTIZATION" "$EXTRA_ARGS"
fi

log_info "All benchmarks complete. Reports saved to $REPORTS_DIR/"
