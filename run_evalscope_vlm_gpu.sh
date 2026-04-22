#!/bin/bash
################################################################################
# GPU-accelerated EvalScope VLM Benchmark Runner
#
# Same as run_evalscope_vlm.sh but uses GPU by default
################################################################################

set -e

MODE=${1:-local}
MODEL_OR_URL=${2:-./Qwen3.5-9B-INT4}
DATASET=${3:-MMMU_DEV_VAL}
LIMIT=${4:-}
MODEL_NAME=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV="/home/intel/jie/Env/bin/python"
OV_BUILD="/home/intel/jie/openvino_35d22/bin/intel64/Release"

export PYTHONPATH="$OV_BUILD/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$OV_BUILD:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD=1
# Bypass corporate proxy for internal network
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,10.*"
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,10.*"

echo "======================================================================"
echo "Qwen3.5 VLM Benchmark Evaluation (GPU Accelerated)"
echo "======================================================================"
echo "Mode:     $MODE"
echo "Dataset:  $DATASET"
if [ -n "$LIMIT" ]; then
    echo "Limit:    $LIMIT samples"
fi

# Check Intel GPU
if command -v sycl-ls > /dev/null 2>&1 && sycl-ls 2>/dev/null | grep -qi "intel"; then
    echo "Device:   Intel GPU detected (via sycl-ls)"
    DEVICE="GPU"
elif command -v clinfo > /dev/null 2>&1 && clinfo 2>/dev/null | grep -qi "intel"; then
    echo "Device:   Intel GPU detected (via clinfo)"
    DEVICE="GPU"
else
    echo "⚠️  WARNING: No Intel GPU detected, falling back to CPU"
    DEVICE="CPU"
fi

echo "======================================================================"
echo ""

# Build output directory name
OUTPUT_DIR="$SCRIPT_DIR/accuracy_results_evalscope_${DATASET}"

# Build command
CMD="$PYTHON_ENV $SCRIPT_DIR/evalscope_qwen3_5_vlm.py"
CMD="$CMD --mode $MODE"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --task-cfg $SCRIPT_DIR/configs/qwen3_5_vl_eval_high_quality.yaml"

if [ "$MODE" = "local" ]; then
    CMD="$CMD --model-path $MODEL_OR_URL"
    CMD="$CMD --device $DEVICE"  # Use GPU
    echo "Model:    $MODEL_OR_URL"
    echo "Device:   $DEVICE"
elif [ "$MODE" = "api" ]; then
    CMD="$CMD --api-url $MODEL_OR_URL"
    echo "API URL:  $MODEL_OR_URL"
    if [ -n "$MODEL_NAME" ]; then
        CMD="$CMD --model-name $MODEL_NAME"
        echo "Model:    $MODEL_NAME"
    fi
fi

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "======================================================================"
echo ""

# Run
eval $CMD

echo ""
echo "======================================================================"
echo "  Evaluation completed!"
echo "======================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "======================================================================"
