#!/bin/bash
################################################################################
# EvalScope VLM Benchmark Runner for Qwen3.5
#
# Runs standard VLM benchmarks (MMMU, LongVideoBench) via EvalScope.
#
# Usage:
#   bash run_evalscope_vlm.sh MODE MODEL_OR_URL DATASET [LIMIT]
#
# Examples:
#   # Local mode (model loaded directly, no vLLM needed)
#   bash run_evalscope_vlm.sh local ./Qwen3.5-0.8B-INT4 MMMU_DEV_VAL 20
#
#   # API mode (connect to existing vLLM/IPEX-LLM service)
#   bash run_evalscope_vlm.sh api http://localhost:41091/v1/chat/completions MMMU_DEV_VAL
#
#   # LongVideoBench with limited samples
#   bash run_evalscope_vlm.sh local ./Qwen3.5-0.8B-INT4 LongVideoBench 10
################################################################################

set -e

# Parameters
MODE=${1:-local}
MODEL_OR_URL=${2:-./Qwen3.5-0.8B-INT4}
DATASET=${3:-MMMU_DEV_VAL}
LIMIT=${4:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV="/home/intel/jie/Env/bin/python"
OV_LIBS="/home/intel/jie/Env/lib/python3.12/site-packages/openvino/libs"

# Set up environment
export LD_LIBRARY_PATH="$OV_LIBS:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT=https://hf-mirror.com
export OV_GPU_DYNAMIC_QUANTIZATION_THRESHOLD=1

echo "======================================================================"
echo "Qwen3.5 VLM Standard Benchmark Evaluation (EvalScope)"
echo "======================================================================"
echo "Mode:     $MODE"
echo "Dataset:  $DATASET"
if [ -n "$LIMIT" ]; then
    echo "Limit:    $LIMIT samples"
fi

# Check if evalscope is installed
if ! $PYTHON_ENV -c "import evalscope" 2>/dev/null; then
    echo ""
    echo "ERROR: evalscope is not installed in the Python environment."
    echo "Install it with:"
    echo "  $PYTHON_ENV -m pip install 'evalscope[all]'"
    echo ""
    exit 1
fi

# Build output directory name
OUTPUT_DIR="$SCRIPT_DIR/accuracy_results_evalscope_${DATASET}"

# Build command
CMD="$PYTHON_ENV $SCRIPT_DIR/evalscope_qwen3_5_vlm.py"
CMD="$CMD --mode $MODE"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --task-cfg $SCRIPT_DIR/configs/qwen3_5_vl_eval.yaml"

if [ "$MODE" = "local" ]; then
    CMD="$CMD --model-path $MODEL_OR_URL"
    CMD="$CMD --device CPU"
    echo "Model:    $MODEL_OR_URL"
elif [ "$MODE" = "api" ]; then
    CMD="$CMD --api-url $MODEL_OR_URL"
    echo "API URL:  $MODEL_OR_URL"
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
echo ""
echo "To compare all results:"
echo "  python compare_results.py"
echo "======================================================================"
