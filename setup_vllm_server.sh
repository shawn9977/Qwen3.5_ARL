#!/bin/bash
################################################################################
# vLLM Server Setup for Qwen3.5-VL Models (NVIDIA GPU)
#
# Usage:
#   # Available models: Qwen3.5-VL family (0.8B, 9B, 27B, 35B-A3B)
#   bash setup_vllm_server.sh Qwen/Qwen3.5-0.8B
#   bash setup_vllm_server.sh Qwen/Qwen3.5-9B
#   bash setup_vllm_server.sh Qwen/Qwen3.5-27B
#   bash setup_vllm_server.sh Qwen/Qwen3.5-35B-A3B
#
#   # Specify model + port
#   bash setup_vllm_server.sh Qwen/Qwen3.5-9B 8000
#
#   # Multi-GPU tensor parallel
#   bash setup_vllm_server.sh Qwen/Qwen3.5-27B 8000 2
#
#   # Disable thinking mode (required for accuracy testing)
#   bash setup_vllm_server.sh Qwen/Qwen3.5-9B 8000 1 0.9 chat_template_no_think.jinja
#
#   # FP8 quantization
#   bash setup_vllm_server.sh Qwen/Qwen3.5-9B 8000 1 0.9 chat_template_no_think.jinja fp8
#
#   # FP16 dtype
#   bash setup_vllm_server.sh Qwen/Qwen3.5-9B 8000 1 0.9 chat_template_no_think.jinja float16
#
# After setup, run accuracy test from Intel machine:
#   bash run_evalscope_vlm_gpu.sh api http://<NVIDIA_IP>:8000 MMMU_DEV_VAL 200
################################################################################

set -e

MODEL=${1:-Qwen/Qwen3.5-9B}
PORT=${2:-8000}
TP_SIZE=${3:-1}
GPU_MEM_UTIL=${4:-0.9}
CHAT_TEMPLATE=${5:-}  # Path to custom chat template (e.g. chat_template_no_think.jinja)
PRECISION=${6:-auto}  # auto(BF16), float16, bfloat16, fp8, awq, gptq, etc.

# HuggingFace mirror for faster downloads in China (comment out if not needed)
export HF_ENDPOINT=https://hf-mirror.com

echo "======================================================================"
echo "  vLLM Server for Qwen3.5-VL"
echo "======================================================================"
echo "  Model:    $MODEL"
echo "  Port:     $PORT"
echo "  TP Size:  $TP_SIZE"
echo "  GPU Mem:  $GPU_MEM_UTIL"
echo "  Precision: $PRECISION"
if [ -n "$CHAT_TEMPLATE" ]; then
    echo "  Template: $CHAT_TEMPLATE (thinking disabled)"
else
    echo "  Template: default (thinking ENABLED - use chat_template_no_think.jinja to disable)"
fi
echo "======================================================================"

# Check NVIDIA GPU
if ! nvidia-smi > /dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. This script requires NVIDIA GPU."
    exit 1
fi

echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Check/install vLLM
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "vLLM not found, installing..."
    pip install vllm --upgrade
    echo ""
fi

echo "vLLM version: $(python3 -c 'import vllm; print(vllm.__version__)')"
echo ""

# Start vLLM server
echo "Starting vLLM server on port $PORT ..."
echo "Press Ctrl+C to stop"
echo "======================================================================"

TEMPLATE_ARG=""
if [ -n "$CHAT_TEMPLATE" ]; then
    if [ ! -f "$CHAT_TEMPLATE" ]; then
        echo "ERROR: Chat template file not found: $CHAT_TEMPLATE"
        exit 1
    fi
    TEMPLATE_ARG="--chat-template $CHAT_TEMPLATE"
    echo "Using custom chat template: $CHAT_TEMPLATE"
fi

# Determine --dtype and --quantization based on PRECISION parameter
# fp8, awq, gptq, etc. use --quantization; others use --dtype
QUANT_TYPES="fp8 awq gptq gptq_marlin awq_marlin squeezellm"
DTYPE_ARG="--dtype auto"
QUANT_ARG=""

if echo "$QUANT_TYPES" | grep -qw "$PRECISION"; then
    QUANT_ARG="--quantization $PRECISION"
    echo "Using quantization: $PRECISION"
elif [ "$PRECISION" != "auto" ]; then
    DTYPE_ARG="--dtype $PRECISION"
fi

vllm serve "$MODEL" \
    --served-model-name Qwen3_5-VL \
    --port "$PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len 8192 \
    --limit-mm-per-prompt '{"image": 7}' \
    $DTYPE_ARG \
    $QUANT_ARG \
    --enforce-eager \
    $TEMPLATE_ARG
