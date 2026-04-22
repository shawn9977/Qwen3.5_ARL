#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="/home/intel/jie/Env/bin/python"
SCRIPT_DIR="/home/intel/jie"

# Custom OpenVINO build for PTL iGPU dynamic shape support
export PYTHONPATH="/home/intel/jie/openvino_35d22/bin/intel64/Release/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/intel/jie/openvino_35d22/bin/intel64/Release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MODEL_XML="${1:-/home/intel/jie/Qwen3.5-35B-A3B-INT4/openvino_language_model.xml}"
DEVICE="${DEVICE:-GPU}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-256}"
WARMUP="${WARMUP:-3}"
ITERS="${ITERS:-20}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
NUM_STREAMS="${NUM_STREAMS:-AUTO}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: Python not found: ${VENV_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_XML}" ]]; then
  echo "ERROR: model xml does not exist: ${MODEL_XML}" >&2
  exit 1
fi

"${VENV_PYTHON}" "${SCRIPT_DIR}/benchmark_qwen3_5_openvino.py" \
  --model-xml "${MODEL_XML}" \
  --device "${DEVICE}" \
  --batch "${BATCH}" \
  --seq-len "${SEQ_LEN}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --decode-tokens "${DECODE_TOKENS}" \
  --num-streams "${NUM_STREAMS}"
