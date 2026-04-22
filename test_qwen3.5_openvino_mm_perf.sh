#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="/home/intel/jie/Env/bin/python"
SCRIPT_DIR="/home/intel/jie"

# Custom OpenVINO build for PTL iGPU dynamic shape support
export PYTHONPATH="/home/intel/jie/openvino_35d22/bin/intel64/Release/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/intel/jie/openvino_35d22/bin/intel64/Release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

MODEL_DIR="${1:-/home/intel/jie/Qwen3.5-35B-A3B-INT4}"
IMAGE_PATH="${2:-/home/intel/jie/test_image.jpg}"
DEVICE="${DEVICE:-GPU}"
INPUT_TEXT_TOKENS="${INPUT_TEXT_TOKENS:-1024}"
NEW_TOKENS="${NEW_TOKENS:-64}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-5}"
NUM_STREAMS="${NUM_STREAMS:-1}"
CSV_OUT="${CSV_OUT:-}"
STRICT_DEVICE="${STRICT_DEVICE:-0}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: Python not found: ${VENV_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: model directory does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "ERROR: image does not exist: ${IMAGE_PATH}" >&2
  exit 1
fi

CMD=(
  "${VENV_PYTHON}" "${SCRIPT_DIR}/benchmark_qwen3_5_mm_realtext.py"
  --model-dir "${MODEL_DIR}"
  --image "${IMAGE_PATH}"
  --device "${DEVICE}"
  --input-text-tokens "${INPUT_TEXT_TOKENS}"
  --new-tokens "${NEW_TOKENS}"
  --warmup "${WARMUP}"
  --iters "${ITERS}"
  --num-streams "${NUM_STREAMS}"
)

if [[ -n "${CSV_OUT}" ]]; then
  CMD+=(--csv-out "${CSV_OUT}")
fi

if [[ "${STRICT_DEVICE}" == "1" ]]; then
  CMD+=(--strict-device)
fi

"${CMD[@]}"
