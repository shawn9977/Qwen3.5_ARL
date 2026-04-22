#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./check_openvino_stack.sh [python_executable]
# Example:
#   ./check_openvino_stack.sh /home/intel/jie/Env/bin/python

TS="$(date +%Y%m%d_%H%M%S)"
REPORT="openvino_stack_report_${TS}.txt"
PY_BIN="${1:-}"

if [[ -z "${PY_BIN}" ]]; then
  if [[ -x "/home/intel/jie/Env/bin/python" ]]; then
    PY_BIN="/home/intel/jie/Env/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PY_BIN="$(command -v python)"
  else
    PY_BIN="python-not-found"
  fi
fi

run_section() {
  local title="$1"
  shift
  {
    echo
    echo "===== ${title} ====="
    "$@"
  } >> "${REPORT}" 2>&1 || true
}

{
  echo "OpenVINO/GPU Runtime Check Report"
  echo "Generated at: $(date -Iseconds)"
  echo "Host: $(hostname)"
  echo "Kernel: $(uname -srmo)"
  echo "Python executable: ${PY_BIN}"
  echo
} > "${REPORT}"

run_section "OS Release" bash -lc 'if command -v lsb_release >/dev/null 2>&1; then lsb_release -a; else cat /etc/os-release; fi'
run_section "PCI GPU" bash -lc 'if command -v lspci >/dev/null 2>&1; then lspci -nn | grep -Ei "vga|3d|display|intel"; else echo "lspci not found"; fi'

run_section "Python Version" bash -lc "if [[ '${PY_BIN}' != 'python-not-found' ]]; then '${PY_BIN}' --version; else echo 'python not found'; fi"

run_section "Python Packages (OpenVINO related)" bash -lc "if [[ '${PY_BIN}' != 'python-not-found' ]]; then '${PY_BIN}' -m pip list --format=freeze | grep -Ei 'openvino|optimum|nncf|intel|onnxruntime|tokenizers|transformers' || true; else echo 'python not found'; fi"

run_section "OpenVINO GPU Probe" bash -lc "if [[ '${PY_BIN}' != 'python-not-found' ]]; then timeout 20s '${PY_BIN}' - <<'PY'
import json
import traceback

try:
    import openvino as ov
    core = ov.Core()
    data = {
        'available_devices': core.available_devices,
    }
    try:
        data['GPU.FULL_DEVICE_NAME'] = core.get_property('GPU', 'FULL_DEVICE_NAME')
    except Exception as e:
        data['GPU.FULL_DEVICE_NAME_error'] = str(e)
    try:
        data['GPU.DEVICE_ARCHITECTURE'] = core.get_property('GPU', 'DEVICE_ARCHITECTURE')
    except Exception as e:
        data['GPU.DEVICE_ARCHITECTURE_error'] = str(e)
    try:
        data['GPU.DRIVER_VERSION'] = core.get_property('GPU', 'DRIVER_VERSION')
    except Exception as e:
        data['GPU.DRIVER_VERSION_error'] = str(e)
    print(json.dumps(data, ensure_ascii=True, indent=2))
except Exception:
    traceback.print_exc()
PY
else
  echo 'python not found'
fi"

run_section "Debian Packages (exact names)" bash -lc '
if command -v dpkg-query >/dev/null 2>&1; then
  pkgs=(
    intel-opencl-icd
    libze-intel-gpu1
    intel-ocloc
    intel-igc-core-2
    intel-igc-opencl-2
    libigc2
    libigdgmm12
    intel-level-zero-npu
    libze-intel-gpu-raytracing
  )
  for p in "${pkgs[@]}"; do
    dpkg-query -W -f="${Package}\t${Version}\t${Status}\n" "$p" 2>/dev/null || echo -e "$p\tnot-installed\tunknown"
  done
else
  echo "dpkg-query not found"
fi'

run_section "Debian Packages (keyword scan)" bash -lc '
if command -v dpkg >/dev/null 2>&1; then
  dpkg -l | grep -Ei "openvino|compute-runtime|intel-opencl|level-zero|libze-intel-gpu|intel-ocloc|igc|gmmlib|neo" || true
else
  echo "dpkg not found"
fi'

run_section "OpenCL ICD Files" bash -lc 'ls -l /etc/OpenCL/vendors 2>/dev/null || echo "OpenCL ICD dir not found"'
run_section "Level Zero Files" bash -lc 'ls -l /usr/lib/x86_64-linux-gnu/libze* 2>/dev/null || echo "libze files not found"'

run_section "clinfo (first 120 lines)" bash -lc '
if command -v clinfo >/dev/null 2>&1; then
  timeout 20s clinfo | sed -n "1,120p"
else
  echo "clinfo not found"
fi'

run_section "sycl-ls" bash -lc '
if command -v sycl-ls >/dev/null 2>&1; then
  timeout 20s sycl-ls
else
  echo "sycl-ls not found"
fi'

run_section "NEO Artifact Directory" bash -lc '
candidates=("./neo" "$HOME/neo" "/home/intel/jie/neo")
for d in "${candidates[@]}"; do
  if [[ -d "$d" ]]; then
    echo "Found: $d"
    find "$d" -maxdepth 1 -mindepth 1 -printf "%f\n" | sort
  fi
done'

{
  echo
  echo "===== Quick Verdict Guide ====="
  echo "1) If openvino import fails: Python environment/package issue."
  echo "2) If GPU missing in available_devices: runtime/driver stack issue."
  echo "3) If FULL_DEVICE_NAME shows [0x....]: often naming/version difference, not always missing NEO."
  echo "4) Compare this report with a known-good machine package versions."
  echo
  echo "Report saved to: ${REPORT}"
} >> "${REPORT}"

echo "Report saved to: ${REPORT}"
