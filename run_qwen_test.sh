#!/bin/bash
# Script to run Qwen test with custom OpenVINO build

set -e  # Exit on error

echo "=== Setting up environment ==="

# Setup Intel oneAPI environment
source /opt/intel/oneapi/setvars.sh --force > /dev/null 2>&1
echo "✓ Intel oneAPI environment loaded"

# Remove system oneDNN from CPATH to avoid conflicts
export CPATH=$(echo $CPATH | tr ':' '\n' | grep -v dnnl | tr '\n' ':' | sed 's/:$//')
echo "✓ CPATH cleaned"

# Setup OpenVINO paths
export PYTHONPATH=/home/intel/jie/openvino/bin/intel64/Release/python
export LD_LIBRARY_PATH=/home/intel/jie/openvino/bin/intel64/Release
echo "✓ OpenVINO paths set"

echo ""
echo "=== OpenVINO Info ==="
python3 << EOF
import sys
sys.path.insert(0, '/home/intel/jie/openvino/bin/intel64/Release/python')
import openvino as ov
print(f"OpenVINO version: {ov.__version__}")
core = ov.Core()
if 'GPU' in core.available_devices:
    print(f"GPU available: {core.get_property('GPU', 'FULL_DEVICE_NAME')}")
else:
    print("WARNING: GPU not available!")
EOF

echo ""
echo "=== Running Qwen test ==="
# Run the test
cd /home/intel/jie
/home/intel/jie/Env/bin/python test_qwen3.5.py
