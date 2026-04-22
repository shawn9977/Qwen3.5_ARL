# Qwen3.5 on Intel Panther Lake iGPU - Getting Started Guide

## Overview

This guide walks through exporting Qwen3.5 vision-language models to OpenVINO IR format using optimum-intel, setting up the runtime environment on an Intel Panther Lake (PTL) platform with integrated Arc GPU, and running performance benchmarks for both text-only and multimodal (image + text) inference.

**Tested platform**: Intel Panther Lake with Arc B390 iGPU

**Tested models**:

| Model | HuggingFace ID | Architecture | Total Params | Active Params |
|-------|---------------|--------------|-------------|---------------|
| Qwen3.5-0.8B | Qwen/Qwen3.5-0.8B | Dense | 0.8B | 0.8B |
| Qwen3.5-9B | Qwen/Qwen3.5-9B | Dense | 9B | 9B |
| Qwen3.5-27B | Qwen/Qwen3.5-27B | Dense | 27B | 27B |
| Qwen3.5-35B-A3B | Qwen/Qwen3.5-35B-A3B | MoE | 35B | 3B |

---

## 1. Model Export with optimum-intel

Qwen3.5 OpenVINO support is based on [optimum-intel PR #1634](https://github.com/huggingface/optimum-intel/pull/1634) (`[OpenVINO] Support Qwen3.5 and Qwen3.5-MoE`).

### 1.1 Install dependencies for export

```bash
pip install git+https://github.com/rkazants/optimum-intel.git@support_qwen3_5
pip install --pre -U openvino openvino-tokenizers nncf \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
pip install transformers==5.2.0
pip install requests torchvision opencv-python
```

### 1.2 Pre-export environment setup

```bash
# For users in China, use HuggingFace mirror to avoid download issues
export HF_ENDPOINT=https://hf-mirror.com

# Set a custom TMPDIR with sufficient disk space
export TMPDIR=/mnt/disk5/temp_space && mkdir -p $TMPDIR
```

> **Why TMPDIR?** During model export, PyTorch and the HuggingFace transformers library load the full-precision model weights into memory, then write intermediate files (ONNX graphs, temporary safetensor shards, etc.) to the system temp directory (`/tmp` by default). For the 35B-A3B model, these intermediate files can exceed 70 GB. On many systems `/tmp` is mounted as a `tmpfs` backed by RAM or lives on a small root partition, which will run out of space and cause the export to fail with `OSError: No space left on device`. Pointing `TMPDIR` to a disk with sufficient free space avoids this.
>
> **Memory requirement**: Exporting the 35B-A3B model also requires substantial system RAM (~200 GB+ peak) to hold the full-precision weights in memory during the conversion and quantization process. For smaller models (0.8B, 9B), default `/tmp` and 32 GB RAM are typically sufficient.

### 1.3 Export commands

All Qwen3.5 models are vision-language models, so the `--task image-text-to-text` flag is required.

```bash
# Qwen3.5-0.8B (INT4)
optimum-cli export openvino \
  -m Qwen/Qwen3.5-0.8B \
  --task image-text-to-text \
  --weight-format int4 \
  Qwen3.5-0.8B-INT4

# Qwen3.5-9B (INT4)
optimum-cli export openvino \
  -m Qwen/Qwen3.5-9B \
  --task image-text-to-text \
  --weight-format int4 \
  Qwen3.5-9B-INT4

# Qwen3.5-27B (INT4)
optimum-cli export openvino \
  -m Qwen/Qwen3.5-27B \
  --task image-text-to-text \
  --weight-format int4 \
  Qwen3.5-27B-INT4

# Qwen3.5-35B-A3B MoE (INT4)
optimum-cli export openvino \
  -m Qwen/Qwen3.5-35B-A3B \
  --task image-text-to-text \
  --weight-format int4 \
  Qwen3.5-35B-A3B-INT4
```

### 1.4 Exported artifacts

Each exported model directory contains:

```
Qwen3.5-*-INT4/
  openvino_language_model.xml/.bin          # Core LLM
  openvino_text_embeddings_model.xml/.bin   # Text embedding lookup
  openvino_vision_embeddings_model.xml/.bin # Vision encoder
  openvino_vision_embeddings_merger_model.xml/.bin
  openvino_vision_embeddings_pos_model.xml/.bin
  openvino_tokenizer.xml/.bin
  openvino_detokenizer.xml/.bin
  config.json, generation_config.json, tokenizer_config.json, ...
```

---

## 2. PTL iGPU Environment Setup

After exporting the models, set up the runtime environment on the Panther Lake platform. Running MoE models on PTL iGPU requires a custom OpenVINO build with source-level fixes.

### 2.1 Create Python virtual environment

```bash
python3 -m venv /home/intel/jie/Env
source /home/intel/jie/Env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
Key package versions:

| Package | Version / Source |
|---------|-----------------|
| transformers | 5.2.0 |
| optimum-intel | 1.27.0.dev0 (git: rkazants/optimum-intel@4602e00) |
| optimum | 2.1.0.dev0 (git: huggingface/optimum@3db78a4) |
| openvino | 2026.2.0.dev20260402 |
| torch | 2.11.0 |
| numpy | 2.2.6 |


### 2.2 Build OpenVINO from source (required for MoE models on PTL iGPU)

The stock OpenVINO GPU plugin has several issues when running MoE models on PTL iGPU:

1. **Dynamic shape assertions** -- KV cache nodes have unbounded dynamic dimensions (`[?,2,?,256]`), causing `has_upper_bound` assertion failures. Source fix: modify `layout.cpp` to use a default upper bound (4096) instead of asserting, and remove related assertions in `engine.cpp`, `ocl_engine.cpp`, `ze_engine.cpp`, `primitive_inst.cpp`.
2. **oneDNN dependency** -- The MoE micro-kernel (`moe_3gemm_swiglu_opt`) requires oneDNN for GPU, which is off by default.
3. **Type mismatch in MoE codegen** -- Ternary operator type mismatch in `moe_3gemm_gen_micro.hpp` and `.cpp` when `input_type == ov::element::dynamic`.

A pre-built patch (`ov_patch/ptl_igpu_moe_fix.patch`) is provided that includes all necessary fixes. The patch applies on top of OpenVINO commit `35d22f2a0e`.

```bash
git clone https://github.com/openvinotoolkit/openvino.git openvino_35d22
cd openvino_35d22
git checkout 35d22f2a0e
git submodule update --init --recursive

# Apply the PTL iGPU MoE fix patch
git apply /home/intel/jie/ov_patch/ptl_igpu_moe_fix.patch

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_ONEDNN_FOR_GPU=ON \
      ..

# Full build (all plugins)
make -j$(nproc)

# Or, to only build the GPU plugin (much faster):
# make -j$(nproc) openvino_intel_gpu_plugin
```

After building, set the environment variables to use the custom build:

```bash
export PYTHONPATH="/home/intel/jie/openvino_35d22/bin/intel64/Release/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/intel/jie/openvino_35d22/bin/intel64/Release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

> **Note**: Dense models (0.8B, 9B, 27B) can run on PTL iGPU with the stock OpenVINO pip package without source fixes. The custom build is only required for the MoE model (35B-A3B).

### 2.3 Verify GPU device

```bash
python -c "
import openvino as ov
core = ov.Core()
print(core.get_property('GPU', 'FULL_DEVICE_NAME'))
"
```

Expected output:
```
Intel(R) Arc(TM) B390 GPU (iGPU)
```

---

## 3. Functional Sanity Test

Before benchmarking, verify the model loads and runs correctly on GPU using `test_qwen3.5.py`.

```bash
python test_qwen3.5.py
```

The script loads the model via `OVModelForVisualCausalLM`, compiles it on GPU, and runs a sample inference. Key points:

- The script passes `ov_config={"CACHE_DIR": ""}` to disable the auto model cache, which avoids a known issue with dynamic shape handling on the GPU plugin.
- Edit the `model_dir` and `device` variables in the script to test different models or devices.

Check that all components are placed on GPU.0:
```
Execution devices:
  - language_model: ['GPU.0']
  - vision_embeddings: ['GPU.0']
  - vision_embeddings_merger: ['GPU.0']
  - vision_embeddings_pos: ['GPU.0']
```

---

## 4. Performance Benchmarks

Two benchmark modes are provided:

| Mode | Script | Description |
|------|--------|-------------|
| Text-only | `test_qwen3.5_openvino_perf.sh` | Raw OpenVINO runtime benchmark on language model |
| Multimodal | `test_qwen3.5_openvino_mm_perf.sh` | End-to-end image + text benchmark via optimum-intel |

### 4.1 Text-only LLM benchmark

Measures prefill and decode latency at the OpenVINO runtime level using the language model IR directly.

```bash
DEVICE=GPU \
BATCH=1 \
SEQ_LEN=1024 \
DECODE_TOKENS=64 \
WARMUP=3 \
ITERS=5 \
NUM_STREAMS=1 \
bash test_qwen3.5_openvino_perf.sh
```

**Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| DEVICE | Target device (GPU / CPU) | GPU |
| BATCH | Batch size | 1 |
| SEQ_LEN | Prefill sequence length | 256 |
| DECODE_TOKENS | Number of autoregressive decode steps | 32 |
| WARMUP | Warmup iterations (not measured) | 3 |
| ITERS | Measured iterations | 20 |
| NUM_STREAMS | OpenVINO NUM_STREAMS setting | AUTO |

### 4.2 Multimodal benchmark

Measures end-to-end TTFT, TPOT, and throughput with image + text input through the full optimum-intel pipeline.

```bash
DEVICE=GPU \
INPUT_TEXT_TOKENS=1024 \
NEW_TOKENS=64 \
WARMUP=3 \
ITERS=5 \
NUM_STREAMS=1 \
CSV_OUT=./mm_perf_1k.csv \
bash test_qwen3.5_openvino_mm_perf.sh
```

**Parameters**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| DEVICE | Target device (GPU / CPU) | GPU |
| INPUT_TEXT_TOKENS | User text token count (before chat template) | 1024 |
| NEW_TOKENS | Generated token count | 64 |
| WARMUP | Warmup iterations | 1 |
| ITERS | Measured iterations | 5 |
| NUM_STREAMS | OpenVINO NUM_STREAMS setting | 1 |
| CSV_OUT | Optional CSV output path | (none) |

### 4.3 Metrics definition

| Metric | Definition |
|--------|-----------|
| **Prefill** | Time to process the full input sequence in one forward pass (populate KV cache) |
| **TTFT** | Time To First Token = prefill latency + first decode step latency |
| **TPOT** | Time Per Output Token = mean latency of decode steps 2..N (excludes 1st decode step) |
| **Throughput** | Total generated tokens / total wall-clock time (prefill + all decode steps) |

---

## 5. Benchmark Results

**Test conditions**: Panther Lake Arc B390 iGPU | All components on GPU.0 | Input=1024 tokens, Output=512 tokens | WARMUP=3, ITERS=10, NUM_STREAMS=1

### 5.1 Text-only input (SEQ_LEN=1024, DECODE_TOKENS=512)

| Metric | 9B-INT4 | 35B-A3B-INT4 (MoE) |
|--------|---------|---------------------|
| Hidden dim | 4096 | 2048 |
| Prefill (mean) | 1232.5 ms | 1632.0 ms |
| TTFT (mean) | 1300.9 ms | 1694.7 ms |
| TPOT (mean) | 51.5 ms | 31.4 ms |
| **Throughput** | **18.54 tok/s** | **28.90 tok/s** |

### 5.2 Multimodal input (Image + 1024 text tokens, NEW_TOKENS=512)

| Metric | 9B-INT4 | 35B-A3B-INT4 (MoE) |
|--------|---------|---------------------|
| Prompt tokens | 1408 | 1408 |
| TTFT (mean) | 1896.8 ms | 2283.2 ms |
| TPOT (mean) | 54.4 ms | 34.5 ms |
| **Throughput** | **17.24 tok/s** | **25.72 tok/s** |

### 5.3 Key observations

- **MoE efficiency**: The 35B-A3B MoE model activates only 3B parameters per token despite having 35B total parameters. On PTL iGPU, its decode speed (31.4 ms TPOT) is 39% faster than the 9B dense model (51.5 ms), while offering significantly greater model capacity.

- **MoE throughput advantage**: With 512 output tokens, the 35B-A3B MoE achieves 28.90 tok/s text-only throughput, 56% higher than 9B dense (18.54 tok/s). The sparse activation pattern makes it the clear winner for longer generation tasks.

- **Vision overhead**: Multimodal input adds ~590 ms to TTFT for both models (vision encoder + merger + positional encoding). The decode-phase TPOT difference is small (2-3 ms) since vision components are not involved during autoregressive generation.

---

## 6. Accuracy Benchmarks

Accuracy testing compares quantized OpenVINO models against the original BF16 model to measure quantization loss. We use the full MMMU_DEV_VAL (Massive Multidiscipline Multimodal Understanding) benchmark (1050 samples) via EvalScope + VLMEvalKit.

### 6.1 Prerequisites

```bash
pip install 'evalscope[all]'
```

### 6.2 vLLM server setup (for BF16 baseline, on NVIDIA GPU machine)

The BF16 baseline runs on an NVIDIA GPU via vLLM. Qwen3.5 has thinking mode enabled by default, which must be disabled for accuracy testing. A custom chat template (`chat_template_no_think.jinja`) is provided for this.

```bash
# Start vLLM with thinking disabled (required for accuracy testing)
bash setup_vllm_server.sh Qwen/Qwen3.5-9B 8000 1 0.9 chat_template_no_think.jinja

# For GPUs with less VRAM (e.g. RTX 4090), lower gpu-memory-utilization
bash setup_vllm_server.sh Qwen/Qwen3.5-9B 8000 1 0.7 chat_template_no_think.jinja

# FP8 quantization on vLLM
bash setup_vllm_server.sh Qwen/Qwen3.5-9B 8000 1 0.7 chat_template_no_think.jinja fp8

# FP16 dtype
bash setup_vllm_server.sh Qwen/Qwen3.5-9B 8000 1 0.7 chat_template_no_think.jinja float16

# Kill vLLM server when done
pkill -f "vllm serve"
```

Parameters: `<model> <port> <tp_size> <gpu_mem_util> <chat_template> <precision>`

| Parameter | Description | Default |
|-----------|-------------|---------|
| model | HuggingFace model ID | Qwen/Qwen3.5-9B |
| port | Server port | 8000 |
| tp_size | Tensor parallel GPU count | 1 |
| gpu_mem_util | GPU memory utilization (0.0-1.0) | 0.9 |
| chat_template | Custom chat template path | (none, thinking enabled) |
| precision | `auto`(BF16), `float16`, `bfloat16`, `fp8`, `awq`, `gptq` | auto |

The `precision` parameter automatically routes to `--dtype` or `--quantization` in vLLM depending on the value (e.g. `fp8` uses `--quantization fp8`, `float16` uses `--dtype float16`).

### 6.3 Running accuracy tests

**API mode** — test BF16 original model via vLLM (on NVIDIA GPU):

```bash
bash run_evalscope_vlm_gpu.sh api http://<NVIDIA_IP>:8000 MMMU_DEV_VAL 1050
```

**Local mode** — test OpenVINO models on Intel GPU:

```bash
# INT4 quantized model
bash run_evalscope_vlm_gpu.sh local ./Qwen3.5-9B-INT4 MMMU_DEV_VAL 1050

# FP16 model
bash run_evalscope_vlm_gpu.sh local ./Qwen3.5-9B-FP16 MMMU_DEV_VAL 1050
```

Results are saved to `accuracy_results_evalscope_MMMU_DEV_VAL/summary_metrics.json`.

### 6.4 Accuracy results (Qwen3.5-9B, MMMU_DEV_VAL, 1050 samples, full benchmark)

| Model Format | Inference Engine | Hardware | Overall (validation) |
|-------------|-----------------|----------|---------------------|
| **BF16** | vLLM | NVIDIA GPU | **52.4%** |
| **FP16** | OpenVINO | Intel PTL iGPU | **54.8%** |
| **INT8** | OpenVINO | Intel PTL iGPU | **52.7%** |
| **INT4** | OpenVINO | Intel PTL iGPU | **48.1%** |

Breakdown by category:

| Category | BF16 (vLLM) | FP16 (OpenVINO) | INT8 (OpenVINO) | INT4 (OpenVINO) | INT8 vs BF16 | INT4 vs BF16 |
|----------|-------------|-----------------|-----------------|-----------------|-------------|-------------|
| Overall | **52.4%** | 54.8% | 52.7% | 48.1% | +0.2pp | -4.3pp |
| Art & Design | **57.5%** | 56.7% | 51.7% | 54.2% | -5.8pp | -3.3pp |
| Business | 47.3% | **53.3%** | 50.7% | 43.3% | +3.3pp | -4.0pp |
| Health & Medicine | 50.7% | **56.7%** | **56.7%** | 52.0% | +6.0pp | +1.3pp |
| Humanities & Social Sci | 70.8% | **72.5%** | 70.0% | 58.3% | -0.8pp | -12.5pp |
| Science | 50.7% | 50.7% | **51.3%** | 45.3% | +0.7pp | -5.3pp |
| Tech & Engineering | 45.2% | **46.2%** | 42.9% | 41.4% | -2.4pp | -3.8pp |

### 6.5 Accuracy results (Qwen3.5-35B-A3B, MMMU_DEV_VAL, 1050 samples, full benchmark)

| Model Format | Inference Engine | Hardware | Overall (validation) |
|-------------|-----------------|----------|---------------------|
| **BF16** | vLLM | NVIDIA A100 GPU | **60.0%** |
| **INT8** | OpenVINO | Intel PTL iGPU | **55.4%** |
| **INT4** | OpenVINO | Intel PTL iGPU | **50.3%** |

Breakdown by category:

| Category | BF16 (vLLM) | INT8 (OpenVINO) | INT4 (OpenVINO) | INT8 vs BF16 | INT4 vs BF16 |
|----------|-------------|-----------------|-----------------|-------------|-------------|
| Overall | **60.0%** | 55.4% | 50.3% | -4.6pp | -9.7pp |
| Art & Design | 65.8% | **66.7%** | 64.2% | +0.8pp | -1.7pp |
| Business | **48.0%** | **48.0%** | 42.7% | 0.0pp | -5.3pp |
| Health & Medicine | **66.7%** | 63.3% | 60.7% | -3.3pp | -6.0pp |
| Humanities & Social Sci | **78.3%** | 67.5% | 65.0% | -10.8pp | -13.3pp |
| Science | **59.3%** | 55.3% | 43.3% | -4.0pp | -16.0pp |
| Tech & Engineering | **50.5%** | 41.9% | 37.1% | -8.6pp | -13.3pp |

### 6.6 Accuracy comparison script

Use `compare_accuracy.py` to generate comparison tables from result directories:

```bash
# Compare specific results with labels
python compare_accuracy.py \
    --results "BF16 (vLLM)=accuracy_results_evalscope_MMMU_DEV_VAL/MMMU_DEV_VAL/20260410_142816" \
              "INT4 (OpenVINO)=accuracy_results_evalscope_MMMU_DEV_VAL/MMMU_DEV_VAL/20260410_150439" \
              "INT8 (OpenVINO)=accuracy_results_evalscope_MMMU_DEV_VAL/MMMU_DEV_VAL/20260410_152404"

# Auto-detect all results under a directory
python compare_accuracy.py --auto-detect accuracy_results_evalscope_MMMU_DEV_VAL/MMMU_DEV_VAL

# Export to CSV
python compare_accuracy.py --results ... --output comparison.csv
```

### 6.7 Key observations

- **FP16 OpenVINO conversion preserves accuracy well** — For 9B, FP16 (54.8%) slightly exceeds BF16 vLLM baseline (52.4%), confirming the OpenVINO conversion is high quality. The small difference is within sampling variance at `temperature=1.0`.

- **INT8 nearly matches BF16 for 9B** — INT8 (52.7%) is only +0.2pp from BF16 (52.4%), making INT8 essentially lossless for the 9B dense model. Some categories even improve (Health & Medicine +6.0pp, Business +3.3pp).

- **INT4 quantization loss is ~4pp for 9B, ~10pp for 35B-A3B** — 9B INT4 loses 4.3pp vs BF16, while 35B-A3B INT4 loses 9.7pp. The MoE model is more sensitive to aggressive quantization.

- **INT8 is a good middle ground for 35B-A3B** — INT8 (55.4%) loses only 4.6pp vs BF16 (60.0%), while INT4 (50.3%) loses 9.7pp. INT8 offers a better accuracy-memory tradeoff for the MoE model.

- **Humanities & Social Science is most sensitive to quantization** — Both 9B (-12.5pp INT4) and 35B-A3B (-13.3pp INT4) show the largest drops in this category.

- **Art & Design is most resilient** — For 35B-A3B, INT8 actually gains +0.8pp. For 9B INT4, the drop is only 3.3pp.

- **0.8B model is too small for MMMU** — Qwen3.5-0.8B achieves ~0% on MMMU regardless of precision, as the model lacks the reasoning capability for this benchmark. Use 9B+ models for meaningful accuracy testing.

---

## 7. Troubleshooting

### 7.1 Common issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `has_upper_bound` assertion on GPU | KV cache nodes have unbounded dynamic dimensions on PTL iGPU | Apply source fixes below, rebuild |
| `unordered_map::at` error | Auto model cache (`CACHE_DIR`) conflicts with dynamic shape handling | Pass `ov_config={"CACHE_DIR": ""}` to disable cache |
| `moe_3gemm_swiglu_opt depends on onednn` | OpenVINO built without oneDNN GPU support | Rebuild with `cmake -DENABLE_ONEDNN_FOR_GPU=ON` |
| Model falls back to CPU | GPU compilation failure, check execution device logs | Verify all fixes above are applied |
| Noisy benchmark numbers | Insufficient warmup or iterations | Increase WARMUP and ITERS |

### 7.2 OpenVINO source fixes for PTL iGPU MoE support

The following files need modification in the OpenVINO source tree. All changes are in the GPU plugin under `src/plugins/intel_gpu/`.

#### Fix 1: Dynamic shape upper bound defaults (layout.cpp)

**File**: `src/plugins/intel_gpu/src/runtime/layout.cpp`

In `get_tensor()` and `get_padded_dims()`, replace the assertion with a default upper bound:

```cpp
// Before:
OPENVINO_ASSERT(!is_dynamic() || has_upper_bound(),
    "[GPU] get_tensor() is called for dynamic shape without upper bound");

// After:
constexpr int64_t DEFAULT_UPPER_BOUND = 4096;
ov::Shape shape;
if (is_dynamic()) {
    for (const auto& dim : size) {
        if (dim.is_dynamic()) {
            auto max_len = dim.get_max_length();
            shape.push_back(max_len == -1 ? DEFAULT_UPPER_BOUND : max_len);
        } else {
            shape.push_back(dim.get_length());
        }
    }
} else {
    shape = size.to_shape();
}
```

#### Fix 2: Remove has_upper_bound assertions

Remove `OPENVINO_ASSERT(layout.has_upper_bound(), ...)` in these files, replacing with graceful handling (return false or allow dynamic layouts):

- `src/plugins/intel_gpu/src/runtime/engine.cpp` -- `check_allocatable()`
- `src/plugins/intel_gpu/src/runtime/ocl/ocl_engine.cpp` -- `allocate_memory()`
- `src/plugins/intel_gpu/src/runtime/ze/ze_engine.cpp` -- `allocate_memory()`
- `src/plugins/intel_gpu/src/graph/primitive_inst.cpp` -- two assertion sites

#### Fix 3: MoE type mismatch in ternary operators

**File**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_gen_micro.hpp` (line 75)
**File**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_gen_micro.cpp` (line 270)

```cpp
// Before:
cfg.out_type = (input_type == ov::element::dynamic) ? ov::element::f16 : input_type;

// After:
cfg.out_type = (input_type == ov::element::dynamic)
    ? ov::element::Type(ov::element::f16)
    : ov::element::Type(input_type);
```
