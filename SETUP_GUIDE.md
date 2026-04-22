# Qwen3.5-35B-A3B OpenVINO 推理环境搭建指南

**机器环境：** intel-AXMB-D150-3  
**GPU：** Intel Arc Graphics (Arrow Lake-P iGPU, arch=v12.74.4)  
**OS：** Ubuntu 24.04, Linux 6.17.0-22  
**Python：** 3.12.13 (conda 环境 `jie`)  
**完成日期：** 2026-04-21  

---

## 一、目标

在ARL机器运行推理环境，
运行 `test_qwen3.5.py` 成功推理 Qwen3.5-35B-A3B 多模态模型。

---

## 二、核心依赖版本

| 包 | 版本 / 来源 |
|---|---|
| transformers | 5.2.0 |
| optimum | 2.1.0.dev0 (git: huggingface/optimum@3db78a4) |
| optimum-intel | 1.27.0.dev0 (git: rkazants/optimum-intel@4602e00) |
| optimum-onnx | 自动由 optimum-intel 依赖拉取 (branch: xadupre/transformers5) |
| openvino | 2026.2.0.dev20260402 (nightly) |
| openvino-tokenizers | 2026.2.0.0.dev20260402 (nightly) |
| nncf | 3.0.0 |
| torch | 2.11.0 |
| numpy | 1.26.4 |

---

## 三、模型导出（Model Export with optimum-intel）

Qwen3.5 OpenVINO 支持基于 optimum-intel PR #1634（[OpenVINO] Support Qwen3.5 and Qwen3.5-MoE）。

### 3.1 安装导出依赖

```bash
conda create -n qwen35 python=3.12
conda activate qwen35

pip install git+https://github.com/rkazants/optimum-intel.git@support_qwen3_5
pip install --pre -U openvino openvino-tokenizers nncf \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
pip install transformers==5.2.0
pip install requests torchvision opencv-python
```

### 3.2 导出前环境配置

```bash
# 国内用户使用 HuggingFace 镜像，避免下载失败
export HF_ENDPOINT=https://hf-mirror.com

# 设置有足够磁盘空间的临时目录
export TMPDIR=/mnt/disk5/temp_space && mkdir -p $TMPDIR
```

**为什么需要设置 TMPDIR？**  
导出过程中，PyTorch 和 transformers 会把全精度权重加载到内存，并将中间文件（ONNX 图、临时 safetensor 分片等）写入系统临时目录（默认 `/tmp`）。对于 35B-A3B 模型，这些中间文件可超过 **70 GB**。`/tmp` 通常挂载为 tmpfs（由 RAM 支撑）或在空间较小的根分区上，空间不足会导致导出失败（`OSError: No space left on device`）。将 TMPDIR 指向有足够空间的磁盘可避免此问题。

**内存要求：** 导出 35B-A3B 模型需要约 **200 GB+ 系统 RAM**（转换和量化过程中保存全精度权重）。较小模型（0.8B、9B）使用默认 `/tmp` 和 32 GB RAM 通常足够。

### 3.3 导出命令

所有 Qwen3.5 模型均为视觉语言模型，必须加 `--task image-text-to-text`。

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

### 3.4 导出产物结构

每个导出模型目录包含：

```
Qwen3.5-*-INT4/
  openvino_language_model.xml/.bin            # 核心 LLM
  openvino_text_embeddings_model.xml/.bin     # 文本 embedding 查找表
  openvino_vision_embeddings_model.xml/.bin   # 视觉编码器
  openvino_vision_embeddings_merger_model.xml/.bin
  openvino_vision_embeddings_pos_model.xml/.bin
  openvino_tokenizer.xml/.bin
  openvino_detokenizer.xml/.bin
  config.json, generation_config.json, tokenizer_config.json, ...
```

---

## 四、ARL推理运行环境搭建

### 4.1 创建 Python 环境

**方案A：conda 环境（本文档使用，ARL 机器）**
```bash
conda create -n jie python=3.12
conda activate jie
pip install --upgrade pip
pip install -r requirements.txt
```



---

### 4.2 OpenVINO 安装方式

| 平台 | 模型 | 方式 |
|---|---|---|
| **ARL iGPU (Arrow Lake-P，本机)** | 所有模型 | pip 安装 stock 版本（见八、完整安装步骤） |


#### 4.2.1  从源码构建 OpenVINO（仅 MoE 模型需要 ）


已提供包含所有修复的补丁：`ov_patch/ptl_igpu_moe_fix.patch`（基于 OpenVINO commit `35d22f2a0e`）。

```bash
git clone https://github.com/openvinotoolkit/openvino.git openvino_35d22
cd openvino_35d22
git checkout 35d22f2a0e
git submodule update --init --recursive

# 应用 PTL iGPU（Panther Lake）MoE 修复补丁
git apply /home/intel/project/qwen35/Qwen3.5_PTL_test/ov_patch/ptl_igpu_moe_fix.patch

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_ONEDNN_FOR_GPU=ON \
      ..

# 完整构建（所有插件）
make -j$(nproc)

# 或者只构建 GPU 插件（速度更快）：
# make -j$(nproc) openvino_intel_gpu_plugin
```

构建完成后设置环境变量：
```bash
export PYTHONPATH="/home/intel/project/qwen35/Qwen3.5_PTL_test/openvino_35d22/bin/intel64/Release/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/home/intel/project/qwen35/Qwen3.5_PTL_test/openvino_35d22/bin/intel64/Release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

---

### 4.3 验证 GPU 设备

```bash
python -c "
import openvino as ov
core = ov.Core()
print(core.get_property('GPU', 'FULL_DEVICE_NAME'))
"
```


预期输出（本机 ARL）：
```
Intel(R) Arc(TM) Graphics (iGPU)
```

---

## 五、功能验证（Functional Sanity Test）

环境搭建完成后，在正式 benchmark 前用 `test_qwen3.5.py` 验证模型能正确加载并推理。

```bash
cd /home/intel/project/qwen35/Qwen3.5_PTL_test
python test_qwen3.5.py
```

脚本通过 `OVModelForVisualCausalLM` 加载模型，在 GPU 上编译，并运行一次样例推理。关键说明：
- 脚本传入 `ov_config={"CACHE_DIR": ""}` 禁用模型缓存，避免 GPU plugin 动态 shape 处理的已知问题。
- 修改脚本中的 `model_dir` 和 `device` 变量可测试不同模型或设备。

验证所有组件均在 GPU.0 上运行：
```
Execution devices:
  - language_model: ['GPU.0']
  - vision_embeddings: ['GPU.0']
  - vision_embeddings_merger: ['GPU.0']
  - vision_embeddings_pos: ['GPU.0']
```

> **注意（本机 ARL）：** 运行 35B-A3B 时如遇 `CL_OUT_OF_RESOURCES` 崩溃，说明 iGPU 显存不足，
> 改用 `device = "AUTO:GPU,CPU"` 或 `device = "CPU"`。

---

## 六、遇到的问题和解决步骤

### 问题1：`No module named 'transformers.onnx'`

**原因：** `optimum-intel` 从 `transformers.onnx.utils` 导入 `ParameterFormat`、
`compute_serialized_parameters_size`，但 transformers ≥ 4.40 已删除该模块。

**修复文件：**  
`/home/intel/miniforge3/envs/jie/lib/python3.12/site-packages/optimum/intel/openvino/utils.py`

将：
```python
from transformers.onnx.utils import ParameterFormat, compute_serialized_parameters_size
```
改为：
```python
try:
    from transformers.onnx.utils import ParameterFormat, compute_serialized_parameters_size
except ImportError:
    class ParameterFormat:
        Float = 4  # bytes per float32 parameter

    def compute_serialized_parameters_size(num_parameters, data_type):
        size = data_type if isinstance(data_type, int) else 4
        return num_parameters * size
```

---

### 问题2：`cannot import name 'is_offline_mode' from 'transformers.utils'`

**原因：** transformers 5.x 将 `is_offline_mode` 移到了 `huggingface_hub`。

**修复文件：**  
`/home/intel/miniforge3/envs/jie/lib/python3.12/site-packages/optimum/intel/openvino/modeling_base.py`

将：
```python
from transformers.file_utils import add_start_docstrings
from transformers.generation import GenerationMixin
from transformers.utils import is_offline_mode
from transformers.utils.hub import cached_file
```
改为：
```python
from transformers.generation import GenerationMixin
try:
    from transformers.utils import is_offline_mode
except ImportError:
    from huggingface_hub import is_offline_mode
try:
    from transformers.file_utils import add_start_docstrings
except ImportError:
    from transformers.utils import add_start_docstrings
from transformers.utils.hub import cached_file
```

---

### 问题3：`No module named 'optimum.exporters.onnx'`

**原因：** optimum 2.x 将 onnx exporters 拆分到独立包 `optimum-onnx`，
但 requirements.txt 里的 optimum-onnx commit hash 与 optimum-intel 依赖的 branch URL 冲突，导致无法安装。

**解决方法：** 安装时分两步，让 optimum-intel 自动拉取 optimum-onnx。见下方安装步骤。

---

### 问题4：`cannot import name '_CAN_RECORD_REGISTRY' from 'transformers.utils.generic'`

**原因：** optimum 的 `_traceable_decorator.py` 使用了 transformers 5.3+ 才有的内部 API。

**修复文件：**  
`/home/intel/miniforge3/envs/jie/lib/python3.12/site-packages/optimum/exporters/onnx/_traceable_decorator.py`

将：
```python
from transformers.utils.generic import _CAN_RECORD_REGISTRY, OutputRecorder, logger
```
改为：
```python
try:
    from transformers.utils.generic import _CAN_RECORD_REGISTRY, OutputRecorder, logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

    _CAN_RECORD_REGISTRY = {}

    class OutputRecorder:
        def __init__(self, target_class=None, index=0, class_name=None, layer_name=None):
            self.target_class = target_class
            self.index = index
            self.class_name = class_name
            self.layer_name = layer_name
```

---

### 问题5：`ValueError: The checkpoint has model type 'qwen3_5_moe' but Transformers does not recognize this architecture`

**原因：** transformers 5.2.0 已支持 `qwen3_5_moe`，但 `AutoProcessor` 通过
`AutoConfig` 加载时触发了一个路径绕过了注册表。实际上安装正确版本的 transformers 后此问题自动消失。

---

### 问题6：`position_ids shape [3,1,472] incompatible with model port shape [4,?,?]`

**原因：** OV 模型编译时 `position_ids` 期望 4D shape `[4, batch, seq]`：
- dim 0：text position ids
- dim 1-3：mrope（temporal, height, width）

但 `get_rope_index()` 返回的是 `[3, batch, seq]`，`prepare_inputs` 中只把 2D 扩展为 3D，
没有再扩展到 4D。

**修复文件：**  
`/home/intel/miniforge3/envs/jie/lib/python3.12/site-packages/optimum/intel/openvino/modeling_visual_language.py`

在第193行之后添加：
```python
# Qwen3_5_moe OV model 期望 position_ids shape [4, batch, seq]
# dim0 = text_position_ids，dim1-3 = mrope（t, h, w）
# 将 [3, batch, seq] 在 dim0 复制一份 text_position_ids，变为 [4, batch, seq]
if self.config.model_type == "qwen3_5_moe" and position_ids.ndim == 3 and position_ids.shape[0] == 3:
    position_ids = np.concatenate([position_ids[:1], position_ids], axis=0)
```

---

## 七、requirements.txt 修复

原始文件有以下错误，需修复：

| 问题 | 修复 |
|---|---|
| `openvino openvino-tokenizers nncf --extra-index-url ...` 多包写一行 | 每包单独一行，`--extra-index-url` 单独一行 |
| `openvino-genai==2026.0.0.0` 与 `openvino-tokenizers==2026.2.x` 版本系列冲突 | 删除 `openvino-genai` 行 |
| `optimum-onnx` 显式 pin 的 commit hash 与 optimum-intel 依赖的 branch URL 冲突 | 删除 `optimum-onnx` 行，让 optimum-intel 自动拉取 |
| `transformers==5.2.0` 与 `optimum-intel` 元数据 `transformers<5.1` 冲突 | 分两步安装绕过（见下方） |

具体修改内容：

**1. 将多包单行拆分并删除冲突包：**

原始写法（错误）：
```
openvino openvino-tokenizers nncf --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
openvino-genai==2026.0.0.0
openvino-telemetry==2025.2.0
openvino-tokenizers==2026.2.0.0.dev20260402
optimum @ git+https://github.com/huggingface/optimum@3db78a41a715a04d4629e21cec4e7b1790c8266f
optimum-intel @ git+https://github.com/rkazants/optimum-intel.git@4602e000f4ca2c0e04a03c3633d30703b6bb0b05
optimum-onnx @ git+https://github.com/huggingface/optimum-onnx.git@7c9ccd7331403147c6b2a9f9440cd31f470a08f4
```

修改后（正确）：
```
--extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
openvino==2026.2.0.dev20260402
openvino-telemetry==2025.2.0
openvino-tokenizers==2026.2.0.0.dev20260402
optimum @ git+https://github.com/huggingface/optimum@3db78a41a715a04d4629e21cec4e7b1790c8266f
optimum-intel @ git+https://github.com/rkazants/optimum-intel.git@4602e000f4ca2c0e04a03c3633d30703b6bb0b05
```

删除说明：
- `openvino-genai==2026.0.0.0`：版本系列是 2026.0，与 openvino-tokenizers 的 2026.2 系列冲突，且本机不需要
- `optimum-onnx @ git+...`：显式 pin 的 commit hash 与 optimum-intel 自动依赖的 branch URL 被 pip 视为两个不同来源而冲突，删除后由 optimum-intel 自动拉取

**2. transformers 版本降级声明（临时）：**

安装前将 requirements.txt 中：
```
transformers==5.2.0
```
暂时改为：
```
transformers>=4.45,<5.1
```
待第一步安装完成后，再单独升级（见下方步骤）。

---

## 八、完整安装步骤

### 步骤1：修复 requirements.txt

按照上方"四、requirements.txt 修复"中的说明修改文件，然后将 `transformers==5.2.0` **临时**改为：
```
transformers>=4.45,<5.1
```

### 步骤2：安装所有非 git 包

```bash
grep -v "git+" requirements.txt | pip install -r /dev/stdin
```

安装结束会出现以下版本兼容警告，**可以忽略**：
```
optimum-intel requires transformers<4.58,>=4.45, but you have transformers 5.2.0
optimum-onnx requires transformers<4.58.0,>=4.36, but you have transformers 5.2.0
```

### 步骤3：将 transformers 升级到 5.2.0

```bash
pip install transformers==5.2.0 --no-deps
```

> `--no-deps`：只升级 transformers 本身，跳过依赖版本约束检查。

### 步骤4：用 `--no-deps` 安装 git 包（跳过版本约束检查）

```bash
pip install --no-deps \
  "optimum @ git+https://github.com/huggingface/optimum@3db78a41a715a04d4629e21cec4e7b1790c8266f" \
  "optimum-intel @ git+https://github.com/rkazants/optimum-intel.git@4602e000f4ca2c0e04a03c3633d30703b6bb0b05"
```

> `--no-deps` 的作用：只安装代码本身，不让 pip 重新验证 `transformers<5.1` 约束。  
> optimum-onnx 会由 optimum-intel 自动拉取 branch `xadupre/transformers5`。

### 步骤5：将 requirements.txt 中 transformers 还原

将之前临时改的 `transformers>=4.45,<5.1` 改回：
```
transformers==5.2.0
```

### 步骤6：打补丁（修复与 transformers 5.2.0 的 API 不兼容）

按照上方"三、遇到的问题和解决步骤"中的修复逐一操作，共 4 个文件。

### 步骤7：验证导入

```bash
python -c "from optimum.intel.openvino import OVModelForVisualCausalLM; print('OK')"
```

期望输出：
```
Multiple distributions found for package optimum. Picked distribution: optimum
OK
```

### 步骤8：运行测试

```bash
cd /home/intel/project/qwen35/Qwen3.5_PTL_test
python test_qwen3.5.py
```

---

## 九、注意事项

- **GPU 显存**：当前机器 iGPU 显存较小（Arrow Lake-P），如果遇到 `CL_OUT_OF_RESOURCES` 崩溃，
  说明显存不足，改用 `device = "CPU"` 或 `device = "AUTO:GPU,CPU"`。
- **模型缓存**：首次运行会编译 GPU kernel，较慢。如遇编译问题可设置 `ov_config={"CACHE_DIR": ""}` 禁用缓存。
- **视频帧警告**：`Asked to sample fps frames per second but no video metadata was provided` 为正常提示，不影响结果。
- **`Multiple distributions found for package optimum`**：因为同时存在 PyPI 版和 git 版，为正常提示，不影响运行。

---

## 十、成功输出示例

```
Compiling model on device: GPU
✓ Model compiled successfully on GPU
Configured device: GPU
Execution devices:
  - language_model: ['GPU.0']
  - vision_embeddings: ['GPU.0']
  - vision_embeddings_merger: ['GPU.0']
  - vision_embeddings_pos: ['GPU.0']
user
Why is this video funny?
assistant
<think>
The user wants to know why the video is funny.
...
</think>
```

---

## 十一、性能基准测试

### 8.1 Text-only LLM 基准（4.1 Text-only LLM benchmark）

直接调用 benchmark Python 脚本（脚本内部路径为旧机器，不能直接用 bash 脚本）：

```bash
cd /home/intel/project/qwen35/Qwen3.5_PTL_test

python benchmark_qwen3_5_openvino.py \
  --model-xml /home/intel/project/qwen35/Qwen3.5-35B-A3B/INT4/openvino_language_model.xml \
  --device GPU \
  --batch 1 \
  --seq-len 1024 \
  --decode-tokens 64 \
  --warmup 3 \
  --iters 5 \
  --num-streams 1
```

> **注意：** 脚本原始代码缺少 `position_ids` 输入（qwen3_5_moe 模型要求 `[4, batch, seq]`），
> 已在 `benchmark_qwen3_5_openvino.py` 中补充，自动检测模型输入并构造正确形状的 `position_ids`。

#### 测试结果（intel-AXMB-D150-3, GPU=Intel Arc Graphics Arrow Lake-P iGPU）

**配置1：SEQ_LEN=1024, DECODE_TOKENS=64, ITERS=5**
```
=== Qwen3.5 OpenVINO Runtime Benchmark ===
Model:   /home/intel/project/qwen35/Qwen3.5-35B-A3B/INT4/openvino_language_model.xml
Device:  GPU
Batch:   1
SeqLen:  1024
Hidden:  2048
Decode:  64 tokens
Warmup:  3
Iters:   5
--- Metrics ---
mean Prefill:    33248.591 ms  (std 146.780, p50 33189.839, p90 33423.059, p95 33430.979)
mean TTFT:       33317.229 ms  (std 146.771, p50 33260.049, p90 33491.366, p95 33499.146)
mean TPOT:          51.807 ms  (std   0.232, p50    51.796, p90    52.047, p95    52.074)
Throughput:           1.75 tokens/s  (std 0.01, p50 1.75, p90 1.76, p95 1.76)
```

**配置2：SEQ_LEN=1024, DECODE_TOKENS=512, ITERS=10**
```bash
python benchmark_qwen3_5_openvino.py \
  --model-xml /home/intel/project/qwen35/Qwen3.5-35B-A3B/INT4/openvino_language_model.xml \
  --device GPU --batch 1 --seq-len 1024 --decode-tokens 512 --warmup 3 --iters 10 --num-streams 1
```
```
=== Qwen3.5 OpenVINO Runtime Benchmark ===
Model:   /home/intel/project/qwen35/Qwen3.5-35B-A3B/INT4/openvino_language_model.xml
Device:  GPU
Batch:   1
SeqLen:  1024
Hidden:  2048
Decode:  512 tokens
Warmup:  3
Iters:   10
--- Metrics ---
mean Prefill:    33411.574 ms  (std 470.545, p50 33221.767, p90 33888.373, p95 34270.838)
mean TTFT:       33484.116 ms  (std 471.006, p50 33293.329, p90 33963.128, p95 34344.954)
mean TPOT:          55.837 ms  (std   0.078, p50    55.843, p90    55.895, p95    55.949)
Throughput:           8.26 tokens/s  (std 0.06, p50 8.28, p90 8.31, p95 8.31)
```

#### 指标说明

| 指标 | 定义 |
|---|---|
| Prefill | 处理全部输入序列的一次前向传播耗时（填充 KV cache） |
| TTFT | Time To First Token = Prefill 延迟 + 第1个 decode step 延迟 |
| TPOT | Time Per Output Token = decode step 2..N 的平均延迟（不含第1步） |
| Throughput | 总生成 token 数 / 总耗时（prefill + 所有 decode steps） |

---

### 8.2 Multimodal 基准（4.2 Multimodal benchmark）

端到端图片 + 文字推理，通过 optimum-intel 完整流水线测试：

```bash
cd /home/intel/project/qwen35/Qwen3.5_PTL_test

python benchmark_qwen3_5_mm_realtext.py \
  --model-dir /home/intel/project/qwen35/Qwen3.5-35B-A3B/INT4 \
  --image /home/intel/project/qwen35/Qwen3.5_PTL_test/test_image.jpg \
  --device GPU \
  --input-text-tokens 1024 \
  --new-tokens 64 \
  --warmup 3 \
  --iters 5 \
  --num-streams 1
```

#### 测试结果（intel-AXMB-D150-3, GPU=Intel Arc Graphics Arrow Lake-P iGPU）

**配置1：INPUT_TEXT_TOKENS=1024, NEW_TOKENS=64, ITERS=5**
```
=== Qwen3.5 Multimodal Benchmark (Real Text Tokens) ===
Model dir:          /home/intel/project/qwen35/Qwen3.5-35B-A3B/INT4
Device:             GPU
Image:              /home/intel/project/qwen35/Qwen3.5_PTL_test/test_image.jpg (800x577, W×H)
User text tokens:   1024 (exact before chat template)
Prompt tokens:      1408 (after chat template)
Generated tokens:   64
Warmup / Iters:     3 / 5
NUM_STREAMS:        1
Execution devices:
  - language_model: ['GPU.0']
  - vision_embeddings: ['GPU.0']
  - vision_embeddings_merger: ['GPU.0']
  - vision_embeddings_pos: ['GPU.0']
--- Metrics ---
mean TTFT:    40227.451 ms  (std 497.703, p50 40311.882, p90 40769.415, p95 40866.450)
mean TPOT:       80.160 ms  (std   1.499, p50    80.670, p90    81.461, p95    81.470)
Throughput:        1.41 tokens/s  (std 0.02, p50 1.41, p90 1.43, p95 1.44)
Generated tokens/iter (observed): min=64, max=64, mean=64.00
```

**配置2：INPUT_TEXT_TOKENS=1024, NEW_TOKENS=512, ITERS=10**
```bash
python benchmark_qwen3_5_mm_realtext.py \
  --model-dir /home/intel/project/qwen35/Qwen3.5-35B-A3B/INT4 \
  --image /home/intel/project/qwen35/Qwen3.5_PTL_test/test_image.jpg \
  --device GPU --input-text-tokens 1024 --new-tokens 512 --warmup 3 --iters 10 --num-streams 1
```
```
=== Qwen3.5 Multimodal Benchmark (Real Text Tokens) ===
Model dir:          /home/intel/project/qwen35/Qwen3.5-35B-A3B/INT4
Device:             GPU
Image:              /home/intel/project/qwen35/Qwen3.5_PTL_test/test_image.jpg (800x577, W×H)
User text tokens:   1024 (exact before chat template)
Prompt tokens:      1408 (after chat template)
Generated tokens:   512
Warmup / Iters:     3 / 10
NUM_STREAMS:        1
Execution devices:
  - language_model: ['GPU.0']
  - vision_embeddings: ['GPU.0']
  - vision_embeddings_merger: ['GPU.0']
  - vision_embeddings_pos: ['GPU.0']
--- Metrics ---
mean TTFT:    40214.586 ms  (std 528.558, p50 40029.695, p90 40736.425, p95 41060.453)
mean TPOT:       83.168 ms  (std   6.028, p50    80.817, p90    90.934, p95    94.714)
Throughput:        6.20 tokens/s  (std 0.22, p50 6.30, p90 6.38, p95 6.38)
Generated tokens/iter (observed): min=512, max=512, mean=512.00
```

> **提示：** `The following generation flags are not valid and may be ignored: ['top_p', 'top_k']` 为正常警告，不影响结果。

#### 汇总对比

| 测试模式 | SeqLen / Prompt tokens | New tokens | Warmup/Iters | TTFT (mean) | TPOT (mean) | Throughput |
|---|---|---|---|---|---|---|
| Text-only (LLM) | 1024 | 64 | 3/5 | 33317 ms | 51.8 ms | 1.75 tok/s |
| Text-only (LLM) | 1024 | 512 | 3/10 | 33484 ms | 55.8 ms | 8.26 tok/s |
| Multimodal (image+text) | 1408 | 64 | 3/5 | 40227 ms | 80.2 ms | 1.41 tok/s |
| Multimodal (image+text) | 1408 | 512 | 3/10 | 40215 ms | 83.2 ms | 6.20 tok/s |
