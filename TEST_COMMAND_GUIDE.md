Qwen3.5 OpenVINO Test Command Guide

Scope
- Workspace: /home/intel/jie
- Python env: /home/intel/jie/Env/bin/python
- Target device in examples: GPU

Quick Start Order
1. Functional sanity test (multimodal chat)
2. Multimodal performance benchmark
3. Text LLM runtime benchmark

1) Functional Sanity Test
Purpose
- Verify model loading and generation path are healthy.
- Script prints configured device and per-component execution devices.

Command
python test_qwen3.5.py

What to check
- Script starts and returns an assistant response.
- Execution devices show GPU.0 for expected components.

2) Multimodal Performance Benchmark
Script
- test_qwen3.5_openvino_mm_perf.sh

Purpose
- Measure TTFT, TPOT, throughput with image + text input.
- Export per-iteration metrics to CSV.

Command
DEVICE=GPU INPUT_TEXT_TOKENS=1024 NEW_TOKENS=64 WARMUP=3 ITERS=5 NUM_STREAMS=1 CSV_OUT=./mm_perf_1k.csv bash test_qwen3.5_openvino_mm_perf.sh

Output
- Console summary: mean TTFT, mean TPOT, throughput.
- CSV file: ./mm_perf_1k.csv

Main parameters
- DEVICE: GPU or AUTO:GPU,CPU
- INPUT_TEXT_TOKENS: user text token length before chat template
- NEW_TOKENS: generated token count
- WARMUP: warmup rounds
- ITERS: measured rounds
- NUM_STREAMS: OpenVINO stream setting
- CSV_OUT: output CSV path

3) Text LLM Runtime Benchmark
Script
- test_qwen3.5_openvino_perf.sh

Purpose
- Measure text model prefill/decode latency and throughput.

Command
DEVICE=GPU BATCH=1 SEQ_LEN=1024 DECODE_TOKENS=64 WARMUP=3 ITERS=5 NUM_STREAMS=1 bash test_qwen3.5_openvino_perf.sh

Output
- Console summary of latency and throughput metrics.

Main parameters
- DEVICE: GPU or CPU
- BATCH: batch size
- SEQ_LEN: prefill sequence length
- DECODE_TOKENS: decode token steps
- WARMUP: warmup rounds
- ITERS: measured rounds
- NUM_STREAMS: OpenVINO stream setting

Recommended run matrix
- Baseline latency: NUM_STREAMS=1
- Throughput comparison: NUM_STREAMS=2,4 and compare metrics
- Stability check: increase ITERS to 50

Troubleshooting
- If GPU compile fails and script falls back, read execution-device lines first.
- If numbers are noisy, increase WARMUP and ITERS.
- Keep model, prompt length, and image fixed when comparing runs.