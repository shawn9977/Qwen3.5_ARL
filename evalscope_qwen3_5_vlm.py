#!/usr/bin/env python3
"""
Qwen3.5 VLM Evaluation via EvalScope + VLMEvalKit

Runs standard VLM benchmarks (MMMU, LongVideoBench) against Qwen3.5 models.
Supports two modes:
  - local: Starts a lightweight OpenAI-compatible API server wrapping OVModelForVisualCausalLM
  - api:   Connects to an existing vLLM or other model serving endpoint

Prerequisites:
  pip install 'evalscope[all]'

Usage:
  # Local mode (no vLLM needed)
  python evalscope_qwen3_5_vlm.py --mode local --model-path ./Qwen3.5-0.8B-INT4 --dataset MMMU_DEV_VAL --limit 20

  # API mode (connect to existing service)
  python evalscope_qwen3_5_vlm.py --mode api --api-url http://localhost:41091/v1/chat/completions --dataset MMMU_DEV_VAL

  # Use custom config
  python evalscope_qwen3_5_vlm.py --task-cfg configs/qwen3_5_vl_eval.yaml --dataset LongVideoBench
"""

import argparse
import json
import os
import sys
import time
import datetime
import signal
import base64
import io
import threading
from pathlib import Path


def seconds_to_hms(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


# ---------------------------------------------------------------------------
# Local model server (for --mode local)
# ---------------------------------------------------------------------------

def start_local_server(model_path, device, port, max_image_side=512):
    """
    Launch a minimal FastAPI server that wraps OVModelForVisualCausalLM
    as an OpenAI-compatible /v1/chat/completions endpoint.
    Returns (server_process, api_url).
    """
    try:
        import uvicorn
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError:
        print("ERROR: FastAPI and uvicorn are required for local mode.")
        print("Install: pip install fastapi uvicorn")
        sys.exit(1)

    from transformers import AutoProcessor
    from optimum.intel.openvino import OVModelForVisualCausalLM
    from PIL import Image

    app = FastAPI()

    # Load model once at startup
    print(f"Loading model from {model_path} on {device}...")
    processor = AutoProcessor.from_pretrained(model_path)
    model = OVModelForVisualCausalLM.from_pretrained(
        model_path, device=device, ov_config={"CACHE_DIR": ""}, compile=False
    )
    model.compile()
    print("Model loaded successfully.")

    class ChatRequest(BaseModel):
        model: str = "qwen3.5-vl"
        messages: list = []
        max_tokens: int = 81920
        temperature: float = 1.0

    class ChatResponse(BaseModel):
        id: str = "chatcmpl-local"
        object: str = "chat.completion"
        created: int = 0
        model: str = "qwen3.5-vl"
        choices: list = []

    def _resize_image(img, max_side):
        """Resize image if it exceeds max_side."""
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
        return img

    def _is_gpu_oom(exc):
        msg = str(exc)
        return any(s in msg for s in [
            "CL_OUT_OF_RESOURCES", "clWaitForEvents failed with -14",
            "clFinish, error code: -5",
        ])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        """Handle OpenAI-compatible chat completion requests with image support."""
        images = []
        user_text_parts = []
        system_prompt = None

        # Extract images and text from messages, preserving roles
        for msg in request.messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # System message: extract as system prompt (string only)
            if role == "system":
                if isinstance(content, str):
                    system_prompt = content
                continue

            # User/assistant message: extract images and text
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        user_text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            b64_data = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64_data)
                            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                            img = _resize_image(img, max_image_side)
                            images.append(img)
            elif isinstance(content, str):
                user_text_parts.append(content)

        # Use system prompt from request, or fallback to default
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. For multiple-choice questions, "
                "respond with only the single letter of the correct answer (e.g. A, B, C, or D)."
            )

        # Build multimodal message for Qwen3.5 processor
        content_parts = []
        for img in images:
            content_parts.append({"type": "image"})
        content_parts.append({"type": "text", "text": " ".join(user_text_parts)})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        # Retry with progressively smaller images on GPU OOM
        retry_sizes = [max_image_side, max_image_side // 2, 192]
        answer = "[INFERENCE_FAILED]"

        for attempt, cur_max_side in enumerate(retry_sizes):
            try:
                cur_images = [_resize_image(img.copy(), cur_max_side) for img in images] if images else []

                if cur_images:
                    inputs = processor(text=[text], images=cur_images, return_tensors="pt")
                else:
                    inputs = processor(text=[text], return_tensors="pt")

                # Use full max_tokens for high-quality evaluation
                # For memory-constrained environments, uncomment the line below:
                # max_tokens = min(request.max_tokens, 512)
                max_tokens = request.max_tokens

                # Sampling parameters aligned with vLLM API mode
                # Note: do NOT use repetition_penalty here — it suppresses
                # answer tokens (A/B/C/D) that appear in the question text,
                # causing wrong or verbose outputs.
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": request.temperature > 0,
                }
                if request.temperature > 0:
                    gen_kwargs["temperature"] = request.temperature
                    gen_kwargs["top_p"] = 0.95
                    gen_kwargs["top_k"] = 20

                output_ids = model.generate(**inputs, **gen_kwargs)
                input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
                answer = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                break
            except RuntimeError as exc:
                if _is_gpu_oom(exc) and attempt < len(retry_sizes) - 1:
                    print(f"  GPU OOM (attempt {attempt+1}), retrying with max_side={retry_sizes[attempt+1]}")
                    continue
                print(f"  Inference error: {exc}")
                answer = f"[ERROR: {str(exc)[:200]}]"
                break

        return ChatResponse(
            created=int(time.time()),
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }]
        )

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # Run server in a thread
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready (use socket to bypass corporate proxy)
    import socket
    for i in range(60):
        try:
            sock = socket.create_connection(("127.0.0.1", port), timeout=2)
            sock.sendall(f"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
            resp = sock.recv(1024)
            sock.close()
            if b"200" in resp:
                break
        except Exception:
            time.sleep(1)
    else:
        print("ERROR: Local server failed to start within 60 seconds")
        sys.exit(1)

    api_url = f"http://localhost:{port}/v1/chat/completions"
    print(f"Local model server ready at {api_url}")
    return server, api_url


# ---------------------------------------------------------------------------
# EvalScope benchmark runner
# ---------------------------------------------------------------------------

def run_evalscope_benchmark(task_cfg_path, dataset, output_dir, limit=None):
    """Run evaluation using EvalScope + VLMEvalKit backend."""
    try:
        from evalscope.config import parse_task_config
        from evalscope.run import run_task
        from evalscope.summarizer import Summarizer
        import vlmeval.run
        import vlmeval.dataset.mmbench_video
        import vlmeval.dataset.mlvu
    except ImportError:
        print("ERROR: evalscope is not installed.")
        print("Install: pip install 'evalscope[all]'")
        sys.exit(1)

    configs = parse_task_config(task_cfg_path)

    eval_updates = {"data": [dataset]}
    if limit is not None:
        eval_updates["limit"] = limit
    configs.eval_config.update(eval_updates)

    global_updates = {"work_dir": os.path.join(output_dir, dataset)}
    configs.update(global_updates)

    start = time.time()
    run_task(task_cfg=configs)
    eval_duration = time.time() - start

    print("\n>> Getting evaluation report...")
    report_list = Summarizer.get_report_from_cfg(task_cfg=configs)
    total_duration = time.time() - start

    print(f"\n>> Report: {report_list}")
    print(f"Duration: total {seconds_to_hms(total_duration)}, eval {seconds_to_hms(eval_duration)}")

    return report_list, total_duration


def convert_to_standard_format(report_list, dataset, output_dir):
    """
    Convert evalscope results to summary_metrics.json format
    compatible with compare_results.py.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract accuracy from report and CSV results.
    # MMMU_DEV_VAL has two splits: "dev" and "validation".
    # The report dict only contains "dev" split, but "validation" is the main benchmark.
    # We read the _acc.csv directly to get the validation split accuracy.
    accuracy = None
    num_samples = 0

    # Try to read accuracy from CSV (more reliable, has both splits)
    import csv
    import glob
    acc_csvs = glob.glob(os.path.join(output_dir, "**/*_acc.csv"), recursive=True)
    if acc_csvs:
        acc_csv = sorted(acc_csvs)[-1]  # latest
        with open(acc_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row.get("split", "")
                overall = row.get("Overall", "0")
                if split == "validation":
                    accuracy = float(overall)
                    break
                elif split == "dev" and accuracy is None:
                    accuracy = float(overall)
        print(f"Read accuracy from CSV: {acc_csv} -> {accuracy}")

    # Get num_samples from the prediction xlsx
    try:
        import pandas as pd
        xlsx_files = glob.glob(os.path.join(output_dir, "**/*_MMMU_DEV_VAL.xlsx"), recursive=True)
        if not xlsx_files:
            xlsx_files = glob.glob(os.path.join(output_dir, "**/*.xlsx"), recursive=True)
            xlsx_files = [f for f in xlsx_files if "_acc" not in f and "_result" not in f]
        if xlsx_files:
            pred_df = pd.read_excel(sorted(xlsx_files)[-1])
            num_samples = len(pred_df)
    except Exception:
        pass

    # Fallback to report dict
    if accuracy is None and report_list and len(report_list) > 0:
        report = report_list[0] if isinstance(report_list, list) else report_list
        if isinstance(report, dict):
            for key, value in report.items():
                if isinstance(value, dict) and "Overall" in value:
                    overall_val = value["Overall"]
                    accuracy = float(overall_val) if isinstance(overall_val, str) else overall_val
                    break
            if accuracy is None:
                accuracy = report.get("accuracy", report.get("score", report.get("overall", None)))

    # Normalize accuracy to 0-1 range if it's in percentage (>1)
    normalized_acc = None
    if accuracy is not None:
        normalized_acc = accuracy / 100.0 if accuracy > 1 else accuracy

    summary = {
        "mean_similarity": normalized_acc,
        "std_similarity": 0.0,
        "min_similarity": normalized_acc,
        "max_similarity": normalized_acc,
        "median_similarity": normalized_acc,
        "num_samples": num_samples,
        "benchmark": dataset,
        "raw_report": report_list,
        "image_mean_similarity": normalized_acc,
        "video_mean_similarity": None,
    }

    summary_path = os.path.join(output_dir, "summary_metrics.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3.5 VLM on standard benchmarks via EvalScope")
    parser.add_argument("--mode", type=str, choices=["local", "api"], default="local",
                        help="local: start model server locally; api: use existing endpoint")
    parser.add_argument("--model-path", type=str, default="./Qwen3.5-0.8B-INT4",
                        help="Local OpenVINO model path (for local mode)")
    parser.add_argument("--device", type=str, default="GPU",
                        help="Device for local model (GPU/CPU)")
    parser.add_argument("--api-url", type=str, default=None,
                        help="API URL (for api mode)")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Model name on vLLM server (for api mode, e.g. Qwen/Qwen3.5-9B)")
    parser.add_argument("--task-cfg", type=str, default="configs/qwen3_5_vl_eval.yaml",
                        help="EvalScope task config YAML")
    parser.add_argument("--dataset", type=str, default="MMMU_DEV_VAL",
                        help="Benchmark dataset (MMMU_DEV_VAL, LongVideoBench, etc.)")
    parser.add_argument("-o", "--output-dir", type=str, default="./accuracy_results_evalscope",
                        help="Output directory")
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="Limit number of evaluation samples")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image size for processing")
    parser.add_argument("--nframe", type=int, default=4,
                        help="Number of video frames")
    parser.add_argument("--port", type=int, default=8901,
                        help="Port for local model server")

    args = parser.parse_args()

    # Bypass corporate proxy for localhost and internal network
    os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,10.*"
    os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1,10.*"

    print("=" * 70)
    print("Qwen3.5 VLM Standard Benchmark Evaluation (EvalScope)")
    print("=" * 70)
    print(f"Mode:     {args.mode}")
    print(f"Dataset:  {args.dataset}")
    print(f"Output:   {args.output_dir}")
    if args.limit:
        print(f"Limit:    {args.limit} samples")

    # Determine API URL
    if args.mode == "local":
        print(f"Model:    {args.model_path}")
        print(f"Device:   {args.device}")
        server, api_url = start_local_server(
            args.model_path, args.device, args.port, args.img_size)
    elif args.mode == "api":
        if args.api_url is None:
            print("ERROR: --api-url is required for api mode")
            sys.exit(1)
        api_url = args.api_url
        # VLMEvalKit POSTs directly to api_base, so ensure full endpoint path
        if not api_url.endswith("/v1/chat/completions"):
            api_url = api_url.rstrip("/") + "/v1/chat/completions"
        server = None
        print(f"API URL:  {api_url}")

    # Always generate a config with the correct api_url and port
    # (the static YAML may have a stale port)
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = Path(args.output_dir) / "eval_config.yaml"
    # Local mode: long timeout (Intel GPU is slow), serial processing (single GPU)
    # API mode: shorter timeout (vLLM is fast), parallel processing
    if args.mode == "local":
        timeout_val = 600
        nproc_val = 1
        max_tokens_val = 128  # MCQ only needs a few tokens; saves GPU time
    else:
        timeout_val = 120
        nproc_val = 4
        max_tokens_val = 2048

    config_content = f"""work_dir: {args.output_dir}
timeout: {timeout_val}
eval_backend: VLMEvalKit
debug: true
eval_config:
  model:
    - type: Qwen3_5-VL
      name: CustomAPIModel
      api_base: {api_url}
      key: EMPTY
      temperature: 1.0
      timeout: {timeout_val}
      retry: 2
      fps: -1
      nframe: {args.nframe}
      img_size: {args.img_size}
      max_tokens: {max_tokens_val}
      video_llm: false
      system_prompt: "You are a helpful assistant. For multiple-choice questions, respond with only the single letter of the correct answer (e.g. A, B, C, or D). Do not explain."
  mode: all
  reuse: false
  nproc: {nproc_val}
"""
    config_path.write_text(config_content)
    print(f"Using config: {config_path} (api_base: {api_url})")

    # Run benchmark
    print(f"\nStarting {args.dataset} evaluation...")
    report_list, duration = run_evalscope_benchmark(
        str(config_path), args.dataset, args.output_dir, args.limit)

    # Convert to standard format
    convert_to_standard_format(report_list, args.dataset, args.output_dir)

    print(f"\nTotal time: {seconds_to_hms(duration)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
