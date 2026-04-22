#!/usr/bin/env python3
import argparse
import csv
import statistics
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.generation.streamers import BaseStreamer
from optimum.intel.openvino import OVModelForVisualCausalLM


class TimingTokenStreamer(BaseStreamer):
    def __init__(self, prompt_token_len: int):
        self.prompt_token_len = prompt_token_len
        self._prompt_seen = 0
        self.generated_tokens = 0
        self.first_token_time: float | None = None

    def put(self, value):
        if isinstance(value, torch.Tensor):
            tokens = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, np.ndarray):
            tokens = np.array(value).reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            tokens = list(value)
        else:
            return

        now = time.perf_counter()
        for _ in tokens:
            if self._prompt_seen < self.prompt_token_len:
                self._prompt_seen += 1
                continue
            if self.first_token_time is None:
                self.first_token_time = now
            self.generated_tokens += 1

    def end(self):
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3.5 multimodal benchmark (image + exact text tokens) with mean TTFT/TPOT"
    )
    parser.add_argument(
        "--model-dir",
        default="/home/intel/jie/Qwen3.5-35B-A3B-INT4",
        help="Qwen3.5 OpenVINO model directory",
    )
    parser.add_argument(
        "--image",
        default="/home/intel/jie/test_image.jpg",
        help="Input image path",
    )
    parser.add_argument("--device", default="GPU", help="GPU/CPU/NPU")
    parser.add_argument("--input-text-tokens", type=int, default=1024, help="Exact user text token count")
    parser.add_argument("--new-tokens", type=int, default=64, help="Generated token count")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations")
    parser.add_argument(
        "--base-prompt",
        default="Please describe this image in detail and summarize key objects and relations.",
        help="Base text for constructing exact token-length prompt",
    )
    parser.add_argument("--num-streams", default="1", help="OpenVINO NUM_STREAMS")
    parser.add_argument(
        "--strict-device",
        action="store_true",
        help="Do not fall back to AUTO when GPU compile fails",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV output path for per-iteration TTFT/TPOT metrics",
    )
    return parser.parse_args()


def percentile_ms(values: list[float], p: float) -> float:
    return float(np.percentile(np.array(values, dtype=np.float64), p))


def summarize(values: list[float]) -> tuple[float, float, float, float, float]:
    mean_v = statistics.fmean(values)
    std_v = statistics.pstdev(values) if len(values) > 1 else 0.0
    p50_v = percentile_ms(values, 50)
    p90_v = percentile_ms(values, 90)
    p95_v = percentile_ms(values, 95)
    return mean_v, std_v, p50_v, p90_v, p95_v


def build_exact_text(tokenizer, base_prompt: str, n_tokens: int) -> str:
    ids = tokenizer(base_prompt, add_special_tokens=False).input_ids
    if len(ids) == 0:
        raise RuntimeError("Base prompt tokenized to zero tokens")

    if len(ids) < n_tokens:
        repeat = (n_tokens + len(ids) - 1) // len(ids)
        ids = (ids * repeat)[:n_tokens]
    else:
        ids = ids[:n_tokens]

    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def write_csv(
    csv_path: Path,
    rows: list[dict],
    device: str,
    model_dir: Path,
    image_path: Path,
    input_text_tokens: int,
    prompt_tokens: int,
    requested_new_tokens: int,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["device", device])
        writer.writerow(["model_dir", str(model_dir)])
        writer.writerow(["image", str(image_path)])
        writer.writerow(["input_text_tokens", input_text_tokens])
        writer.writerow(["prompt_tokens", prompt_tokens])
        writer.writerow(["requested_new_tokens", requested_new_tokens])
        writer.writerow([])
        writer.writerow(
            [
                "iteration",
                "ttft_ms",
                "tpot_ms",
                "throughput_tokens_per_s",
                "generated_tokens",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["iteration"],
                    f"{r['ttft_ms']:.6f}",
                    f"{r['tpot_ms']:.6f}",
                    f"{r['throughput_toks_s']:.6f}",
                    r["generated_tokens"],
                ]
            )


def main() -> None:
    args = parse_args()
    if args.input_text_tokens <= 0 or args.new_tokens <= 0 or args.warmup < 0 or args.iters <= 0:
        raise ValueError("input-text-tokens/new-tokens/iters must be >0 and warmup >= 0")

    model_dir = Path(args.model_dir)
    image_path = Path(args.image)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)

    ov_config = {"CACHE_DIR": ""}
    if args.num_streams is not None:
        ov_config["NUM_STREAMS"] = args.num_streams

    effective_device = args.device

    def load_model_for_device(device_name: str) -> OVModelForVisualCausalLM:
        return OVModelForVisualCausalLM.from_pretrained(
            str(model_dir),
            device=device_name,
            ov_config=ov_config,
            trust_remote_code=True,
            compile=False,
        )

    try:
        model = load_model_for_device(args.device)
        model.compile()
    except RuntimeError as exc:
        # Some OpenVINO GPU plugin/runtime combinations fail to compile this VLM.
        # Automatically fall back to AUTO so CPU can be used for unsupported parts.
        can_fallback = (
            not args.strict_device
            and args.device.upper() == "GPU"
            and "unordered_map::at" in str(exc)
        )
        if not can_fallback:
            raise

        fallback_device = "AUTO:GPU,CPU"
        print(
            "WARNING: compile failed on GPU with 'unordered_map::at'. "
            f"Retrying with {fallback_device}."
        )
        model = load_model_for_device(fallback_device)
        model.compile()
        effective_device = fallback_device

    def _read_execution_devices(obj):
        if obj is None:
            return None
        try:
            if hasattr(obj, "get_compiled_model"):
                compiled = obj.get_compiled_model()
            else:
                compiled = obj
            return compiled.get_property("EXECUTION_DEVICES")
        except Exception:
            return None

    component_exec_devices: dict[str, object] = {}
    for comp_name in [
        "language_model",
        "vision_embeddings",
        "vision_embeddings_merger",
        "vision_embeddings_pos",
    ]:
        comp = getattr(model, comp_name, None)
        req = getattr(comp, "request", None) if comp is not None else None
        devices = _read_execution_devices(req)
        if devices is not None:
            component_exec_devices[comp_name] = devices

    user_text = build_exact_text(processor.tokenizer, args.base_prompt, args.input_text_tokens)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt")

    prompt_token_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0

    print("=== Qwen3.5 Multimodal Benchmark (Real Text Tokens) ===")
    print(f"Model dir:          {model_dir}")
    print(f"Device:             {effective_device}")
    print(f"Image:              {image_path}")
    print(f"User text tokens:   {args.input_text_tokens} (exact before chat template)")
    print(f"Prompt tokens:      {prompt_token_len} (after chat template)")
    print(f"Generated tokens:   {args.new_tokens}")
    print(f"Warmup / Iters:     {args.warmup} / {args.iters}")
    print(f"NUM_STREAMS:        {args.num_streams}")
    if component_exec_devices:
        print("Execution devices:")
        for name, devices in component_exec_devices.items():
            print(f"  - {name}: {devices}")

    def run_once(measure: bool):
        streamer = TimingTokenStreamer(prompt_token_len)
        result = {}

        def _target():
            result["out"] = model.generate(
                **inputs,
                max_new_tokens=args.new_tokens,
                do_sample=False,
                streamer=streamer,
            )

        t0 = time.perf_counter()
        th = threading.Thread(target=_target, daemon=True)
        th.start()
        th.join()
        t1 = time.perf_counter()

        if not measure:
            return None

        if streamer.first_token_time is None:
            raise RuntimeError("No generated token observed; cannot compute TTFT/TPOT")

        total_ms = (t1 - t0) * 1000.0
        ttft_ms = (streamer.first_token_time - t0) * 1000.0
        gen_tokens = max(streamer.generated_tokens, 1)
        tpot_ms = ((t1 - streamer.first_token_time) * 1000.0) / max(gen_tokens - 1, 1)
        throughput = gen_tokens / (total_ms / 1000.0)
        return ttft_ms, tpot_ms, throughput, gen_tokens

    for _ in range(args.warmup):
        run_once(measure=False)

    ttft_all: list[float] = []
    tpot_all: list[float] = []
    throughput_all: list[float] = []
    gen_token_counts: list[int] = []
    per_iter_rows: list[dict] = []

    for idx in range(args.iters):
        ttft_ms, tpot_ms, throughput, gen_tokens = run_once(measure=True)
        ttft_all.append(ttft_ms)
        tpot_all.append(tpot_ms)
        throughput_all.append(throughput)
        gen_token_counts.append(gen_tokens)
        per_iter_rows.append(
            {
                "iteration": idx + 1,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "throughput_toks_s": throughput,
                "generated_tokens": gen_tokens,
            }
        )

    ttft_mean, ttft_std, ttft_p50, ttft_p90, ttft_p95 = summarize(ttft_all)
    tpot_mean, tpot_std, tpot_p50, tpot_p90, tpot_p95 = summarize(tpot_all)
    thr_mean, thr_std, thr_p50, thr_p90, thr_p95 = summarize(throughput_all)

    print("--- Metrics ---")
    print(f"mean TTFT: {ttft_mean:.3f} ms  (std {ttft_std:.3f}, p50 {ttft_p50:.3f}, p90 {ttft_p90:.3f}, p95 {ttft_p95:.3f})")
    print(f"mean TPOT: {tpot_mean:.3f} ms  (std {tpot_std:.3f}, p50 {tpot_p50:.3f}, p90 {tpot_p90:.3f}, p95 {tpot_p95:.3f})")
    print(f"Throughput: {thr_mean:.2f} tokens/s  (std {thr_std:.2f}, p50 {thr_p50:.2f}, p90 {thr_p90:.2f}, p95 {thr_p95:.2f})")
    print(f"Generated tokens/iter (observed): min={min(gen_token_counts)}, max={max(gen_token_counts)}, mean={statistics.fmean(gen_token_counts):.2f}")

    if args.csv_out is not None:
        csv_path = Path(args.csv_out)
        write_csv(
            csv_path=csv_path,
            rows=per_iter_rows,
            device=effective_device,
            model_dir=model_dir,
            image_path=image_path,
            input_text_tokens=args.input_text_tokens,
            prompt_tokens=prompt_token_len,
            requested_new_tokens=args.new_tokens,
        )
        print(f"CSV exported: {csv_path}")


if __name__ == "__main__":
    main()
