#!/usr/bin/env python3
import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone OpenVINO Runtime benchmark for Qwen3.5 OpenVINO language model"
    )
    parser.add_argument(
        "--model-xml",
        default="/home/intel/jie/Qwen3.5-35B-A3B-INT4/openvino_language_model.xml",
        help="Path to openvino_language_model.xml",
    )
    parser.add_argument("--device", default="GPU", help="Target device, e.g. GPU/CPU/NPU")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length used per infer")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Measured iterations")
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=32,
        help="Number of autoregressive decode steps used to compute TTFT/TPOT",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--num-streams",
        default=None,
        help="Optional OpenVINO NUM_STREAMS value (e.g. 1, 2, AUTO)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional OpenVINO CACHE_DIR for compiled model cache",
    )
    return parser.parse_args()


def percentile_ms(latencies_ms: list[float], p: float) -> float:
    return float(np.percentile(np.array(latencies_ms, dtype=np.float64), p))


def summarize_ms(values: list[float]) -> tuple[float, float, float, float, float]:
    mean_ms = statistics.fmean(values)
    std_ms = statistics.pstdev(values) if len(values) > 1 else 0.0
    p50_ms = percentile_ms(values, 50)
    p90_ms = percentile_ms(values, 90)
    p95_ms = percentile_ms(values, 95)
    return mean_ms, std_ms, p50_ms, p90_ms, p95_ms


def main() -> None:
    args = parse_args()

    if args.batch <= 0 or args.seq_len <= 0 or args.warmup < 0 or args.iters <= 0 or args.decode_tokens <= 0:
        raise ValueError("Invalid benchmark arguments: batch/seq-len/iters must be >0 and warmup >=0")

    model_xml = Path(args.model_xml)
    if not model_xml.exists():
        raise FileNotFoundError(f"Model XML not found: {model_xml}")

    rng = np.random.default_rng(args.seed)

    core = ov.Core()
    model = core.read_model(str(model_xml))

    inputs = {t.get_any_name(): t for t in model.inputs}
    required = {"attention_mask", "inputs_embeds", "beam_idx"}
    if not required.issubset(inputs.keys()):
        raise RuntimeError(
            f"Unexpected model inputs {list(inputs.keys())}, expected at least {sorted(required)}"
        )

    hidden_dim = int(inputs["inputs_embeds"].partial_shape[2].get_length())

    compile_config: dict[str, str] = {"CACHE_DIR": ""}
    if args.num_streams is not None:
        compile_config["NUM_STREAMS"] = args.num_streams
    if args.cache_dir is not None:
        compile_config["CACHE_DIR"] = args.cache_dir

    compiled = core.compile_model(model, args.device, compile_config)
    req = compiled.create_infer_request()

    prefill_attention_mask = np.ones((args.batch, args.seq_len), dtype=np.int64)
    prefill_inputs_embeds = rng.standard_normal((args.batch, args.seq_len, hidden_dim), dtype=np.float32)
    beam_idx = np.arange(args.batch, dtype=np.int32)
    decode_embeds = rng.standard_normal((args.batch, args.decode_tokens, hidden_dim), dtype=np.float32)

    # Build position_ids if the model requires it (e.g. qwen3_5_moe uses [4, batch, seq])
    position_ids_shape = None
    if "position_ids" in inputs:
        pos_shape = inputs["position_ids"].partial_shape
        pos_dim0 = pos_shape[0].get_length() if pos_shape[0].is_static else 3
        position_ids_shape = (pos_dim0, args.batch, args.seq_len)

    prefill_position_ids = (
        np.tile(np.arange(args.seq_len, dtype=np.int64)[None, None, :], (position_ids_shape[0], args.batch, 1))
        if position_ids_shape is not None else None
    )

    prefill_inputs = {
        "attention_mask": prefill_attention_mask,
        "inputs_embeds": prefill_inputs_embeds,
        "beam_idx": beam_idx,
    }
    if prefill_position_ids is not None:
        prefill_inputs["position_ids"] = prefill_position_ids

    print("=== Qwen3.5 OpenVINO Runtime Benchmark ===")
    print(f"Model:   {model_xml}")
    print(f"Device:  {args.device}")
    print(f"Batch:   {args.batch}")
    print(f"SeqLen:  {args.seq_len}")
    print(f"Hidden:  {hidden_dim}")
    print(f"Decode:  {args.decode_tokens} tokens")
    print(f"Warmup:  {args.warmup}")
    print(f"Iters:   {args.iters}")

    for _ in range(args.warmup):
        req.reset_state()
        req.infer(prefill_inputs)
        current_len = args.seq_len
        for step in range(args.decode_tokens):
            current_len += 1
            decode_inputs = {
                "attention_mask": np.ones((args.batch, current_len), dtype=np.int64),
                "inputs_embeds": decode_embeds[:, step : step + 1, :],
                "beam_idx": beam_idx,
            }
            if prefill_position_ids is not None:
                decode_inputs["position_ids"] = np.tile(
                    np.array([[[current_len - 1]]], dtype=np.int64),
                    (position_ids_shape[0], args.batch, 1)
                )
            req.infer(decode_inputs)

    prefill_latencies_ms: list[float] = []
    ttft_latencies_ms: list[float] = []
    tpot_latencies_ms: list[float] = []
    gen_throughput_toks_s: list[float] = []

    for _ in range(args.iters):
        req.reset_state()

        t0 = time.perf_counter()
        req.infer(prefill_inputs)
        t1 = time.perf_counter()
        prefill_ms = (t1 - t0) * 1000.0

        decode_latencies_ms: list[float] = []
        current_len = args.seq_len
        for step in range(args.decode_tokens):
            current_len += 1
            decode_inputs = {
                "attention_mask": np.ones((args.batch, current_len), dtype=np.int64),
                "inputs_embeds": decode_embeds[:, step : step + 1, :],
                "beam_idx": beam_idx,
            }
            if prefill_position_ids is not None:
                decode_inputs["position_ids"] = np.tile(
                    np.array([[[current_len - 1]]], dtype=np.int64),
                    (position_ids_shape[0], args.batch, 1)
                )
            td0 = time.perf_counter()
            req.infer(decode_inputs)
            td1 = time.perf_counter()
            decode_latencies_ms.append((td1 - td0) * 1000.0)

        ttft_ms = prefill_ms + decode_latencies_ms[0]
        tpot_ms = (
            statistics.fmean(decode_latencies_ms[1:])
            if len(decode_latencies_ms) > 1
            else decode_latencies_ms[0]
        )
        total_gen_ms = prefill_ms + sum(decode_latencies_ms)
        iter_throughput = (args.batch * args.decode_tokens) / (total_gen_ms / 1000.0)

        prefill_latencies_ms.append(prefill_ms)
        ttft_latencies_ms.append(ttft_ms)
        tpot_latencies_ms.append(tpot_ms)
        gen_throughput_toks_s.append(iter_throughput)

    prefill_mean, prefill_std, prefill_p50, prefill_p90, prefill_p95 = summarize_ms(prefill_latencies_ms)
    ttft_mean, ttft_std, ttft_p50, ttft_p90, ttft_p95 = summarize_ms(ttft_latencies_ms)
    tpot_mean, tpot_std, tpot_p50, tpot_p90, tpot_p95 = summarize_ms(tpot_latencies_ms)

    thr_mean = statistics.fmean(gen_throughput_toks_s)
    thr_std = statistics.pstdev(gen_throughput_toks_s) if len(gen_throughput_toks_s) > 1 else 0.0
    thr_p50 = percentile_ms(gen_throughput_toks_s, 50)
    thr_p90 = percentile_ms(gen_throughput_toks_s, 90)
    thr_p95 = percentile_ms(gen_throughput_toks_s, 95)

    print("--- Metrics ---")
    print(f"mean Prefill: {prefill_mean:.3f} ms  (std {prefill_std:.3f}, p50 {prefill_p50:.3f}, p90 {prefill_p90:.3f}, p95 {prefill_p95:.3f})")
    print(f"mean TTFT: {ttft_mean:.3f} ms  (std {ttft_std:.3f}, p50 {ttft_p50:.3f}, p90 {ttft_p90:.3f}, p95 {ttft_p95:.3f})")
    print(f"mean TPOT: {tpot_mean:.3f} ms  (std {tpot_std:.3f}, p50 {tpot_p50:.3f}, p90 {tpot_p90:.3f}, p95 {tpot_p95:.3f})")
    print(f"Throughput: {thr_mean:.2f} tokens/s  (std {thr_std:.2f}, p50 {thr_p50:.2f}, p90 {thr_p90:.2f}, p95 {thr_p95:.2f})")


if __name__ == "__main__":
    main()
