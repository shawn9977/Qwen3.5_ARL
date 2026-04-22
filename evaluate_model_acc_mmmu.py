"""MMMU_DEV_VAL benchmark evaluation using a vLLM inference server.

Evaluates model accuracy on the MMMU (Massive Multidiscipline Multimodal
Understanding) benchmark using the dev + validation splits (1050 samples:
150 dev + 900 validation across 30 subjects).

Inference is performed by sending requests to a running vLLM server via
its OpenAI-compatible HTTP API (default: http://localhost:8000).

Usage:
    # Start the vLLM server first:
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct

    python evaluate_model_acc.py \\
        --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --vllm-url http://localhost:8000 \\
        --splits dev validation \\
        --output results_mmmu/

    # Evaluate only specific subjects
    python evaluate_model_acc.py --model Qwen/Qwen2.5-VL-7B-Instruct --subjects Math Physics

    # Limit samples for a quick smoke-test
    python evaluate_model_acc.py --model Qwen/Qwen2.5-VL-7B-Instruct --limit 50
"""
from __future__ import annotations

import argparse
import ast
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_BASE = Path(__file__).resolve().parent / "results_mmmu"

# ---------------------------------------------------------------------------
# MMMU constants
# ---------------------------------------------------------------------------
# All 30 subjects in the MMMU benchmark
MMMU_SUBJECTS: List[str] = [
    "Accounting", "Agriculture", "Architecture_and_Engineering",
    "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
    "Chemistry", "Clinical_Medicine", "Computer_Science", "Design",
    "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature",
    "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering",
    "Music", "Pharmacy", "Physics", "Psychology", "Public_Health",
    "Sociology",
]

# Six high-level disciplines for grouped reporting
MMMU_DISCIPLINES: Dict[str, List[str]] = {
    "Art & Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Health & Medicine": [
        "Basic_Medical_Science", "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health",
    ],
    "Humanities & Social Science": [
        "Geography", "History", "Literature", "Psychology", "Sociology",
    ],
    "Science": [
        "Agriculture", "Biology", "Chemistry", "Energy_and_Power",
        "Materials", "Physics",
    ],
    "Technology & Engineering": [
        "Architecture_and_Engineering", "Computer_Science", "Electronics",
        "Math", "Mechanical_Engineering",
    ],
}

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _parse_options(options_field) -> List[str]:
    """Parse the 'options' field which may be a list or a string repr of a list."""
    if isinstance(options_field, list):
        return [str(o) for o in options_field]
    if isinstance(options_field, str):
        try:
            parsed = ast.literal_eval(options_field)
            if isinstance(parsed, list):
                return [str(o) for o in parsed]
        except (ValueError, SyntaxError):
            pass
        # Fallback: split by common delimiters
        return [s.strip() for s in re.split(r'\n|;', options_field) if s.strip()]
    return []


def get_sample_images(sample: dict) -> List[Image.Image]:
    """Extract all non-None images (image_1 … image_7) from an MMMU sample."""
    images: List[Image.Image] = []
    for i in range(1, 8):
        key = f"image_{i}"
        val = sample.get(key)
        if val is None:
            continue
        if isinstance(val, Image.Image):
            images.append(val.convert("RGB"))
        else:
            try:
                images.append(Image.fromarray(np.array(val)).convert("RGB"))
            except Exception:
                pass
    return images


def resize_to_fixed(image: Image.Image, max_side: int) -> Image.Image:
    """Resize image so its longest side equals max_side, preserving aspect ratio.

    Useful for capping payload size sent to the vLLM server.
    """
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    if w >= h:
        new_w = max_side
        new_h = max(1, round(h * max_side / w))
    else:
        new_h = max_side
        new_w = max(1, round(w * max_side / h))
    return image.resize((new_w, new_h), Image.LANCZOS)


def build_mmmu_prompt(sample: dict, n_shot_examples: List[dict] = ()) -> str:
    """Build the VL prompt for one MMMU sample.

    Format mirrors the standard MMMU evaluation protocol:
        <image>
        Question text
        A. option1
        B. option2
        ...
        Answer with the option's letter from the given choices directly.

    For few-shot examples (optional) the format is:
        <image>
        Question text
        A. ...
        Answer: X

    Image placeholders (<image N>) in the question text are normalised to
    a single <image> tag so the VL model can locate the image embedding.
    """
    def _format_one(s, include_answer: bool = False) -> str:
        q = re.sub(r'<image\s*\d*>', '<image>', s["question"]).strip()
        opts = _parse_options(s["options"])
        letters = "ABCDEFGH"
        opt_lines: List[str] = []
        for idx, opt in enumerate(opts):
            letter = letters[idx] if idx < len(letters) else str(idx)
            # Options may already be prefixed with "A. …" — keep as-is
            if re.match(r'^[A-H][.)]\s', opt):
                opt_lines.append(opt)
            else:
                opt_lines.append(f"{letter}. {opt}")
        body = q + "\n" + "\n".join(opt_lines)
        if include_answer:
            body += f"\nAnswer: {s['answer'].strip().upper()}"
        return body

    parts: List[str] = []
    for ex in n_shot_examples:
        parts.append(_format_one(ex, include_answer=True))
    # Final test question
    parts.append(_format_one(sample, include_answer=False))
    parts.append("Answer with the option's letter from the given choices directly.")
    return "\n\n".join(parts)


def load_mmmu_dataset(
    splits: List[str],
    subjects: List[str],
) -> List[dict]:
    """Load MMMU samples from HuggingFace datasets.

    Returns a flat list of dicts, each augmented with '__subject__' and
    '__split__' keys.  Duplicate IDs within the same subject/split are
    de-duplicated.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        sys.exit(
            "ERROR: 'datasets' package not found.\n"
            "Install it with:  pip install datasets"
        )

    all_samples: List[dict] = []
    for subject in subjects:
        print(f"  [{subject}]", end=" ", flush=True)
        try:
            ds = load_dataset("MMMU/MMMU", subject, trust_remote_code=True)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue

        count = 0
        for split in splits:
            if split not in ds:
                continue
            for sample in ds[split]:
                row = dict(sample)
                row["__subject__"] = subject
                row["__split__"] = split
                all_samples.append(row)
                count += 1
        print(f"{count} samples")

    return all_samples


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def parse_mcq_answer(response: str, valid_letters: str = "ABCDEFGH") -> str:
    """Extract the answer letter from the model's response.

    Mirrors the multi-strategy approach in mmlu_redux.py, extended to A-H
    for MMMU's larger option sets.
    """
    if not response:
        return ""

    # Strip <think>…</think> blocks produced by reasoning models before parsing,
    # so stray option letters inside the chain-of-thought don't pollute extraction.
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    # If stripping left nothing (pure thinking response), fall back to the text
    # that appears after the last </think> tag.
    if not clean:
        m = re.search(r'</think>(.*)', response, re.DOTALL)
        clean = m.group(1).strip() if m else response

    vl = valid_letters  # "ABCDEFGH" or subset

    # Strategy 1: Explicit answer patterns (highest confidence — use last match)
    answer_patterns = [
        rf'(?:answer|Answer)\s*(?:is|:)\s*\**([{vl}])\**',
        rf'(?:the\s+answer|The\s+answer)\s+is\s+\**([{vl}])\**',
        rf'(?:correct\s+(?:answer|option))\s*(?:is|:)\s*\**([{vl}])\**',
        rf'(?:choose|select|pick)\s+\**([{vl}])\**',
        rf'\b([{vl}])\s+is\s+correct',
    ]
    for pattern in answer_patterns:
        matches = list(re.finditer(pattern, clean, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).upper()

    # Strategy 2: Response starts directly with the answer letter
    first = clean.strip()
    if first and first[0].upper() in vl and (len(first) == 1 or first[1] in ' \t\n\r.。,，)'):
        return first[0].upper()

    # Strategy 3: Last standalone valid letter in the response
    matches = list(re.finditer(rf'\b([{vl}])\b', clean))
    if matches:
        return matches[-1].group(1).upper()

    # Strategy 4: Last occurrence of any valid letter character
    for ch in reversed(clean):
        if ch.upper() in vl:
            return ch.upper()

    return ""


class _VLLMServerAdapter:
    """Sends inference requests to a running vLLM server (OpenAI-compatible HTTP API)."""

    def __init__(self, model_name: str, base_url: str = "http://localhost:8000"):
        from openai import OpenAI
        self._model = model_name
        self._client = OpenAI(base_url=f"{base_url.rstrip('/')}/v1", api_key="EMPTY")

    def generate(self, images: List[Image.Image], prompt: str, max_new_tokens: int) -> str:
        """Generate response from vLLM with multiple images.

        Args:
            images: List of PIL Image objects
            prompt: Text prompt with <image> placeholders
            max_new_tokens: Maximum tokens to generate
        """
        # Encode each image as base64 PNG data URL
        image_content = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img)
            else:
                pil_img = img
            
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            data_url = f"data:image/png;base64,{b64}"
            image_content.append({"type": "image_url", "image_url": {"url": data_url}})

        # Replace <image N> placeholders with "image N" for clarity
        text = re.sub(r'<image\s*(\d+)?>', lambda m: f"image {m.group(1)}" if m.group(1) else "images", prompt).strip()

        # Build message with all images
        user_content = image_content + [{"type": "text", "text": text}]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. For multiple-choice questions, "
                           "respond with only the single letter of the correct answer (e.g. A, B, C, or D).",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Results persistence helpers
# ---------------------------------------------------------------------------

def load_existing_answers(answers_file: Path) -> Dict[str, dict]:
    existing: Dict[str, dict] = {}
    if answers_file.exists():
        with open(answers_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        existing[entry["id"]] = entry
                    except json.JSONDecodeError:
                        pass
    return existing


def append_answer(answers_file: Path, entry: dict) -> None:
    with open(answers_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: List[dict],
) -> dict:
    """Compute per-subject, per-discipline, and overall accuracy."""
    per_subject: Dict[str, dict] = {}
    for pred in predictions:
        subj = pred["subject"]
        if subj not in per_subject:
            per_subject[subj] = {"correct": 0, "total": 0, "empty": 0}
        per_subject[subj]["total"] += 1
        if pred["predicted"] == pred["answer"]:
            per_subject[subj]["correct"] += 1
        if not pred["predicted"]:
            per_subject[subj]["empty"] += 1

    # Subject accuracies
    for s in per_subject.values():
        s["accuracy"] = s["correct"] / max(s["total"], 1)

    # Discipline (group) accuracies
    per_discipline: Dict[str, dict] = {}
    for disc, disc_subjects in MMMU_DISCIPLINES.items():
        correct = sum(per_subject.get(s, {}).get("correct", 0) for s in disc_subjects)
        total = sum(per_subject.get(s, {}).get("total", 0) for s in disc_subjects)
        per_discipline[disc] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / max(total, 1),
        }

    total_correct = sum(s["correct"] for s in per_subject.values())
    total_questions = sum(s["total"] for s in per_subject.values())
    empty_answers = sum(s["empty"] for s in per_subject.values())
    all_accs = [s["accuracy"] for s in per_subject.values()]
    macro_avg = sum(all_accs) / max(len(all_accs), 1)
    micro_avg = total_correct / max(total_questions, 1)

    return {
        "micro_avg": micro_avg,
        "macro_avg": macro_avg,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "empty_answers": empty_answers,
        "per_subject": per_subject,
        "per_discipline": per_discipline,
    }


def format_summary(metrics: dict, model_path: str,
                   splits: List[str], inference_time: Optional[float]) -> str:
    lines = [
        "=" * 70,
        "MMMU_DEV_VAL Evaluation Results",
        "=" * 70,
        f"  Model:        {model_path}",
        f"  Splits:       {', '.join(splits)}",
        f"  Questions:    {metrics['total_questions']}",
        f"  Correct:      {metrics['total_correct']}",
        f"  Empty:        {metrics['empty_answers']}",
    ]
    if inference_time is not None:
        per_q = inference_time / max(metrics["total_questions"], 1)
        lines.append(f"  Inference:    {inference_time:.1f}s total, {per_q:.1f}s/question")

    lines += [
        "",
        f"  Micro Average (overall):  {metrics['micro_avg']:.4f}"
        f"  ({metrics['total_correct']}/{metrics['total_questions']})",
        f"  Macro Average (per subj): {metrics['macro_avg']:.4f}",
        "",
        "  Per Discipline:",
    ]
    for disc, info in sorted(metrics["per_discipline"].items()):
        lines.append(
            f"    {disc:<40s}  {info['accuracy']:.4f}"
            f"  ({info['correct']}/{info['total']})"
        )

    lines += ["", "  Per Subject:"]
    sorted_subjects = sorted(
        metrics["per_subject"].items(), key=lambda x: x[1]["accuracy"], reverse=True
    )
    for subject, info in sorted_subjects:
        lines.append(
            f"    {subject:<45s}  {info['accuracy']:.4f}"
            f"  ({info['correct']}/{info['total']})"
        )

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> float:
    model_name = args.model
    vllm_url: str = args.vllm_url

    print(f"[config] model       = {model_name}")
    print(f"[config] vllm_url    = {vllm_url}")
    print(f"[config] splits      = {args.splits}")
    print(f"[config] max_tokens  = {args.max_new_tokens}")
    print()

    # --- Load dataset ---
    subjects = args.subjects if args.subjects else MMMU_SUBJECTS
    print(f"[data] Loading MMMU ({', '.join(args.splits)}) — {len(subjects)} subjects ...")
    samples = load_mmmu_dataset(args.splits, subjects)
    if not samples:
        sys.exit("ERROR: No samples loaded. Check dataset availability and subject names.")
    print(f"[data] Total: {len(samples)} samples\n")

    if args.limit and args.limit < len(samples):
        print(f"[data] Limiting to first {args.limit} samples (--limit)")
        samples = samples[: args.limit]
        print()

    # --- Setup output directory ---
    out_dir = Path(args.output) if args.output else RESULTS_BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_file = out_dir / "answers.jsonl"
    results_json = out_dir / "results.json"

    # --- Load existing answers (resume support) ---
    existing = load_existing_answers(answers_file)
    if existing:
        print(f"[resume] Loaded {len(existing)} existing answers from {answers_file}")

    remaining = [s for s in samples if s.get("id") not in existing]
    print(f"[eval] {len(remaining)} samples to process  "
          f"({len(existing)} already done)\n")

    # --- Initialise vLLM adapter ---
    pipe = _VLLMServerAdapter(model_name=model_name, base_url=vllm_url)
    print(f"[pipeline] Using vLLM server at {vllm_url} (model={model_name})\n")

    # --- Inference loop ---
    inference_start = time.time()
    total_done = len(existing)
    total_samples = len(samples)

    for idx, sample in enumerate(remaining):
        sample_id = sample.get("id", f"{sample['__subject__']}_{idx}")
        answer = (sample.get("answer") or "").strip().upper()

        # Determine valid letters for this sample's options
        opts = _parse_options(sample.get("options", []))
        valid_letters = "ABCDEFGH"[: max(len(opts), 1)]

        # Skip samples without a ground-truth answer (test set)
        if not answer:
            continue

        # Get images
        sample_images = get_sample_images(sample)
        if not sample_images:
            print(f"  [{total_done+1}/{total_samples}] {sample_id}: no image — skipping")
            continue

        # Optionally resize each image
        if args.image_size:
            sample_images = [resize_to_fixed(img, args.image_size) for img in sample_images]
        
        prompt = build_mmmu_prompt(sample)

        elapsed = time.time() - inference_start
        eta = (elapsed / max(idx, 1)) * (len(remaining) - idx) if idx > 0 else 0
        print(
            f"  [{total_done+1}/{total_samples}] {sample_id}"
            f"  elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
            end="",
            flush=True,
        )

        try:
            raw = pipe.generate(sample_images, prompt, args.max_new_tokens)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            raw = ""
        print("\n", raw)

        predicted = parse_mcq_answer(raw, valid_letters=valid_letters)
        is_correct = predicted == answer
        print(
            f"  -> {predicted or '?'} (gt={answer})"
            f"{'  OK' if is_correct else '  WRONG'}"
        )

        entry = {
            "id": sample_id,
            "subject": sample["__subject__"],
            "split": sample["__split__"],
            "answer": answer,
            "predicted": predicted,
            "correct": is_correct,
            "response": raw[:1000],  # store first 1000 chars for inspection
        }
        existing[sample_id] = entry
        append_answer(answers_file, entry)
        total_done += 1

    inference_time = time.time() - inference_start

    # --- Compute metrics ---
    predictions = list(existing.values())
    # Only keep samples that have ground-truth answers (skip test-set entries)
    predictions = [p for p in predictions if p.get("answer")]

    metrics = compute_metrics(predictions)
    summary = format_summary(metrics, model_name, args.splits, inference_time)
    print()
    print(summary)

    # --- Save final results ---
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_name,
                "splits": args.splits,
                "metrics": {
                    k: v for k, v in metrics.items()
                    if k not in ("per_subject", "per_discipline")
                },
                "per_subject": metrics["per_subject"],
                "per_discipline": metrics["per_discipline"],
                "summary": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[output] Results saved to {results_json}")
    print(f"[output] Per-sample answers in {answers_file}")

    return metrics["micro_avg"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model accuracy on MMMU_DEV_VAL using a vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name as served by the vLLM server "
             "(e.g. Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["dev", "validation"],
        choices=["dev", "validation", "test"],
        help="Dataset splits to use (default: dev validation)",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        metavar="SUBJECT",
        help="Subjects to evaluate (default: all 30). "
             "E.g. --subjects Math Physics Chemistry",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=81920,
        help="Maximum new tokens to generate per sample (default: 16). "
             "Keep small for MCQ; increase to 256+ if the model writes explanations.",
    )
    parser.add_argument(
        "--output", default=None,
        metavar="DIR",
        help="Output directory for results.json and answers.jsonl "
             f"(default: {RESULTS_BASE})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        metavar="N",
        help="Evaluate only the first N samples (for quick testing)",
    )
    parser.add_argument(
        "--image-size", type=int, default=None,
        metavar="N",
        help="Resize each image so its longest side is at most N pixels before "
             "sending to the vLLM server (e.g. 448, 896). Reduces payload size. "
             "Default: no resizing.",
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000",
        metavar="URL",
        help="Base URL of the vLLM OpenAI-compatible server "
             "(default: http://localhost:8000).",
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
