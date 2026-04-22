#!/usr/bin/env python3
"""
Compare accuracy results from multiple EvalScope MMMU_DEV_VAL runs.

Usage:
    # Compare specific result directories with labels
    python compare_accuracy.py \
        --results "BF16 (vLLM)=accuracy_results_evalscope_MMMU_DEV_VAL/MMMU_DEV_VAL/20260410_142816" \
                  "INT4 (OpenVINO)=accuracy_results_evalscope_MMMU_DEV_VAL/MMMU_DEV_VAL/20260410_150439" \
                  "INT8 (OpenVINO)=accuracy_results_evalscope_MMMU_DEV_VAL/MMMU_DEV_VAL/20260410_152404"

    # Auto-detect all results under a base directory
    python compare_accuracy.py --auto-detect accuracy_results_evalscope_MMMU_DEV_VAL/MMMU_DEV_VAL

    # Choose split (default: validation)
    python compare_accuracy.py --split dev --results ...

    # Export to CSV
    python compare_accuracy.py --results ... --output comparison.csv
"""

import argparse
import csv
import glob
import os
import sys

import pandas as pd


def read_acc_csv(result_dir, split="validation"):
    """Read accuracy from _acc.csv in result directory."""
    acc_csvs = glob.glob(os.path.join(result_dir, "**/*_acc.csv"), recursive=True)
    if not acc_csvs:
        print(f"  WARNING: No _acc.csv found in {result_dir}")
        return None

    acc_csv = sorted(acc_csvs)[-1]
    results = {}
    with open(acc_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                for key, val in row.items():
                    if key != "split":
                        try:
                            results[key] = float(val)
                        except ValueError:
                            results[key] = val
                break

    if not results:
        print(f"  WARNING: Split '{split}' not found in {acc_csv}")
        # Try the other split
        with open(acc_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, val in row.items():
                    if key != "split":
                        try:
                            results[key] = float(val)
                        except ValueError:
                            results[key] = val
                results["_actual_split"] = row["split"]
                break

    return results


def read_prediction_stats(result_dir):
    """Read prediction statistics from xlsx file."""
    xlsx_files = glob.glob(os.path.join(result_dir, "**/*_MMMU_DEV_VAL.xlsx"), recursive=True)
    # Exclude openai_result files
    xlsx_files = [f for f in xlsx_files if "openai_result" not in f]
    if not xlsx_files:
        return None

    df = pd.read_excel(sorted(xlsx_files)[-1])
    total = len(df)
    preds = [str(p).strip() for p in df["prediction"]]
    failed = sum(1 for p in preds if "Failed" in p or "ERROR" in p)

    from collections import Counter
    pred_dist = Counter(preds).most_common(6)

    return {
        "total_samples": total,
        "failed": failed,
        "success_rate": f"{(total - failed) / total * 100:.1f}%",
        "top_predictions": pred_dist,
    }


def format_pct(val):
    """Format float as percentage string."""
    if isinstance(val, float):
        return f"{val * 100:.1f}%"
    return str(val)


def print_comparison(all_results, labels, split):
    """Print formatted comparison table."""
    if not all_results:
        print("No results to compare.")
        return

    # Get all subjects from the first result
    all_subjects = list(all_results[0].keys())
    # Remove internal keys
    all_subjects = [s for s in all_subjects if not s.startswith("_")]

    # Separate individual subjects and category subjects
    categories = ["Art & Design", "Business", "Health & Medicine",
                   "Humanities & Social Science", "Science", "Tech & Engineering"]
    individual_subjects = [s for s in all_subjects if s not in categories and s != "Overall"]

    # Print header
    print("\n" + "=" * 100)
    print(f"  Accuracy Comparison (split: {split}, {len(labels)} configurations)")
    print("=" * 100)

    # --- Overview table ---
    print(f"\n{'Subject':<30}", end="")
    for label in labels:
        print(f" {label:>20}", end="")
    print()
    print("-" * (30 + 21 * len(labels)))

    # Overall first
    print(f"{'Overall':<30}", end="")
    best_overall = max(r.get("Overall", 0) for r in all_results)
    for r in all_results:
        val = r.get("Overall", 0)
        marker = " *" if val == best_overall and len(all_results) > 1 else "  "
        print(f" {format_pct(val):>18}{marker}", end="")
    print()
    print("-" * (30 + 21 * len(labels)))

    # Categories
    for cat in categories:
        if cat in all_subjects:
            print(f"{cat:<30}", end="")
            best = max(r.get(cat, 0) for r in all_results)
            for r in all_results:
                val = r.get(cat, 0)
                marker = " *" if val == best and len(all_results) > 1 else "  "
                print(f" {format_pct(val):>18}{marker}", end="")
            print()

    print("-" * (30 + 21 * len(labels)))

    # Individual subjects
    print(f"\n{'--- Per Subject ---':<30}")
    for subj in sorted(individual_subjects):
        print(f"{subj:<30}", end="")
        vals = [r.get(subj, 0) for r in all_results]
        best = max(vals)
        for val in vals:
            marker = " *" if val == best and len(all_results) > 1 else "  "
            print(f" {format_pct(val):>18}{marker}", end="")
        print()

    # --- Prediction stats ---
    print(f"\n{'--- Prediction Stats ---':<30}")
    for i, label in enumerate(labels):
        stats = all_results[i].get("_pred_stats")
        if stats:
            print(f"  {label}: {stats['total_samples']} samples, "
                  f"{stats['failed']} failed ({stats['success_rate']} success)")
            top = ", ".join(f"{k}:{v}" for k, v in stats["top_predictions"])
            print(f"    Distribution: {top}")

    # --- Delta analysis ---
    if len(all_results) >= 2:
        print(f"\n{'--- Delta vs ' + labels[0] + ' ---':<30}")
        base = all_results[0]
        print(f"{'Subject':<30}", end="")
        for label in labels[1:]:
            print(f" {label:>20}", end="")
        print()
        print("-" * (30 + 21 * (len(labels) - 1)))

        for subj in ["Overall"] + categories:
            if subj not in all_subjects and subj != "Overall":
                continue
            base_val = base.get(subj, 0)
            print(f"{subj:<30}", end="")
            for r in all_results[1:]:
                val = r.get(subj, 0)
                delta = (val - base_val) * 100
                sign = "+" if delta >= 0 else ""
                print(f" {sign}{delta:>17.1f}pp", end="")
            print()

    print("\n" + "=" * 100)
    print("  * = best in row")
    print("=" * 100)


def export_csv(all_results, labels, split, output_path):
    """Export comparison to CSV file."""
    all_subjects = list(all_results[0].keys())
    all_subjects = [s for s in all_subjects if not s.startswith("_")]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["Subject"] + labels
        writer.writerow(header)
        for subj in ["Overall"] + sorted([s for s in all_subjects if s != "Overall"]):
            row = [subj]
            for r in all_results:
                val = r.get(subj, 0)
                row.append(f"{val * 100:.1f}" if isinstance(val, float) else str(val))
            writer.writerow(row)
    print(f"\nExported to: {output_path}")


def auto_detect_results(base_dir):
    """Auto-detect result directories under base_dir."""
    results = []
    if not os.path.isdir(base_dir):
        print(f"ERROR: Directory not found: {base_dir}")
        return results

    for entry in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            # Check if it contains result files
            acc_csvs = glob.glob(os.path.join(full_path, "**/*_acc.csv"), recursive=True)
            if acc_csvs:
                results.append((entry, full_path))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare accuracy results from multiple EvalScope runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare with labels
  python compare_accuracy.py \\
      --results "BF16 (vLLM)=path/to/bf16_results" \\
                "INT4 (OV)=path/to/int4_results" \\
                "INT8 (OV)=path/to/int8_results"

  # Auto-detect all results
  python compare_accuracy.py --auto-detect accuracy_results_evalscope_MMMU_DEV_VAL/MMMU_DEV_VAL

  # Export to CSV
  python compare_accuracy.py --results ... --output comparison.csv
        """,
    )
    parser.add_argument(
        "--results", nargs="+",
        help='Result dirs in "label=path" format (e.g., "BF16=path/to/results")',
    )
    parser.add_argument(
        "--auto-detect",
        help="Auto-detect all result directories under this path",
    )
    parser.add_argument(
        "--split", default="validation",
        help="Which split to compare (default: validation)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Export comparison to CSV file",
    )
    args = parser.parse_args()

    if not args.results and not args.auto_detect:
        parser.print_help()
        sys.exit(1)

    # Parse result directories
    entries = []
    if args.auto_detect:
        entries = auto_detect_results(args.auto_detect)
        if not entries:
            print(f"No results found under {args.auto_detect}")
            sys.exit(1)
        print(f"Auto-detected {len(entries)} result directories:")
        for label, path in entries:
            print(f"  {label}: {path}")
    elif args.results:
        for item in args.results:
            if "=" in item:
                label, path = item.split("=", 1)
            else:
                label = os.path.basename(item.rstrip("/"))
                path = item
            entries.append((label.strip(), path.strip()))

    # Read results
    labels = []
    all_results = []
    for label, path in entries:
        print(f"\nReading: {label} ({path})")
        acc = read_acc_csv(path, args.split)
        if acc is None:
            print(f"  Skipping {label}: no accuracy data found")
            continue
        pred_stats = read_prediction_stats(path)
        acc["_pred_stats"] = pred_stats
        labels.append(label)
        all_results.append(acc)
        print(f"  Overall ({args.split}): {format_pct(acc.get('Overall', 0))}")

    if not all_results:
        print("\nNo valid results found.")
        sys.exit(1)

    # Print comparison
    print_comparison(all_results, labels, args.split)

    # Export if requested
    if args.output:
        export_csv(all_results, labels, args.split, args.output)


if __name__ == "__main__":
    main()
