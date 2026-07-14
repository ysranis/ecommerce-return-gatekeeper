#!/usr/bin/env python3
"""
Script 04 — Before vs. After Comparison Report

Loads the four evaluation JSON files produced by script 03 (two baseline runs
and two fine-tuned runs) and generates a side-by-side comparison report showing
the delta for all five metrics.

Runs locally on Mac — no GPU, no ML dependencies. Pure stdlib JSON.

Output files:
    results/comparison_report.json  — machine-readable (for Week 4 dashboard)
    results/comparison_report.md    — human-readable Markdown table (for portfolio)

Usage:
    python scripts/04_compare_results.py
    python scripts/04_compare_results.py --results-dir results --output-dir results
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

METRICS = [
    "json_validity_rate",
    "intent_accuracy",
    "gatekeeper_accuracy",
    "slot_f1",
    "hallucination_rate",
]

METRIC_LABELS = {
    "json_validity_rate": "JSON validity",
    "intent_accuracy": "Intent accuracy",
    "gatekeeper_accuracy": "Gatekeeper acc.",
    "slot_f1": "Slot F1",
    "hallucination_rate": "Hallucination rt.",
}

# For hallucination_rate, a negative delta is improvement — we flag this for display
LOWER_IS_BETTER = {"hallucination_rate"}

MODELS = [
    {
        "slug": "qwen",
        "label": "Qwen-2.5-7B (Track A — Accuracy-Optimized)",
        "baseline_file": "baseline_results_qwen.json",
        "finetuned_file": "finetuned_results_qwen.json",
    },
    {
        "slug": "llama",
        "label": "Llama-3.2-3B (Track B — Speed-Optimized)",
        "baseline_file": "baseline_results_llama.json",
        "finetuned_file": "finetuned_results_llama.json",
    },
]


def _load_result(path: Path) -> dict:
    """Load and return a result JSON file. Exit with a clear message if missing."""
    if not path.exists():
        sys.exit(
            f"ERROR: {path} not found.\n"
            f"Run scripts/03_baseline_eval.py for both models before and after fine-tuning."
        )
    with open(path) as f:
        return json.load(f)


def _format_metric(metric: str, value: float) -> str:
    """Format a metric value for display."""
    if metric == "slot_f1":
        return f"{value:.3f}"
    return f"{value:.1%}"


def _format_delta(metric: str, delta: float) -> str:
    """Format a delta value, marking improvement direction."""
    if metric == "slot_f1":
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.3f}"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1%}"


def _delta_is_improvement(metric: str, delta: float) -> bool:
    """Return True if the delta represents an improvement."""
    if metric in LOWER_IS_BETTER:
        return delta < 0
    return delta > 0


def build_comparison(results_dir: Path) -> dict:
    """Load all 4 result files and compute per-model metric deltas."""
    comparison = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }

    for model_cfg in MODELS:
        slug = model_cfg["slug"]
        baseline_path = results_dir / model_cfg["baseline_file"]
        finetuned_path = results_dir / model_cfg["finetuned_file"]

        baseline_data = _load_result(baseline_path)
        finetuned_data = _load_result(finetuned_path)

        baseline_summary = baseline_data["summary"]
        finetuned_summary = finetuned_data["summary"]

        metrics = {}
        for metric in METRICS:
            b_val = baseline_summary[metric]
            f_val = finetuned_summary[metric]
            delta = round(f_val - b_val, 4)
            metrics[metric] = {
                "baseline": b_val,
                "finetuned": f_val,
                "delta": delta,
                "improved": _delta_is_improvement(metric, delta),
            }

        comparison["models"][slug] = {
            "label": model_cfg["label"],
            "baseline_model": baseline_data["metadata"]["model"],
            "finetuned_adapter": finetuned_data["metadata"].get("adapter_path"),
            "baseline_file": str(baseline_path),
            "finetuned_file": str(finetuned_path),
            "metrics": metrics,
        }

    return comparison


def write_json_report(comparison: dict, output_path: Path) -> None:
    """Write the machine-readable comparison JSON."""
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Written: {output_path}")


def write_markdown_report(comparison: dict, output_path: Path) -> None:
    """Write the human-readable Markdown comparison report."""
    lines = [
        "# E-commerce Return Gatekeeper — Before vs. After Fine-Tuning",
        "",
        f"_Generated: {comparison['generated_at']}_",
        "",
    ]

    for slug, model_data in comparison["models"].items():
        lines.append(f"## {model_data['label']}")
        lines.append("")
        lines.append(f"**Base model:** `{model_data['baseline_model']}`")
        if model_data["finetuned_adapter"]:
            lines.append(f"**Adapter:** `{model_data['finetuned_adapter']}`")
        lines.append("")

        # Table header
        lines.append("| Metric | Baseline | Fine-tuned | Delta |")
        lines.append("|---|---|---|---|")

        for metric in METRICS:
            m = model_data["metrics"][metric]
            label = METRIC_LABELS[metric]
            b_str = _format_metric(metric, m["baseline"])
            f_str = _format_metric(metric, m["finetuned"])
            d_str = _format_delta(metric, m["delta"])

            # Bold the delta to make it stand out
            lines.append(f"| {label} | {b_str} | {f_str} | **{d_str}** |")

        lines.append("")

    # Summary section
    lines.append("## Key Takeaways")
    lines.append("")
    for slug, model_data in comparison["models"].items():
        improved_count = sum(
            1 for m in model_data["metrics"].values() if m["improved"]
        )
        lines.append(
            f"- **{model_data['label']}**: "
            f"{improved_count}/{len(METRICS)} metrics improved after fine-tuning"
        )
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {output_path}")


def print_summary(comparison: dict) -> None:
    """Print a console summary of the comparison."""
    for slug, model_data in comparison["models"].items():
        print(f"\n{'='*60}")
        print(f"  {model_data['label']}")
        print(f"{'='*60}")
        print(f"  {'Metric':<22} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>10}")
        print(f"  {'─'*54}")

        for metric in METRICS:
            m = model_data["metrics"][metric]
            label = METRIC_LABELS[metric]
            b_str = _format_metric(metric, m["baseline"])
            f_str = _format_metric(metric, m["finetuned"])
            d_str = _format_delta(metric, m["delta"])
            arrow = "↑" if m["improved"] else "↓"
            print(f"  {label:<22} {b_str:>10} {f_str:>12} {d_str:>8} {arrow}")


def main(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading evaluation result files…")
    comparison = build_comparison(results_dir)

    json_output = output_dir / "comparison_report.json"
    md_output = output_dir / "comparison_report.md"

    write_json_report(comparison, json_output)
    write_markdown_report(comparison, md_output)
    print_summary(comparison)

    print(f"\n{'='*60}")
    print("Comparison report complete.")
    print(f"  JSON : {json_output}")
    print(f"  MD   : {md_output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate before/after comparison report from eval result files."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing the 4 evaluation JSON files (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write comparison_report.json and .md (default: results/)",
    )
    main(parser.parse_args())
