#!/usr/bin/env python3
"""
Script 07 — Week 3: Full 5-Model Benchmark

Loads the 4 Week 2 evaluation result files, runs DeepSeek-V3 teacher
inference on the 150-row test set, judges all 750 outputs with DeepSeek-V3,
computes bootstrap 95% CIs, and writes three report files.

Runs locally on Mac — no GPU required.

Required env vars:
    DEEPSEEK_API_KEY   — for both teacher eval and LLM judge (DeepSeek-V3)

Usage:
    python scripts/07_evaluate_models.py
    python scripts/07_evaluate_models.py --skip-teacher --skip-judge
    python scripts/07_evaluate_models.py --n-resamples 500 --concurrency 5
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Model config ──────────────────────────────────────────────────────────────

MODEL_CONFIG = [
    {
        "slug": "base_qwen",
        "label": "Qwen-2.5-7B (base)",
        "source_file": "baseline_results_qwen.json",
    },
    {
        "slug": "ft_qwen",
        "label": "Qwen-2.5-7B fine-tuned (Track A)",
        "source_file": "finetuned_results_qwen.json",
    },
    {
        "slug": "base_llama",
        "label": "Llama-3.2-3B (base)",
        "source_file": "baseline_results_llama.json",
    },
    {
        "slug": "ft_llama",
        "label": "Llama-3.2-3B fine-tuned (Track B)",
        "source_file": "finetuned_results_llama.json",
    },
]

TEACHER_SLUG = "teacher_deepseek"
TEACHER_LABEL = "DeepSeek-V3 (teacher)"

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

LOWER_IS_BETTER = {"hallucination_rate"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_result_file(path: Path) -> dict:
    if not path.exists():
        sys.exit(
            f"ERROR: {path} not found.\n"
            f"Run Week 2 scripts (03_baseline_eval.py) before proceeding."
        )
    with open(path) as f:
        return json.load(f)


def _strip_code_fences(text: str) -> str:
    import re
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _make_empty_teacher_data() -> dict:
    """Return a placeholder teacher result dict (used when --skip-teacher and no file exists)."""
    return {
        "metadata": {
            "model": "deepseek-chat (DeepSeek-V3)",
            "adapter_path": None,
            "run_type": "teacher",
            "slug": TEACHER_SLUG,
            "eval_timestamp": datetime.now(timezone.utc).isoformat(),
            "test_file": "data/test.jsonl",
            "total_rows": 0,
            "note": "placeholder — run without --skip-teacher to populate",
        },
        "summary": {
            "json_validity_rate": 0.0,
            "intent_accuracy": 0.0,
            "gatekeeper_accuracy": 0.0,
            "slot_f1": 0.0,
            "hallucination_rate": 0.0,
            "json_valid_count": 0,
            "intent_correct_count": 0,
            "gatekeeper_correct_count": 0,
            "total_evaluated": 0,
        },
        "rows": [],
    }


# ── Teacher eval ──────────────────────────────────────────────────────────────

def _compute_slot_f1(gt_order, gt_invoice, pred_order, pred_invoice) -> float:
    """Micro-F1 for order_id + invoice_id slots."""
    tp = fp = fn = 0
    for gt_val, pred_val in [(gt_order, pred_order), (gt_invoice, pred_invoice)]:
        gt_has = gt_val is not None and str(gt_val).strip() != ""
        pred_has = pred_val is not None and str(pred_val).strip() != ""
        if gt_has and pred_has:
            if str(gt_val).strip().lower() == str(pred_val).strip().lower():
                tp += 1
            else:
                fp += 1
                fn += 1
        elif pred_has and not gt_has:
            fp += 1
        elif gt_has and not pred_has:
            fn += 1
    denom = 2 * tp + fp + fn
    return round(2 * tp / denom, 4) if denom > 0 else 1.0


def _check_hallucination(message: str, pred_order, pred_invoice) -> bool:
    msg_lower = message.lower()
    for val in [pred_order, pred_invoice]:
        if val and str(val).strip():
            if str(val).strip().lower() not in msg_lower:
                return True
    return False


async def _run_teacher_eval(
    test_rows: list[dict],
    checkpoint_path: str,
    api_key: str,
    concurrency: int,
) -> dict:
    """Run DeepSeek-V3 on all test rows. Returns result dict matching Week 2 schema."""
    from scripts.lib.checkpoint import load_checkpoint, append_row
    from scripts.lib.deepseek_client import DeepSeekClient
    from scripts.lib.prompts import LABELING_SYSTEM_PROMPT
    from scripts.lib.validator import validate

    client = DeepSeekClient(api_key=api_key, concurrency=concurrency)
    done = {r["seed_id"]: r for r in load_checkpoint(checkpoint_path)}
    print(f"[teacher] Loaded {len(done)} cached rows.")

    semaphore = asyncio.Semaphore(concurrency)

    async def _eval_one(row: dict) -> dict:
        seed_id = row["seed_id"]
        if seed_id in done:
            return done[seed_id]

        raw = await client.generate(
            user_prompt=row["synthetic_message"],
            system_prompt=LABELING_SYSTEM_PROMPT,
            max_tokens=512,
            temperature=0.0,
        )
        raw_stripped = _strip_code_fences(raw)
        json_valid, parsed = validate(raw_stripped)

        pred_intent = parsed.get("intent_action") if parsed else None
        pred_gk = parsed.get("gatekeeper_status") if parsed else None
        pred_slots = parsed.get("extracted_slots", {}) if parsed else {}
        pred_order = pred_slots.get("order_id")
        pred_invoice = pred_slots.get("invoice_id")

        gt_order = row.get("extracted_slots", {}).get("order_id")
        gt_invoice = row.get("extracted_slots", {}).get("invoice_id")

        result_row = {
            "seed_id": seed_id,
            "synthetic_message": row["synthetic_message"],
            "gt_intent_action": row.get("intent_action"),
            "gt_gatekeeper_status": row.get("gatekeeper_status"),
            "gt_order_id": gt_order,
            "gt_invoice_id": gt_invoice,
            "raw_output": raw,
            "json_valid": json_valid,
            "pred_intent_action": pred_intent,
            "pred_gatekeeper_status": pred_gk,
            "pred_order_id": pred_order,
            "pred_invoice_id": pred_invoice,
            "intent_correct": json_valid and pred_intent == row.get("intent_action"),
            "gatekeeper_correct": json_valid and pred_gk == row.get("gatekeeper_status"),
            "slot_f1": _compute_slot_f1(gt_order, gt_invoice, pred_order, pred_invoice) if json_valid else 0.0,
            "hallucinated_slots": _check_hallucination(row["synthetic_message"], pred_order, pred_invoice) if json_valid else False,
            "error": None,
        }

        idx = len(done) + 1
        status = "OK" if json_valid else "FAIL"
        print(f"[teacher] [{idx:03d}/150] {seed_id}  json={status}")
        append_row(checkpoint_path, result_row)
        done[seed_id] = result_row
        return result_row

    remaining_rows = [row for row in test_rows if row["seed_id"] not in done]

    async def bounded(row):
        async with semaphore:
            return await _eval_one(row)

    new_results = await asyncio.gather(*[bounded(row) for row in remaining_rows])

    # Merge: preserve order from test_rows
    all_rows = []
    for row in test_rows:
        sid = row["seed_id"]
        if sid in done:
            all_rows.append(done[sid])

    # Build summary
    n = len(all_rows)
    json_valid_count = sum(1 for r in all_rows if r["json_valid"])
    intent_correct_count = sum(1 for r in all_rows if r["intent_correct"])
    gk_correct_count = sum(1 for r in all_rows if r["gatekeeper_correct"])
    slot_f1_sum = sum(r["slot_f1"] for r in all_rows)
    halluc_count = sum(1 for r in all_rows if r["hallucinated_slots"])

    return {
        "metadata": {
            "model": "deepseek-chat (DeepSeek-V3)",
            "adapter_path": None,
            "run_type": "teacher",
            "slug": TEACHER_SLUG,
            "eval_timestamp": datetime.now(timezone.utc).isoformat(),
            "test_file": "data/test.jsonl",
            "total_rows": n,
        },
        "summary": {
            "json_validity_rate": round(json_valid_count / n, 4) if n else 0.0,
            "intent_accuracy": round(intent_correct_count / n, 4) if n else 0.0,
            "gatekeeper_accuracy": round(gk_correct_count / n, 4) if n else 0.0,
            "slot_f1": round(slot_f1_sum / n, 4) if n else 0.0,
            "hallucination_rate": round(halluc_count / n, 4) if n else 0.0,
            "json_valid_count": json_valid_count,
            "intent_correct_count": intent_correct_count,
            "gatekeeper_correct_count": gk_correct_count,
            "total_evaluated": n,
        },
        "rows": all_rows,
    }


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _compute_summary_with_ci(
    model_data: dict,
    judge_scores: dict,
    model_slug: str,
    n_resamples: int,
) -> dict:
    """Compute per-metric bootstrap CIs + judge score CI for one model."""
    from scripts.lib.bootstrap import bootstrap_ci

    rows = model_data["rows"]
    label = model_data.get("label", model_slug)

    # Guard: empty rows (e.g. placeholder teacher) — return zeros
    if not rows:
        zero_ci = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        return {
            "label": label,
            "metrics": {m: dict(zero_ci) for m in METRICS},
            "judge_score": None,
        }

    per_metric_values = {
        "json_validity_rate": [1.0 if r["json_valid"] else 0.0 for r in rows],
        "intent_accuracy": [1.0 if r["intent_correct"] else 0.0 for r in rows],
        "gatekeeper_accuracy": [1.0 if r["gatekeeper_correct"] else 0.0 for r in rows],
        "slot_f1": [r["slot_f1"] for r in rows],
        "hallucination_rate": [1.0 if r.get("hallucinated_slots") else 0.0 for r in rows],
    }

    metrics = {}
    for metric, values in per_metric_values.items():
        metrics[metric] = bootstrap_ci(values, n_resamples=n_resamples)

    # Judge scores
    judge_values = []
    for row in rows:
        key = f"{row['seed_id']}:{model_slug}"
        if key in judge_scores:
            judge_values.append(float(judge_scores[key]["score"]))

    judge_ci = bootstrap_ci(judge_values, n_resamples=n_resamples) if judge_values else None

    return {"label": label, "metrics": metrics, "judge_score": judge_ci}


# ── Report writers ────────────────────────────────────────────────────────────

def _fmt_pct(val: float) -> str:
    return f"{val:.1%}"

def _fmt_f1(val: float) -> str:
    return f"{val:.3f}"

def _fmt_metric(metric: str, val: float) -> str:
    return _fmt_f1(val) if metric == "slot_f1" else _fmt_pct(val)

def _fmt_delta(metric: str, delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    if metric == "slot_f1":
        return f"{sign}{delta:.3f}"
    return f"{sign}{delta:.1%}"


def write_eval_results(all_model_data: dict, judge_scores: dict, output_path: Path) -> None:
    """Write unified eval_results.json."""
    # Restructure judge_scores: {seed_id: {model_slug: {score, reason}}}
    judge_by_row: dict = {}
    for key, result in judge_scores.items():
        seed_id, model_slug = key.rsplit(":", 1)
        judge_by_row.setdefault(seed_id, {})[model_slug] = result

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            slug: {
                "label": data["label"],
                "source_file": data.get("source_file", ""),
                "summary": data["summary"],
                "rows": data["rows"],
            }
            for slug, data in all_model_data.items()
        },
        "judge_scores": judge_by_row,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Written: {output_path}")


def write_eval_summary(
    summary_by_model: dict,
    n_resamples: int,
    output_path: Path,
) -> None:
    """Write eval_summary.json with bootstrap CIs."""
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_test_rows": 150,
        "n_resamples": n_resamples,
        "models": summary_by_model,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Written: {output_path}")


def write_cross_comparison_table(summary_by_model: dict, output_path: Path) -> None:
    """Write cross_comparison_table.md — the portfolio money table."""
    lines = [
        "# E-commerce Return Gatekeeper — Full 5-Model Benchmark",
        "",
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Before vs. After Fine-Tuning + Teacher Comparison",
        "",
        "| Metric | Base Qwen | FT Qwen (A) | Δ(A) | Base Llama | FT Llama (B) | Δ(B) | DeepSeek-V3 (Teacher) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    slugs = ["base_qwen", "ft_qwen", "base_llama", "ft_llama", TEACHER_SLUG]
    model_data = {s: summary_by_model.get(s, {}) for s in slugs}

    for metric in METRICS:
        label = METRIC_LABELS[metric]
        vals = {}
        for slug in slugs:
            m = model_data[slug].get("metrics", {}).get(metric, {})
            vals[slug] = m.get("mean", 0.0)

        delta_a = vals["ft_qwen"] - vals["base_qwen"]
        delta_b = vals["ft_llama"] - vals["base_llama"]

        row = (
            f"| {label} "
            f"| {_fmt_metric(metric, vals['base_qwen'])} "
            f"| {_fmt_metric(metric, vals['ft_qwen'])} "
            f"| **{_fmt_delta(metric, delta_a)}** "
            f"| {_fmt_metric(metric, vals['base_llama'])} "
            f"| {_fmt_metric(metric, vals['ft_llama'])} "
            f"| **{_fmt_delta(metric, delta_b)}** "
            f"| {_fmt_metric(metric, vals[TEACHER_SLUG])} |"
        )
        lines.append(row)

    # Judge score row
    judge_vals = {}
    for slug in slugs:
        js = model_data[slug].get("judge_score")
        judge_vals[slug] = js["mean"] if js else 0.0

    if any(judge_vals.values()):
        delta_a_j = judge_vals["ft_qwen"] - judge_vals["base_qwen"]
        delta_b_j = judge_vals["ft_llama"] - judge_vals["base_llama"]
        lines.append(
            f"| LLM Judge (avg 1-5) "
            f"| {judge_vals['base_qwen']:.2f} "
            f"| {judge_vals['ft_qwen']:.2f} "
            f"| **{delta_a_j:+.2f}** "
            f"| {judge_vals['base_llama']:.2f} "
            f"| {judge_vals['ft_llama']:.2f} "
            f"| **{delta_b_j:+.2f}** "
            f"| {judge_vals[TEACHER_SLUG]:.2f} |"
        )

    lines += [
        "",
        "## 95% Bootstrap Confidence Intervals",
        "",
        "| Metric | FT Qwen CI | FT Llama CI | Teacher CI |",
        "|---|---|---|---|",
    ]
    for metric in METRICS:
        label = METRIC_LABELS[metric]

        def _ci_str(slug, metric=metric):
            m = model_data[slug].get("metrics", {}).get(metric, {})
            lo, hi = m.get("ci_lower", 0), m.get("ci_upper", 0)
            if metric == "slot_f1":
                return f"[{lo:.3f}, {hi:.3f}]"
            return f"[{lo:.1%}, {hi:.1%}]"

        lines.append(
            f"| {label} | {_ci_str('ft_qwen')} | {_ci_str('ft_llama')} | {_ci_str(TEACHER_SLUG)} |"
        )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Written: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    load_dotenv()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load 4 existing Week 2 result files ──────────────────────────
    print(f"\n{'='*56}")
    print("WEEK 3: FULL 5-MODEL BENCHMARK")
    print(f"{'='*56}\n")

    all_model_data: dict = {}
    for cfg in MODEL_CONFIG:
        path = results_dir / cfg["source_file"]
        data = _load_result_file(path)
        all_model_data[cfg["slug"]] = {
            "label": cfg["label"],
            "source_file": str(path),
            "summary": data["summary"],
            "rows": data["rows"],
        }
        print(f"[load] {cfg['slug']}: {len(data['rows'])} rows loaded from {path.name}")

    # ── Step 2: Teacher eval ─────────────────────────────────────────────────
    teacher_path = output_dir / "teacher_results_deepseek.json"
    teacher_checkpoint = output_dir / "teacher_eval_checkpoint.jsonl"

    if args.skip_teacher and teacher_path.exists():
        print(f"\n[teacher] Skipping — loading existing {teacher_path.name}")
        with open(teacher_path) as f:
            teacher_data = json.load(f)

    elif args.skip_teacher and not teacher_path.exists():
        print(
            f"\n[teacher] Skipping — {teacher_path.name} not found. "
            "Using empty placeholder (run without --skip-teacher to populate)."
        )
        teacher_data = _make_empty_teacher_data()

    else:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if not deepseek_key:
            sys.exit("ERROR: DEEPSEEK_API_KEY not set. Required for teacher eval.")

        test_path = data_dir / "test.jsonl"
        if not test_path.exists():
            sys.exit(f"ERROR: {test_path} not found.")

        from scripts.lib.checkpoint import load_checkpoint
        test_rows = load_checkpoint(str(test_path))
        print(f"\n[teacher] Running DeepSeek-V3 on {len(test_rows)} test rows...")

        teacher_data = asyncio.run(
            _run_teacher_eval(test_rows, str(teacher_checkpoint), deepseek_key, args.concurrency)
        )
        with open(teacher_path, "w") as f:
            json.dump(teacher_data, f, indent=2)
        print(f"[teacher] Saved: {teacher_path}")

    all_model_data[TEACHER_SLUG] = {
        "label": TEACHER_LABEL,
        "source_file": str(teacher_path),
        "summary": teacher_data["summary"],
        "rows": teacher_data["rows"],
    }

    print("\n[teacher] Summary:")
    s = teacher_data["summary"]
    if s["total_evaluated"] == 0:
        print("  (placeholder — no teacher eval data)")
    else:
        print(f"  JSON validity  : {s['json_validity_rate']:.1%}")
        print(f"  Intent acc.    : {s['intent_accuracy']:.1%}")
        print(f"  Gatekeeper acc.: {s['gatekeeper_accuracy']:.1%}")
        print(f"  Slot F1        : {s['slot_f1']:.3f}")
        print(f"  Halluc. rate   : {s['hallucination_rate']:.1%}")

    # ── Step 3: LLM Judge ────────────────────────────────────────────────────
    judge_checkpoint = output_dir / "judge_scores_checkpoint.jsonl"
    judge_scores: dict = {}

    if not args.skip_judge:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if not deepseek_key:
            print("[WARN] DEEPSEEK_API_KEY not set — skipping judge scoring.")
        else:
            from scripts.lib.judge import run_judge
            rows_by_model = {slug: data["rows"] for slug, data in all_model_data.items()}
            n_judge_calls = sum(len(v) for v in rows_by_model.values())
            print(f"\n[judge] Running DeepSeek-V3 judge on {n_judge_calls} outputs...")
            judge_scores = asyncio.run(
                run_judge(rows_by_model, str(judge_checkpoint), deepseek_key, args.concurrency)
            )

    # ── Step 4: Bootstrap CIs ────────────────────────────────────────────────
    print("\n[bootstrap] Computing 95% CIs...")
    summary_by_model: dict = {}
    for slug, data in all_model_data.items():
        summary_by_model[slug] = _compute_summary_with_ci(
            data, judge_scores, slug, args.n_resamples
        )
    print("[bootstrap] Done.")

    # ── Step 5: Write output files ───────────────────────────────────────────
    print("\n[output] Writing report files...")
    write_eval_results(all_model_data, judge_scores, output_dir / "eval_results.json")
    write_eval_summary(summary_by_model, args.n_resamples, output_dir / "eval_summary.json")
    write_cross_comparison_table(summary_by_model, output_dir / "cross_comparison_table.md")

    # ── Print console summary ─────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*56}")
    for slug in ["base_qwen", "ft_qwen", "base_llama", "ft_llama", TEACHER_SLUG]:
        m = summary_by_model.get(slug, {})
        label = m.get("label", slug)
        intent = m.get("metrics", {}).get("intent_accuracy", {}).get("mean", 0)
        gk = m.get("metrics", {}).get("gatekeeper_accuracy", {}).get("mean", 0)
        judge = m.get("judge_score") or {}
        judge_str = f" | judge={judge.get('mean', 0):.2f}" if judge else ""
        print(f"  {label:<42} intent={intent:.1%}  gk={gk:.1%}{judge_str}")
    print(f"\n  eval_results.json         -> {output_dir / 'eval_results.json'}")
    print(f"  eval_summary.json         -> {output_dir / 'eval_summary.json'}")
    print(f"  cross_comparison_table.md -> {output_dir / 'cross_comparison_table.md'}")
    print(f"{'='*56}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Week 3: Full 5-model benchmark with LLM judge and bootstrap CIs."
    )
    parser.add_argument("--results-dir", default="results", help="Week 2 result files dir")
    parser.add_argument("--data-dir", default="data", help="Dir containing test.jsonl")
    parser.add_argument("--output-dir", default="results", help="Output dir for report files")
    parser.add_argument("--concurrency", type=int, default=10, help="Parallel API calls")
    parser.add_argument("--n-resamples", type=int, default=1000, help="Bootstrap resamples")
    parser.add_argument("--skip-teacher", action="store_true", help="Skip DeepSeek teacher eval")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge scoring")
    main(parser.parse_args())
