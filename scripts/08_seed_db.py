#!/usr/bin/env python3
"""
Script 08 — Seed Vercel Postgres from Week 3 eval results.

Reads results/eval_summary.json and results/eval_results.json,
creates tables (idempotent), and upserts all rows.

Prerequisites:
  pip install psycopg2-binary python-dotenv
  Set DATABASE_URL (or POSTGRES_URL) in .env or environment.

Usage:
  python scripts/08_seed_db.py
  python scripts/08_seed_db.py --dry-run   # print row counts only, no DB write
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.seed_db_helpers import parse_model_summaries, parse_eval_rows


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _get_db_url() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
    if not url:
        sys.exit(
            "ERROR: DATABASE_URL (or POSTGRES_URL) not set.\n"
            "Add it to your .env file or environment before running this script."
        )
    return url


def seed(summary_path: str, results_path: str, dry_run: bool = False) -> None:
    print(f"Loading {summary_path}...")
    summary_data = _load_json(summary_path)
    print(f"Loading {results_path}...")
    results_data = _load_json(results_path)

    model_rows = parse_model_summaries(summary_data)
    eval_rows = parse_eval_rows(results_data)

    print(f"  model_summaries rows : {len(model_rows)}")
    print(f"  eval_rows rows       : {len(eval_rows)}")

    if dry_run:
        print("\n[DRY RUN] No DB writes performed.")
        return

    import psycopg2
    from psycopg2.extras import execute_values

    url = _get_db_url()
    conn = psycopg2.connect(url)
    cur = conn.cursor()

    # Create tables (schema lives under dashboard/src/lib/ per Next.js --src-dir layout)
    schema_path = Path(__file__).parent.parent / "dashboard" / "src" / "lib" / "schema.sql"
    cur.execute(schema_path.read_text())

    # Upsert model_summaries
    cols = [
        "slug", "label",
        "json_validity_mean", "json_validity_ci_lower", "json_validity_ci_upper",
        "intent_accuracy_mean", "intent_accuracy_ci_lower", "intent_accuracy_ci_upper",
        "gatekeeper_accuracy_mean", "gatekeeper_accuracy_ci_lower", "gatekeeper_accuracy_ci_upper",
        "slot_f1_mean", "slot_f1_ci_lower", "slot_f1_ci_upper",
        "hallucination_rate_mean", "hallucination_rate_ci_lower", "hallucination_rate_ci_upper",
        "judge_score_mean", "judge_score_ci_lower", "judge_score_ci_upper",
    ]
    values = [[r[c] for c in cols] for r in model_rows]
    execute_values(
        cur,
        f"""INSERT INTO model_summaries ({', '.join(cols)}) VALUES %s
            ON CONFLICT (slug) DO UPDATE SET
            {', '.join(f'{c} = EXCLUDED.{c}' for c in cols if c != 'slug')}""",
        values,
    )
    print(f"Upserted {len(model_rows)} model_summaries rows.")

    # Upsert eval_rows
    ecols = [
        "seed_id", "model_slug", "synthetic_message",
        "gt_intent_action", "gt_gatekeeper_status", "gt_order_id", "gt_invoice_id",
        "json_valid", "pred_intent_action", "pred_gatekeeper_status",
        "pred_order_id", "pred_invoice_id",
        "intent_correct", "gatekeeper_correct", "slot_f1", "hallucinated_slots",
        "judge_score", "judge_reason",
    ]
    evalvals = [[r[c] for c in ecols] for r in eval_rows]
    execute_values(
        cur,
        f"""INSERT INTO eval_rows ({', '.join(ecols)}) VALUES %s
            ON CONFLICT (seed_id, model_slug) DO UPDATE SET
            {', '.join(f'{c} = EXCLUDED.{c}' for c in ecols if c not in ('seed_id','model_slug'))}""",
        evalvals,
    )
    print(f"Upserted {len(eval_rows)} eval_rows rows.")

    conn.commit()
    cur.close()
    conn.close()
    print("\nDone. Database seeded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Vercel Postgres from eval results.")
    parser.add_argument("--summary", default="results/eval_summary.json")
    parser.add_argument("--results", default="results/eval_results.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seed(args.summary, args.results, dry_run=args.dry_run)
