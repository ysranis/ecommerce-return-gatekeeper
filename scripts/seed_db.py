"""
Seed Vercel/Neon Postgres with eval data.

Usage:
    export POSTGRES_URL="postgres://user:pass@host/db?sslmode=require"
    pip install psycopg2-binary
    python scripts/seed_db.py
"""

import json
import os
import sys
from pathlib import Path

try:
    import psycopg2
    from psycopg2.extras import execute_batch
except ImportError:
    sys.exit("Missing dependency. Run: pip install psycopg2-binary")

POSTGRES_URL = os.environ.get("POSTGRES_URL")
if not POSTGRES_URL:
    sys.exit("POSTGRES_URL environment variable not set.")

ROOT = Path(__file__).parent.parent
SCHEMA = ROOT / "dashboard" / "src" / "lib" / "schema.sql"
SUMMARY = ROOT / "results" / "eval_summary.json"
EVAL    = ROOT / "results" / "eval_results.json"

print("Connecting to Postgres...")
conn = psycopg2.connect(POSTGRES_URL)
cur = conn.cursor()

# ── Create tables ────────────────────────────────────────────────────────────
print("Creating tables...")
cur.execute(SCHEMA.read_text())

# ── Seed model_summaries ─────────────────────────────────────────────────────
print("Seeding model_summaries...")
summary = json.loads(SUMMARY.read_text())

cur.execute("DELETE FROM model_summaries")
for slug, m in summary["models"].items():
    met = m["metrics"]
    js  = m["judge_score"]
    cur.execute("""
        INSERT INTO model_summaries (
            slug, label,
            json_validity_mean, json_validity_ci_lower, json_validity_ci_upper,
            intent_accuracy_mean, intent_accuracy_ci_lower, intent_accuracy_ci_upper,
            gatekeeper_accuracy_mean, gatekeeper_accuracy_ci_lower, gatekeeper_accuracy_ci_upper,
            slot_f1_mean, slot_f1_ci_lower, slot_f1_ci_upper,
            hallucination_rate_mean, hallucination_rate_ci_lower, hallucination_rate_ci_upper,
            judge_score_mean, judge_score_ci_lower, judge_score_ci_upper
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (slug) DO UPDATE SET
            label                        = EXCLUDED.label,
            json_validity_mean           = EXCLUDED.json_validity_mean,
            json_validity_ci_lower       = EXCLUDED.json_validity_ci_lower,
            json_validity_ci_upper       = EXCLUDED.json_validity_ci_upper,
            intent_accuracy_mean         = EXCLUDED.intent_accuracy_mean,
            intent_accuracy_ci_lower     = EXCLUDED.intent_accuracy_ci_lower,
            intent_accuracy_ci_upper     = EXCLUDED.intent_accuracy_ci_upper,
            gatekeeper_accuracy_mean     = EXCLUDED.gatekeeper_accuracy_mean,
            gatekeeper_accuracy_ci_lower = EXCLUDED.gatekeeper_accuracy_ci_lower,
            gatekeeper_accuracy_ci_upper = EXCLUDED.gatekeeper_accuracy_ci_upper,
            slot_f1_mean                 = EXCLUDED.slot_f1_mean,
            slot_f1_ci_lower             = EXCLUDED.slot_f1_ci_lower,
            slot_f1_ci_upper             = EXCLUDED.slot_f1_ci_upper,
            hallucination_rate_mean      = EXCLUDED.hallucination_rate_mean,
            hallucination_rate_ci_lower  = EXCLUDED.hallucination_rate_ci_lower,
            hallucination_rate_ci_upper  = EXCLUDED.hallucination_rate_ci_upper,
            judge_score_mean             = EXCLUDED.judge_score_mean,
            judge_score_ci_lower         = EXCLUDED.judge_score_ci_lower,
            judge_score_ci_upper         = EXCLUDED.judge_score_ci_upper
    """, (
        slug, m["label"],
        met["json_validity_rate"]["mean"], met["json_validity_rate"]["ci_lower"], met["json_validity_rate"]["ci_upper"],
        met["intent_accuracy"]["mean"],    met["intent_accuracy"]["ci_lower"],    met["intent_accuracy"]["ci_upper"],
        met["gatekeeper_accuracy"]["mean"],met["gatekeeper_accuracy"]["ci_lower"],met["gatekeeper_accuracy"]["ci_upper"],
        met["slot_f1"]["mean"],            met["slot_f1"]["ci_lower"],            met["slot_f1"]["ci_upper"],
        met["hallucination_rate"]["mean"], met["hallucination_rate"]["ci_lower"], met["hallucination_rate"]["ci_upper"],
        js["mean"], js["ci_lower"], js["ci_upper"],
    ))

print(f"  Inserted {len(summary['models'])} model summaries.")

# ── Seed eval_rows ───────────────────────────────────────────────────────────
print("Seeding eval_rows...")
eval_data = json.loads(EVAL.read_text())

rows = []
for slug, model_data in eval_data["models"].items():
    for row in model_data["rows"]:
        rows.append((
            row["seed_id"],
            slug,
            row.get("synthetic_message"),
            row.get("gt_intent_action"),
            row.get("gt_gatekeeper_status"),
            row.get("gt_order_id"),
            row.get("gt_invoice_id"),
            row.get("json_valid"),
            row.get("pred_intent_action"),
            row.get("pred_gatekeeper_status"),
            row.get("pred_order_id"),
            row.get("pred_invoice_id"),
            row.get("intent_correct"),
            row.get("gatekeeper_correct"),
            row.get("slot_f1"),
            row.get("hallucinated_slots"),
            row.get("judge_score"),    # NULL if not present
            row.get("judge_reason"),   # NULL if not present
        ))

execute_batch(cur, """
    INSERT INTO eval_rows (
        seed_id, model_slug, synthetic_message,
        gt_intent_action, gt_gatekeeper_status, gt_order_id, gt_invoice_id,
        json_valid, pred_intent_action, pred_gatekeeper_status, pred_order_id, pred_invoice_id,
        intent_correct, gatekeeper_correct, slot_f1, hallucinated_slots,
        judge_score, judge_reason
    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    ON CONFLICT (seed_id, model_slug) DO UPDATE SET
        synthetic_message      = EXCLUDED.synthetic_message,
        gt_intent_action       = EXCLUDED.gt_intent_action,
        gt_gatekeeper_status   = EXCLUDED.gt_gatekeeper_status,
        gt_order_id            = EXCLUDED.gt_order_id,
        gt_invoice_id          = EXCLUDED.gt_invoice_id,
        json_valid             = EXCLUDED.json_valid,
        pred_intent_action     = EXCLUDED.pred_intent_action,
        pred_gatekeeper_status = EXCLUDED.pred_gatekeeper_status,
        pred_order_id          = EXCLUDED.pred_order_id,
        pred_invoice_id        = EXCLUDED.pred_invoice_id,
        intent_correct         = EXCLUDED.intent_correct,
        gatekeeper_correct     = EXCLUDED.gatekeeper_correct,
        slot_f1                = EXCLUDED.slot_f1,
        hallucinated_slots     = EXCLUDED.hallucinated_slots,
        judge_score            = EXCLUDED.judge_score,
        judge_reason           = EXCLUDED.judge_reason
""", rows, page_size=200)

print(f"  Inserted {len(rows)} eval rows.")

conn.commit()
cur.close()
conn.close()
print("Done. Database seeded successfully.")
