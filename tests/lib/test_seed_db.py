import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.seed_db_helpers import parse_model_summaries, parse_eval_rows


SUMMARY_FIXTURE = {
    "models": {
        "base_qwen": {
            "label": "Qwen-2.5-7B (base)",
            "metrics": {
                "json_validity_rate": {"mean": 0.99, "ci_lower": 0.98, "ci_upper": 1.0},
                "intent_accuracy": {"mean": 0.61, "ci_lower": 0.53, "ci_upper": 0.69},
                "gatekeeper_accuracy": {"mean": 0.33, "ci_lower": 0.26, "ci_upper": 0.41},
                "slot_f1": {"mean": 0.92, "ci_lower": 0.88, "ci_upper": 0.95},
                "hallucination_rate": {"mean": 0.02, "ci_lower": 0.0, "ci_upper": 0.05},
            },
            "judge_score": {"mean": 3.08, "ci_lower": 2.92, "ci_upper": 3.24},
        }
    }
}

ROWS_FIXTURE = {
    "models": {
        "base_qwen": {
            "label": "Qwen-2.5-7B (base)",
            "rows": [
                {
                    "seed_id": "bitext_001",
                    "synthetic_message": "I want my money back",
                    "gt_intent_action": "get_refund",
                    "gt_gatekeeper_status": "APPROVE",
                    "gt_order_id": "AX-123",
                    "gt_invoice_id": "INV-456",
                    "json_valid": True,
                    "pred_intent_action": "get_refund",
                    "pred_gatekeeper_status": "APPROVE",
                    "pred_order_id": "AX-123",
                    "pred_invoice_id": "INV-456",
                    "intent_correct": True,
                    "gatekeeper_correct": True,
                    "slot_f1": 1.0,
                    "hallucinated_slots": False,
                    "error": None,
                    "judge_score": 5,
                    "judge_reason": "Perfect match",
                }
            ],
        }
    }
}


def test_parse_model_summaries_returns_one_row_per_model():
    rows = parse_model_summaries(SUMMARY_FIXTURE)
    assert len(rows) == 1
    assert rows[0]["slug"] == "base_qwen"


def test_parse_model_summaries_flattens_metrics():
    rows = parse_model_summaries(SUMMARY_FIXTURE)
    r = rows[0]
    assert r["json_validity_mean"] == 0.99
    assert r["json_validity_ci_lower"] == 0.98
    assert r["judge_score_mean"] == 3.08


def test_parse_eval_rows_returns_one_row_per_seed_per_model():
    rows = parse_eval_rows(ROWS_FIXTURE)
    assert len(rows) == 1
    assert rows[0]["seed_id"] == "bitext_001"
    assert rows[0]["model_slug"] == "base_qwen"


def test_parse_eval_rows_includes_judge_fields():
    rows = parse_eval_rows(ROWS_FIXTURE)
    assert rows[0]["judge_score"] == 5
    assert rows[0]["judge_reason"] == "Perfect match"


def test_parse_eval_rows_handles_null_judge_score():
    fixture = json.loads(json.dumps(ROWS_FIXTURE))
    fixture["models"]["base_qwen"]["rows"][0]["judge_score"] = None
    fixture["models"]["base_qwen"]["rows"][0]["judge_reason"] = None
    rows = parse_eval_rows(fixture)
    assert rows[0]["judge_score"] is None
    assert rows[0]["judge_reason"] is None
