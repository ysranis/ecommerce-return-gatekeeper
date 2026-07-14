"""
Parsing helpers for scripts/08_seed_db.py.
Extracted so they can be tested without a live DB connection.
"""


def parse_model_summaries(data: dict) -> list[dict]:
    """Parse eval_summary.json into model_summaries table rows."""
    rows = []
    for slug, model in data["models"].items():
        m = model["metrics"]
        j = model["judge_score"]
        rows.append({
            "slug": slug,
            "label": model["label"],
            "json_validity_mean": m["json_validity_rate"]["mean"],
            "json_validity_ci_lower": m["json_validity_rate"]["ci_lower"],
            "json_validity_ci_upper": m["json_validity_rate"]["ci_upper"],
            "intent_accuracy_mean": m["intent_accuracy"]["mean"],
            "intent_accuracy_ci_lower": m["intent_accuracy"]["ci_lower"],
            "intent_accuracy_ci_upper": m["intent_accuracy"]["ci_upper"],
            "gatekeeper_accuracy_mean": m["gatekeeper_accuracy"]["mean"],
            "gatekeeper_accuracy_ci_lower": m["gatekeeper_accuracy"]["ci_lower"],
            "gatekeeper_accuracy_ci_upper": m["gatekeeper_accuracy"]["ci_upper"],
            "slot_f1_mean": m["slot_f1"]["mean"],
            "slot_f1_ci_lower": m["slot_f1"]["ci_lower"],
            "slot_f1_ci_upper": m["slot_f1"]["ci_upper"],
            "hallucination_rate_mean": m["hallucination_rate"]["mean"],
            "hallucination_rate_ci_lower": m["hallucination_rate"]["ci_lower"],
            "hallucination_rate_ci_upper": m["hallucination_rate"]["ci_upper"],
            "judge_score_mean": j["mean"],
            "judge_score_ci_lower": j["ci_lower"],
            "judge_score_ci_upper": j["ci_upper"],
        })
    return rows


def parse_eval_rows(data: dict) -> list[dict]:
    """Parse eval_results.json into eval_rows table rows."""
    rows = []
    for slug, model in data["models"].items():
        for row in model.get("rows", []):
            rows.append({
                "seed_id": row["seed_id"],
                "model_slug": slug,
                "synthetic_message": row.get("synthetic_message"),
                "gt_intent_action": row.get("gt_intent_action"),
                "gt_gatekeeper_status": row.get("gt_gatekeeper_status"),
                "gt_order_id": row.get("gt_order_id"),
                "gt_invoice_id": row.get("gt_invoice_id"),
                "json_valid": row.get("json_valid"),
                "pred_intent_action": row.get("pred_intent_action"),
                "pred_gatekeeper_status": row.get("pred_gatekeeper_status"),
                "pred_order_id": row.get("pred_order_id"),
                "pred_invoice_id": row.get("pred_invoice_id"),
                "intent_correct": row.get("intent_correct"),
                "gatekeeper_correct": row.get("gatekeeper_correct"),
                "slot_f1": row.get("slot_f1"),
                "hallucinated_slots": row.get("hallucinated_slots"),
                "judge_score": row.get("judge_score"),
                "judge_reason": row.get("judge_reason"),
            })
    return rows
