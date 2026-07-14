"""LLM-as-a-Judge using DeepSeek-V3.

Async, checkpointed. Reads existing scores from checkpoint JSONL on startup
and skips already-judged outputs. Writes each new score immediately after
completion so the checkpoint survives interruptions.

Checkpoint line format (one JSON object per line):
    {"key": "bitext_1589:base_qwen", "score": 4, "reason": "..."}

Key format: f"{seed_id}:{model_slug}"
Model slugs: base_qwen, ft_qwen, base_llama, ft_llama, teacher_deepseek
"""
import asyncio
import json
import re
from pathlib import Path

# Judge rubric (keep under 400 tokens total prompt)
_RUBRIC = (
    "5=Correct intent+gatekeeper_status, valid JSON, coherent response. "
    "4=Correct intent+gatekeeper_status, valid JSON, minor slot miss. "
    "3=Correct JSON+intent but wrong gatekeeper_status, or multiple slot failures. "
    "2=Partially parseable JSON, structural issues. "
    "1=Hallucinated fields, unparseable output, or critical policy breach."
)


def build_judge_prompt(candidate: dict, ground_truth: dict) -> str:
    """Build the judge prompt for a single candidate output.

    Args:
        candidate:    Predicted fields: intent_action, gatekeeper_status, order_id, invoice_id.
        ground_truth: Same fields from the golden label.

    Returns:
        Prompt string (target: < 400 tokens).
    """
    gt_str = json.dumps(ground_truth, ensure_ascii=False)
    cand_str = json.dumps(candidate, ensure_ascii=False)
    return (
        f"Rate the candidate vs ground truth. "
        f'Output only: {{"score": <1-5>, "reason": "<one sentence>"}}\n\n'
        f"RUBRIC: {_RUBRIC}\n\n"
        f"Ground truth: {gt_str}\n"
        f"Candidate: {cand_str}"
    )


def _parse_judge_response(text: str) -> dict:
    """Parse the judge's JSON response. Returns score=1 on any parse failure."""
    # Strip code fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text.strip())
        score = int(parsed["score"])
        reason = str(parsed.get("reason", ""))
        if score not in (1, 2, 3, 4, 5):
            raise ValueError(f"Score out of range: {score}")
        return {"score": score, "reason": reason}
    except Exception as exc:
        return {"score": 1, "reason": f"parse_error: {str(exc)[:80]}"}


def _load_checkpoint(path: str) -> dict:
    """Load existing judge scores from checkpoint JSONL. Returns {key: result}."""
    scores = {}
    p = Path(path)
    if not p.exists():
        return scores
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                    scores[obj["key"]] = {"score": obj["score"], "reason": obj["reason"]}
                except (json.JSONDecodeError, KeyError):
                    pass
    return scores


def _append_score(path: str, key: str, score: int, reason: str) -> None:
    """Append a single judge score to the checkpoint JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps({"key": key, "score": score, "reason": reason}) + "\n")


async def _judge_one(
    key: str,
    candidate: dict,
    ground_truth: dict,
    client,
    checkpoint_path: str,
) -> tuple[str, dict]:
    """Judge a single output using DeepSeekClient. Returns (key, result)."""
    prompt = build_judge_prompt(candidate, ground_truth)
    try:
        raw = await client.generate(
            user_prompt=prompt,
            system_prompt="You are an expert QA evaluator. Output only valid JSON.",
            max_tokens=80,
            temperature=0.0,
        )
    except Exception as exc:
        raw = f'{{"score": 1, "reason": "api_error: {str(exc)[:60]}"}}'
    result = _parse_judge_response(raw)
    _append_score(checkpoint_path, key, result["score"], result["reason"])
    return key, result


async def run_judge(
    rows_by_model: dict,
    checkpoint_path: str,
    api_key: str,
    concurrency: int = 10,
) -> dict:
    """Run DeepSeek-V3 as judge on all model outputs, with checkpointing.

    Args:
        rows_by_model:   {model_slug: [row_dict, ...]} where each row has
                         seed_id, pred_intent_action, pred_gatekeeper_status,
                         pred_order_id, pred_invoice_id, gt_intent_action,
                         gt_gatekeeper_status, gt_order_id, gt_invoice_id.
        checkpoint_path: Path to JSONL checkpoint file.
        api_key:         DeepSeek API key.
        concurrency:     Max parallel judge calls (default 10).

    Returns:
        {f"{seed_id}:{model_slug}": {"score": int, "reason": str}}
    """
    from scripts.lib.deepseek_client import DeepSeekClient

    scores = _load_checkpoint(checkpoint_path)
    already_done = set(scores.keys())
    print(f"[judge] Loaded {len(already_done)} cached scores from checkpoint.")

    client = DeepSeekClient(api_key=api_key, concurrency=concurrency)

    tasks = []
    for model_slug, rows in rows_by_model.items():
        for row in rows:
            key = f"{row['seed_id']}:{model_slug}"
            if key in already_done:
                continue
            candidate = {
                "intent_action": row.get("pred_intent_action"),
                "gatekeeper_status": row.get("pred_gatekeeper_status"),
                "order_id": row.get("pred_order_id"),
                "invoice_id": row.get("pred_invoice_id"),
            }
            ground_truth = {
                "intent_action": row["gt_intent_action"],
                "gatekeeper_status": row["gt_gatekeeper_status"],
                "order_id": row["gt_order_id"],
                "invoice_id": row["gt_invoice_id"],
            }
            tasks.append(
                _judge_one(key, candidate, ground_truth, client, checkpoint_path)
            )

    total = len(tasks)
    print(f"[judge] Running {total} new judge calls (concurrency={concurrency})…")

    results = await asyncio.gather(*tasks)
    for key, result in results:
        scores[key] = result

    print(f"[judge] Done. Total scored: {len(scores)}")
    return scores
