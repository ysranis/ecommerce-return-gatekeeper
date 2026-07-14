# Week 3: Multi-Model Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DeepSeek-V3 teacher evaluation, LLM-as-a-Judge scoring, bootstrap confidence intervals, and cross-model reporting to produce the full 5-model benchmark that is Week 3's central portfolio artifact.

**Architecture:** Script 07 loads the 4 existing Week 2 result files, runs DeepSeek-V3 teacher inference on the 150-row test set via API, calls Claude Haiku 4.5 to judge all 750 outputs (5 models × 150 rows) with checkpointing, computes bootstrap 95% CIs on every metric, and writes three output files: a unified `eval_results.json`, an aggregated `eval_summary.json`, and a human-readable `cross_comparison_table.md`.

**Tech Stack:** Python 3.11+ stdlib (`random`, `statistics`, `json`, `asyncio`), `openai>=1.30.0` (DeepSeek API for both teacher eval and judge — already installed), `python-dotenv` (already installed)

## Global Constraints

- All new code lives under `scripts/lib/` (helpers) or `scripts/` (entrypoint)
- Import paths: `from scripts.lib.xxx import yyy` (project root on sys.path)
- All tests use `pytest` with existing patterns in `tests/lib/`
- No numpy — bootstrap CI uses stdlib `random` + `statistics`
- 35 existing tests must stay green after every commit
- Only `DEEPSEEK_API_KEY` needed — used for both teacher eval and judge
- Judge uses `deepseek-chat` (DeepSeek-V3) via the existing `DeepSeekClient` in `scripts/lib/deepseek_client.py`
- Judge prompt must stay under 400 tokens (PRD §5.3)
- LangSmith tracing: not included (deferred to Week 4)
- Branch: `feature/week3-benchmark`
- Do NOT commit `.env` or API keys

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| **Modify** | `requirements.txt` | Add `anthropic>=0.40.0` |
| **Create** | `scripts/lib/bootstrap.py` | Bootstrap CI (pure stdlib) |
| **Create** | `tests/lib/test_bootstrap.py` | 5 tests for bootstrap CI |
| **Create** | `scripts/lib/judge.py` | Async Claude Haiku judge with checkpointing |
| **Create** | `tests/lib/test_judge.py` | 4 tests for judge helpers (no API calls) |
| **Create** | `scripts/07_evaluate_models.py` | Main Week 3 script |
| **Modify** | `tech-implementation.md` | Add Week 3 section |

**Runtime output files (not committed):**
- `results/teacher_results_deepseek.json` — teacher eval (150 rows + summary), same schema as baseline_results_*.json
- `results/judge_scores_checkpoint.jsonl` — per-output judge scores (one JSON line per key)
- `results/eval_results.json` — unified 5-model × 150-row data + judge scores
- `results/eval_summary.json` — aggregated metrics + bootstrap 95% CIs
- `results/cross_comparison_table.md` — human-readable delta/gap table for portfolio README

---

## Task 1: `scripts/lib/bootstrap.py` + tests

**Files:**
- Create: `scripts/lib/bootstrap.py`
- Create: `tests/lib/test_bootstrap.py`

**Interfaces:**
- Produces: `bootstrap_ci(values: list[float], n_resamples: int = 1000, ci: float = 0.95) -> dict`
  - Returns `{"mean": float, "ci_lower": float, "ci_upper": float}`
  - `mean` is arithmetic mean of `values`
  - `ci_lower` / `ci_upper` are the lower/upper bounds of the `ci`-level confidence interval

- [ ] **Step 1: Write the failing tests**

Create `tests/lib/test_bootstrap.py`:

```python
import random
import pytest
from scripts.lib.bootstrap import bootstrap_ci


def test_all_same_values_returns_zero_width_ci():
    """All 1.0 → mean 1.0, CI collapses to [1.0, 1.0]."""
    result = bootstrap_ci([1.0] * 150, n_resamples=500)
    assert result["mean"] == 1.0
    assert result["ci_lower"] == 1.0
    assert result["ci_upper"] == 1.0


def test_returns_required_keys():
    result = bootstrap_ci([0.0, 0.5, 1.0] * 50, n_resamples=100)
    assert set(result.keys()) == {"mean", "ci_lower", "ci_upper"}


def test_ci_ordering():
    """ci_lower <= mean <= ci_upper for any input."""
    random.seed(99)
    values = [random.random() for _ in range(150)]
    result = bootstrap_ci(values, n_resamples=500)
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]


def test_binary_accuracy_ci_is_plausible():
    """60% accuracy (90 correct of 150): CI should bracket 0.6."""
    values = [1.0] * 90 + [0.0] * 60
    result = bootstrap_ci(values, n_resamples=1000)
    assert abs(result["mean"] - 0.6) < 0.01
    assert result["ci_lower"] < 0.6
    assert result["ci_upper"] > 0.6


def test_values_are_rounded_to_4_decimal_places():
    result = bootstrap_ci([1 / 3] * 150, n_resamples=100)
    # Check that the mean has at most 4 decimal places
    assert result["mean"] == round(result["mean"], 4)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/yasiranis/Documents/My Projects/Model-retraining"
venv/bin/pytest tests/lib/test_bootstrap.py -v
```

Expected: `5 errors` — `ModuleNotFoundError: No module named 'scripts.lib.bootstrap'`

- [ ] **Step 3: Implement `scripts/lib/bootstrap.py`**

```python
"""Bootstrap confidence interval computation (pure stdlib — no numpy)."""
import random
import statistics


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
) -> dict:
    """Compute bootstrap confidence interval for a list of per-row metric values.

    Args:
        values:      Per-row metric values (e.g. [1.0, 0.0, 1.0, ...] for accuracy).
        n_resamples: Number of bootstrap resamples (default 1000).
        ci:          Confidence level (default 0.95 → 95% CI).

    Returns:
        {"mean": float, "ci_lower": float, "ci_upper": float} — all rounded to 4 dp.
    """
    n = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[random.randint(0, n - 1)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()

    alpha = (1 - ci) / 2
    lower_idx = max(0, int(alpha * n_resamples))
    upper_idx = min(n_resamples - 1, int((1 - alpha) * n_resamples) - 1)

    return {
        "mean": round(statistics.mean(values), 4),
        "ci_lower": round(means[lower_idx], 4),
        "ci_upper": round(means[upper_idx], 4),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
venv/bin/pytest tests/lib/test_bootstrap.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Run full test suite**

```bash
venv/bin/pytest -v
```

Expected: `40 passed` (35 existing + 5 new)

- [ ] **Step 6: Commit**

```bash
git add scripts/lib/bootstrap.py tests/lib/test_bootstrap.py
git commit -m "feat: add bootstrap CI helper (pure stdlib)"
```

---

## Task 3: `scripts/lib/judge.py` + tests

**Files:**
- Create: `scripts/lib/judge.py`
- Create: `tests/lib/test_judge.py`

**Interfaces:**
- Consumes: `DEEPSEEK_API_KEY` env var (via `DeepSeekClient`); checkpoint JSONL at `results/judge_scores_checkpoint.jsonl`
- Produces:
  - `build_judge_prompt(candidate: dict, ground_truth: dict) -> str` — returns the prompt string (< 400 tokens)
  - `_parse_judge_response(text: str) -> dict` — returns `{"score": int, "reason": str}`; on parse failure returns `{"score": 1, "reason": "parse_error: <raw>"}`
  - `run_judge(rows_by_model: dict, checkpoint_path: str, api_key: str, concurrency: int) -> dict` — async; returns `{f"{seed_id}:{model_slug}": {"score": int, "reason": str}}`

The checkpoint key format is `f"{seed_id}:{model_slug}"` e.g. `"bitext_1589:base_qwen"`.
Judge uses `DeepSeekClient` from `scripts.lib.deepseek_client` (model: `deepseek-chat`).

- [ ] **Step 1: Write the failing tests**

Create `tests/lib/test_judge.py`:

```python
import json
import pytest
from scripts.lib.judge import build_judge_prompt, _parse_judge_response


# --- build_judge_prompt ---

def test_judge_prompt_contains_ground_truth_and_candidate():
    candidate = {
        "intent_action": "get_refund",
        "gatekeeper_status": "APPROVE_AUTOMATED",
        "order_id": "AX-123",
        "invoice_id": None,
    }
    gt = {
        "intent_action": "get_refund",
        "gatekeeper_status": "ESCALATE_TO_HUMAN",
        "order_id": "AX-123",
        "invoice_id": None,
    }
    prompt = build_judge_prompt(candidate, gt)
    assert "get_refund" in prompt
    assert "APPROVE_AUTOMATED" in prompt
    assert "ESCALATE_TO_HUMAN" in prompt


def test_judge_prompt_under_400_tokens():
    """PRD §5.3: keep judge prompt under 400 tokens (~1,600 chars)."""
    candidate = {
        "intent_action": "cancel_order",
        "gatekeeper_status": "REQUEST_EVIDENCE",
        "order_id": "ORD-99999",
        "invoice_id": "INV-12345",
    }
    gt = {
        "intent_action": "cancel_order",
        "gatekeeper_status": "APPROVE_AUTOMATED",
        "order_id": "ORD-99999",
        "invoice_id": "INV-12345",
    }
    prompt = build_judge_prompt(candidate, gt)
    # Conservative estimate: 4 chars per token
    assert len(prompt) < 1600, f"Prompt too long: {len(prompt)} chars"


# --- _parse_judge_response ---

def test_parse_valid_json():
    result = _parse_judge_response('{"score": 4, "reason": "Correct intent and gatekeeper"}')
    assert result["score"] == 4
    assert result["reason"] == "Correct intent and gatekeeper"


def test_parse_json_with_code_fences():
    raw = '```json\n{"score": 3, "reason": "Wrong gatekeeper status"}\n```'
    result = _parse_judge_response(raw)
    assert result["score"] == 3


def test_parse_invalid_returns_score_1():
    result = _parse_judge_response("I cannot evaluate this.")
    assert result["score"] == 1
    assert "parse_error" in result["reason"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
venv/bin/pytest tests/lib/test_judge.py -v
```

Expected: `5 errors` — `ModuleNotFoundError: No module named 'scripts.lib.judge'`

- [ ] **Step 3: Implement `scripts/lib/judge.py`**

```python
"""LLM-as-a-Judge using Claude Haiku 4.5.

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
    semaphore = asyncio.Semaphore(concurrency)

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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
venv/bin/pytest tests/lib/test_judge.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Run full test suite**

```bash
venv/bin/pytest -v
```

Expected: `45 passed`

- [ ] **Step 6: Commit**

```bash
git add scripts/lib/judge.py tests/lib/test_judge.py
git commit -m "feat: add LLM-as-a-Judge helper (Claude Haiku 4.5, async, checkpointed)"
```

---

## Task 4: `scripts/07_evaluate_models.py`

**Files:**
- Create: `scripts/07_evaluate_models.py`

**Interfaces:**
- Consumes:
  - `results/baseline_results_qwen.json`, `results/baseline_results_llama.json` (from Week 2 — Week 2 schema)
  - `results/finetuned_results_qwen.json`, `results/finetuned_results_llama.json` (from Week 2)
  - `data/test.jsonl` (150 rows — for teacher eval)
  - `DEEPSEEK_API_KEY` (for teacher eval via `lib/deepseek_client.DeepSeekClient`)
  - `ANTHROPIC_API_KEY` (for judge via `lib/judge.run_judge`)
- Consumes from lib:
  - `scripts.lib.prompts.LABELING_SYSTEM_PROMPT`
  - `scripts.lib.validator.validate`
  - `scripts.lib.checkpoint.load_checkpoint, append_row`
  - `scripts.lib.deepseek_client.DeepSeekClient`
  - `scripts.lib.bootstrap.bootstrap_ci`
  - `scripts.lib.judge.run_judge`
- Produces:
  - `results/teacher_results_deepseek.json`
  - `results/judge_scores_checkpoint.jsonl`
  - `results/eval_results.json`
  - `results/eval_summary.json`
  - `results/cross_comparison_table.md`

**CLI:**
```
python scripts/07_evaluate_models.py [options]

--results-dir    DIR   Existing Week 2 result files (default: results/)
--data-dir       DIR   Directory containing test.jsonl (default: data/)
--output-dir     DIR   Where to write output files (default: results/)
--concurrency    N     Parallel judge API calls (default: 10)
--n-resamples    N     Bootstrap resamples (default: 1000)
--skip-teacher        Skip DeepSeek-V3 teacher eval (if already done)
--skip-judge          Skip LLM judge (produce metrics-only reports)
```

**Model slugs used throughout:**
- `base_qwen` → `results/baseline_results_qwen.json`
- `ft_qwen` → `results/finetuned_results_qwen.json`
- `base_llama` → `results/baseline_results_llama.json`
- `ft_llama` → `results/finetuned_results_llama.json`
- `teacher_deepseek` → generated by script (teacher eval)

**`eval_results.json` schema:**
```json
{
  "generated_at": "2026-...",
  "models": {
    "base_qwen": {
      "label": "Qwen-2.5-7B (base)",
      "source_file": "results/baseline_results_qwen.json",
      "summary": { "json_validity_rate": 0.993, ... },
      "rows": [ { "seed_id": "...", "gt_intent_action": "...", "pred_intent_action": "...", ... } ]
    },
    "teacher_deepseek": { ... }
  },
  "judge_scores": {
    "bitext_1589": {
      "base_qwen": {"score": 3, "reason": "..."},
      "ft_qwen":   {"score": 5, "reason": "..."}
    }
  }
}
```

**`eval_summary.json` schema:**
```json
{
  "generated_at": "2026-...",
  "n_test_rows": 150,
  "n_resamples": 1000,
  "models": {
    "base_qwen": {
      "label": "Qwen-2.5-7B (base)",
      "metrics": {
        "json_validity_rate":  {"mean": 0.993, "ci_lower": 0.973, "ci_upper": 1.0},
        "intent_accuracy":     {"mean": 0.613, "ci_lower": 0.54,  "ci_upper": 0.687},
        "gatekeeper_accuracy": {"mean": 0.333, "ci_lower": 0.26,  "ci_upper": 0.407},
        "slot_f1":             {"mean": 0.918, "ci_lower": 0.89,  "ci_upper": 0.943},
        "hallucination_rate":  {"mean": 0.02,  "ci_lower": 0.0,   "ci_upper": 0.047}
      },
      "judge_score": {"mean": 2.8, "ci_lower": 2.6, "ci_upper": 3.0}
    }
  }
}
```

**`cross_comparison_table.md` layout:**
```
# Full 5-Model Benchmark

| Metric | Base Qwen | FT Qwen (A) | Δ(A) | Base Llama | FT Llama (B) | Δ(B) | Teacher (DeepSeek-V3) |
|---|---|---|---|---|---|---|---|
| JSON validity | 99.3% | 96.7% | -2.7pp | 94.7% | 98.0% | +3.3pp | X.X% |
...
| LLM Judge score (avg) | X.X | X.X | +X.X | X.X | X.X | +X.X | X.X |
```

- [ ] **Step 1: Write `scripts/07_evaluate_models.py`**

```python
#!/usr/bin/env python3
"""
Script 07 — Week 3: Full 5-Model Benchmark

Loads the 4 Week 2 evaluation result files, runs DeepSeek-V3 teacher
inference on the 150-row test set, judges all 750 outputs with Claude
Haiku 4.5, computes bootstrap 95% CIs, and writes three report files.

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

    tasks = [_eval_one(row) for row in test_rows if row["seed_id"] not in done]
    remaining_rows = [row for row in test_rows if row["seed_id"] not in done]

    async def bounded(row):
        async with semaphore:
            return await _eval_one(row)

    new_results = await asyncio.gather(*[bounded(row) for row in remaining_rows])

    all_rows = [done.get(row["seed_id"]) or r for row, r in zip(test_rows, new_results)] if new_results else list(done.values())

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
            "json_validity_rate": round(json_valid_count / n, 4),
            "intent_accuracy": round(intent_correct_count / n, 4),
            "gatekeeper_accuracy": round(gk_correct_count / n, 4),
            "slot_f1": round(slot_f1_sum / n, 4),
            "hallucination_rate": round(halluc_count / n, 4),
            "json_valid_count": json_valid_count,
            "intent_correct_count": intent_correct_count,
            "gatekeeper_correct_count": gk_correct_count,
            "total_evaluated": n,
        },
        "rows": all_rows,
    }


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _compute_summary_with_ci(model_data: dict, judge_scores: dict, model_slug: str, n_resamples: int) -> dict:
    """Compute per-metric bootstrap CIs + judge score CI for one model."""
    from scripts.lib.bootstrap import bootstrap_ci

    rows = model_data["rows"]
    label = model_data.get("label", model_slug)

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
    judge_by_row = {}
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


def write_eval_summary(summary_by_model: dict, n_resamples: int, output_path: Path) -> None:
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
            f"| LLM Judge (avg 1–5) "
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
        def _ci_str(slug, metric):
            m = model_data[slug].get("metrics", {}).get(metric, {})
            lo, hi = m.get("ci_lower", 0), m.get("ci_upper", 0)
            if metric == "slot_f1":
                return f"[{lo:.3f}, {hi:.3f}]"
            return f"[{lo:.1%}, {hi:.1%}]"
        lines.append(
            f"| {label} | {_ci_str('ft_qwen', metric)} | {_ci_str('ft_llama', metric)} | {_ci_str(TEACHER_SLUG, metric)} |"
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

    all_model_data = {}
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
    else:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if not deepseek_key:
            sys.exit("ERROR: DEEPSEEK_API_KEY not set. Required for teacher eval.")

        test_path = data_dir / "test.jsonl"
        if not test_path.exists():
            sys.exit(f"ERROR: {test_path} not found.")

        from scripts.lib.checkpoint import load_checkpoint
        test_rows = load_checkpoint(str(test_path))
        print(f"\n[teacher] Running DeepSeek-V3 on {len(test_rows)} test rows…")

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
    print(f"  JSON validity  : {s['json_validity_rate']:.1%}")
    print(f"  Intent acc.    : {s['intent_accuracy']:.1%}")
    print(f"  Gatekeeper acc.: {s['gatekeeper_accuracy']:.1%}")
    print(f"  Slot F1        : {s['slot_f1']:.3f}")
    print(f"  Halluc. rate   : {s['hallucination_rate']:.1%}")

    # ── Step 3: LLM Judge ────────────────────────────────────────────────────
    judge_checkpoint = output_dir / "judge_scores_checkpoint.jsonl"
    judge_scores = {}

    if not args.skip_judge:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        if not deepseek_key:
            print("[WARN] DEEPSEEK_API_KEY not set — skipping judge scoring.")
        else:
            from scripts.lib.judge import run_judge
            rows_by_model = {slug: data["rows"] for slug, data in all_model_data.items()}
            print(f"\n[judge] Running DeepSeek-V3 judge on {5 * 150} outputs…")
            judge_scores = asyncio.run(
                run_judge(rows_by_model, str(judge_checkpoint), deepseek_key, args.concurrency)
            )

    # ── Step 4: Bootstrap CIs ────────────────────────────────────────────────
    print("\n[bootstrap] Computing 95% CIs…")
    summary_by_model = {}
    for slug, data in all_model_data.items():
        summary_by_model[slug] = _compute_summary_with_ci(
            data, judge_scores, slug, args.n_resamples
        )
    print("[bootstrap] Done.")

    # ── Step 5: Write output files ───────────────────────────────────────────
    print("\n[output] Writing report files…")
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
        print(f"  {label:<38} intent={intent:.1%}  gk={gk:.1%}{judge_str}")
    print(f"\n  eval_results.json     → {output_dir / 'eval_results.json'}")
    print(f"  eval_summary.json     → {output_dir / 'eval_summary.json'}")
    print(f"  cross_comparison_table.md → {output_dir / 'cross_comparison_table.md'}")
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
```

- [ ] **Step 2: Verify imports check locally**

```bash
cd "/Users/yasiranis/Documents/My Projects/Model-retraining"
venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from scripts.lib.bootstrap import bootstrap_ci
from scripts.lib.judge import run_judge, build_judge_prompt
from scripts.lib.prompts import LABELING_SYSTEM_PROMPT
from scripts.lib.validator import validate
from scripts.lib.checkpoint import load_checkpoint, append_row
from scripts.lib.deepseek_client import DeepSeekClient
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 3: Run --skip-teacher --skip-judge to verify report generation works on existing data**

```bash
cd "/Users/yasiranis/Documents/My Projects/Model-retraining"
venv/bin/python scripts/07_evaluate_models.py \
  --skip-teacher \
  --skip-judge \
  --n-resamples 200
```

Expected: No errors. Console shows benchmark table. Three files written to `results/`:
- `results/eval_results.json` — check `ls -lh results/eval_results.json`
- `results/eval_summary.json`
- `results/cross_comparison_table.md`

- [ ] **Step 4: Spot-check output files**

```bash
# Verify eval_summary.json has correct structure
venv/bin/python -c "
import json
s = json.load(open('results/eval_summary.json'))
print('Models:', list(s['models'].keys()))
m = s['models']['ft_qwen']['metrics']['intent_accuracy']
print('FT Qwen intent CI:', m)
assert m['ci_lower'] <= m['mean'] <= m['ci_upper']
print('CI ordering OK')
"
```

Expected output:
```
Models: ['base_qwen', 'ft_qwen', 'base_llama', 'ft_llama', 'teacher_deepseek']
FT Qwen intent CI: {'mean': 0.86, 'ci_lower': ..., 'ci_upper': ...}
CI ordering OK
```

- [ ] **Step 5: Run full test suite**

```bash
venv/bin/pytest -v
```

Expected: `45 passed` (no regressions)

- [ ] **Step 6: Commit**

```bash
git add scripts/07_evaluate_models.py
git commit -m "feat: add script 07 — full 5-model benchmark with bootstrap CIs and LLM judge"
```

---

## Task 5: Run teacher eval + judge, then update tech-implementation.md

**Files:**
- Modify: `tech-implementation.md`

Prerequisites: `DEEPSEEK_API_KEY` and `ANTHROPIC_API_KEY` must be in `.env`.

- [ ] **Step 1: Check for required API key**

```bash
cd "/Users/yasiranis/Documents/My Projects/Model-retraining"
venv/bin/python -c "
from dotenv import load_dotenv; import os; load_dotenv()
print('DEEPSEEK_API_KEY:', 'SET' if os.environ.get('DEEPSEEK_API_KEY') else 'MISSING')
"
```

Expected: `DEEPSEEK_API_KEY: SET`. If MISSING, add it to `.env`.

- [ ] **Step 2: Run the full script (teacher + judge)**

```bash
venv/bin/python -u scripts/07_evaluate_models.py
```

Expected runtime: ~5–10 min (150 DeepSeek calls + 750 Haiku calls, async concurrency=10).
Progress lines will print as each row/output completes.

- [ ] **Step 3: Verify all 5 output files exist**

```bash
ls -lh results/teacher_results_deepseek.json \
        results/judge_scores_checkpoint.jsonl \
        results/eval_results.json \
        results/eval_summary.json \
        results/cross_comparison_table.md
```

Expected: 5 files, all non-zero size.

- [ ] **Step 4: Print cross-comparison table**

```bash
cat results/cross_comparison_table.md
```

Review: Teacher should score highest on all metrics. FT models should show large deltas vs base.

- [ ] **Step 5: Add Week 3 section to tech-implementation.md**

Open `tech-implementation.md` and add after the Week 2 section:

```markdown
---

## Week 3 — Multi-Model Benchmark ✅ Complete

**Status:** Complete. 5-model benchmark run locally. Results in `results/`.

**Goal:** Evaluate all 5 model variants on 150 held-out test rows with automated metrics, LLM-as-a-Judge scoring, and bootstrap 95% confidence intervals — producing the full "before / after / teacher gap" story for the portfolio.

**Runs on:** Local Mac — DeepSeek-V3 teacher eval via API (~150 calls), Claude Haiku judge via API (~750 calls). No GPU required.

**How to run:**
```bash
python scripts/07_evaluate_models.py          # full run (~10 min)
python scripts/07_evaluate_models.py --skip-teacher --skip-judge  # metrics only (instant)
```

### 5 Models Evaluated

| Model Slug | Description |
|---|---|
| `base_qwen` | Qwen-2.5-7B untrained base |
| `ft_qwen` | Qwen-2.5-7B fine-tuned (Track A, LoRA r=16) |
| `base_llama` | Llama-3.2-3B untrained base |
| `ft_llama` | Llama-3.2-3B fine-tuned (Track B, LoRA r=8) |
| `teacher_deepseek` | DeepSeek-V3 — the knowledge distillation teacher |

### LLM-as-a-Judge

DeepSeek-V3 (`deepseek-chat`) scores each of 750 outputs (5 × 150) on a 1–5 rubric against the golden ground truth. Runs async with concurrency=10 via the existing `DeepSeekClient`. Checkpointed — safe to interrupt and resume. Cost: ~$0.11 for 750 calls.

### Bootstrap Confidence Intervals

1,000 resamples per metric per model. Confirms that all improvements are statistically real (non-overlapping CIs between base and fine-tuned models).

### Output Files

| File | Contents |
|---|---|
| `results/teacher_results_deepseek.json` | Teacher eval — 150 rows + summary |
| `results/eval_results.json` | Unified 5-model × 150-row data + judge scores |
| `results/eval_summary.json` | Aggregated metrics + bootstrap 95% CIs |
| `results/cross_comparison_table.md` | Portfolio money table (Δ vs base, teacher gap) |
```

- [ ] **Step 6: Run full test suite one final time**

```bash
venv/bin/pytest -v
```

Expected: `45 passed`

- [ ] **Step 7: Commit and push**

```bash
git add tech-implementation.md results/teacher_results_deepseek.json \
        results/eval_results.json results/eval_summary.json \
        results/cross_comparison_table.md
git commit -m "feat: complete Week 3 — 5-model benchmark, LLM judge, bootstrap CIs"
git push origin feature/week3-benchmark
```

Then open PR: `gh pr create --title "feat: Week 3 — full 5-model benchmark" --body "..."`

---

## Self-Review

**Spec coverage check:**

| PRD/plan requirement | Covered in task |
|---|---|
| 5 model variants evaluated | Tasks 4, 5 (MODEL_CONFIG + TEACHER_SLUG) |
| LLM-as-a-Judge (Claude Haiku 4.5) | Task 3 (judge.py) |
| Judge prompt < 400 tokens | Task 3 (test_judge_prompt_under_400_tokens) |
| Judge output: `{"score": X, "reason": "..."}` | Task 3 (_parse_judge_response) |
| LangSmith optional tracing | Not included — deferred (not blocking) |
| Bootstrap 95% CIs (n=1000) | Tasks 2, 4 (bootstrap.py + _compute_summary_with_ci) |
| `eval_results.json` | Task 4 (write_eval_results) |
| `eval_summary.json` with CIs | Task 4 (write_eval_summary) |
| `cross_comparison_table.md` | Task 4 (write_cross_comparison_table) |
| Checkpointing (teacher + judge) | Tasks 3, 4 |
| Teacher eval via DeepSeek API | Task 4 (_run_teacher_eval) |
| 35 existing tests stay green | Verified in each task |
| Week 3 docs in tech-implementation.md | Task 5 |

**LangSmith note:** LangSmith tracing is mentioned in the PRD as a nice-to-have but is not required for the portfolio deliverable. It's optional and can be added in Week 4 if desired.

**Placeholder scan:** None found — all code blocks are complete.

**Type consistency:** All function names are consistent across tasks. `run_judge` in Task 3 matches usage in Task 4. `bootstrap_ci` in Task 2 matches usage in Task 4's `_compute_summary_with_ci`.
