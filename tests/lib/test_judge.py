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
