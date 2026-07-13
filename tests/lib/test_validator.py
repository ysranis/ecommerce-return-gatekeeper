import json
import pytest
from scripts.lib.validator import REQUIRED_KEYS, validate


def _valid_row() -> dict:
    """Return a minimal valid golden row with all required keys."""
    return {
        "chain_of_thought": "Customer ordered 15 days ago; within 30-day window.",
        "intent_action": "get_refund",
        "extracted_slots": {"order_id": "AX-1234", "invoice_id": None,
                            "return_window_days": 15, "item_condition": None},
        "gatekeeper_status": "APPROVE_AUTOMATED",
        "confidence_score": 0.95,
        "user_facing_response": "Your refund has been approved.",
    }


def test_validate_returns_true_with_all_required_keys():
    row = _valid_row()
    valid, parsed = validate(json.dumps(row))
    assert valid is True
    assert parsed == row


def test_validate_returns_false_for_invalid_json():
    valid, parsed = validate("not json {{{")
    assert valid is False
    assert parsed is None


def test_validate_returns_false_for_empty_string():
    valid, parsed = validate("")
    assert valid is False
    assert parsed is None


def test_validate_returns_false_when_one_key_missing():
    row = _valid_row()
    del row["gatekeeper_status"]
    valid, parsed = validate(json.dumps(row))
    assert valid is False
    assert parsed is None


def test_validate_returns_false_when_multiple_keys_missing():
    valid, parsed = validate(json.dumps({"chain_of_thought": "only one key"}))
    assert valid is False
    assert parsed is None


def test_required_keys_contains_six_entries():
    assert len(REQUIRED_KEYS) == 6


def test_validate_preserves_extra_keys():
    row = _valid_row()
    row["extra_key"] = "extra_value"
    valid, parsed = validate(json.dumps(row))
    assert valid is True
    assert parsed["extra_key"] == "extra_value"
