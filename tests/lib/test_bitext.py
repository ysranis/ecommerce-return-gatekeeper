import pytest
from scripts.lib.bitext import TARGET_INTENTS, filter_bitext, sample_balanced


def _make_rows(intents: list[str], per_intent: int) -> list[dict]:
    rows = []
    for intent in intents:
        for i in range(per_intent):
            rows.append({"original_message": f"{intent} msg {i}", "intent": intent})
    return rows


def test_filter_bitext_keeps_target_intents():
    rows = [
        {"original_message": "msg1", "intent": "get_refund"},
        {"original_message": "msg2", "intent": "other_intent"},
        {"original_message": "msg3", "intent": "cancel_order"},
    ]
    result = filter_bitext(rows, TARGET_INTENTS)
    assert len(result) == 2
    assert all(r["intent"] in TARGET_INTENTS for r in result)


def test_filter_bitext_removes_non_target_intents():
    rows = [{"original_message": "msg", "intent": "unknown"}]
    result = filter_bitext(rows, TARGET_INTENTS)
    assert result == []


def test_filter_bitext_empty_input():
    assert filter_bitext([], TARGET_INTENTS) == []


def test_sample_balanced_returns_correct_total():
    rows = _make_rows(TARGET_INTENTS, 500)
    result = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    assert len(result) == len(TARGET_INTENTS) * 400


def test_sample_balanced_equal_distribution():
    rows = _make_rows(TARGET_INTENTS, 500)
    result = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    from collections import Counter
    counts = Counter(r["intent"] for r in result)
    for intent in TARGET_INTENTS:
        assert counts[intent] == 400


def test_sample_balanced_is_reproducible():
    rows = _make_rows(TARGET_INTENTS, 500)
    r1 = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    r2 = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    assert [r["original_message"] for r in r1] == [r["original_message"] for r in r2]


def test_sample_balanced_different_seeds_differ():
    rows = _make_rows(TARGET_INTENTS, 500)
    r1 = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    r2 = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=99)
    # With 2000 rows sampled randomly, these will differ with overwhelming probability
    assert [r["original_message"] for r in r1] != [r["original_message"] for r in r2]


def test_sample_balanced_takes_all_when_fewer_than_per_intent():
    # Only 200 rows per intent — should take all 200
    rows = _make_rows(TARGET_INTENTS, 200)
    result = sample_balanced(rows, TARGET_INTENTS, per_intent=400, seed=42)
    assert len(result) == len(TARGET_INTENTS) * 200


def test_target_intents_contains_five_entries():
    assert len(TARGET_INTENTS) == 5
    assert "get_refund" in TARGET_INTENTS
    assert "cancel_order" in TARGET_INTENTS
    assert "track_refund" in TARGET_INTENTS
    assert "complaint" in TARGET_INTENTS
    assert "check_refund_policy" in TARGET_INTENTS
