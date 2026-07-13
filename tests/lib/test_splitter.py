import pytest
from scripts.lib.splitter import check_quality_gate, split_dataset


def _rows(n: int) -> list[dict]:
    return [{"seed_id": str(i)} for i in range(n)]


def test_split_produces_correct_counts():
    train, val, test = split_dataset(_rows(1500))
    assert len(train) == 1200
    assert len(val) == 150
    assert len(test) == 150


def test_split_covers_all_rows():
    rows = _rows(1500)
    train, val, test = split_dataset(rows)
    assert len(train) + len(val) + len(test) == 1500


def test_split_no_overlap():
    rows = _rows(1500)
    train, val, test = split_dataset(rows)
    train_ids = {r["seed_id"] for r in train}
    val_ids = {r["seed_id"] for r in val}
    test_ids = {r["seed_id"] for r in test}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_is_reproducible():
    rows = _rows(1500)
    t1, v1, te1 = split_dataset(rows, seed=42)
    t2, v2, te2 = split_dataset(rows, seed=42)
    assert t1 == t2
    assert v1 == v2
    assert te1 == te2


def test_split_differs_with_different_seed():
    rows = _rows(1500)
    t1, _, _ = split_dataset(rows, seed=42)
    t2, _, _ = split_dataset(rows, seed=99)
    assert t1 != t2


def test_quality_gate_passes_at_exactly_1500():
    check_quality_gate(_rows(1500), min_rows=1500)  # must not raise


def test_quality_gate_passes_above_threshold():
    check_quality_gate(_rows(2000), min_rows=1500)  # must not raise


def test_quality_gate_fails_below_threshold():
    with pytest.raises(SystemExit):
        check_quality_gate(_rows(1499), min_rows=1500)


def test_quality_gate_fails_at_zero():
    with pytest.raises(SystemExit):
        check_quality_gate([], min_rows=1500)
