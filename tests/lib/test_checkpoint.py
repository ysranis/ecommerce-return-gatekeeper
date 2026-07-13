import json
import pytest
from scripts.lib.checkpoint import append_row, load_checkpoint


def test_load_checkpoint_returns_empty_list_when_file_missing(tmp_path):
    result = load_checkpoint(str(tmp_path / "missing.jsonl"))
    assert result == []


def test_load_checkpoint_returns_rows_from_existing_file(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text('{"id": 1}\n{"id": 2}\n')
    result = load_checkpoint(str(path))
    assert result == [{"id": 1}, {"id": 2}]


def test_load_checkpoint_skips_blank_lines(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n')
    result = load_checkpoint(str(path))
    assert result == [{"id": 1}, {"id": 2}]


def test_append_row_creates_file_and_writes_row(tmp_path):
    path = tmp_path / "out.jsonl"
    append_row(str(path), {"seed_id": "bitext_0001", "value": "test"})
    lines = [l for l in path.read_text().strip().split("\n") if l]
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"seed_id": "bitext_0001", "value": "test"}


def test_append_row_appends_multiple_rows(tmp_path):
    path = tmp_path / "out.jsonl"
    append_row(str(path), {"seed_id": "a"})
    append_row(str(path), {"seed_id": "b"})
    lines = [l for l in path.read_text().strip().split("\n") if l]
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"seed_id": "a"}
    assert json.loads(lines[1]) == {"seed_id": "b"}


def test_load_then_append_roundtrip(tmp_path):
    path = str(tmp_path / "data.jsonl")
    append_row(path, {"seed_id": "x", "value": 42})
    rows = load_checkpoint(path)
    assert rows == [{"seed_id": "x", "value": 42}]
