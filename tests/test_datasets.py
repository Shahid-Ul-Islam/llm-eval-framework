from pathlib import Path

import pytest

from llm_eval.datasets import DatasetError, load_dataset

ROOT = Path(__file__).resolve().parents[1]


def test_sample_datasets_load():
    coding = load_dataset(ROOT / "datasets" / "coding.jsonl")
    reasoning = load_dataset(ROOT / "datasets" / "reasoning.jsonl")
    assert len(coding) >= 8 and len(reasoning) >= 8
    assert any("sql" in ex.tags for ex in coding)
    assert all(ex.expected is not None for ex in reasoning)


def test_tag_filter():
    sql_only = load_dataset(ROOT / "datasets" / "coding.jsonl", tags=["sql"])
    assert sql_only and all("sql" in ex.tags for ex in sql_only)


def test_resolve_by_name(monkeypatch):
    monkeypatch.chdir(ROOT)
    assert load_dataset("reasoning")


def test_default_id_and_errors(tmp_path):
    f = tmp_path / "d.jsonl"
    f.write_text('{"prompt": "hi", "expected": "x"}\n\n', encoding="utf-8")
    [ex] = load_dataset(f)
    assert ex.id == "d-1" and ex.tags == []

    f.write_text('{"id": "a", "prompt": "1"}\n{"id": "a", "prompt": "2"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(f)

    f.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(DatasetError, match=":1: invalid JSON"):
        load_dataset(f)

    f.write_text('{"id": "a"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="prompt"):
        load_dataset(f)
