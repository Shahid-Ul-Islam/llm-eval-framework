import sqlite3

import pytest

from llm_eval.models import RunRecord, ScoreResult, Usage
from llm_eval.storage import ResultStore, RunInfo, load_jsonl, new_run_id


def _record(run_id, ex_id, score, cost=0.01):
    return RunRecord(
        run_id=run_id,
        example_id=ex_id,
        model="mock:echo",
        template="baseline",
        prompt="p",
        output="o",
        tags=["t1"],
        scores=[ScoreResult(scorer="exact", score=score, passed=score == 1.0)],
        usage=Usage(input_tokens=3, output_tokens=2, cost_usd=cost),
    )


def test_roundtrip_jsonl_and_sqlite(tmp_path):
    store = ResultStore(tmp_path)
    rid = new_run_id()
    store.start_run(RunInfo(run_id=rid, dataset="d", models=["mock:echo"], scorers=["exact"]))
    store.add_record(_record(rid, "a", 1.0))
    store.add_record(_record(rid, "b", 0.0, cost=0.02))
    info = store.finish_run(rid)

    assert info.status == "completed" and info.n_records == 2
    assert info.total_cost_usd == pytest.approx(0.03)
    assert info.models == ["mock:echo"]

    from_db = store.load_records(rid)
    from_file = load_jsonl(store.jsonl_path(rid))
    assert [r.example_id for r in from_db] == ["a", "b"]
    assert from_db == from_file

    con = sqlite3.connect(store.db_path)
    assert con.execute("SELECT AVG(score) FROM scores WHERE run_id=?", (rid,)).fetchone()[0] == 0.5
    con.close()
    store.close()


def test_list_runs_latest_and_unknown(tmp_path):
    store = ResultStore(tmp_path)
    assert store.latest_run_id() is None
    for rid in ["r1", "r2"]:
        store.start_run(RunInfo(run_id=rid))
    assert store.latest_run_id() == "r2"
    assert [r.run_id for r in store.list_runs()] == ["r2", "r1"]
    assert store.load_records(["r1", "r2"]) == []
    with pytest.raises(KeyError):
        store.load_records("nope")
    store.close()
