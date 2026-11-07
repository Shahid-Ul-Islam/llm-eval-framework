"""Result persistence: one JSONL file per run + a shared SQLite database.

Layout (under `results_dir`):
    results.db          runs, results, scores tables (queryable with any SQLite tool)
    <run_id>.jsonl      one RunRecord per line (portable, append-only)

The full RunRecord JSON is stored in `results.record`; other columns are denormalized
for convenient SQL queries.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from llm_eval.models import RunRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    dataset TEXT,
    models TEXT NOT NULL,
    templates TEXT NOT NULL,
    scorers TEXT NOT NULL,
    n_records INTEGER DEFAULT 0,
    total_cost_usd REAL,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    example_id TEXT NOT NULL,
    model TEXT NOT NULL,
    template TEXT NOT NULL,
    tags TEXT NOT NULL,
    mean_score REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_s REAL,
    cached INTEGER,
    error TEXT,
    created_at TEXT NOT NULL,
    record TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    result_id INTEGER NOT NULL REFERENCES results(id),
    run_id TEXT NOT NULL,
    scorer TEXT NOT NULL,
    score REAL NOT NULL,
    passed INTEGER,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id, scorer);
"""


class RunInfo(BaseModel):
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    status: str = "running"
    dataset: str | None = None
    models: list[str] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list)
    scorers: list[str] = Field(default_factory=list)
    n_records: int = 0
    total_cost_usd: float | None = None
    notes: str | None = None


def new_run_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


class ResultStore:
    def __init__(self, results_dir: str | Path):
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "results.db"
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ write

    def jsonl_path(self, run_id: str) -> Path:
        return self.dir / f"{run_id}.jsonl"

    def start_run(self, info: RunInfo) -> None:
        self._conn.execute(
            "INSERT INTO runs (run_id, created_at, status, dataset, models, templates, scorers,"
            " notes) VALUES (?,?,?,?,?,?,?,?)",
            (
                info.run_id,
                info.created_at.isoformat(),
                info.status,
                info.dataset,
                json.dumps(info.models),
                json.dumps(info.templates),
                json.dumps(info.scorers),
                info.notes,
            ),
        )
        self._conn.commit()
        self.jsonl_path(info.run_id).touch()

    def add_record(self, record: RunRecord) -> None:
        with self.jsonl_path(record.run_id).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(record.model_dump_json() + "\n")
        cur = self._conn.execute(
            "INSERT INTO results (run_id, example_id, model, template, tags, mean_score,"
            " input_tokens, output_tokens, cost_usd, latency_s, cached, error, created_at,"
            " record) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.run_id,
                record.example_id,
                record.model,
                record.template,
                json.dumps(record.tags),
                record.mean_score,
                record.usage.input_tokens,
                record.usage.output_tokens,
                record.usage.cost_usd,
                record.latency_s,
                int(record.cached),
                record.error,
                record.created_at.isoformat(),
                record.model_dump_json(),
            ),
        )
        self._conn.executemany(
            "INSERT INTO scores (result_id, run_id, scorer, score, passed, error)"
            " VALUES (?,?,?,?,?,?)",
            [
                (
                    cur.lastrowid,
                    record.run_id,
                    s.scorer,
                    s.score,
                    None if s.passed is None else int(s.passed),
                    s.error,
                )
                for s in record.scores
            ],
        )
        self._conn.commit()

    def finish_run(self, run_id: str, status: str = "completed") -> RunInfo:
        n, cost = self._conn.execute(
            "SELECT COUNT(*), SUM(cost_usd) FROM results WHERE run_id = ?", (run_id,)
        ).fetchone()
        self._conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, n_records = ?, total_cost_usd = ?"
            " WHERE run_id = ?",
            (status, datetime.now(UTC).isoformat(), n, cost, run_id),
        )
        self._conn.commit()
        return self.get_run(run_id)

    # ------------------------------------------------------------------ read

    def _row_to_info(self, row: sqlite3.Row) -> RunInfo:
        data = dict(row)
        for key in ("models", "templates", "scorers"):
            data[key] = json.loads(data[key])
        return RunInfo.model_validate(data)

    def get_run(self, run_id: str) -> RunInfo:
        row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run {run_id!r}")
        return self._row_to_info(row)

    def list_runs(self, limit: int | None = None) -> list[RunInfo]:
        sql = "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC"
        params: tuple = ()
        if limit:
            sql += " LIMIT ?"
            params = (limit,)
        return [self._row_to_info(r) for r in self._conn.execute(sql, params)]

    def latest_run_id(self) -> str | None:
        runs = self.list_runs(limit=1)
        return runs[0].run_id if runs else None

    def load_records(self, run_ids: str | Iterable[str]) -> list[RunRecord]:
        ids = [run_ids] if isinstance(run_ids, str) else list(run_ids)
        for rid in ids:
            self.get_run(rid)  # raises on unknown id
        placeholders = ",".join("?" * len(ids))
        rows = self._conn.execute(
            f"SELECT record FROM results WHERE run_id IN ({placeholders}) ORDER BY id", ids
        )
        return [RunRecord.model_validate_json(r["record"]) for r in rows]

    def close(self) -> None:
        self._conn.close()


def load_jsonl(path: str | Path) -> list[RunRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        return [RunRecord.model_validate_json(line) for line in fh if line.strip()]
