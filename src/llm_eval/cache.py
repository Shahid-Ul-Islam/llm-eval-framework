"""SQLite-backed response cache keyed by a hash of (provider, request)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from llm_eval.models import GenerationRequest, ModelResponse

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


def cache_key(provider: str, request: GenerationRequest) -> str:
    payload = {"provider": provider, **request.cache_key_payload()}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, provider: str, request: GenerationRequest) -> ModelResponse | None:
        row = self._conn.execute(
            "SELECT response FROM responses WHERE key = ?", (cache_key(provider, request),)
        ).fetchone()
        return ModelResponse.model_validate_json(row[0]) if row else None

    def set(self, provider: str, request: GenerationRequest, response: ModelResponse) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (key, provider, model, response) VALUES (?,?,?,?)",
            (cache_key(provider, request), provider, request.model, response.model_dump_json()),
        )
        self._conn.commit()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def clear(self) -> None:
        self._conn.execute("DELETE FROM responses")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
