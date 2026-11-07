"""Execute generated code against per-item tests in a subprocess with a timeout.

Args (`scorer_args.unit_test`):
  language: "python" (default) or "sql"
  timeout:  seconds (default 10)
  python -> tests: list of statements (usually `assert ...`), each run in a fresh copy
            of the solution's namespace. Score = fraction passing.
  sql    -> setup: SQL script creating/populating tables (SQLite),
            expected_rows: list of rows, ordered: compare row order (default false).

ISOLATION WARNING: the subprocess (`python -I`, temp working dir, timeout) protects
against hangs and crashes, NOT malicious code. It is not a security sandbox. Run
untrusted models' code inside a container/VM.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from llm_eval.models import Example, ScoreResult
from llm_eval.scorers.base import BaseScorer, ScoringContext, register_scorer
from llm_eval.scorers.extract import extract_code

PY_HARNESS = r"""
import json, sys
src = open(sys.argv[1], encoding="utf-8").read()
tests = json.load(open(sys.argv[2], encoding="utf-8"))
ns = {"__name__": "solution"}
try:
    exec(compile(src, "solution.py", "exec"), ns)
except BaseException as e:
    print(json.dumps({"error": f"solution failed to load: {type(e).__name__}: {e}"}))
    sys.exit(0)
results = []
for t in tests:
    try:
        exec(compile(t, "test", "exec"), dict(ns))
        results.append({"test": t, "passed": True})
    except BaseException as e:
        results.append({"test": t, "passed": False, "error": f"{type(e).__name__}: {e}"})
print(json.dumps({"results": results}))
"""

SQL_HARNESS = r"""
import json, sqlite3, sys
query = open(sys.argv[1], encoding="utf-8").read()
spec = json.load(open(sys.argv[2], encoding="utf-8"))
con = sqlite3.connect(":memory:")
try:
    con.executescript(spec["setup"])
    rows = con.execute(query).fetchall()
except Exception as e:
    print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
    sys.exit(0)
print(json.dumps({"rows": [list(r) for r in rows]}))
"""


def _norm_row(row: list[Any]) -> list[Any]:
    return [round(v, 6) if isinstance(v, float) else v for v in row]


def _rows_equal(got: list[list[Any]], expected: list[list[Any]], ordered: bool) -> bool:
    g = [_norm_row(r) for r in got]
    e = [_norm_row(r) for r in expected]
    if not ordered:
        g, e = sorted(g, key=repr), sorted(e, key=repr)
    return g == e


def run_harness(harness: str, code: str, payload: Any, timeout: float) -> dict[str, Any]:
    """Run a harness script in an isolated subprocess; returns its JSON result."""
    with tempfile.TemporaryDirectory(prefix="llm_eval_") as tmp:
        d = Path(tmp)
        (d / "harness.py").write_text(harness, encoding="utf-8")
        (d / "solution.txt").write_text(code, encoding="utf-8")
        (d / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "harness.py", "solution.txt", "payload.json"],
                cwd=d,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout}s", "timeout": True}
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if not lines:
        return {"error": f"no output (exit {proc.returncode}): {proc.stderr[-500:]}"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"error": f"unparseable harness output: {lines[-1][:200]}"}


@register_scorer
class UnitTestScorer(BaseScorer):
    name = "unit_test"

    def applicable(self, example: Example) -> bool:
        args = self.args(example)
        lang = args.get("language", "python")
        if lang == "python":
            return bool(args.get("tests"))
        if lang == "sql":
            return "setup" in args and "expected_rows" in args
        return False

    async def score(self, example: Example, output: str, ctx: ScoringContext) -> ScoreResult:
        args = self.args(example)
        lang = args.get("language", "python")
        timeout = float(args.get("timeout", 10))
        code = extract_code(output, lang)
        if lang == "sql":
            return await self._score_sql(code, args, timeout)
        return await self._score_python(code, args, timeout)

    async def _score_python(self, code: str, args: dict, timeout: float) -> ScoreResult:
        tests = args["tests"]
        res = await asyncio.to_thread(run_harness, PY_HARNESS, code, tests, timeout)
        if "error" in res:
            return self.result(0.0, False, reason=res["error"], total=len(tests))
        results = res["results"]
        passed = sum(r["passed"] for r in results)
        failures = [r for r in results if not r["passed"]][:5]
        return self.result(
            passed / len(results),
            passed == len(results),
            passed_tests=passed,
            total=len(results),
            failures=failures,
        )

    async def _score_sql(self, query: str, args: dict, timeout: float) -> ScoreResult:
        res = await asyncio.to_thread(run_harness, SQL_HARNESS, query, args, timeout)
        if "error" in res:
            return self.result(0.0, False, reason=res["error"])
        ok = _rows_equal(res["rows"], args["expected_rows"], bool(args.get("ordered")))
        return self.result(1.0 if ok else 0.0, ok, rows=res["rows"][:20])
