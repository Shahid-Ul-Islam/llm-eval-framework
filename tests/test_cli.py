"""End-to-end CLI tests using the offline mock provider."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_eval.cli import app
from llm_eval.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
cli = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)  # so dataset names and prompts/templates.yaml resolve
    monkeypatch.setenv("LLM_EVAL_CACHE_PATH", str(tmp_path / "cache.db"))
    monkeypatch.setenv("LLM_EVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LLM_EVAL_JUDGE_MODEL", "mock:judge")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def invoke(*args):
    result = cli.invoke(app, list(args))
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


def test_run_report_compare_end_to_end(isolated_env):
    out = invoke(
        "run",
        "-d",
        "reasoning",
        "-m",
        "mock:echo",
        "-s",
        "exact,regex",
        "-t",
        "baseline",
        "-t",
        "concise",
        "--limit",
        "3",
    )
    assert "3 examples x 1 model(s) x 2 template(s) = 6 calls" in out
    assert "| mock:echo | baseline | 3 |" in out
    assert "Run id:" in out

    out = invoke("runs")
    assert "reasoning" in out and "baseline,concise" in out

    out = invoke("report", "--format", "both")
    assert "# LLM Eval Report" in out
    reports = list((isolated_env / "results" / "reports").iterdir())
    assert {p.suffix for p in reports} == {".md", ".html"}

    out = invoke("compare", "--by", "template")
    assert "`baseline` (A) vs `concise` (B)" in out

    # Second identical run is fully cached; then compare the two runs.
    out = invoke(
        "run",
        "-d",
        "reasoning",
        "-m",
        "mock:echo",
        "-s",
        "exact,regex",
        "-t",
        "baseline,concise",
        "--limit",
        "3",
    )
    assert "| mock:echo | 0 | 6 |" in out  # 0 API calls, 6 cache hits
    out = invoke("compare", "--by", "run")
    assert "Paired N" in out


def test_run_with_judge_and_unit_tests(isolated_env):
    out = invoke(
        "run",
        "-d",
        "coding",
        "-m",
        "mock:echo",
        "-s",
        "unit_test,json_schema,llm_judge",
        "--tags",
        "sql",
        "--no-cache",
    )
    # judge replies with the echoed prompt (not JSON) -> recorded as judge errors, not crashes
    assert "mock:judge" in out
    out = invoke("report", "-f", "md")
    assert "llm_judge" in out and "unit_test" in out


@pytest.mark.parametrize(
    "args, message",
    [
        (["run", "-d", "nope", "-m", "mock:echo"], "Dataset not found"),
        (["run", "-d", "reasoning", "-m", "gpt-4o"], "provider:model"),
        (["run", "-d", "reasoning", "-m", "foo:bar"], "Unknown provider"),
        (["run", "-d", "reasoning", "-m", "mock:echo", "-s", "bogus"], "Unknown scorer"),
        (["run", "-d", "reasoning", "-m", "mock:echo", "-t", "nope"], "Unknown template"),
        (["report"], "need 1 run"),
        (["compare", "--by", "nope"], "--by must be"),
    ],
)
def test_cli_errors(args, message):
    result = cli.invoke(app, args)
    assert result.exit_code == 2
    assert message in result.output


def test_missing_openai_key_is_reported_per_item(isolated_env, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()
    out = invoke("run", "-d", "reasoning", "-m", "openai:gpt-4o-mini", "--limit", "1")
    assert "generation error" in out and "OPENAI_API_KEY" in out
