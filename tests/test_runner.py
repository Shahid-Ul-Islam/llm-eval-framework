from pathlib import Path

import pytest

from llm_eval.client import LLMClient
from llm_eval.datasets import load_dataset
from llm_eval.models import Example, PromptTemplate
from llm_eval.providers.mock import MockProvider
from llm_eval.runner import Runner
from llm_eval.scorers import get_scorer
from llm_eval.storage import ResultStore, load_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_runner(settings, pricing):
    def _make(scorers=("exact", "regex"), **mock_kwargs):
        mock = MockProvider(settings, pricing, **mock_kwargs)
        client = LLMClient(settings, pricing=pricing, providers={"mock": mock})
        store = ResultStore(settings.results_dir)
        runner = Runner(client, store, [get_scorer(s) for s in scorers], concurrency=3)
        return runner, mock, store

    return _make


async def test_run_matrix_scores_and_persists(make_runner):
    # "Oracle" mock answers the reasoning set correctly for the 'good' template only.
    examples = load_dataset(ROOT / "datasets" / "reasoning.jsonl")
    answers = {ex.prompt: ex.expected for ex in examples}

    def reply(req):
        if req.system == "good":
            return f"Reasoning...\nAnswer: {answers[req.prompt]}"
        return "Answer: 0"

    runner, _mock, store = make_runner(responses=reply)
    templates = [PromptTemplate(name="good", system="good"), PromptTemplate(name="bad")]
    seen = []
    runner.on_record = seen.append

    info = await runner.run(examples, ["mock:oracle"], templates, dataset="reasoning")

    assert info.status == "completed" and info.n_records == 20 == len(seen)
    records = store.load_records(info.run_id)
    good = [r for r in records if r.template == "good"]
    bad = [r for r in records if r.template == "bad"]
    assert all(r.mean_score == 1.0 for r in good)
    assert all(r.mean_score == 0.0 for r in bad)
    assert {s.scorer for r in records for s in r.scores} == {"exact", "regex"}
    assert load_jsonl(store.jsonl_path(info.run_id)) == records


async def test_generation_errors_are_recorded_not_raised(make_runner):
    runner, _, store = make_runner()
    ex = [Example(id="a", prompt="p", expected="p")]
    info = await runner.run(ex, ["mock:echo", "mock:error"], [PromptTemplate(name="baseline")])
    by_model = {r.model: r for r in store.load_records(info.run_id)}
    assert by_model["mock:echo"].error is None and by_model["mock:echo"].mean_score == 1.0
    assert "forced error" in by_model["mock:error"].error and by_model["mock:error"].scores == []


async def test_inapplicable_scorers_skipped_and_cache_used(make_runner):
    runner, mock, store = make_runner(scorers=("exact", "unit_test"))
    ex = [Example(id="a", prompt="hello", expected="hello")]
    tpl = [PromptTemplate(name="baseline")]
    first = await runner.run(ex, ["mock:echo"], tpl)
    second = await runner.run(ex, ["mock:echo"], tpl)
    [r1] = store.load_records(first.run_id)
    [r2] = store.load_records(second.run_id)
    assert [s.scorer for s in r1.scores] == ["exact"]
    assert not r1.cached and r2.cached and len(mock.calls) == 1


async def test_crashing_scorer_becomes_error_score(make_runner):
    class Boom(type(get_scorer("exact"))):
        name = "boom"

        async def score(self, example, output, ctx):
            raise RuntimeError("kaboom")

    runner, _, store = make_runner()
    runner.scorers = [Boom()]
    info = await runner.run(
        [Example(id="a", prompt="x", expected="x")], ["mock:echo"], [PromptTemplate(name="b")]
    )
    [rec] = store.load_records(info.run_id)
    assert rec.scores[0].error == "RuntimeError: kaboom" and rec.mean_score is None
