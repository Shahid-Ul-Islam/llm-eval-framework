from llm_eval.config import Settings
from llm_eval.models import RunRecord, ScoreResult, Usage
from llm_eval.pricing import PricingTable


def test_usage_addition_keeps_unknown_cost_as_none():
    assert (Usage(input_tokens=1) + Usage(output_tokens=2)).cost_usd is None
    total = Usage(input_tokens=1, cost_usd=0.5) + Usage(output_tokens=2)
    assert total.input_tokens == 1 and total.output_tokens == 2 and total.cost_usd == 0.5


def test_run_record_mean_score():
    rec = RunRecord(run_id="r", example_id="e", model="mock:echo", template="t", prompt="p")
    assert rec.mean_score is None
    rec.scores = [ScoreResult(scorer="a", score=1.0), ScoreResult(scorer="b", score=0.0)]
    assert rec.mean_score == 0.5


def test_pricing_default_and_prefix_lookup():
    table = PricingTable()
    assert table.cost("claude-opus-5-5", 1_000_000, 1_000_000) == 24.0
    assert table.cost("mock:echo", 100, 100) == 0.0
    assert table.cost("unknown-model", 10, 10) is None


def test_pricing_override_file(tmp_path):
    f = tmp_path / "prices.yaml"
    f.write_text("gpt-4o-mini: {input: 1.0, output: 2.0}\nnew-model: {input: 3, output: 4}\n")
    table = PricingTable.load(f)
    assert table.cost("gpt-4o-mini", 1_000_000, 0) == 1.0
    assert table.cost("new-model", 0, 1_000_000) == 4.0


def test_settings_reads_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # avoid picking up a real .env
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_EVAL_CONCURRENCY", "9")
    s = Settings()
    assert s.openai_api_key == "sk-test"
    assert s.concurrency == 9
