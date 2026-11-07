import pytest

from llm_eval.compare import compare, render_comparison_markdown, variants
from llm_eval.models import RunRecord, ScoreResult, Usage
from llm_eval.report import aggregate, build_report, render_html, render_markdown
from llm_eval.storage import RunInfo


def rec(
    ex,
    model="m1",
    template="A",
    score=1.0,
    tags=("x",),
    run="r1",
    error=None,
    cost=0.01,
    scorer="exact",
):
    scores = [] if error else [ScoreResult(scorer=scorer, score=score, passed=score == 1.0)]
    return RunRecord(
        run_id=run,
        example_id=ex,
        model=model,
        template=template,
        prompt="p",
        tags=list(tags),
        scores=scores,
        error=error,
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=cost),
        latency_s=0.5,
    )


RECORDS = [
    rec("e1", template="A", score=0.0, tags=("math",)),
    rec("e2", template="A", score=1.0, tags=("math", "logic")),
    rec("e3", template="A", score=0.0, tags=("logic",)),
    rec("e1", template="B", score=1.0, tags=("math",)),
    rec("e2", template="B", score=1.0, tags=("math", "logic")),
    rec("e3", template="B", score=1.0, tags=("logic",)),
    rec("e4", template="B", error="boom", tags=("logic",)),
]


def test_aggregate_overall_and_by_tag():
    overall = {a.key: a for a in aggregate(RECORDS, ["model", "template"])}
    a, b = overall[("m1", "A")], overall[("m1", "B")]
    assert a.n == 3 and a.mean_score == pytest.approx(1 / 3) and a.pass_rate == pytest.approx(1 / 3)
    assert b.n == 4 and b.mean_score == 1.0 and b.gen_errors == 1
    assert b.cost_usd == pytest.approx(0.04) and b.input_tokens == 40
    by_tag = {a.key: a for a in aggregate(RECORDS, ["tag", "template"])}
    assert by_tag[("logic", "A")].n == 2 and by_tag[("logic", "A")].mean_score == 0.5
    with pytest.raises(ValueError):
        aggregate(RECORDS, ["nope"])


def test_aggregate_by_scorer_does_not_double_count_usage():
    r = rec("e1")
    r.scores.append(ScoreResult(scorer="regex", score=0.0, passed=False))
    r.scores.append(ScoreResult(scorer="llm_judge", score=0.0, error="judge failed"))
    aggs = {a.key: a for a in aggregate([r], ["model", "scorer"])}
    assert aggs[("m1", "exact")].mean_score == 1.0
    assert aggs[("m1", "regex")].pass_rate == 0.0
    assert aggs[("m1", "llm_judge")].score_errors == 1 and aggs[("m1", "llm_judge")].scored == 0
    total = aggregate([r], ["model"])[0]
    assert total.cost_usd == pytest.approx(0.01) and total.mean_score == 0.5


def test_markdown_and_html_reports():
    data = build_report([RunInfo(run_id="r1", dataset="d", n_records=7)], RECORDS)
    md = render_markdown(data)
    assert "| m1 | A | 3 | 0.333 | 33% |" in md
    assert "## Mean score per tag" in md and "| logic |" in md
    html = render_html(data)
    assert "<table>" in html and "m1" in html and "0.333" in html
    # autoescaping
    data2 = build_report([], [rec("e1", model="<script>")])
    assert "<script>" not in render_html(data2)


def test_compare_templates():
    comps = compare(RECORDS, "template", "A", "B")
    [c] = comps
    assert c.group == "m1" and c.n == 3  # e4 errored on B, unpaired
    assert (c.wins, c.losses, c.ties) == (2, 0, 1)
    assert c.delta == pytest.approx(2 / 3)
    assert c.p_value == pytest.approx(0.5)
    assert c.tag_deltas["logic"] == [0.0, 1.0]
    md = render_comparison_markdown(comps, "template")
    assert "+0.667" in md and "Delta per tag" in md
    assert variants(RECORDS, "template") == ["A", "B"]


def test_compare_models_and_runs():
    recs = [rec("e1", model="m1"), rec("e1", model="m2", score=0.0)]
    [c] = compare(recs, "model", "m1", "m2")
    assert c.group == "A" and c.losses == 1
    runs = [rec("e1", run="r1", score=0.0), rec("e1", run="r2", score=1.0)]
    [c] = compare(runs, "run", "r1", "r2")
    assert c.group == "m1 / A" and c.wins == 1
    assert "No paired" in render_comparison_markdown([], "run")
