"""Aggregate run records and render Markdown / HTML reports."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from jinja2 import Environment, select_autoescape

from llm_eval.models import RunRecord
from llm_eval.storage import RunInfo

GROUP_KEYS = ("model", "template", "tag", "scorer", "run_id")


@dataclass
class Aggregate:
    key: tuple[str, ...]
    n: int = 0  # records (or score items when grouped by scorer)
    scored: int = 0  # items with a valid score
    score_sum: float = 0.0
    passed: int = 0
    pass_known: int = 0
    gen_errors: int = 0
    score_errors: int = 0
    cost_usd: float = 0.0
    cost_known: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    latency_sum: float = 0.0
    cached: int = 0
    _seen_records: set[int] = field(default_factory=set, repr=False)

    @property
    def mean_score(self) -> float | None:
        return self.score_sum / self.scored if self.scored else None

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.pass_known if self.pass_known else None

    @property
    def avg_latency(self) -> float | None:
        live = self.n - self.cached - self.gen_errors
        return self.latency_sum / live if live > 0 else None


def record_passed(record: RunRecord) -> bool | None:
    """A record passes when every valid score with a pass/fail verdict passed."""
    verdicts = [s.passed for s in record.scores if s.error is None and s.passed is not None]
    return all(verdicts) if verdicts else None


def _keys_for(record: RunRecord, by: Sequence[str]) -> list[tuple[str, ...]]:
    parts: list[list[str]] = []
    for k in by:
        if k == "tag":
            parts.append(record.tags or ["(untagged)"])
        elif k == "scorer":
            parts.append([s.scorer for s in record.scores] or ["(none)"])
        else:
            parts.append([str(getattr(record, k))])
    keys: list[tuple[str, ...]] = [()]
    for options in parts:
        keys = [(*k, o) for k in keys for o in options]
    return keys


def aggregate(records: Iterable[RunRecord], by: Sequence[str]) -> list[Aggregate]:
    """Group records by any of `model`, `template`, `tag`, `scorer`, `run_id`.

    With `scorer` in `by`, each score item is counted separately; otherwise each
    record contributes its mean score. Records count once per tag they carry.
    """
    unknown = set(by) - set(GROUP_KEYS)
    if unknown:
        raise ValueError(f"Unknown group key(s) {unknown}; choose from {GROUP_KEYS}")
    groups: dict[tuple[str, ...], Aggregate] = {}
    for idx, rec in enumerate(records):
        for key in _keys_for(rec, by):
            agg = groups.setdefault(key, Aggregate(key=key))
            agg.n += 1
            if "scorer" in by:
                name = key[list(by).index("scorer")]
                items = [s for s in rec.scores if s.scorer == name]
                for s in items:
                    if s.error:
                        agg.score_errors += 1
                        continue
                    agg.scored += 1
                    agg.score_sum += s.score
                    if s.passed is not None:
                        agg.pass_known += 1
                        agg.passed += int(s.passed)
            else:
                if rec.mean_score is not None:
                    agg.scored += 1
                    agg.score_sum += rec.mean_score
                verdict = record_passed(rec)
                if verdict is not None:
                    agg.pass_known += 1
                    agg.passed += int(verdict)
                agg.score_errors += sum(1 for s in rec.scores if s.error)
            if rec.error:
                agg.gen_errors += 1
            # usage is per record: avoid double counting when a record fans out per scorer
            if idx not in agg._seen_records:
                agg._seen_records.add(idx)
                if rec.usage.cost_usd is not None:
                    agg.cost_known = True
                    agg.cost_usd += rec.usage.cost_usd
                agg.input_tokens += rec.usage.input_tokens
                agg.output_tokens += rec.usage.output_tokens
                agg.latency_sum += rec.latency_s
                agg.cached += int(rec.cached)
    return sorted(groups.values(), key=lambda a: a.key)


# ---------------------------------------------------------------------------- formatting


def fmt_score(v: float | None) -> str:
    return "-" if v is None else f"{v:.3f}"


def fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.0f}%"


def fmt_cost(a: Aggregate) -> str:
    return f"${a.cost_usd:.4f}" if a.cost_known else "n/a"


def fmt_latency(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}s"


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    def esc(c: str) -> str:
        return str(c).replace("|", "\\|")

    lines = ["| " + " | ".join(map(esc, headers)) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(map(esc, r)) + " |" for r in rows]
    return "\n".join(lines)


@dataclass
class ReportData:
    runs: list[RunInfo]
    overall: list[Aggregate]
    by_scorer: list[Aggregate]
    by_tag: list[Aggregate]
    variants: list[tuple[str, str]]  # (model, template) column order for tag pivot
    tags: list[str]
    tag_scores: dict[tuple[str, str, str], Aggregate]  # (tag, model, template)


def build_report(runs: Sequence[RunInfo], records: Sequence[RunRecord]) -> ReportData:
    overall = aggregate(records, ["model", "template"])
    by_tag = aggregate(records, ["tag", "model", "template"])
    variants = [(a.key[0], a.key[1]) for a in overall]
    return ReportData(
        runs=list(runs),
        overall=overall,
        by_scorer=aggregate(records, ["model", "template", "scorer"]),
        by_tag=by_tag,
        variants=variants,
        tags=sorted({a.key[0] for a in by_tag}),
        tag_scores={a.key: a for a in by_tag},
    )


def render_overall_table(data: ReportData) -> str:
    return markdown_table(
        [
            "Model",
            "Template",
            "N",
            "Mean score",
            "Pass rate",
            "Gen errors",
            "Cost",
            "Tokens in/out",
            "Avg latency",
            "Cached",
        ],
        [
            [
                a.key[0],
                a.key[1],
                str(a.n),
                fmt_score(a.mean_score),
                fmt_pct(a.pass_rate),
                str(a.gen_errors),
                fmt_cost(a),
                f"{a.input_tokens}/{a.output_tokens}",
                fmt_latency(a.avg_latency),
                str(a.cached),
            ]
            for a in data.overall
        ],
    )


def render_markdown(data: ReportData) -> str:
    out = ["# LLM Eval Report", ""]
    for r in data.runs:
        out.append(
            f"- **Run** `{r.run_id}` - dataset `{r.dataset}`, status {r.status}, "
            f"{r.n_records} records, created {r.created_at:%Y-%m-%d %H:%M} UTC"
        )
    out += ["", "## Overall (per model / template)", "", render_overall_table(data)]
    out += ["", "## Per scorer", ""]
    out.append(
        markdown_table(
            ["Model", "Template", "Scorer", "Items", "Mean score", "Pass rate", "Scorer errors"],
            [
                [
                    *a.key,
                    str(a.scored + a.score_errors),
                    fmt_score(a.mean_score),
                    fmt_pct(a.pass_rate),
                    str(a.score_errors),
                ]
                for a in data.by_scorer
                if a.key[2] != "(none)"
            ],
        )
    )
    out += ["", "## Mean score per tag", ""]
    headers = ["Tag"] + [f"{m} / {t}" for m, t in data.variants]
    rows = []
    for tag in data.tags:
        row = [tag]
        for m, t in data.variants:
            agg = data.tag_scores.get((tag, m, t))
            row.append(f"{fmt_score(agg.mean_score)} (n={agg.n})" if agg else "-")
        rows.append(row)
    out.append(markdown_table(headers, rows))
    out.append("")
    return "\n".join(out)


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Eval Report</title>
<style>
  :root { --fg:#1f2328; --muted:#59636e; --border:#d1d9e0; --bg:#fff; --head:#f6f8fa; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e6edf3; --muted:#9198a1; --border:#3d444d; --bg:#0d1117; --head:#151b23; }
  }
  body { font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg);
         background: var(--bg); margin: 0 auto; padding: 24px 16px; max-width: 1200px; }
  h1 { font-size: 22px; } h2 { font-size: 17px; margin-top: 32px; }
  .meta { color: var(--muted); margin: 2px 0; }
  .wrap { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  th, td { border: 1px solid var(--border); padding: 6px 10px; text-align: left;
           white-space: nowrap; }
  th { background: var(--head); }
  td.num { text-align: right; }
  td.score { text-align: right; font-weight: 600; }
  .bar { display:inline-block; height:8px; border-radius:4px; margin-left:8px;
         vertical-align: middle; background: hsl(var(--h) 65% 50%); }
</style></head><body>
<h1>LLM Eval Report</h1>
{% for r in data.runs %}
<p class="meta">Run <code>{{ r.run_id }}</code> &middot; dataset <code>{{ r.dataset }}</code>
 &middot; {{ r.status }} &middot; {{ r.n_records }} records
 &middot; {{ r.created_at.strftime('%Y-%m-%d %H:%M') }} UTC</p>
{% endfor %}

{% macro score_cell(v) -%}
{%- if v is none %}<td class="score">-</td>{% else -%}
<td class="score">{{ '%.3f' % v }}<span class="bar"
 style="width:{{ (v * 60) | round | int }}px; --h:{{ (v * 120) | round | int }}"></span></td>
{%- endif %}
{%- endmacro %}

<h2>Overall (per model / template)</h2>
<div class="wrap"><table>
<tr><th>Model</th><th>Template</th><th>N</th><th>Mean score</th><th>Pass rate</th>
<th>Gen errors</th><th>Cost</th><th>Tokens in/out</th><th>Avg latency</th><th>Cached</th></tr>
{% for a in data.overall %}
<tr><td>{{ a.key[0] }}</td><td>{{ a.key[1] }}</td><td class="num">{{ a.n }}</td>
{{ score_cell(a.mean_score) }}<td class="num">{{ pct(a.pass_rate) }}</td>
<td class="num">{{ a.gen_errors }}</td><td class="num">{{ cost(a) }}</td>
<td class="num">{{ a.input_tokens }}/{{ a.output_tokens }}</td>
<td class="num">{{ latency(a.avg_latency) }}</td><td class="num">{{ a.cached }}</td></tr>
{% endfor %}
</table></div>

<h2>Per scorer</h2>
<div class="wrap"><table>
<tr><th>Model</th><th>Template</th><th>Scorer</th><th>Items</th><th>Mean score</th>
<th>Pass rate</th><th>Scorer errors</th></tr>
{% for a in data.by_scorer if a.key[2] != '(none)' %}
<tr><td>{{ a.key[0] }}</td><td>{{ a.key[1] }}</td><td>{{ a.key[2] }}</td>
<td class="num">{{ a.scored + a.score_errors }}</td>{{ score_cell(a.mean_score) }}
<td class="num">{{ pct(a.pass_rate) }}</td><td class="num">{{ a.score_errors }}</td></tr>
{% endfor %}
</table></div>

<h2>Mean score per tag</h2>
<div class="wrap"><table>
<tr><th>Tag</th>{% for m, t in data.variants %}<th>{{ m }}<br>{{ t }}</th>{% endfor %}</tr>
{% for tag in data.tags %}
<tr><td>{{ tag }}</td>
{% for m, t in data.variants %}{% set a = data.tag_scores.get((tag, m, t)) %}
{% if a %}{{ score_cell(a.mean_score) }}{% else %}<td class="score">-</td>{% endif %}
{% endfor %}</tr>
{% endfor %}
</table></div>
</body></html>
"""


def render_html(data: ReportData) -> str:
    env = Environment(autoescape=select_autoescape(default=True))
    template = env.from_string(_HTML)
    return template.render(data=data, pct=fmt_pct, cost=fmt_cost, latency=fmt_latency)
