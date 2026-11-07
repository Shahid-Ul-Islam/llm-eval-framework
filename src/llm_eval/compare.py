"""A/B comparison of two variants on the examples they share.

Compare along one dimension while holding the others fixed:
  template: A/B prompt variants, paired per (example, model)
  model:    two models, paired per (example, template)
  run:      two runs (e.g. before/after a change), paired per (example, model, template)
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from llm_eval.models import RunRecord
from llm_eval.report import fmt_score, markdown_table

DIMENSIONS = ("template", "model", "run")
_ATTR = {"template": "template", "model": "model", "run": "run_id"}


@dataclass
class Comparison:
    group: str  # the held-fixed context, e.g. the model when comparing templates
    a: str
    b: str
    n: int = 0
    a_sum: float = 0.0
    b_sum: float = 0.0
    wins: int = 0  # B better than A
    losses: int = 0
    ties: int = 0
    tag_deltas: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @property
    def a_mean(self) -> float | None:
        return self.a_sum / self.n if self.n else None

    @property
    def b_mean(self) -> float | None:
        return self.b_sum / self.n if self.n else None

    @property
    def delta(self) -> float | None:
        return None if not self.n else (self.b_sum - self.a_sum) / self.n

    @property
    def p_value(self) -> float | None:
        """Two-sided exact sign test on wins vs losses (ties dropped)."""
        k, n = min(self.wins, self.losses), self.wins + self.losses
        if n == 0:
            return None
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
        return min(1.0, 2 * tail)


def variants(records: Sequence[RunRecord], dimension: str) -> list[str]:
    return sorted({str(getattr(r, _ATTR[dimension])) for r in records})


def compare(
    records: Sequence[RunRecord], dimension: str, a: str, b: str, *, eps: float = 1e-9
) -> list[Comparison]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    attr = _ATTR[dimension]
    context_attrs = [x for x in ("model", "template") if x != attr]

    def context(r: RunRecord) -> str:
        return " / ".join(str(getattr(r, c)) for c in context_attrs)

    def pair_key(r: RunRecord) -> tuple[str, ...]:
        return (r.example_id, *(str(getattr(r, c)) for c in context_attrs))

    side: dict[str, dict[tuple[str, ...], RunRecord]] = {a: {}, b: {}}
    for r in records:
        v = str(getattr(r, attr))
        if v in side and r.mean_score is not None:
            side[v][pair_key(r)] = r

    results: dict[str, Comparison] = {}
    for key in side[a].keys() & side[b].keys():
        ra, rb = side[a][key], side[b][key]
        cmp = results.setdefault(context(ra), Comparison(group=context(ra), a=a, b=b))
        sa, sb = ra.mean_score, rb.mean_score
        cmp.n += 1
        cmp.a_sum += sa
        cmp.b_sum += sb
        if sb > sa + eps:
            cmp.wins += 1
        elif sa > sb + eps:
            cmp.losses += 1
        else:
            cmp.ties += 1
        for tag in ra.tags:
            cmp.tag_deltas[tag].append(sb - sa)
    return sorted(results.values(), key=lambda c: c.group)


def render_comparison_markdown(comparisons: Sequence[Comparison], dimension: str) -> str:
    if not comparisons:
        return "_No paired examples with valid scores to compare._\n"
    a, b = comparisons[0].a, comparisons[0].b
    held = " / ".join(x for x in ("model", "template") if x != _ATTR[dimension])
    out = [f"# A/B comparison by {dimension}: `{a}` (A) vs `{b}` (B)", ""]
    out.append(
        markdown_table(
            [
                held.title(),
                "Paired N",
                "A mean",
                "B mean",
                "Delta (B-A)",
                "B wins",
                "B losses",
                "Ties",
                "Sign-test p",
            ],
            [
                [
                    c.group,
                    str(c.n),
                    fmt_score(c.a_mean),
                    fmt_score(c.b_mean),
                    f"{c.delta:+.3f}",
                    str(c.wins),
                    str(c.losses),
                    str(c.ties),
                    "-" if c.p_value is None else f"{c.p_value:.3f}",
                ]
                for c in comparisons
            ],
        )
    )
    out += ["", "## Delta per tag (B - A)", ""]
    rows = []
    for c in comparisons:
        for tag, deltas in sorted(c.tag_deltas.items()):
            rows.append([c.group, tag, str(len(deltas)), f"{sum(deltas) / len(deltas):+.3f}"])
    out.append(markdown_table([held.title(), "Tag", "N", "Mean delta"], rows))
    out.append("")
    return "\n".join(out)
