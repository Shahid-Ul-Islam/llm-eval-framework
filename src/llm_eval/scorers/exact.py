"""Exact match against `example.expected` after normalization.

Args (`scorer_args.exact`): `case_sensitive` (default false), `extract_answer`
(default true: compare the last `Answer: ...` line when present).
"""

from __future__ import annotations

import re

from llm_eval.models import Example, ScoreResult
from llm_eval.scorers.base import BaseScorer, ScoringContext, register_scorer
from llm_eval.scorers.extract import extract_answer


def normalize(text: str, case_sensitive: bool = False) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.strip("`*\"' ").rstrip(".").strip()
    return text if case_sensitive else text.lower()


@register_scorer
class ExactMatchScorer(BaseScorer):
    name = "exact"

    def applicable(self, example: Example) -> bool:
        return example.expected is not None

    async def score(self, example: Example, output: str, ctx: ScoringContext) -> ScoreResult:
        args = self.args(example)
        cs = args.get("case_sensitive", False)
        answer = extract_answer(output) if args.get("extract_answer", True) else output
        expected = example.expected if isinstance(example.expected, list) else [example.expected]
        got = normalize(answer, cs)
        ok = any(got == normalize(e, cs) for e in expected)
        return self.result(1.0 if ok else 0.0, ok, answer=answer[:200])
