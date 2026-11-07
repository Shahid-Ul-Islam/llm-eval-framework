"""Regex search over the output.

Args (`scorer_args.regex`): `pattern` (required), `ignore_case` (default false),
`negate` (default false: pass when the pattern does NOT match).
"""

from __future__ import annotations

import re

from llm_eval.models import Example, ScoreResult
from llm_eval.scorers.base import BaseScorer, ScoringContext, register_scorer


@register_scorer
class RegexScorer(BaseScorer):
    name = "regex"

    def applicable(self, example: Example) -> bool:
        return bool(self.args(example).get("pattern"))

    async def score(self, example: Example, output: str, ctx: ScoringContext) -> ScoreResult:
        args = self.args(example)
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        try:
            matched = re.search(args["pattern"], output, flags) is not None
        except re.error as e:
            return self.result(0.0, None).model_copy(update={"error": f"bad pattern: {e}"})
        ok = matched != bool(args.get("negate", False))
        return self.result(1.0 if ok else 0.0, ok, pattern=args["pattern"])
