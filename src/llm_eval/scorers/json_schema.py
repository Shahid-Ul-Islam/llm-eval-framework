"""JSON-schema validity of the output.

Args (`scorer_args.json_schema`): `schema` (required JSON Schema). JSON is taken from a
```json fence or the first parseable object/array in the output.
"""

from __future__ import annotations

import jsonschema

from llm_eval.models import Example, ScoreResult
from llm_eval.scorers.base import BaseScorer, ScoringContext, register_scorer
from llm_eval.scorers.extract import extract_json


@register_scorer
class JsonSchemaScorer(BaseScorer):
    name = "json_schema"

    def applicable(self, example: Example) -> bool:
        return isinstance(self.args(example).get("schema"), dict)

    async def score(self, example: Example, output: str, ctx: ScoringContext) -> ScoreResult:
        schema = self.args(example)["schema"]
        try:
            data = extract_json(output)
        except ValueError:
            return self.result(0.0, False, reason="no parseable JSON")
        validator = jsonschema.Draft202012Validator(schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
        if errors:
            msgs = [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5]]
            return self.result(0.0, False, errors=msgs)
        return self.result(1.0, True)
