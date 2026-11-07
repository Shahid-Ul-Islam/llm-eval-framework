"""Scorer abstraction and registry.

A scorer turns (example, model output) into a `ScoreResult` with `score` in [0, 1].
Scorers that need per-item configuration read it from `example.scorer_args[<name>]`
and report `applicable() == False` for items that lack it, so one run can mix
datasets and scorers freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from llm_eval.models import Example, ScoreResult

if TYPE_CHECKING:
    from llm_eval.client import LLMClient
    from llm_eval.config import Settings


@dataclass
class ScoringContext:
    """Shared resources for scorers that need them (e.g. the LLM judge)."""

    client: LLMClient | None = None
    settings: Settings | None = None


class BaseScorer(ABC):
    name: ClassVar[str]

    def args(self, example: Example) -> dict[str, Any]:
        return example.scorer_args.get(self.name, {})

    def applicable(self, example: Example) -> bool:
        return True

    @abstractmethod
    async def score(self, example: Example, output: str, ctx: ScoringContext) -> ScoreResult:
        """Score one output. Infrastructure failures go in `ScoreResult.error`."""

    def result(self, score: float, passed: bool | None = None, **details: Any) -> ScoreResult:
        return ScoreResult(scorer=self.name, score=score, passed=passed, details=details)


_REGISTRY: dict[str, type[BaseScorer]] = {}


def register_scorer[S: type[BaseScorer]](cls: S) -> S:
    _REGISTRY[cls.name] = cls
    return cls


def available_scorers() -> list[str]:
    _ensure_builtin_scorers()
    return sorted(_REGISTRY)


def get_scorer(name: str) -> BaseScorer:
    _ensure_builtin_scorers()
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"Unknown scorer {name!r}. Available: {available_scorers()}") from None


def _ensure_builtin_scorers() -> None:
    from llm_eval.scorers import (  # noqa: F401
        exact,
        json_schema,
        llm_judge,
        regex,
        unit_test,
    )
