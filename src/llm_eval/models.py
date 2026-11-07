"""Core data models shared across the framework."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Example(BaseModel):
    """One benchmark item loaded from a JSONL dataset."""

    id: str
    prompt: str
    expected: Any = None
    tags: list[str] = Field(default_factory=list)
    # Per-item scorer configuration, keyed by scorer name, e.g.
    # {"regex": {"pattern": "..."}, "unit_test": {"tests": "...", "language": "python"}}
    scorer_args: dict[str, dict[str, Any]] = Field(default_factory=dict)


class PromptTemplate(BaseModel):
    """A named prompt template. `{prompt}` is replaced with the example prompt."""

    name: str
    template: str = "{prompt}"
    system: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None  # None when the model's price is unknown

    def __add__(self, other: Usage) -> Usage:
        cost = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
        )


class GenerationRequest(BaseModel):
    """Provider-agnostic request."""

    model: str
    prompt: str
    system: str | None = None
    max_tokens: int = 2048
    temperature: float | None = None  # ignored by providers that reject sampling params

    def cache_key_payload(self) -> dict[str, Any]:
        return self.model_dump()


class ModelResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    cached: bool = False


class ScoreResult(BaseModel):
    """Normalized score: `score` is always in [0, 1]."""

    scorer: str
    score: float
    passed: bool | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RunRecord(BaseModel):
    """One (example, model, template) evaluation result."""

    run_id: str
    example_id: str
    model: str  # full spec, e.g. "openai:gpt-4o-mini"
    template: str
    prompt: str  # rendered prompt actually sent
    output: str | None = None
    expected: Any = None
    tags: list[str] = Field(default_factory=list)
    scores: list[ScoreResult] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    latency_s: float = 0.0
    cached: bool = False
    error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def mean_score(self) -> float | None:
        if not self.scores:
            return None
        return sum(s.score for s in self.scores) / len(self.scores)
