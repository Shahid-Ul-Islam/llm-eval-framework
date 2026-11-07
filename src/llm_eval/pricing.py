"""Token pricing table and cost calculation.

Prices are USD per 1M tokens. Defaults are a snapshot and WILL drift: override or extend
them with a JSON/YAML file via `LLM_EVAL_PRICING_FILE`, e.g.

    gpt-4o-mini: {input: 0.15, output: 0.60}
    my-finetune: {input: 1.0, output: 2.0}
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel


class Price(BaseModel):
    input: float  # USD per 1M input tokens
    output: float  # USD per 1M output tokens


DEFAULT_PRICES: dict[str, Price] = {
    # Anthropic (first-party API rates)
    "claude-fable-5-1": Price(input=10.00, output=50.00),
    "claude-opus-5-5": Price(input=4.00, output=20.00),
    "claude-opus-5": Price(input=5.00, output=25.00),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00),
    "claude-sonnet-5": Price(input=2.00, output=10.00),
    "claude-sonnet-4-6": Price(input=3.00, output=15.00),
    "claude-haiku-5-5": Price(input=0.10, output=0.50),
    "claude-haiku-4-5": Price(input=1.00, output=5.00),
    # OpenAI (verify against current OpenAI pricing; override as needed)
    "gpt-4o": Price(input=2.50, output=10.00),
    "gpt-4o-mini": Price(input=0.15, output=0.60),
    "gpt-4.1": Price(input=2.00, output=8.00),
    "gpt-4.1-mini": Price(input=0.40, output=1.60),
    # Offline mock provider is free
    "mock": Price(input=0.0, output=0.0),
}


class PricingTable:
    def __init__(self, prices: dict[str, Price] | None = None):
        self.prices: dict[str, Price] = dict(DEFAULT_PRICES if prices is None else prices)

    @classmethod
    def load(cls, override_file: Path | None = None) -> PricingTable:
        table = cls()
        if override_file is not None:
            table.update_from_file(override_file)
        return table

    def update_from_file(self, path: Path) -> None:
        text = Path(path).read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(f"Pricing file {path} must contain a mapping of model -> prices")
        for model, price in data.items():
            self.prices[str(model)] = Price.model_validate(price)

    def get(self, model: str) -> Price | None:
        if model in self.prices:
            return self.prices[model]
        # Fall back to the longest known prefix (e.g. dated snapshots, "mock:echo" -> "mock").
        candidates = [k for k in self.prices if model.startswith(k)]
        return self.prices[max(candidates, key=len)] if candidates else None

    def cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.get(model)
        if price is None:
            return None
        return (input_tokens * price.input + output_tokens * price.output) / 1_000_000
