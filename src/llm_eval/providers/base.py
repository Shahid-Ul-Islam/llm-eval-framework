"""Provider abstraction, error types and registry.

A provider turns a `GenerationRequest` into a `Completion` (text + token counts).
`BaseProvider.generate()` adds timing and cost; retries and caching are layered on
top in `llm_eval.client`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from llm_eval.config import Settings, get_settings
from llm_eval.models import GenerationRequest, ModelResponse, Usage
from llm_eval.pricing import PricingTable


class ProviderError(Exception):
    """Non-retryable provider failure (bad request, auth, unknown model...)."""


class TransientError(ProviderError):
    """Retryable failure (timeouts, connection errors, 5xx, overloaded)."""


class RateLimitError(TransientError):
    """HTTP 429. `retry_after` (seconds) is honored by the retry layer when present."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderConfigError(ProviderError):
    """Provider is misconfigured (e.g. missing API key)."""


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class BaseProvider(ABC):
    """Subclass this, set `name`, implement `_complete`, and decorate with `@register_provider`."""

    name: ClassVar[str]

    def __init__(self, settings: Settings | None = None, pricing: PricingTable | None = None):
        self.settings = settings or get_settings()
        self.pricing = pricing or PricingTable.load(self.settings.pricing_file)

    @abstractmethod
    async def _complete(self, request: GenerationRequest) -> Completion:
        """Call the backend. Raise `RateLimitError`/`TransientError`/`ProviderError`."""

    async def generate(self, request: GenerationRequest) -> ModelResponse:
        start = time.perf_counter()
        completion = await self._complete(request)
        latency = time.perf_counter() - start
        return ModelResponse(
            text=completion.text,
            provider=self.name,
            model=request.model,
            usage=Usage(
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                cost_usd=self.cost(request.model, completion),
            ),
            latency_s=latency,
        )

    def cost(self, model: str, completion: Completion) -> float | None:
        args = (completion.input_tokens, completion.output_tokens)
        cost = self.pricing.cost(model, *args)
        return cost if cost is not None else self.pricing.cost(f"{self.name}:{model}", *args)

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release network resources. Override if the provider holds a client."""


# --------------------------------------------------------------------------- registry

_REGISTRY: dict[str, type[BaseProvider]] = {}


def register_provider[P: type[BaseProvider]](cls: P) -> P:
    _REGISTRY[cls.name] = cls
    return cls


def available_providers() -> list[str]:
    _ensure_builtin_providers()
    return sorted(_REGISTRY)


def get_provider_class(name: str) -> type[BaseProvider]:
    _ensure_builtin_providers()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ProviderError(
            f"Unknown provider {name!r}. Available: {available_providers()}"
        ) from None


def parse_model_spec(spec: str) -> tuple[str, str]:
    """`"openai:gpt-4o-mini"` -> `("openai", "gpt-4o-mini")`."""
    provider, sep, model = spec.partition(":")
    if not sep or not provider or not model:
        raise ProviderError(f"Model spec must look like 'provider:model', got {spec!r}")
    return provider, model


def _ensure_builtin_providers() -> None:
    # Imported lazily so the registry is populated without import cycles.
    from llm_eval.providers import mock  # noqa: F401
