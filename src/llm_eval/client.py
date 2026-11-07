"""High-level client: model spec -> provider, with caching, retries and usage tracking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from llm_eval.cache import ResponseCache
from llm_eval.config import Settings, get_settings
from llm_eval.models import GenerationRequest, ModelResponse, Usage
from llm_eval.pricing import PricingTable
from llm_eval.providers.base import BaseProvider, get_provider_class, parse_model_spec
from llm_eval.reliability import RetryPolicy, with_retries


@dataclass
class UsageStats:
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    errors: int = 0
    usage: Usage = field(default_factory=lambda: Usage(cost_usd=0.0))


class LLMClient:
    """Shared entry point for all model calls (runner and LLM judge).

    `providers` lets callers inject pre-built provider instances (e.g. a configured
    MockProvider in tests); otherwise providers are created lazily from the registry.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        pricing: PricingTable | None = None,
        cache: ResponseCache | None = None,
        use_cache: bool = True,
        providers: dict[str, BaseProvider] | None = None,
    ):
        self.settings = settings or get_settings()
        self.pricing = pricing or PricingTable.load(self.settings.pricing_file)
        if cache is None and use_cache:
            cache = ResponseCache(self.settings.cache_path)
        self.cache = cache if use_cache else None
        self._providers: dict[str, BaseProvider] = dict(providers or {})
        self.policy = RetryPolicy(
            max_retries=self.settings.max_retries,
            base_delay=self.settings.retry_base_delay,
            max_delay=self.settings.retry_max_delay,
        )
        self.stats: dict[str, UsageStats] = defaultdict(UsageStats)

    def provider(self, name: str) -> BaseProvider:
        if name not in self._providers:
            cls = get_provider_class(name)
            self._providers[name] = cls(self.settings, self.pricing)
        return self._providers[name]

    async def generate(
        self,
        spec: str,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float | None = None,
    ) -> ModelResponse:
        provider_name, model = parse_model_spec(spec)
        request = GenerationRequest(
            model=model,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        stats = self.stats[spec]

        if self.cache is not None:
            hit = self.cache.get(provider_name, request)
            if hit is not None:
                stats.cache_hits += 1
                # Cached responses cost nothing this run; token counts are kept for reference.
                usage = hit.usage.model_copy(update={"cost_usd": 0.0})
                return hit.model_copy(update={"cached": True, "latency_s": 0.0, "usage": usage})

        provider = self.provider(provider_name)

        def _count_retry(*_):
            stats.retries += 1

        stats.calls += 1
        try:
            response = await with_retries(
                lambda: provider.generate(request), self.policy, on_retry=_count_retry
            )
        except Exception:
            stats.errors += 1
            raise
        stats.usage = stats.usage + response.usage
        if self.cache is not None:
            self.cache.set(provider_name, request, response)
        return response

    def total_usage(self) -> Usage:
        total = Usage(cost_usd=0.0)
        for s in self.stats.values():
            total = total + s.usage
        return total

    async def aclose(self) -> None:
        for p in self._providers.values():
            await p.aclose()
        if self.cache is not None:
            self.cache.close()
