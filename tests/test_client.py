import pytest

from llm_eval.cache import ResponseCache, cache_key
from llm_eval.client import LLMClient
from llm_eval.models import GenerationRequest, ModelResponse
from llm_eval.providers import ProviderError, RateLimitError, TransientError
from llm_eval.providers.mock import MockProvider
from llm_eval.reliability import RetryPolicy, with_retries

# ---------------------------------------------------------------- retry policy


async def test_with_retries_recovers_and_honors_retry_after():
    calls, sleeps = 0, []

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError(retry_after=7)
        if calls == 2:
            raise TransientError("boom")
        return "ok"

    async def fake_sleep(s):
        sleeps.append(s)

    policy = RetryPolicy(max_retries=3, base_delay=1, max_delay=30, jitter=False)
    assert await with_retries(fn, policy, sleep=fake_sleep) == "ok"
    assert sleeps == [7, 2]  # retry-after, then base * 2**1


async def test_with_retries_gives_up_and_skips_fatal_errors():
    async def noop(_):
        pass

    async def always_transient():
        raise TransientError("x")

    with pytest.raises(TransientError):
        await with_retries(always_transient, RetryPolicy(max_retries=2), sleep=noop)

    calls = 0

    async def fatal():
        nonlocal calls
        calls += 1
        raise ProviderError("bad request")

    with pytest.raises(ProviderError):
        await with_retries(fatal, RetryPolicy(max_retries=5), sleep=noop)
    assert calls == 1


def test_retry_after_is_capped():
    policy = RetryPolicy(max_delay=10)
    assert policy.delay_for(0, RateLimitError(retry_after=999)) == 10


# ---------------------------------------------------------------- cache


def test_cache_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path / "c.db")
    req = GenerationRequest(model="m", prompt="p")
    assert cache.get("mock", req) is None
    cache.set("mock", req, ModelResponse(text="hi", provider="mock", model="m"))
    assert cache.get("mock", req).text == "hi"
    assert cache.get("other", req) is None
    assert len(cache) == 1
    assert cache_key("mock", req) != cache_key("mock", req.model_copy(update={"system": "s"}))
    cache.close()


# ---------------------------------------------------------------- client


async def test_client_retries_then_caches(settings, pricing):
    mock = MockProvider(settings, pricing, rate_limit_times=1, fail_times=1)
    client = LLMClient(settings, pricing=pricing, providers={"mock": mock})

    r1 = await client.generate("mock:echo", "hello world")
    assert r1.text == "hello world" and not r1.cached
    r2 = await client.generate("mock:echo", "hello world")
    assert r2.cached and r2.usage.cost_usd == 0.0

    stats = client.stats["mock:echo"]
    assert stats.calls == 1 and stats.retries == 2 and stats.cache_hits == 1
    assert len(mock.calls) == 3
    assert client.total_usage().output_tokens == 2
    await client.aclose()


async def test_client_without_cache_and_errors(settings, pricing):
    mock = MockProvider(settings, pricing)
    client = LLMClient(settings, pricing=pricing, use_cache=False, providers={"mock": mock})
    await client.generate("mock:echo", "x")
    await client.generate("mock:echo", "x")
    assert len(mock.calls) == 2
    with pytest.raises(ProviderError):
        await client.generate("mock:error", "x")
    assert client.stats["mock:error"].errors == 1
