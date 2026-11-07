import pytest

from llm_eval.models import GenerationRequest
from llm_eval.providers import (
    ProviderError,
    RateLimitError,
    TransientError,
    available_providers,
    get_provider_class,
    parse_model_spec,
)
from llm_eval.providers.mock import MockProvider


def test_parse_model_spec():
    assert parse_model_spec("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert parse_model_spec("mock:a:b") == ("mock", "a:b")
    for bad in ["gpt-4o", ":x", "mock:"]:
        with pytest.raises(ProviderError):
            parse_model_spec(bad)


def test_registry():
    assert "mock" in available_providers()
    assert get_provider_class("mock") is MockProvider
    with pytest.raises(ProviderError, match="Unknown provider"):
        get_provider_class("nope")


async def test_mock_echo_usage_and_cost(settings, pricing):
    p = MockProvider(settings, pricing)
    resp = await p.generate(GenerationRequest(model="echo", prompt="one two three", system="s"))
    assert resp.text == "one two three"
    assert resp.provider == "mock" and resp.model == "echo"
    assert resp.usage.input_tokens == 4 and resp.usage.output_tokens == 3
    assert resp.usage.cost_usd == 0.0  # "mock" price via provider-prefixed lookup


async def test_mock_canned_and_failures(settings, pricing):
    p = MockProvider(
        settings,
        pricing,
        responses={"capital": "Paris"},
        rate_limit_times=1,
        fail_times=1,
        retry_after=2.5,
    )
    req = GenerationRequest(model="any", prompt="capital of France?")
    with pytest.raises(RateLimitError) as ei:
        await p.generate(req)
    assert ei.value.retry_after == 2.5
    with pytest.raises(TransientError):
        await p.generate(req)
    assert (await p.generate(req)).text == "Paris"
    with pytest.raises(ProviderError):
        await p.generate(GenerationRequest(model="error", prompt="x"))
    assert len(p.calls) == 4
