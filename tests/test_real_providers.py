"""OpenAI/Anthropic providers tested offline with fake SDK clients (no network)."""

from types import SimpleNamespace as NS

import anthropic
import httpx2 as httpx
import openai
import pytest

from llm_eval.models import GenerationRequest
from llm_eval.providers import (
    ProviderConfigError,
    ProviderError,
    RateLimitError,
    TransientError,
    get_provider_class,
)
from llm_eval.providers._sdk_errors import map_sdk_error
from llm_eval.providers.anthropic_provider import REFUSAL_MARKER, AnthropicProvider
from llm_eval.providers.openai_provider import OpenAIProvider


class _Recorder:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.kwargs = result, exc, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return self.result


def _status_error(sdk, cls, status, headers=None):
    req = httpx.Request("POST", "https://example.invalid")
    resp = httpx.Response(status, headers=headers or {}, request=req)
    return getattr(sdk, cls)("err", response=resp, body=None)


def test_builtin_providers_registered():
    assert get_provider_class("openai") is OpenAIProvider
    assert get_provider_class("anthropic") is AnthropicProvider


async def test_anthropic_request_and_usage(settings, pricing):
    msg = NS(
        content=[NS(type="thinking", thinking=""), NS(type="text", text="4")],
        stop_reason="end_turn",
        usage=NS(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=2,
        ),
    )
    rec = _Recorder(msg)
    p = AnthropicProvider(settings, pricing, client=NS(messages=rec))
    r = await p.generate(
        GenerationRequest(model="claude-opus-5-5", prompt="2+2?", system="s", temperature=0.3)
    )
    assert r.text == "4"
    assert rec.kwargs["system"] == "s" and "temperature" not in rec.kwargs
    assert rec.kwargs["messages"] == [{"role": "user", "content": "2+2?"}]
    assert r.usage.input_tokens == 12 and r.usage.output_tokens == 5
    assert r.usage.cost_usd == pytest.approx((12 * 4 + 5 * 20) / 1e6)


async def test_anthropic_refusal_marker(settings, pricing):
    msg = NS(content=[], stop_reason="refusal", usage=NS(input_tokens=1, output_tokens=0))
    p = AnthropicProvider(settings, pricing, client=NS(messages=_Recorder(msg)))
    r = await p.generate(GenerationRequest(model="claude-opus-5-5", prompt="x"))
    assert r.text == REFUSAL_MARKER


async def test_anthropic_rate_limit_mapped(settings, pricing):
    exc = _status_error(anthropic, "RateLimitError", 429, {"retry-after": "3"})
    p = AnthropicProvider(settings, pricing, client=NS(messages=_Recorder(exc=exc)))
    with pytest.raises(RateLimitError) as ei:
        await p.generate(GenerationRequest(model="m", prompt="x"))
    assert ei.value.retry_after == 3.0


async def test_openai_request_and_usage(settings, pricing):
    resp = NS(
        choices=[NS(message=NS(content="hi"))], usage=NS(prompt_tokens=7, completion_tokens=3)
    )
    rec = _Recorder(resp)
    p = OpenAIProvider(settings, pricing, client=NS(chat=NS(completions=rec)))
    r = await p.generate(
        GenerationRequest(
            model="gpt-4o-mini", prompt="q", system="sys", max_tokens=50, temperature=0.0
        )
    )
    assert r.text == "hi" and r.usage.input_tokens == 7 and r.usage.output_tokens == 3
    assert rec.kwargs["messages"][0] == {"role": "system", "content": "sys"}
    assert rec.kwargs["max_completion_tokens"] == 50 and rec.kwargs["temperature"] == 0.0
    assert r.usage.cost_usd == pytest.approx((7 * 0.15 + 3 * 0.60) / 1e6)


async def test_openai_missing_key(settings, pricing):
    with pytest.raises(ProviderConfigError):
        await OpenAIProvider(settings, pricing).generate(GenerationRequest(model="m", prompt="x"))


@pytest.mark.parametrize("sdk", [openai, anthropic])
def test_error_mapping(sdk):
    assert isinstance(
        map_sdk_error(_status_error(sdk, "InternalServerError", 500), sdk, "p"), TransientError
    )
    bad = map_sdk_error(_status_error(sdk, "BadRequestError", 400), sdk, "p")
    assert type(bad) is ProviderError
    conn = sdk.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    assert isinstance(map_sdk_error(conn, sdk, "p"), TransientError)
    overloaded = map_sdk_error(_status_error(sdk, "APIStatusError", 529), sdk, "p")
    assert isinstance(overloaded, TransientError)
