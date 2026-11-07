"""Deterministic offline provider for tests and dry runs.

Models (use as `mock:<model>`):
  echo   - returns the prompt unchanged
  empty  - returns an empty string
  error  - always raises a non-retryable ProviderError
  <any>  - returns a canned response if `responses` matches, else echoes

Programmatic knobs (tests): `responses` (substring -> reply, or a callable),
`fail_times` / `rate_limit_times` (fail the first N calls with a transient error).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from llm_eval.models import GenerationRequest
from llm_eval.providers.base import (
    BaseProvider,
    Completion,
    ProviderError,
    RateLimitError,
    TransientError,
    register_provider,
)


def _tokens(text: str) -> int:
    return len(text.split())


@register_provider
class MockProvider(BaseProvider):
    name = "mock"

    def __init__(
        self,
        *args,
        responses: dict[str, str] | Callable[[GenerationRequest], str] | None = None,
        fail_times: int = 0,
        rate_limit_times: int = 0,
        retry_after: float | None = 0.0,
        delay_s: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.responses = responses
        self.fail_times = fail_times
        self.rate_limit_times = rate_limit_times
        self.retry_after = retry_after
        self.delay_s = delay_s
        self.calls: list[GenerationRequest] = []

    def _reply(self, request: GenerationRequest) -> str:
        if request.model == "empty":
            return ""
        if callable(self.responses):
            return self.responses(request)
        if self.responses:
            for needle, reply in self.responses.items():
                if needle in request.prompt:
                    return reply
        return request.prompt

    async def _complete(self, request: GenerationRequest) -> Completion:
        self.calls.append(request)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if request.model == "error":
            raise ProviderError("mock: forced error")
        if self.rate_limit_times > 0:
            self.rate_limit_times -= 1
            raise RateLimitError("mock: rate limited", retry_after=self.retry_after)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TransientError("mock: transient failure")
        text = self._reply(request)
        prompt_tokens = _tokens(request.prompt) + _tokens(request.system or "")
        return Completion(text=text, input_tokens=prompt_tokens, output_tokens=_tokens(text))
