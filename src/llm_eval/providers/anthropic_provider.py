"""Anthropic (Claude) provider using the official async SDK.

Model spec: `anthropic:<model-id>`, e.g. `anthropic:claude-opus-5-5`.
Credentials: `ANTHROPIC_API_KEY` (or any credential source the SDK resolves itself,
such as an `ant auth login` profile).

Notes:
- SDK-level retries are disabled; `llm_eval.reliability` owns retry/backoff.
- Sampling params (`temperature`) are not sent: current Claude models reject them.
- Server-side refusal fallbacks are deliberately NOT enabled - a benchmark must measure
  the model it names, not a fallback. Refusals are returned as an output marker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_eval.models import GenerationRequest
from llm_eval.providers._sdk_errors import map_sdk_error
from llm_eval.providers.base import BaseProvider, Completion, register_provider

if TYPE_CHECKING:
    import anthropic

REFUSAL_MARKER = "[refusal]"


@register_provider
class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, *args, client: anthropic.AsyncAnthropic | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = client

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            import anthropic

            kwargs: dict = {"max_retries": 0, "timeout": self.settings.request_timeout}
            if self.settings.anthropic_api_key:
                kwargs["api_key"] = self.settings.anthropic_api_key
            self._client = anthropic.AsyncAnthropic(**kwargs)
        return self._client

    async def _complete(self, request: GenerationRequest) -> Completion:
        params: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system:
            params["system"] = request.system
        import anthropic

        try:
            message = await self.client.messages.create(**params)
        except anthropic.AnthropicError as e:
            raise map_sdk_error(e, anthropic, self.name) from e

        text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
        if message.stop_reason == "refusal" and not text:
            text = REFUSAL_MARKER
        u = message.usage
        input_tokens = (
            (u.input_tokens or 0)
            + (getattr(u, "cache_creation_input_tokens", None) or 0)
            + (getattr(u, "cache_read_input_tokens", None) or 0)
        )
        return Completion(text=text, input_tokens=input_tokens, output_tokens=u.output_tokens)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
