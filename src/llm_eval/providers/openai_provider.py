"""OpenAI provider (Chat Completions) using the official async SDK.

Model spec: `openai:<model>`, e.g. `openai:gpt-4o-mini`. Credentials: `OPENAI_API_KEY`.
SDK-level retries are disabled; `llm_eval.reliability` owns retry/backoff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_eval.models import GenerationRequest
from llm_eval.providers._sdk_errors import map_sdk_error
from llm_eval.providers.base import (
    BaseProvider,
    Completion,
    ProviderConfigError,
    register_provider,
)

if TYPE_CHECKING:
    import openai


@register_provider
class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, *args, client: openai.AsyncOpenAI | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = client

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            import openai

            if not self.settings.openai_api_key:
                raise ProviderConfigError("OPENAI_API_KEY is not set (env var or .env)")
            self._client = openai.AsyncOpenAI(
                api_key=self.settings.openai_api_key,
                max_retries=0,
                timeout=self.settings.request_timeout,
            )
        return self._client

    async def _complete(self, request: GenerationRequest) -> Completion:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        params: dict = {
            "model": request.model,
            "messages": messages,
            "max_completion_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        import openai

        try:
            resp = await self.client.chat.completions.create(**params)
        except openai.OpenAIError as e:
            raise map_sdk_error(e, openai, self.name) from e

        text = (resp.choices[0].message.content or "") if resp.choices else ""
        usage = resp.usage
        return Completion(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
