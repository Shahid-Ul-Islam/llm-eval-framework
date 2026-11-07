"""Map OpenAI/Anthropic SDK exceptions (same shape in both SDKs) to framework errors."""

from __future__ import annotations

from types import ModuleType

from llm_eval.providers.base import ProviderError, RateLimitError, TransientError

_RETRYABLE_STATUS = {408, 409, 425, 500, 502, 503, 504, 529}


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def map_sdk_error(exc: Exception, sdk: ModuleType, provider: str) -> ProviderError:
    """Most-specific first: rate limit -> connection/timeout -> status code -> other."""
    msg = f"{provider}: {type(exc).__name__}: {exc}"
    if isinstance(exc, sdk.RateLimitError):
        return RateLimitError(msg, retry_after=_retry_after(exc))
    if isinstance(exc, sdk.APIConnectionError):  # includes APITimeoutError
        return TransientError(msg)
    if isinstance(exc, sdk.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in _RETRYABLE_STATUS or (status is not None and status >= 500):
            return TransientError(msg)
        return ProviderError(msg)
    return ProviderError(msg)
