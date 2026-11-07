"""Retry with exponential backoff + jitter, honoring provider `Retry-After` hints."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from llm_eval.providers.base import RateLimitError, TransientError

log = logging.getLogger(__name__)


@dataclass
class RetryPolicy:
    max_retries: int = 4
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True

    def delay_for(self, attempt: int, error: Exception) -> float:
        """Seconds to wait before retry number `attempt` (0-based)."""
        if isinstance(error, RateLimitError) and error.retry_after is not None:
            return min(max(error.retry_after, 0.0), self.max_delay)
        delay = min(self.base_delay * (2**attempt), self.max_delay)
        if self.jitter:
            delay *= random.uniform(0.5, 1.0)
        return delay


async def with_retries[T](
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    on_retry: Callable[[int, Exception, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call `fn`, retrying `TransientError` (incl. rate limits) up to `policy.max_retries`."""
    attempt = 0
    while True:
        try:
            return await fn()
        except TransientError as e:
            if attempt >= policy.max_retries:
                raise
            delay = policy.delay_for(attempt, e)
            log.warning("retry %d/%d in %.1fs: %s", attempt + 1, policy.max_retries, delay, e)
            if on_retry:
                on_retry(attempt, e, delay)
            await sleep(delay)
            attempt += 1
