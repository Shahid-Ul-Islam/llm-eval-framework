"""Runtime settings, read from environment variables and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings can be set via env vars (prefix `LLM_EVAL_`) or `.env`.

    API keys use the providers' conventional names (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_EVAL_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "LLM_EVAL_OPENAI_API_KEY")
    )
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "LLM_EVAL_ANTHROPIC_API_KEY"),
    )

    results_dir: Path = Path("results")
    cache_path: Path = Path(".llm_eval_cache/cache.db")
    concurrency: int = 4
    max_retries: int = 4
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0
    request_timeout: float = 120.0
    # Optional JSON/YAML file mapping model -> {input: $/1M, output: $/1M}
    pricing_file: Path | None = None
    judge_model: str = "anthropic:claude-opus-5-5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
