import pytest

from llm_eval.config import Settings
from llm_eval.pricing import PricingTable


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=None,
        anthropic_api_key=None,
        results_dir=tmp_path / "results",
        cache_path=tmp_path / "cache" / "cache.db",
        concurrency=4,
        max_retries=3,
        retry_base_delay=0.0,
        retry_max_delay=0.0,
        judge_model="mock:judge",
    )


@pytest.fixture
def pricing() -> PricingTable:
    return PricingTable()
