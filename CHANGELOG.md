# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project skeleton: `pyproject.toml` (hatchling, ruff, pytest config, `llm-eval` entry point),
  `src/llm_eval/` package, Python `.gitignore`, this changelog.
- Core pydantic models (`Example`, `PromptTemplate`, `GenerationRequest`, `ModelResponse`,
  `Usage`, `ScoreResult`, `RunRecord`), env/`.env` settings, and a pricing table with
  hardcoded defaults plus JSON/YAML override file (`LLM_EVAL_PRICING_FILE`).
- `.env.example`, `.gitattributes` (LF line endings).
- JSONL dataset loader with validation (line-numbered errors, duplicate-id check),
  tag filtering and name resolution (`coding` -> `datasets/coding.jsonl`).
- Sample datasets: `coding.jsonl` (6 Python, 3 SQL, 1 JSON task with per-item test
  specs) and `reasoning.jsonl` (10 logic/math items with exact + regex answers).
- Prompt templates: YAML-defined named variants (`baseline`, `concise`, `expert`) with
  optional system prompt; safe `{prompt}`/`{id}`/`{tags}` substitution for A/B runs.
- Provider abstraction: `BaseProvider` (async `_complete`, timing + cost in `generate`),
  error hierarchy (`ProviderError`, `TransientError`, `RateLimitError` with `retry_after`),
  registry with `provider:model` spec parsing.
- Offline `mock` provider (`echo`, `empty`, `error` models; canned responses and
  injectable rate-limit/transient failures for tests).
- Reliability: `RetryPolicy` + `with_retries` (exponential backoff with jitter, honors
  `Retry-After` on rate limits, capped at `retry_max_delay`; fatal errors not retried).
- SQLite response cache keyed by SHA-256 of provider + request.
- `LLMClient`: resolves `provider:model` specs, applies cache and retries, and tracks
  per-model calls, cache hits, retries, errors, tokens and cost.
