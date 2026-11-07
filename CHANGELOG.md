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
- OpenAI (Chat Completions) and Anthropic (Messages) providers on the official async
  SDKs: SDK retries disabled in favor of our retry layer, SDK errors mapped to
  rate-limit/transient/fatal, token usage and cost captured, lazy SDK imports.
  Anthropic: no sampling params sent, `[refusal]` marker on refusals, no silent
  fallback to another model.
- Scorers with registry and per-item `scorer_args`: `exact` (normalized, `Answer:` line
  extraction), `regex`, `json_schema` (Draft 2020-12, JSON extracted from fences/text),
  `unit_test` (Python asserts or SQLite query vs expected rows, run in an isolated
  subprocess with timeout; not a security sandbox), and `llm_judge` (rubric:
  correctness/clarity/safety 1-5, normalized to 0-1, configurable judge model).
- `ScoreResult.error`: scoring-infrastructure failures are recorded but excluded from
  averages.
- Result storage: per-run JSONL file plus SQLite `results.db` (`runs`, `results`,
  `scores` tables) with run listing, latest-run lookup and record loading.
- `Runner`: evaluates dataset x models x templates with bounded concurrency; generation
  failures and crashing scorers are recorded per item instead of aborting the run;
  inapplicable scorers are skipped; interrupted runs are marked `aborted`.
- Reporting: aggregation by model/template/tag/scorer/run (mean score, pass rate, errors,
  cost, tokens, latency, cache hits); Markdown tables and a self-contained HTML report
  (autoescaped, light/dark aware, per-tag pivot).
- A/B comparison by template, model or run on paired examples: means, delta, win/loss/tie,
  exact sign-test p-value and per-tag deltas.
- Typer CLI `llm-eval` (also `python -m llm_eval`): `run` (dataset x models x templates
  with scorers, tag filter, limit, concurrency, `--no-cache`, judge model; prints score
  and usage/cost summary), `compare` (A/B by template, model or run), `report` (Markdown
  + HTML files) and `runs` (list recent runs). Input errors exit with code 2.

### Changed
- Default `max_tokens` raised from 2048 to 16000 so thinking models are not truncated.
- SQLite cache and results DB use WAL + `synchronous=NORMAL` (4x faster writes on Windows).
