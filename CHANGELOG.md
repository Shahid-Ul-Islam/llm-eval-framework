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
