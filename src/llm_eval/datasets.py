"""JSONL dataset loading.

Each line is a JSON object: {"id", "prompt", "expected", "tags", "scorer_args"}.
`id` is optional (defaults to "<file-stem>-<line>").
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from llm_eval.models import Example

DEFAULT_DATASET_DIR = Path("datasets")


class DatasetError(ValueError):
    pass


def resolve_dataset_path(name_or_path: str | Path, search_dir: Path = DEFAULT_DATASET_DIR) -> Path:
    """Accept a file path, or a bare name like `coding` resolved to `datasets/coding.jsonl`."""
    p = Path(name_or_path)
    if p.is_file():
        return p
    candidate = search_dir / f"{p.name}.jsonl"
    if candidate.is_file():
        return candidate
    raise DatasetError(f"Dataset not found: {name_or_path} (also tried {candidate})")


def load_dataset(path: str | Path, tags: Iterable[str] | None = None) -> list[Example]:
    """Load and validate a JSONL dataset, optionally keeping only examples with any of `tags`."""
    path = resolve_dataset_path(path)
    examples: list[Example] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno}: invalid JSON: {e}") from e
            data.setdefault("id", f"{path.stem}-{lineno}")
            try:
                ex = Example.model_validate(data)
            except ValidationError as e:
                raise DatasetError(f"{path}:{lineno}: {e}") from e
            if ex.id in seen:
                raise DatasetError(f"{path}:{lineno}: duplicate id {ex.id!r}")
            seen.add(ex.id)
            examples.append(ex)

    if tags:
        wanted = set(tags)
        examples = [ex for ex in examples if wanted.intersection(ex.tags)]
    return examples
