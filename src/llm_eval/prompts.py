"""Prompt templates: load named variants from YAML and render them for an example."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import yaml

from llm_eval.models import Example, PromptTemplate

DEFAULT_TEMPLATES_FILE = Path("prompts/templates.yaml")
BASELINE = PromptTemplate(name="baseline", template="{prompt}")

_PLACEHOLDER = re.compile(r"\{(prompt|id|tags)\}")


class TemplateError(ValueError):
    pass


def load_templates(path: str | Path | None = None) -> dict[str, PromptTemplate]:
    """Load all templates from a YAML mapping `name -> {template, system}`.

    Always contains `baseline` (identity template), even if the file is missing.
    """
    templates = {BASELINE.name: BASELINE}
    p = Path(path) if path is not None else DEFAULT_TEMPLATES_FILE
    if not p.is_file():
        if path is not None:
            raise TemplateError(f"Templates file not found: {p}")
        return templates
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TemplateError(f"{p} must be a mapping of template name -> definition")
    for name, spec in data.items():
        if isinstance(spec, str):
            spec = {"template": spec}
        templates[str(name)] = PromptTemplate(name=str(name), **spec)
    return templates


def select_templates(
    available: dict[str, PromptTemplate], names: Iterable[str] | None
) -> list[PromptTemplate]:
    if not names:
        return [available.get("baseline", BASELINE)]
    missing = [n for n in names if n not in available]
    if missing:
        raise TemplateError(f"Unknown template(s): {missing}. Available: {sorted(available)}")
    return [available[n] for n in names]


def render(template: PromptTemplate, example: Example) -> str:
    values = {"prompt": example.prompt, "id": example.id, "tags": ", ".join(example.tags)}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template.template)
