from pathlib import Path

import pytest

from llm_eval.models import Example, PromptTemplate
from llm_eval.prompts import TemplateError, load_templates, render, select_templates

ROOT = Path(__file__).resolve().parents[1]


def test_render_substitutes_only_known_placeholders():
    t = PromptTemplate(name="t", template='[{id}|{tags}] {prompt} {"json": {other}}')
    ex = Example(id="e1", prompt="Q?", tags=["a", "b"])
    assert render(t, ex) == '[e1|a, b] Q? {"json": {other}}'


def test_prompt_containing_braces_is_not_reinterpreted():
    t = PromptTemplate(name="t", template="X {prompt}")
    assert render(t, Example(id="e", prompt="{id}")) == "X {id}"


def test_load_repo_templates():
    templates = load_templates(ROOT / "prompts" / "templates.yaml")
    assert {"baseline", "concise", "expert"} <= set(templates)
    assert templates["expert"].system


def test_missing_default_file_gives_baseline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert list(load_templates()) == ["baseline"]
    with pytest.raises(TemplateError):
        load_templates(tmp_path / "nope.yaml")


def test_select_templates():
    available = {"baseline": PromptTemplate(name="baseline"), "x": PromptTemplate(name="x")}
    assert [t.name for t in select_templates(available, None)] == ["baseline"]
    assert [t.name for t in select_templates(available, ["x", "baseline"])] == ["x", "baseline"]
    with pytest.raises(TemplateError, match="Unknown"):
        select_templates(available, ["nope"])
