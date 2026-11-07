"""Helpers to pull answers, code and JSON out of free-form model output."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```[ \t]*([\w+-]*)[^\n]*\n(.*?)```", re.DOTALL)
_ANSWER_LINE = re.compile(
    r"^\W*answer[*_\s]*[:\-][*_\s]*(.+?)[*_\s]*$", re.IGNORECASE | re.MULTILINE
)


def extract_answer(text: str) -> str:
    """Return the last `Answer: ...` line's value, or the whole text if there is none."""
    matches = _ANSWER_LINE.findall(text)
    return matches[-1] if matches else text


def extract_code(text: str, language: str | None = None) -> str:
    """Return the last fenced block tagged `language` (else last fence, else the text)."""
    blocks = _FENCE.findall(text)
    if not blocks:
        return text.strip()
    if language:
        tagged = [body for tag, body in blocks if tag.lower() == language.lower()]
        if tagged:
            return tagged[-1].strip()
    return blocks[-1][1].strip()


def extract_json(text: str) -> Any:
    """Parse JSON from a ```json fence or the first decodable object/array in `text`.

    Raises ValueError if nothing parses.
    """
    candidates = [body for tag, body in _FENCE.findall(text) if tag.lower() in ("json", "")]
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, i)
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in output")
