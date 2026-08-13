"""Text rewriting with offset remapping + integrity checks.

``apply_replacements`` is lifted verbatim (behaviour-preserving) from the prior
robustness job: it rebuilds the text left-to-right and recomputes each span's
``begin``/``end`` from the actual rendered string, so offsets can never drift.
``replace_one`` is the single-target convenience used by the OFAT expander.
"""
from __future__ import annotations

from typing import Any


def apply_replacements(
    text: str,
    spans: list[dict[str, Any]],
    replacement_texts: list[str],
    extra_meta: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    ordered = sorted(
        zip(spans, replacement_texts, extra_meta or [{} for _ in spans]),
        key=lambda item: int(item[0]["begin"]),
    )
    parts: list[str] = []
    new_spans: list[dict[str, Any]] = []
    cursor = 0
    for span, replacement, meta in ordered:
        begin = int(span["begin"])
        end = int(span["end"])
        if begin < cursor:
            raise ValueError("Cannot apply overlapping replacements")
        parts.append(text[cursor:begin])
        new_begin = sum(len(part) for part in parts)
        parts.append(replacement)
        new_end = new_begin + len(replacement)
        new_spans.append({"begin": new_begin, "end": new_end, "label": span["label"], "text": replacement, **meta})
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), new_spans


def replace_one(
    text: str, span: dict[str, Any], replacement: str
) -> tuple[str, dict[str, Any]]:
    """Replace exactly one span's text; return (new_text, new_span) with the
    target's recomputed offsets. Other spans are intentionally not tracked —
    the analyzer only scores the perturbed target."""
    new_text, new_spans = apply_replacements(text, [span], [replacement])
    return new_text, new_spans[0]


def offsets_valid(text: str, span: dict[str, Any]) -> bool:
    begin, end = int(span["begin"]), int(span["end"])
    return 0 <= begin < end <= len(text) and text[begin:end] == span["text"]


def assert_offsets(text: str, span: dict[str, Any]) -> None:
    if not offsets_valid(text, span):
        begin, end = int(span["begin"]), int(span["end"])
        raise AssertionError(
            f"offset integrity failed: text[{begin}:{end}]={text[begin:end]!r} != {span.get('text')!r}"
        )
