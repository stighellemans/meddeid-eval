"""Canonical span/label helpers.

Extracted and lightly generalised from
``deid_training/job_templates/deid_bert_gpu_robustness_job/code/robustness.py``
so the harness reads the canonical schema (``spans`` + ``label`` /
``category`` / ``subtype``).
"""
from __future__ import annotations

import re
from typing import Any

# Map canonical Name labels -> the stability "role" dimension.
ROLE_BY_LABEL = {
    "namepatient": "patient",
    "namecaregiver": "caregiver",
}


def label_value(span: dict[str, Any]) -> str:
    return str(span.get("label") or span.get("Category") or span.get("category") or span.get("type") or "")


def label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", label.lower())


def span_label_keys(span: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for key in [
        "label", "Category", "category", "type", "entity_label", "group", "SubType", "subtype", "subcategory",
    ]:
        value = span.get(key)
        if value is None:
            continue
        normalized = label_key(str(value))
        if normalized:
            keys.add(normalized)
    category = span.get("Category") or span.get("category")
    subtype = span.get("SubType") or span.get("subtype") or span.get("type")
    if category is not None and subtype is not None:
        combined = label_key(f"{category}:{subtype}")
        if combined:
            keys.add(combined)
    return keys


def label_matches(span: dict[str, Any], aliases: set[str]) -> bool:
    for candidate in span_label_keys(span):
        if candidate in aliases:
            return True
        if any(alias in candidate for alias in aliases):
            return True
    return False


def aliases_for(labels: list[str] | tuple[str, ...]) -> set[str]:
    return {label_key(label) for label in labels}


def raw_spans(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical gold-span list."""
    value = row.get("spans")
    return [span for span in value if isinstance(span, dict)] if isinstance(value, list) else []


def canonical_label(span: dict[str, Any]) -> str:
    """Best-effort canonical ``Category:Subtype`` (or ``Category``) label."""
    lbl = span.get("label")
    if lbl:
        return str(lbl)
    category = span.get("Category") or span.get("category")
    subtype = span.get("SubType") or span.get("subtype")
    if category and subtype:
        return f"{category}:{subtype}"
    return str(category or "")


def normalize_gold_spans(row: dict[str, Any], aliases: set[str]) -> list[dict[str, Any]]:
    """Spans matching ``aliases``, normalised to ``{begin,end,label,text}`` and
    re-validated against the row text (drops out-of-range/empty spans)."""
    text = str(row.get("text", ""))
    out: list[dict[str, Any]] = []
    for span in raw_spans(row):
        if not label_matches(span, aliases):
            continue
        try:
            begin = int(span.get("begin", span.get("start", -1)))
            end = int(span.get("end", -1))
        except (TypeError, ValueError):
            continue
        if begin < 0 or end <= begin or end > len(text):
            continue
        out.append({"begin": begin, "end": end, "label": canonical_label(span), "text": text[begin:end]})
    return out


def role_of(label: str) -> str | None:
    return ROLE_BY_LABEL.get(label_key(label))


def doc_id_of(row: dict[str, Any], idx: int) -> str:
    return str(row.get("document_id") or row.get("doc_id") or row.get("id") or f"doc_{idx}")
