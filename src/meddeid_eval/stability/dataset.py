"""Dataset loading with optional text join.

Most inputs are self-contained (`{document_id, text, spans}`). Some historical
gold files split the text out: `gold.jsonl` carries spans only, and the text lives in a
separate `dataset_texts.json` map keyed by `document_id`. `load_rows` reconciles
both: it fills each record's `text` from the record itself (`text`/`plain_text`)
or, failing that, from a `text_source` map — so the perturbation stage always
sees `{text, spans}`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl
from .spans import doc_id_of


def load_text_map(path: str | Path) -> dict[str, str]:
    """document_id -> text, from either a JSON map ({id: {text: ...}} or
    {id: "..."}) or a JSONL of {document_id, text} records."""
    path = Path(path)
    out: dict[str, str] = {}
    if path.suffix == ".jsonl":
        for r in read_jsonl(path):
            tid = doc_id_of(r, len(out))
            out[tid] = str(r.get("text") or r.get("plain_text") or "")
        return out
    data = read_json(path)
    if isinstance(data, list):
        for i, r in enumerate(data):
            out[doc_id_of(r, i)] = str(r.get("text") or r.get("plain_text") or "")
    elif isinstance(data, dict):
        for tid, v in data.items():
            if isinstance(v, dict):
                out[str(tid)] = str(v.get("text") or v.get("plain_text") or "")
            else:
                out[str(tid)] = str(v or "")
    return out


def load_rows(dataset: str | Path, text_source: str | Path | None = None) -> list[dict[str, Any]]:
    """Load records and guarantee each has a non-empty `text` where possible.

    Resolution per record: existing `text` -> `plain_text` -> `text_source` map
    (by document_id). Records still lacking text are returned as-is (the caller
    skips their spans, which are out of range against an empty string)."""
    rows = read_jsonl(dataset)
    text_map = load_text_map(text_source) if text_source else {}
    for idx, row in enumerate(rows):
        if row.get("text"):
            continue
        if row.get("plain_text"):
            row["text"] = row["plain_text"]
            continue
        if text_map:
            row["text"] = text_map.get(doc_id_of(row, idx), "")
    return rows


def coverage(rows: list[dict[str, Any]]) -> dict[str, int]:
    with_text = sum(1 for r in rows if r.get("text"))
    return {"rows": len(rows), "with_text": with_text, "without_text": len(rows) - with_text}
