from __future__ import annotations

import json

from meddeid_eval.stability.dataset import coverage, load_rows, load_text_map


def _write(p, lines):
    p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")


def test_text_map_from_json_map(tmp_path):
    p = tmp_path / "texts.json"
    p.write_text(json.dumps({"a": {"text": "hello", "patient": {}}, "b": "world"}), encoding="utf-8")
    m = load_text_map(p)
    assert m == {"a": "hello", "b": "world"}


def test_text_map_from_jsonl(tmp_path):
    p = tmp_path / "texts.jsonl"
    _write(p, [{"document_id": "a", "text": "hi"}, {"document_id": "b", "text": "yo"}])
    assert load_text_map(p) == {"a": "hi", "b": "yo"}


def test_load_rows_joins_split_text(tmp_path):
    # historical gold shape: spans with no text plus a separate text map
    gold = tmp_path / "gold.jsonl"
    _write(gold, [
        {"document_id": "d1", "spans": [{"begin": 0, "end": 3, "label": "Name:Patient", "text": "Jan"}]},
        {"document_id": "d2", "spans": []},
    ])
    texts = tmp_path / "dataset_texts.json"
    texts.write_text(json.dumps({"d1": {"text": "Jan is here"}, "d2": {"text": "niets"}}), encoding="utf-8")
    rows = load_rows(gold, texts)
    assert rows[0]["text"] == "Jan is here"
    assert rows[1]["text"] == "niets"
    assert coverage(rows)["without_text"] == 0


def test_load_rows_plain_text_fallback(tmp_path):
    ds = tmp_path / "cap.jsonl"
    _write(ds, [{"document_id": "d1", "plain_text": "van plain_text", "spans": []}])
    rows = load_rows(ds)
    assert rows[0]["text"] == "van plain_text"


def test_load_rows_prefers_existing_text(tmp_path):
    ds = tmp_path / "ds.jsonl"
    _write(ds, [{"document_id": "d1", "text": "real", "plain_text": "ignore me"}])
    rows = load_rows(ds, None)
    assert rows[0]["text"] == "real"
