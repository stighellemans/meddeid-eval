from __future__ import annotations

from collections import Counter
from pathlib import Path

from meddeid_eval.stability.config import DateCfg, NameCfg, StabilityConfig, ValueShiftCfg
from meddeid_eval.stability.expand import iter_variants
from meddeid_eval.stability.offsets import offsets_valid


def _cfg() -> StabilityConfig:
    return StabilityConfig(
        dataset=Path("unused.jsonl"),
        output_dir=Path("unused_out"),
        seed=42,
        name=NameCfg(other_trials=2),
        date=DateCfg(value_shift=ValueShiftCfg(year_min=1990, year_max=2000, step=5)),
    )


def _rows():
    text = "Patient Jan Janssens, geboren 12-05-1983, gezien vandaag."
    name_begin = text.index("Jan Janssens")
    name_end = name_begin + len("Jan Janssens")
    date_begin = text.index("12-05-1983")
    date_end = date_begin + len("12-05-1983")
    return [{
        "document_id": "doc-1",
        "text": text,
        "spans": [
            {"begin": name_begin, "end": name_end, "label": "Name:Patient", "text": "Jan Janssens"},
            {"begin": date_begin, "end": date_end, "label": "Date", "text": "12-05-1983"},
        ],
    }]


def test_iter_variants_offsets_and_dimensions(lookups):
    cfg = _cfg()
    stats: Counter = Counter()
    variants = list(iter_variants(_rows(), cfg, lookups, stats))
    assert variants, "expected variants"

    # every emitted variant must have intact target offsets
    for v in variants:
        t = v["target"]
        assert offsets_valid(v["text"], t), (v["variant_id"], t)

    dims_by_kind = {"name": set(), "date": set()}
    for v in variants:
        t = v["target"]
        dims_by_kind.setdefault(t["kind"], set()).add(t["dimension"])

    assert "baseline" in dims_by_kind["name"]
    assert {"capitalization", "format", "name_source"} <= dims_by_kind["name"]
    assert "baseline" in dims_by_kind["date"]
    assert {"date_value_shift", "date_format"} <= dims_by_kind["date"]


def test_role_tracked(lookups):
    cfg = _cfg()
    variants = list(iter_variants(_rows(), cfg, lookups, Counter()))
    name_roles = {v["target"]["role"] for v in variants if v["target"]["kind"] == "name"}
    assert name_roles == {"patient"}


def test_baseline_text_unchanged(lookups):
    cfg = _cfg()
    original = _rows()[0]["text"]
    for v in iter_variants(_rows(), cfg, lookups, Counter()):
        if v["target"]["dimension"] == "baseline":
            assert v["text"] == original
