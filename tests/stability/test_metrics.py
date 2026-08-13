from __future__ import annotations

from meddeid_eval.stability.metrics import (
    aggregate,
    category_key,
    classify_target,
    paired_degradation_stats,
)


def _t(begin, end, label):
    return {"begin": begin, "end": end, "label": label, "text": "x"}


def test_category_key():
    assert category_key("Name:Patient") == "name"
    assert category_key("Date") == "date"


def test_classify_target_levels():
    target = _t(3, 15, "Name:Patient")
    pred_full_other_subtype = [{"begin": 3, "end": 15, "label": "Name:Caregiver", "category": "Name"}]
    assert classify_target(target, pred_full_other_subtype, "category") == "full"
    assert classify_target(target, pred_full_other_subtype, "label") == "missed"

    assert classify_target(target, [{"begin": 3, "end": 10, "label": "Name:Patient", "category": "Name"}], "category") == "partial"
    assert classify_target(target, [{"begin": 20, "end": 25, "label": "Name:Patient", "category": "Name"}], "category") == "missed"
    assert classify_target(target, [{"begin": 3, "end": 15, "label": "Date", "category": "Date"}], "category") == "missed"


def test_paired_degradation_stats_basic():
    stats = paired_degradation_stats({"a": 1.0, "b": 1.0}, {"a": 1.0, "b": 0.0}, seed=1, iterations=200)
    assert stats["n_pairs"] == 2
    assert abs(stats["mean_degradation"] - 0.5) < 1e-9


def test_aggregate_toy():
    def target(span_index, dim, value, begin, end, text, label="Name:Patient", role="patient", kind="name"):
        return {"begin": begin, "end": end, "label": label, "text": text, "role": role,
                "kind": kind, "span_index": span_index, "dimension": dim,
                "dimension_value": value, "base_text": "Jan"}

    variants = [
        {"variant_id": "d::0::baseline::original", "document_id": "d",
         "target": target(0, "baseline", "original", 0, 3, "Jan")},
        {"variant_id": "d::0::capitalization::lower", "document_id": "d",
         "target": target(0, "capitalization", "lower", 0, 3, "jan")},
    ]
    preds = {
        "d::0::baseline::original": [{"begin": 0, "end": 3, "label": "Name:Patient", "category": "Name"}],
        "d::0::capitalization::lower": [],  # missed
    }
    agg = aggregate(variants, preds, seed=1, level="category")
    assert agg["baseline_recall"] == 1.0
    assert agg["n_targets"] == 1
    lower = [r for r in agg["rows"] if r["dimension"] == "capitalization" and r["dimension_value"] == "lower"][0]
    assert lower["full_plus_partial_recall"] == 0.0
    assert abs(agg["degradation"]["capitalization"]["patient"]["mean_degradation"] - 1.0) < 1e-9
