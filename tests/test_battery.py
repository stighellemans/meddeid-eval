from __future__ import annotations

import pytest

from meddeid_eval.battery import document_clustered_bootstrap, validate_inputs


def _document(document_id: str, *, predicted: bool = False) -> dict:
    spans = [{"begin": 0, "end": 3, "label": "Name:Patient"}]
    return {
        "document_id": document_id,
        "text": "Jan note",
        "spans": spans if not predicted or document_id == "d1" else [],
    }


def test_document_bootstrap_is_paired_and_deterministic() -> None:
    gold = [_document("d1"), _document("d2")]
    baseline = [_document("d1", predicted=True), _document("d2", predicted=True)]
    candidate = [_document("d1"), _document("d2")]

    result = document_clustered_bootstrap(
        gold,
        {"baseline": baseline, "candidate": candidate},
        replicates=100,
        seed=7,
    )

    assert result["documents"] == 2
    assert result["replicates"] == 100
    delta = next(
        row for row in result["paired_differences"] if row["metric"] == "exact_recall"
    )
    assert delta["estimate"] == 0.5
    assert delta["ci_lower"] == 0.0
    assert delta["ci_upper"] == 1.0
    label_delta = next(
        row
        for row in result["paired_differences"]
        if row["metric"] == "exact_label_accuracy_matched"
    )
    assert label_delta["estimate"] == 0.0
    assert label_delta["valid_replicates"] == 73


def test_bootstrap_handles_no_matched_spans_for_label_accuracy() -> None:
    gold = [_document("d1")]
    empty = [{"document_id": "d1", "text": "Jan note", "spans": []}]

    result = document_clustered_bootstrap(
        gold,
        {"baseline": empty, "candidate": empty},
        replicates=10,
        seed=7,
    )

    estimate = next(
        row
        for row in result["estimates"]
        if row["metric"] == "exact_label_accuracy_matched"
    )
    delta = next(
        row
        for row in result["paired_differences"]
        if row["metric"] == "exact_label_accuracy_matched"
    )
    assert estimate["estimate"] is None
    assert estimate["valid_replicates"] == 0
    assert delta["estimate"] is None
    assert delta["valid_replicates"] == 0


def test_battery_rejects_nonmatching_document_sets() -> None:
    with pytest.raises(ValueError, match="do not exactly match"):
        validate_inputs([_document("d1")], {"candidate": [_document("d2")]})
