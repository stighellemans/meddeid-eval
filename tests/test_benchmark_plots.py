from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from meddeid_eval.benchmark_plots import render_comparison_plots


def _payload(name: str, recall: float, seconds: float) -> dict:
    return {
        "run": {"name": name, "seconds": seconds, "device": "cpu"},
        "core_pii_recall": recall,
        "exact_f1": recall - 0.05,
        "non_pii_redaction_rate": 0.01,
        "details": {
            "recall_by_gold_label": [
                {
                    "gold_label": "Name:Patient",
                    "matched_core_pii_chars": 90,
                    "total_core_pii_chars": 100,
                    "core_pii_recall": recall,
                },
            ],
            "recall_by_subannotation_category": [
                {
                    "subannotation_category": "given",
                    "matched_core_pii_chars": 45,
                    "total_core_pii_chars": 50,
                    "core_pii_recall": recall,
                },
            ],
            "non_pii_redaction_by_predicted_label": [
                {"prediction_label": "Profession", "non_pii_redacted_chars": 12},
            ],
            "exact_label_confusion": [
                {
                    "gold_label": "Name:Patient",
                    "prediction_label": "Name:Patient",
                    "spans": 9,
                },
                {
                    "gold_label": "Name:Patient",
                    "prediction_label": "Name:Caregiver",
                    "spans": 1,
                },
            ],
        },
    }


def test_render_comparison_plots_creates_complete_family(tmp_path) -> None:
    paths = render_comparison_plots(
        [_payload("System A", 0.95, 10), _payload("System B", 0.85, 100)],
        tmp_path,
    )
    stems = {path.stem for path in paths}
    assert stems == {
        "performance_overview",
        "recall_by_gold_label",
        "recall_by_subannotation",
        "non_pii_redactions",
        "exact_label_confusion",
        "accuracy_vs_runtime",
    }
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.stat().st_size > 1_000 for path in paths)
