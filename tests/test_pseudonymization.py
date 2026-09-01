from __future__ import annotations

import csv

from meddeid_eval.pseudonymization import (
    EvaluationSettings,
    aggregate_tables,
    evaluate_predicted_pseudonymization,
    validate_safe_export,
    write_safe_export,
)


def _fixture_rows():
    text = "Datum 01/02/2020, leeftijd 40 jaar."
    date_begin = text.index("01/02/2020")
    age_begin = text.index("40 jaar")
    gold = [
        {
            "document_id": "private-document-id",
            "text": text,
            "spans": [
                {
                    "begin": date_begin,
                    "end": date_begin + len("01/02/2020"),
                    "label": "Date",
                },
                {
                    "begin": age_begin,
                    "end": age_begin + len("40 jaar"),
                    "label": "Age_Birthdate",
                },
            ],
        }
    ]
    predictions = [
        {
            "document_id": "private-document-id",
            "spans": [
                {
                    "begin": date_begin,
                    "end": date_begin + len("01/02/2020"),
                    "label": "Date",
                },
                {
                    "begin": age_begin,
                    "end": age_begin + len("40 jaar"),
                    "label": "Name:Patient",
                },
            ],
        }
    ]
    return gold, predictions


def test_end_to_end_evaluation_uses_detection_label_and_transformation():
    gold, predictions = _fixture_rows()
    settings = EvaluationSettings(
        language_profile="nl-BE",
        document_creation_date="2025-01-15",
        date_shift_days=371,
        generate_missing_replacements=True,
    )

    outcomes = evaluate_predicted_pseudonymization(gold, predictions, settings)

    assert outcomes[0].end_to_end_valid is True
    assert outcomes[1].end_to_end_valid is False
    assert outcomes[1].fully_redacted is True
    assert outcomes[1].failure_reason == "incorrect_label"
    summary, _ = aggregate_tables(outcomes)
    assert summary[0]["end_to_end_failure_rate"] == 0.5
    assert summary[0]["residual_exposure_rate"] == 0.0


def test_existing_replacement_is_checked_instead_of_recomputed():
    gold, predictions = _fixture_rows()
    predictions[0]["spans"][0]["replacement"] = "[not the shifted date]"
    settings = EvaluationSettings(
        language_profile="nl-BE",
        document_creation_date="2025-01-15",
        date_shift_days=371,
    )

    outcomes = evaluate_predicted_pseudonymization(gold, predictions, settings)

    assert outcomes[0].failure_reason == "unexpected_replacement"


def test_safe_export_contains_no_text_or_document_ids(tmp_path):
    gold, predictions = _fixture_rows()
    settings = EvaluationSettings(
        language_profile="nl-BE",
        document_creation_date="2025-01-15",
        date_shift_days=371,
        generate_missing_replacements=True,
    )
    outcomes = evaluate_predicted_pseudonymization(gold, predictions, settings)

    manifest = write_safe_export(
        tmp_path, outcomes, settings, prediction_source="synthetic@meta"
    )

    assert manifest["privacy_check"] == "passed"
    assert validate_safe_export(tmp_path)["files"] == [
        "failure_reasons.csv",
        "methodology.json",
        "summary.csv",
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(tmp_path.iterdir())
    )
    assert "private-document-id" not in combined
    assert "01/02/2020" not in combined
    with (tmp_path / "summary.csv").open(encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle))["end_to_end_failure_rate"] == "0.5"
