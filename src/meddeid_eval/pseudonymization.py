"""End-to-end evaluation of date and age pseudonymization from predicted spans."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from meddeid_core.age_policy import load_age_granularity_policy


EVALUATED_LABELS = ("Date", "Age_Birthdate")
SUMMARY_LABELS = ("Overall", *EVALUATED_LABELS)
FAILURE_REASONS = (
    "no_prediction_overlap",
    "incomplete_prediction_coverage",
    "fragmented_prediction_coverage",
    "incorrect_label",
    "missing_replacement",
    "placeholder_fallback",
    "unsupported_or_invalid_format",
    "unexpected_replacement",
    "invalid_transformation",
)
SUMMARY_COLUMNS = (
    "label",
    "gold_spans",
    "end_to_end_valid",
    "end_to_end_failed",
    "end_to_end_valid_rate",
    "end_to_end_failure_rate",
    "fully_redacted",
    "residual_exposure",
    "fully_redacted_rate",
    "residual_exposure_rate",
)
FAILURE_COLUMNS = (
    "label",
    "failure_reason",
    "count",
    "fraction_of_label",
)
EXPORT_FILES = {"summary.csv", "failure_reasons.csv", "methodology.json"}
OPTIONAL_EXPORT_FILES = {"privacy_manifest.json"}
_SAFE_SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@+-]*")


@dataclass(frozen=True)
class EvaluationSettings:
    language_profile: str
    document_creation_date: str
    date_shift_days: int
    generate_missing_replacements: bool = False

    def validate(self) -> None:
        if self.language_profile not in {"nl-BE", "nl-NL", "en-GB", "en-US"}:
            raise ValueError(
                "language_profile must be nl-BE, nl-NL, en-GB, or en-US"
            )
        date.fromisoformat(self.document_creation_date)
        if isinstance(self.date_shift_days, bool) or not isinstance(
            self.date_shift_days, int
        ):
            raise ValueError("date_shift_days must be an integer")
        if self.date_shift_days == 0:
            raise ValueError("date_shift_days must be non-zero")


@dataclass(frozen=True)
class GoldTarget:
    document_id: str
    label: str
    begin: int
    end: int


@dataclass(frozen=True)
class Prediction:
    document_id: str
    prediction_index: int
    label: str
    begin: int
    end: int
    replacement: str | None


@dataclass(frozen=True)
class Outcome:
    label: str
    end_to_end_valid: bool
    fully_redacted: bool
    failure_reason: str | None


def _profile(profile_id: str) -> Any:
    normalized = profile_id.strip().replace("_", "-")
    if normalized.lower().startswith("nl-"):
        from meddeid_language_nl import get_profile

        return get_profile(normalized)
    if normalized.lower().startswith("en-"):
        from meddeid_language_en import get_profile

        return get_profile(normalized)
    raise ValueError(
        f"unsupported language profile {profile_id!r}; use nl-BE, nl-NL, en-GB, or en-US"
    )


def _document_id(row: Mapping[str, Any]) -> str:
    return str(row.get("document_id") or row.get("doc_id") or "").strip()


def _spans(row: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    value = row.get("spans")
    if value is None:
        value = row.get("entities") or []
    if not isinstance(value, list):
        raise ValueError("document spans must be a list")
    return value


def _load_targets_and_predictions(
    gold_rows: Sequence[Mapping[str, Any]],
    predicted_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[GoldTarget], list[Prediction], dict[str, str]]:
    texts: dict[str, str] = {}
    targets: list[GoldTarget] = []
    for row in gold_rows:
        document_id = _document_id(row)
        text = row.get("text")
        if not document_id or not isinstance(text, str):
            raise ValueError("every gold row requires document_id and text")
        texts[document_id] = text
        for raw in _spans(row):
            label = str(raw.get("label") or "").strip()
            if label not in EVALUATED_LABELS:
                continue
            begin, end = int(raw["begin"]), int(raw["end"])
            if not 0 <= begin < end <= len(text):
                raise ValueError(f"gold range outside document {document_id!r}")
            targets.append(GoldTarget(document_id, label, begin, end))

    predictions: list[Prediction] = []
    seen_documents: set[str] = set()
    for row in predicted_rows:
        document_id = _document_id(row)
        if document_id not in texts:
            raise ValueError(
                f"prediction document {document_id!r} is absent from the gold input"
            )
        seen_documents.add(document_id)
        text = texts[document_id]
        for prediction_index, raw in enumerate(_spans(row)):
            begin, end = int(raw["begin"]), int(raw["end"])
            if not 0 <= begin < end <= len(text):
                raise ValueError(f"prediction range outside document {document_id!r}")
            replacement = raw.get("replacement")
            if replacement is not None and not isinstance(replacement, str):
                raise ValueError("prediction replacement must be a string or null")
            predictions.append(
                Prediction(
                    document_id=document_id,
                    prediction_index=prediction_index,
                    label=str(raw.get("label") or "").strip(),
                    begin=begin,
                    end=end,
                    replacement=replacement,
                )
            )
    missing = set(texts) - seen_documents
    if missing:
        raise ValueError(
            f"predictions are incomplete: {len(missing)} of {len(texts)} documents missing"
        )
    return targets, predictions, texts


def _transformation_results(
    predictions: Sequence[Prediction],
    texts: Mapping[str, str],
    settings: EvaluationSettings,
) -> dict[tuple[str, int], tuple[bool, str | None]]:
    profile = _profile(settings.language_profile)
    policy = load_age_granularity_policy()
    results: dict[tuple[str, int], tuple[bool, str | None]] = {}
    for prediction in predictions:
        if prediction.label not in EVALUATED_LABELS:
            continue
        text = texts[prediction.document_id]
        candidate = profile.replace_date(
            text[prediction.begin : prediction.end],
            label=prediction.label,
            date_shift_days=settings.date_shift_days,
            context_before=text[max(0, prediction.begin - 80) : prediction.begin],
            context_after=text[
                prediction.end : min(len(text), prediction.end + 80)
            ],
            document_creation_date=settings.document_creation_date,
            age_granularity_policy=policy,
        )
        expected = f"[{candidate.body}]" if candidate is not None else None
        actual = prediction.replacement
        if actual is None and settings.generate_missing_replacements:
            actual = expected or f"[{prediction.label}]"

        if actual is None:
            result = (False, "missing_replacement")
        elif candidate is None:
            result = (False, "unsupported_or_invalid_format")
        elif actual == f"[{prediction.label}]":
            result = (False, "placeholder_fallback")
        elif actual != expected:
            result = (False, "unexpected_replacement")
        else:
            result = (True, None)
        results[(prediction.document_id, prediction.prediction_index)] = result
    return results


def _interval_is_covered(begin: int, end: int, spans: Sequence[Prediction]) -> bool:
    cursor = begin
    for span in sorted(spans, key=lambda value: (value.begin, value.end)):
        if span.end <= cursor:
            continue
        if span.begin > cursor:
            return False
        cursor = max(cursor, span.end)
        if cursor >= end:
            return True
    return False


def _score_targets(
    targets: Sequence[GoldTarget],
    predictions: Sequence[Prediction],
    transformations: Mapping[tuple[str, int], tuple[bool, str | None]],
    redacted_predictions: set[tuple[str, int]],
) -> list[Outcome]:
    by_document: dict[str, list[Prediction]] = defaultdict(list)
    for prediction in predictions:
        by_document[prediction.document_id].append(prediction)

    outcomes: list[Outcome] = []
    for target in targets:
        overlapping = [
            prediction
            for prediction in by_document.get(target.document_id, [])
            if prediction.begin < target.end and target.begin < prediction.end
        ]
        redaction_overlaps = [
            prediction
            for prediction in overlapping
            if (prediction.document_id, prediction.prediction_index)
            in redacted_predictions
        ]
        detection_fully_covered = _interval_is_covered(
            target.begin, target.end, overlapping
        )
        fully_redacted = _interval_is_covered(
            target.begin, target.end, redaction_overlaps
        )
        containing = [
            prediction
            for prediction in overlapping
            if prediction.begin <= target.begin and prediction.end >= target.end
        ]
        same_label = [
            prediction for prediction in containing if prediction.label == target.label
        ]
        if any(
            transformations.get(
                (prediction.document_id, prediction.prediction_index), (False, None)
            )[0]
            for prediction in same_label
        ):
            outcomes.append(Outcome(target.label, True, True, None))
            continue
        if not overlapping:
            reason = "no_prediction_overlap"
        elif not detection_fully_covered:
            reason = "incomplete_prediction_coverage"
        elif not containing:
            reason = "fragmented_prediction_coverage"
        elif not same_label:
            reason = "incorrect_label"
        else:
            reason = (
                transformations.get(
                    (same_label[0].document_id, same_label[0].prediction_index),
                    (False, "invalid_transformation"),
                )[1]
                or "invalid_transformation"
            )
            if reason not in FAILURE_REASONS:
                reason = "invalid_transformation"
        outcomes.append(Outcome(target.label, False, fully_redacted, reason))
    return outcomes


def evaluate_predicted_pseudonymization(
    gold_rows: Sequence[Mapping[str, Any]],
    predicted_rows: Sequence[Mapping[str, Any]],
    settings: EvaluationSettings,
) -> list[Outcome]:
    """Evaluate the complete predicted-span to transformation chain."""
    settings.validate()
    targets, predictions, texts = _load_targets_and_predictions(
        gold_rows, predicted_rows
    )
    transformations = _transformation_results(predictions, texts, settings)
    redacted_predictions = {
        (prediction.document_id, prediction.prediction_index)
        for prediction in predictions
        if prediction.replacement is not None
        or settings.generate_missing_replacements
    }
    return _score_targets(
        targets, predictions, transformations, redacted_predictions
    )


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def aggregate_tables(
    outcomes: Sequence[Outcome],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[Outcome]] = defaultdict(list)
    by_label["Overall"].extend(outcomes)
    for outcome in outcomes:
        by_label[outcome.label].append(outcome)
    summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for label in SUMMARY_LABELS:
        group = by_label.get(label, [])
        total = len(group)
        valid = sum(outcome.end_to_end_valid for outcome in group)
        fully_redacted = sum(outcome.fully_redacted for outcome in group)
        summary.append(
            {
                "label": label,
                "gold_spans": total,
                "end_to_end_valid": valid,
                "end_to_end_failed": total - valid,
                "end_to_end_valid_rate": _fraction(valid, total),
                "end_to_end_failure_rate": _fraction(total - valid, total),
                "fully_redacted": fully_redacted,
                "residual_exposure": total - fully_redacted,
                "fully_redacted_rate": _fraction(fully_redacted, total),
                "residual_exposure_rate": _fraction(total - fully_redacted, total),
            }
        )
        counts = Counter(
            outcome.failure_reason
            for outcome in group
            if outcome.failure_reason is not None
        )
        for reason in FAILURE_REASONS:
            if counts[reason]:
                failures.append(
                    {
                        "label": label,
                        "failure_reason": reason,
                        "count": counts[reason],
                        "fraction_of_label": _fraction(counts[reason], total),
                    }
                )
    return summary, failures


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _numeric(value: str, *, integer: bool = False) -> None:
    try:
        number = int(value) if integer else float(value)
    except ValueError as error:
        raise ValueError(f"expected numeric aggregate, got {value!r}") from error
    if not integer and (math.isnan(number) or math.isinf(number)):
        raise ValueError("aggregate values must be finite")


def validate_safe_export(export_dir: str | Path) -> dict[str, Any]:
    """Validate the aggregate allowlist before results leave a clinical system."""
    output_dir = Path(export_dir)
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    unexpected = actual - EXPORT_FILES - OPTIONAL_EXPORT_FILES
    missing = EXPORT_FILES - actual
    if unexpected or missing:
        raise ValueError(
            f"export allowlist violation: unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )
    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUMMARY_COLUMNS:
            raise ValueError("unexpected summary columns")
        for row in reader:
            if row["label"] not in SUMMARY_LABELS:
                raise ValueError("uncontrolled summary label")
            for field in (
                "gold_spans",
                "end_to_end_valid",
                "end_to_end_failed",
                "fully_redacted",
                "residual_exposure",
            ):
                _numeric(row[field], integer=True)
            for field in (
                "end_to_end_valid_rate",
                "end_to_end_failure_rate",
                "fully_redacted_rate",
                "residual_exposure_rate",
            ):
                _numeric(row[field])
    with (output_dir / "failure_reasons.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FAILURE_COLUMNS:
            raise ValueError("unexpected failure-reason columns")
        for row in reader:
            if row["label"] not in SUMMARY_LABELS:
                raise ValueError("uncontrolled failure label")
            if row["failure_reason"] not in FAILURE_REASONS:
                raise ValueError("uncontrolled failure reason")
            _numeric(row["count"], integer=True)
            _numeric(row["fraction_of_label"])
    methodology = json.loads(
        (output_dir / "methodology.json").read_text(encoding="utf-8")
    )
    expected_methodology_keys = {
        "schema_version",
        "generated_at",
        "evaluation_scope",
        "prediction_source",
        "language_profile",
        "document_creation_date",
        "date_shift_days",
        "birthdate_replacement_mode",
        "generated_missing_replacements",
        "contains_source_text",
        "contains_document_identifiers",
    }
    if set(methodology) != expected_methodology_keys:
        raise ValueError("unexpected methodology fields")
    if methodology.get("evaluation_scope") != "predicted_spans_end_to_end":
        raise ValueError("unexpected evaluation scope")
    if not _SAFE_SOURCE_ID_RE.fullmatch(str(methodology.get("prediction_source", ""))):
        raise ValueError("unsafe prediction source id")
    if methodology.get("contains_source_text") is not False:
        raise ValueError("export permits source text")
    if methodology.get("contains_document_identifiers") is not False:
        raise ValueError("export permits document identifiers")
    datetime.fromisoformat(str(methodology["generated_at"]))
    date.fromisoformat(str(methodology["document_creation_date"]))
    if methodology["language_profile"] not in {"nl-BE", "nl-NL", "en-GB", "en-US"}:
        raise ValueError("uncontrolled language profile")
    if not isinstance(methodology["date_shift_days"], int) or isinstance(
        methodology["date_shift_days"], bool
    ):
        raise ValueError("date_shift_days must be an integer")
    if methodology["birthdate_replacement_mode"] != "age":
        raise ValueError("unexpected birthdate replacement mode")
    if not isinstance(methodology["generated_missing_replacements"], bool):
        raise ValueError("generated_missing_replacements must be boolean")
    return {"files": sorted(EXPORT_FILES)}


def write_safe_export(
    export_dir: str | Path,
    outcomes: Sequence[Outcome],
    settings: EvaluationSettings,
    *,
    prediction_source: str,
) -> dict[str, Any]:
    """Write aggregate-only CSV/JSON artifacts and a verified privacy manifest."""
    if not _SAFE_SOURCE_ID_RE.fullmatch(prediction_source):
        raise ValueError("prediction source must be a safe identifier")
    output_dir = Path(export_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    unexpected = existing - EXPORT_FILES - OPTIONAL_EXPORT_FILES
    if unexpected:
        raise ValueError(
            f"refusing to clean export directory with unrelated files: {sorted(unexpected)}"
        )
    for filename in EXPORT_FILES | OPTIONAL_EXPORT_FILES:
        (output_dir / filename).unlink(missing_ok=True)
    summary, failures = aggregate_tables(outcomes)
    _write_csv(output_dir / "summary.csv", summary, SUMMARY_COLUMNS)
    _write_csv(output_dir / "failure_reasons.csv", failures, FAILURE_COLUMNS)
    methodology = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evaluation_scope": "predicted_spans_end_to_end",
        "prediction_source": prediction_source,
        "language_profile": settings.language_profile,
        "document_creation_date": settings.document_creation_date,
        "date_shift_days": settings.date_shift_days,
        "birthdate_replacement_mode": "age",
        "generated_missing_replacements": settings.generate_missing_replacements,
        "contains_source_text": False,
        "contains_document_identifiers": False,
    }
    (output_dir / "methodology.json").write_text(
        json.dumps(methodology, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = validate_safe_export(output_dir)
    manifest = {
        "schema_version": 1,
        "privacy_check": "passed",
        "contains_source_text": False,
        "contains_document_identifiers": False,
        "files": report["files"],
        "controlled_string_fields": {
            "label": list(SUMMARY_LABELS),
            "failure_reason": list(FAILURE_REASONS),
        },
    }
    (output_dir / "privacy_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
