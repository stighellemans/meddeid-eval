from __future__ import annotations

import csv
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import score_documents

_METRICS = (
    ("exact_precision", "higher"),
    ("exact_recall", "higher"),
    ("exact_f1", "higher"),
    ("character_precision", "higher"),
    ("character_recall", "higher"),
    ("core_pii_recall", "higher"),
    ("exact_label_accuracy_matched", "higher"),
    ("non_pii_redaction_rate", "lower"),
)


def validate_inputs(gold_rows: list[dict], systems: Mapping[str, list[dict]]) -> None:
    if not gold_rows:
        raise ValueError("gold JSONL is empty")

    def indexed(rows: list[dict], name: str) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for row in rows:
            document_id = str(row.get("document_id") or row.get("doc_id") or "")
            if not document_id:
                raise ValueError(f"{name} contains a row without document_id")
            if document_id in result:
                raise ValueError(
                    f"{name} contains duplicate document_id {document_id!r}"
                )
            result[document_id] = row
        return result

    gold = indexed(gold_rows, "gold")
    for name, rows in systems.items():
        predictions = indexed(rows, name)
        missing = sorted(set(gold) - set(predictions))
        extra = sorted(set(predictions) - set(gold))
        if missing or extra:
            raise ValueError(
                f"{name} document IDs do not exactly match gold "
                f"(missing={len(missing)}, extra={len(extra)})"
            )
        for document_id, gold_row in gold.items():
            predicted_text = predictions[document_id].get("text")
            if predicted_text is not None and predicted_text != gold_row.get("text"):
                raise ValueError(f"{name} text differs from gold for {document_id!r}")


def _ratio(numerator: int, denominator: int, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _aggregate_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    counts = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "exact_true_positive",
            "exact_gold_spans",
            "exact_predicted_spans",
            "matched_label_characters",
            "gold_label_characters",
            "predicted_label_characters",
            "core_pii_matched_characters",
            "core_pii_characters",
            "label_matched_spans",
            "label_correct_spans",
            "non_pii_redacted_chars",
            "non_pii_characters",
        )
    }
    precision = _ratio(
        counts["exact_true_positive"], counts["exact_predicted_spans"], 1.0
    )
    recall = _ratio(counts["exact_true_positive"], counts["exact_gold_spans"], 1.0)
    return {
        "exact_precision": precision,
        "exact_recall": recall,
        "exact_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "character_precision": _ratio(
            counts["matched_label_characters"],
            counts["predicted_label_characters"],
            1.0,
        ),
        "character_recall": _ratio(
            counts["matched_label_characters"], counts["gold_label_characters"], 1.0
        ),
        "core_pii_recall": _ratio(
            counts["core_pii_matched_characters"], counts["core_pii_characters"], 1.0
        ),
        "exact_label_accuracy_matched": (
            counts["label_correct_spans"] / counts["label_matched_spans"]
            if counts["label_matched_spans"]
            else None
        ),
        "non_pii_redaction_rate": _ratio(
            counts["non_pii_redacted_chars"], counts["non_pii_characters"], 0.0
        ),
    }


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def document_clustered_bootstrap(
    gold_rows: list[dict],
    systems: Mapping[str, list[dict]],
    *,
    replicates: int = 10_000,
    seed: int = 20_260_821,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be at least 1")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence level must be between 0 and 1")
    validate_inputs(gold_rows, systems)

    gold_by_id = {
        str(row.get("document_id") or row.get("doc_id")): row for row in gold_rows
    }
    prediction_maps = {
        name: {str(row.get("document_id") or row.get("doc_id")): row for row in rows}
        for name, rows in systems.items()
    }
    document_ids = list(gold_by_id)
    per_document = {
        name: [
            score_documents([gold_by_id[document_id]], [predictions[document_id]])
            for document_id in document_ids
        ]
        for name, predictions in prediction_maps.items()
    }
    points = {name: _aggregate_counts(rows) for name, rows in per_document.items()}
    samples = {name: {metric: [] for metric, _ in _METRICS} for name in systems}
    rng = random.Random(seed)
    for _ in range(replicates):
        indices = [rng.randrange(len(document_ids)) for _ in document_ids]
        for name, rows in per_document.items():
            metrics = _aggregate_counts([rows[index] for index in indices])
            for metric, _ in _METRICS:
                samples[name][metric].append(metrics[metric])

    alpha = (1 - confidence_level) / 2
    estimates = []
    for name in systems:
        for metric, direction in _METRICS:
            values = [
                value for value in samples[name][metric] if value is not None
            ]
            estimates.append(
                {
                    "system": name,
                    "metric": metric,
                    "direction": direction,
                    "estimate": points[name][metric],
                    "ci_lower": _percentile(values, alpha),
                    "ci_upper": _percentile(values, 1 - alpha),
                    "valid_replicates": len(values),
                }
            )

    differences = []
    names = list(systems)
    if len(names) >= 2:
        baseline = names[0]
        for candidate in names[1:]:
            for metric, direction in _METRICS:
                values = [
                    candidate_value - baseline_value
                    for candidate_value, baseline_value in zip(
                        samples[candidate][metric], samples[baseline][metric]
                    )
                    if candidate_value is not None and baseline_value is not None
                ]
                differences.append(
                    {
                        "baseline": baseline,
                        "candidate": candidate,
                        "metric": metric,
                        "direction": direction,
                        "estimate": (
                            points[candidate][metric] - points[baseline][metric]
                            if points[candidate][metric] is not None
                            and points[baseline][metric] is not None
                            else None
                        ),
                        "ci_lower": _percentile(values, alpha),
                        "ci_upper": _percentile(values, 1 - alpha),
                        "valid_replicates": len(values),
                    }
                )
    return {
        "method": "complete-document clustered percentile bootstrap",
        "documents": len(document_ids),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "estimates": estimates,
        "paired_differences": differences,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["system"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    destination: Path,
    payloads: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        str(payload.get("run", {}).get("name") or f"System {i + 1}")
        for i, payload in enumerate(payloads)
    ]
    summary_fields = [metric for metric, _ in _METRICS]
    tables: dict[str, list[dict[str, Any]]] = {
        "summary.csv": [
            {"system": name, **{field: payload.get(field) for field in summary_fields}}
            for name, payload in zip(names, payloads)
        ],
        "bootstrap_estimates.csv": list(bootstrap["estimates"]),
        "bootstrap_paired_differences.csv": list(bootstrap["paired_differences"]),
    }
    for detail_key, filename in (
        ("exact_by_label", "exact_by_label.csv"),
        ("recall_by_gold_label", "recall_by_gold_label.csv"),
        ("recall_by_subannotation_category", "recall_by_subannotation_category.csv"),
        (
            "non_pii_redaction_by_predicted_label",
            "non_pii_redaction_by_predicted_label.csv",
        ),
        ("exact_label_confusion", "exact_label_confusion.csv"),
        ("matched_label_confusion", "matched_label_confusion.csv"),
        ("label_confusion_chars", "label_confusion_characters.csv"),
    ):
        rows = []
        for name, payload in zip(names, payloads):
            rows.extend(
                {"system": name, **row}
                for row in payload.get("details", {}).get(detail_key, [])
            )
        tables[filename] = rows
    paths = []
    for filename, rows in tables.items():
        path = destination / filename
        _write_csv(path, rows)
        paths.append(path)
    return paths


def write_report(
    path: Path,
    *,
    profiles: Sequence[str],
    baseline_model: str,
    payloads: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
) -> None:
    names = [
        str(payload.get("run", {}).get("name") or f"System {i + 1}")
        for i, payload in enumerate(payloads)
    ]
    intervals = {(row["system"], row["metric"]): row for row in bootstrap["estimates"]}
    lines = [
        "# MedDeID evaluation battery",
        "",
        f"Language profile: {', '.join(profiles)}  ",
        f"Automatic baseline: `{baseline_model}`  ",
        *(
            [f"Resolved baseline revision: `{payloads[0]['run']['revision']}`  "]
            if payloads[0].get("run", {}).get("revision")
            else []
        ),
        f"Documents: {bootstrap['documents']}  ",
        f"Uncertainty: {bootstrap['replicates']:,} complete-document bootstrap replicates (seed {bootstrap['seed']})",
        "",
        "| System | Exact-span F1 | Core PII recall | Label accuracy | Non-PII redaction |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, payload in zip(names, payloads):
        rendered = []
        for metric in (
            "exact_f1",
            "core_pii_recall",
            "exact_label_accuracy_matched",
            "non_pii_redaction_rate",
        ):
            row = intervals[(name, metric)]
            if payload[metric] is None or row["ci_lower"] is None:
                rendered.append("NA")
            else:
                rendered.append(
                    f"{100 * float(payload[metric]):.1f}% "
                    f"({100 * row['ci_lower']:.1f}–{100 * row['ci_upper']:.1f}%)"
                )
        lines.append(f"| {name} | {' | '.join(rendered)} |")
    lines.extend(
        [
            "",
            "Exact-span F1 requires both boundary and primary label to match. Core PII recall is label-agnostic character coverage. Label accuracy is the exact primary-label match among deterministically one-to-one matched overlapping spans. Non-PII redaction is a false-positive rate, so lower is better.",
            "",
            "Subannotation categories are a gold-only analysis layer: the model does not predict them. The battery therefore reports detected-versus-missed coverage by subcategory, while primary predicted labels receive a full confusion matrix with explicit Missed and Spurious cells.",
            "",
            "See `tables/` for machine-readable results and `plots/` for PNG/PDF figures.",
            "",
        ]
    )
    if bootstrap["documents"] < 2:
        lines.extend(
            [
                (
                    "> **Smoke-test warning:** fewer than two test documents were "
                    "supplied. Bootstrap intervals therefore cannot represent "
                    "between-document uncertainty; do not use this run for a "
                    "model-quality claim."
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
