from __future__ import annotations

from collections import defaultdict
from typing import Any

_CORE_EXCLUDED_CATEGORIES = {
    "additional_info",
    "formatting",
    "medical_info",
    "time",
    "title",
}


def _span_key(span: dict[str, Any]) -> tuple[int, int, str]:
    return int(span["begin"]), int(span["end"]), str(span["label"])


def _core_segments(row: dict[str, Any], gold_spans: list[dict[str, Any]]):
    """Yield ``(gold_label, category, positions)`` for core-PII segments."""
    has_nested_contract = any("subannotations" in span for span in gold_spans)
    if has_nested_contract:
        for span in gold_spans:
            label = str(span.get("label", "Unassigned"))
            for item in span.get("subannotations") or []:
                category = (
                    str(item.get("category", "uncategorized")).strip()
                    or "uncategorized"
                )
                if category.lower() in _CORE_EXCLUDED_CATEGORIES:
                    continue
                yield label, category, set(range(int(item["begin"]), int(item["end"])))
        return
    for item in row.get("subannotations") or []:
        category = str(item.get("category", "uncategorized")).strip() or "uncategorized"
        if category.lower() in _CORE_EXCLUDED_CATEGORIES:
            continue
        yield "Unassigned", category, set(range(int(item["begin"]), int(item["end"])))


def _recall_rows(counts: dict[str, list[int]], key: str) -> list[dict[str, Any]]:
    rows = []
    for name, (matched, total) in counts.items():
        rows.append(
            {
                key: name,
                "matched_core_pii_chars": matched,
                "total_core_pii_chars": total,
                "core_pii_recall": matched / total if total else None,
            }
        )
    return sorted(
        rows, key=lambda row: (-row["total_core_pii_chars"], str(row[key]).casefold())
    )


def score_documents(
    gold_rows: list[dict], predicted_rows: list[dict]
) -> dict[str, Any]:
    gold = {str(row.get("document_id") or row.get("doc_id")): row for row in gold_rows}
    predicted = {
        str(row.get("document_id") or row.get("doc_id")): row for row in predicted_rows
    }
    exact_tp = exact_gold = exact_pred = 0
    gold_chars = predicted_chars = matched_chars = 0
    core_gold = core_matched = 0
    non_pii_chars = non_pii_redacted_chars = 0
    recall_by_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    recall_by_subannotation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    non_pii_by_prediction: dict[str, int] = defaultdict(int)
    label_confusion: dict[tuple[str, str], int] = defaultdict(int)
    exact_label_confusion: dict[tuple[str, str], int] = defaultdict(int)

    for doc_id, row in gold.items():
        gold_spans = row.get("spans") or []
        predicted_row = predicted.get(doc_id) or {}
        pred_spans = predicted_row.get("spans") or []
        gold_keys = {_span_key(span) for span in gold_spans}
        pred_keys = {_span_key(span) for span in pred_spans}
        exact_tp += len(gold_keys & pred_keys)
        exact_gold += len(gold_keys)
        exact_pred += len(pred_keys)
        predictions_by_boundary: dict[tuple[int, int], list[str]] = defaultdict(list)
        for span in pred_spans:
            predictions_by_boundary[(int(span["begin"]), int(span["end"]))].append(
                str(span["label"])
            )
        for span in gold_spans:
            boundary = (int(span["begin"]), int(span["end"]))
            for prediction_label in predictions_by_boundary.get(boundary, []):
                exact_label_confusion[(str(span["label"]), prediction_label)] += 1

        by_label_gold: dict[str, set[int]] = defaultdict(set)
        by_label_pred: dict[str, set[int]] = defaultdict(set)
        for span in gold_spans:
            by_label_gold[str(span["label"])].update(
                range(int(span["begin"]), int(span["end"]))
            )
        for span in pred_spans:
            by_label_pred[str(span["label"])].update(
                range(int(span["begin"]), int(span["end"]))
            )
        for label in set(by_label_gold) | set(by_label_pred):
            gold_positions = by_label_gold.get(label, set())
            predicted_positions = by_label_pred.get(label, set())
            gold_chars += len(gold_positions)
            predicted_chars += len(predicted_positions)
            matched_chars += len(gold_positions & predicted_positions)
        predicted_redaction_chars = (
            set().union(*by_label_pred.values()) if by_label_pred else set()
        )
        gold_redaction_chars = (
            set().union(*by_label_gold.values()) if by_label_gold else set()
        )
        document_length = len(str(row.get("text", "")))
        if document_length == 0:
            document_length = max(
                [0]
                + [int(span["end"]) for span in gold_spans]
                + [int(span["end"]) for span in pred_spans]
            )
        document_non_pii = set(range(document_length)) - gold_redaction_chars
        non_pii_chars += len(document_non_pii)
        non_pii_redacted_chars += len(predicted_redaction_chars & document_non_pii)
        for label, positions in by_label_pred.items():
            non_pii_by_prediction[label] += len(positions & document_non_pii)
        for gold_label, gold_positions in by_label_gold.items():
            for prediction_label, prediction_positions in by_label_pred.items():
                overlap = len(gold_positions & prediction_positions)
                if overlap:
                    label_confusion[(gold_label, prediction_label)] += overlap

        core_chars: set[int] = set()
        label_segments: dict[str, set[int]] = defaultdict(set)
        category_segments: dict[str, set[int]] = defaultdict(set)
        for gold_label, category, positions in _core_segments(row, gold_spans):
            core_chars.update(positions)
            label_segments[gold_label].update(positions)
            category_segments[category].update(positions)
        core_gold += len(core_chars)
        core_matched += len(core_chars & predicted_redaction_chars)
        for label, positions in label_segments.items():
            recall_by_label[label][0] += len(positions & predicted_redaction_chars)
            recall_by_label[label][1] += len(positions)
        for category, positions in category_segments.items():
            recall_by_subannotation[category][0] += len(
                positions & predicted_redaction_chars
            )
            recall_by_subannotation[category][1] += len(positions)

    precision = exact_tp / exact_pred if exact_pred else 1.0
    recall = exact_tp / exact_gold if exact_gold else 1.0
    return {
        "documents": len(gold),
        "exact_true_positive": exact_tp,
        "exact_precision": precision,
        "exact_recall": recall,
        "exact_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "character_precision": matched_chars / predicted_chars
        if predicted_chars
        else 1.0,
        "character_recall": matched_chars / gold_chars if gold_chars else 1.0,
        "core_pii_recall": core_matched / core_gold if core_gold else 1.0,
        "non_pii_redacted_chars": non_pii_redacted_chars,
        "non_pii_characters": non_pii_chars,
        "non_pii_redaction_rate": non_pii_redacted_chars / non_pii_chars
        if non_pii_chars
        else 0.0,
        "details": {
            "recall_by_gold_label": _recall_rows(recall_by_label, "gold_label"),
            "recall_by_subannotation_category": _recall_rows(
                recall_by_subannotation, "subannotation_category"
            ),
            "non_pii_redaction_by_predicted_label": [
                {"prediction_label": label, "non_pii_redacted_chars": count}
                for label, count in sorted(
                    non_pii_by_prediction.items(),
                    key=lambda item: (-item[1], item[0].casefold()),
                )
            ],
            "label_confusion_chars": [
                {
                    "gold_label": gold_label,
                    "prediction_label": prediction_label,
                    "chars": count,
                }
                for (gold_label, prediction_label), count in sorted(
                    label_confusion.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "exact_label_confusion": [
                {
                    "gold_label": gold_label,
                    "prediction_label": prediction_label,
                    "spans": count,
                }
                for (gold_label, prediction_label), count in sorted(
                    exact_label_confusion.items(), key=lambda item: (-item[1], item[0])
                )
            ],
        },
    }
