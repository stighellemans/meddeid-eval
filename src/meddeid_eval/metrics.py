from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

_CORE_EXCLUDED_CATEGORIES = {
    "additional_info",
    "formatting",
    "medical_info",
    "time",
    "title",
}

MISSED_LABEL = "<missed>"
SPURIOUS_LABEL = "<spurious>"


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


def _record_exact_boundary_confusion(
    gold_spans: list[dict[str, Any]],
    pred_spans: list[dict[str, Any]],
    counts: dict[tuple[str, str], int],
) -> None:
    """Record one-to-one label outcomes for exact span boundaries."""
    gold_by_boundary: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    pred_by_boundary: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for span in gold_spans:
        gold_by_boundary[(int(span["begin"]), int(span["end"]))][
            str(span["label"])
        ] += 1
    for span in pred_spans:
        pred_by_boundary[(int(span["begin"]), int(span["end"]))][
            str(span["label"])
        ] += 1

    for boundary in set(gold_by_boundary) | set(pred_by_boundary):
        remaining_gold = gold_by_boundary[boundary].copy()
        remaining_pred = pred_by_boundary[boundary].copy()
        for label in remaining_gold.keys() & remaining_pred.keys():
            matched = min(remaining_gold[label], remaining_pred[label])
            if matched:
                counts[(label, label)] += matched
                remaining_gold[label] -= matched
                remaining_pred[label] -= matched

        gold_labels = sorted(remaining_gold.elements(), key=str.casefold)
        pred_labels = sorted(remaining_pred.elements(), key=str.casefold)
        paired = min(len(gold_labels), len(pred_labels))
        for index in range(paired):
            counts[(gold_labels[index], pred_labels[index])] += 1
        for label in gold_labels[paired:]:
            counts[(label, MISSED_LABEL)] += 1
        for label in pred_labels[paired:]:
            counts[(SPURIOUS_LABEL, label)] += 1


def _core_positions_for_matching(
    span: dict[str, Any], *, has_nested_contract: bool
) -> set[int]:
    if not has_nested_contract:
        return set(range(int(span["begin"]), int(span["end"])))
    positions: set[int] = set()
    for item in span.get("subannotations") or []:
        category = str(item.get("category", "uncategorized")).strip().lower()
        if category in _CORE_EXCLUDED_CATEGORIES:
            continue
        positions.update(range(int(item["begin"]), int(item["end"])))
    return positions


def _record_overlap_label_confusion(
    gold_spans: list[dict[str, Any]],
    pred_spans: list[dict[str, Any]],
    counts: dict[tuple[str, str], int],
) -> tuple[int, int]:
    """Match spans geometrically and return ``(matched, exact-label-correct)``."""
    has_nested_contract = any("subannotations" in span for span in gold_spans)
    matchable_gold = [
        (
            index,
            span,
            _core_positions_for_matching(
                span, has_nested_contract=has_nested_contract
            ),
        )
        for index, span in enumerate(gold_spans)
    ]
    matchable_gold = [item for item in matchable_gold if item[2]]
    candidates: list[tuple[int, float, float, int, int]] = []
    for gold_index, (_, _, gold_positions) in enumerate(matchable_gold):
        for prediction_index, prediction in enumerate(pred_spans):
            prediction_positions = set(
                range(int(prediction["begin"]), int(prediction["end"]))
            )
            overlap = len(gold_positions & prediction_positions)
            if not overlap:
                continue
            candidates.append(
                (
                    overlap,
                    overlap / len(gold_positions),
                    overlap / len(prediction_positions),
                    gold_index,
                    prediction_index,
                )
            )

    matched_gold: set[int] = set()
    matched_predictions: set[int] = set()
    correct = 0
    for _, _, _, gold_index, prediction_index in sorted(
        candidates,
        key=lambda candidate: (
            -candidate[0],
            -candidate[1],
            -candidate[2],
            candidate[3],
            candidate[4],
        ),
    ):
        if gold_index in matched_gold or prediction_index in matched_predictions:
            continue
        matched_gold.add(gold_index)
        matched_predictions.add(prediction_index)
        gold_label = str(matchable_gold[gold_index][1]["label"])
        prediction_label = str(pred_spans[prediction_index]["label"])
        counts[(gold_label, prediction_label)] += 1
        correct += int(gold_label == prediction_label)

    for gold_index, (_, span, _) in enumerate(matchable_gold):
        if gold_index not in matched_gold:
            counts[(str(span["label"]), MISSED_LABEL)] += 1
    for prediction_index, span in enumerate(pred_spans):
        if prediction_index not in matched_predictions:
            counts[(SPURIOUS_LABEL, str(span["label"]))] += 1
    return len(matched_gold), correct


def _exact_by_label_rows(
    true_positives: Counter[str],
    gold_counts: Counter[str],
    prediction_counts: Counter[str],
) -> list[dict[str, Any]]:
    rows = []
    for label in set(gold_counts) | set(prediction_counts):
        tp = int(true_positives[label])
        gold = int(gold_counts[label])
        predicted = int(prediction_counts[label])
        precision = tp / predicted if predicted else None
        recall = tp / gold if gold else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else 0.0
        )
        rows.append(
            {
                "label": label,
                "exact_true_positive": tp,
                "gold_spans": gold,
                "predicted_spans": predicted,
                "exact_precision": precision,
                "exact_recall": recall,
                "exact_f1": f1,
            }
        )
    return sorted(rows, key=lambda row: str(row["label"]).casefold())


def score_documents(
    gold_rows: list[dict], predicted_rows: list[dict]
) -> dict[str, Any]:
    gold = {str(row.get("document_id") or row.get("doc_id")): row for row in gold_rows}
    predicted = {
        str(row.get("document_id") or row.get("doc_id")): row for row in predicted_rows
    }
    exact_tp = exact_gold = exact_pred = 0
    label_matched_spans = label_correct_spans = 0
    gold_chars = predicted_chars = matched_chars = 0
    core_gold = core_matched = 0
    non_pii_chars = non_pii_redacted_chars = 0
    recall_by_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    recall_by_subannotation: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    non_pii_by_prediction: dict[str, int] = defaultdict(int)
    label_confusion: dict[tuple[str, str], int] = defaultdict(int)
    exact_label_confusion: dict[tuple[str, str], int] = defaultdict(int)
    matched_label_confusion: dict[tuple[str, str], int] = defaultdict(int)
    exact_tp_by_label: Counter[str] = Counter()
    exact_gold_by_label: Counter[str] = Counter()
    exact_pred_by_label: Counter[str] = Counter()

    for doc_id, row in gold.items():
        gold_spans = row.get("spans") or []
        predicted_row = predicted.get(doc_id) or {}
        pred_spans = predicted_row.get("spans") or []
        gold_keys = {_span_key(span) for span in gold_spans}
        pred_keys = {_span_key(span) for span in pred_spans}
        exact_tp += len(gold_keys & pred_keys)
        exact_gold += len(gold_keys)
        exact_pred += len(pred_keys)
        exact_gold_by_label.update(key[2] for key in gold_keys)
        exact_pred_by_label.update(key[2] for key in pred_keys)
        exact_tp_by_label.update(key[2] for key in gold_keys & pred_keys)
        _record_exact_boundary_confusion(gold_spans, pred_spans, exact_label_confusion)
        matched, correct = _record_overlap_label_confusion(
            gold_spans, pred_spans, matched_label_confusion
        )
        label_matched_spans += matched
        label_correct_spans += correct

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
        "exact_gold_spans": exact_gold,
        "exact_predicted_spans": exact_pred,
        "exact_precision": precision,
        "exact_recall": recall,
        "exact_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "label_matched_spans": label_matched_spans,
        "label_correct_spans": label_correct_spans,
        "exact_label_accuracy_matched": (
            label_correct_spans / label_matched_spans
            if label_matched_spans
            else None
        ),
        "matched_label_characters": matched_chars,
        "gold_label_characters": gold_chars,
        "predicted_label_characters": predicted_chars,
        "character_precision": matched_chars / predicted_chars
        if predicted_chars
        else 1.0,
        "character_recall": matched_chars / gold_chars if gold_chars else 1.0,
        "core_pii_matched_characters": core_matched,
        "core_pii_characters": core_gold,
        "core_pii_recall": core_matched / core_gold if core_gold else 1.0,
        "non_pii_redacted_chars": non_pii_redacted_chars,
        "non_pii_characters": non_pii_chars,
        "non_pii_redaction_rate": non_pii_redacted_chars / non_pii_chars
        if non_pii_chars
        else 0.0,
        "details": {
            "exact_by_label": _exact_by_label_rows(
                exact_tp_by_label, exact_gold_by_label, exact_pred_by_label
            ),
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
            "matched_label_confusion": [
                {
                    "gold_label": gold_label,
                    "prediction_label": prediction_label,
                    "spans": count,
                }
                for (gold_label, prediction_label), count in sorted(
                    matched_label_confusion.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
        },
    }
