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


def _subannotations(row: dict[str, Any], gold_spans: list[dict[str, Any]]):
    """Yield canonical nested segments or a legacy top-level segment list."""
    has_nested_contract = any("subannotations" in span for span in gold_spans)
    if not has_nested_contract:
        yield from row.get("subannotations") or []
        return

    for span in gold_spans:
        for item in span.get("subannotations") or []:
            yield item


def score_documents(gold_rows: list[dict], predicted_rows: list[dict]) -> dict[str, float | int]:
    gold = {str(row.get("document_id") or row.get("doc_id")): row for row in gold_rows}
    predicted = {str(row.get("document_id") or row.get("doc_id")): row for row in predicted_rows}
    exact_tp = exact_gold = exact_pred = 0
    gold_chars = predicted_chars = matched_chars = 0
    core_gold = core_matched = 0

    for doc_id, row in gold.items():
        gold_spans = row.get("spans") or []
        predicted_row = predicted.get(doc_id) or {}
        pred_spans = predicted_row.get("spans") or []
        gold_keys = {_span_key(span) for span in gold_spans}
        pred_keys = {_span_key(span) for span in pred_spans}
        exact_tp += len(gold_keys & pred_keys)
        exact_gold += len(gold_keys)
        exact_pred += len(pred_keys)

        by_label_gold: dict[str, set[int]] = defaultdict(set)
        by_label_pred: dict[str, set[int]] = defaultdict(set)
        for span in gold_spans:
            by_label_gold[str(span["label"])].update(range(int(span["begin"]), int(span["end"])))
        for span in pred_spans:
            by_label_pred[str(span["label"])].update(range(int(span["begin"]), int(span["end"])))
        for label in set(by_label_gold) | set(by_label_pred):
            gold_chars += len(by_label_gold[label])
            predicted_chars += len(by_label_pred[label])
            matched_chars += len(by_label_gold[label] & by_label_pred[label])
        predicted_redaction_chars = set().union(*by_label_pred.values()) if by_label_pred else set()
        core_chars: set[int] = set()
        for item in _subannotations(row, gold_spans):
            category = str(item.get("category", "")).strip().lower()
            if category not in _CORE_EXCLUDED_CATEGORIES:
                core_chars.update(range(int(item["begin"]), int(item["end"])))
        core_gold += len(core_chars)
        core_matched += len(core_chars & predicted_redaction_chars)

    precision = exact_tp / exact_pred if exact_pred else 1.0
    recall = exact_tp / exact_gold if exact_gold else 1.0
    return {
        "documents": len(gold),
        "exact_true_positive": exact_tp,
        "exact_precision": precision,
        "exact_recall": recall,
        "exact_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "character_precision": matched_chars / predicted_chars if predicted_chars else 1.0,
        "character_recall": matched_chars / gold_chars if gold_chars else 1.0,
        "core_pii_recall": core_matched / core_gold if core_gold else 1.0,
    }
