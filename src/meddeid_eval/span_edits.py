"""Dependency-free annotation-effort metric used by evaluation and training."""

from __future__ import annotations

from typing import Any


def evaluate_span_edits(
    pred_spans: list[dict[str, Any]],
    gold_spans: list[dict[str, Any]],
    *,
    label_key: str = "label",
) -> dict[str, Any]:
    """Count additions, deletions, and label edits on exact span boundaries."""

    def by_range(spans: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
        result: dict[tuple[int, int], dict[str, Any]] = {}
        for span in spans:
            key = (int(span["begin"]), int(span["end"]))
            if key in result:
                raise ValueError(f"duplicate span boundary {key}")
            result[key] = span
        return result

    predicted = by_range(pred_spans)
    gold = by_range(gold_spans)
    operations: list[dict[str, Any]] = []
    for begin, end in sorted(set(predicted) & set(gold)):
        pred_label = str(predicted[(begin, end)].get(label_key, ""))
        gold_label = str(gold[(begin, end)].get(label_key, ""))
        if pred_label != gold_label:
            operations.append({"op": "Edit", "begin": begin, "end": end, "from": pred_label, "to": gold_label})
    for begin, end in sorted(set(gold) - set(predicted)):
        operations.append({"op": "Addition", "begin": begin, "end": end, "label": gold[(begin, end)].get(label_key, "")})
    for begin, end in sorted(set(predicted) - set(gold)):
        operations.append({"op": "Deletion", "begin": begin, "end": end, "label": predicted[(begin, end)].get(label_key, "")})
    counts = {name: sum(item["op"] == name for item in operations) for name in ("Addition", "Deletion", "Edit")}
    counts["total_ops"] = len(operations)
    return {"counts": counts, "operations": operations}

