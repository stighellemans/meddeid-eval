"""Stability metrics.

Recovery of a perturbed **target** span is classified full / partial / missed
against the model's predictions. Matching is at the **category** level by default
(is the name still caught as a Name? the date as a Date?), with an exact
**label** level available as a stricter view. ``paired_degradation_stats`` (with
bootstrap CI + sign-flip permutation p) is lifted from the robustness job.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .spans import label_key


def category_key(label: str) -> str:
    """Category part of a ``Category:Subtype`` label, normalised."""
    return label_key(str(label).split(":", 1)[0])


def _pred_category(span: dict[str, Any]) -> str:
    cat = span.get("category")
    if cat:
        return label_key(str(cat))
    return category_key(str(span.get("label", "")))


def _pred_label_key(span: dict[str, Any]) -> str:
    return label_key(str(span.get("label", "")))


def classify_target(target: dict[str, Any], predicted: list[dict[str, Any]], level: str = "category") -> str:
    """full (exact offsets) / partial (overlap) / missed, among predictions whose
    ``level`` key matches the target's."""
    if level == "label":
        want = label_key(str(target["label"]))
        cands = [p for p in predicted if _pred_label_key(p) == want]
    else:
        want = category_key(str(target["label"]))
        cands = [p for p in predicted if _pred_category(p) == want]
    tb, te = int(target["begin"]), int(target["end"])
    if any(int(p.get("begin", -1)) == tb and int(p.get("end", -1)) == te for p in cands):
        return "full"
    if any(max(tb, int(p.get("begin", -1))) < min(te, int(p.get("end", -1))) for p in cands):
        return "partial"
    return "missed"


def outcome_score(outcome: str, score_type: str) -> float:
    if score_type == "full":
        return 1.0 if outcome == "full" else 0.0
    return 1.0 if outcome in {"full", "partial"} else 0.0


def _empty_counts() -> dict[str, int]:
    return {"total": 0, "full": 0, "partial": 0, "missed": 0}


def _add(counts: dict[str, int], outcome: str) -> None:
    counts["total"] += 1
    counts[outcome] += 1


def _recalls(counts: dict[str, int]) -> dict[str, float]:
    denom = counts["total"] or 1
    return {
        "full_recall": counts["full"] / denom,
        "partial_recall": counts["partial"] / denom,
        "full_plus_partial_recall": (counts["full"] + counts["partial"]) / denom,
    }


def paired_degradation_stats(original: dict[str, float], transformed: dict[str, float],
                             seed: int, iterations: int = 5000) -> dict[str, Any]:
    keys = sorted(set(original) & set(transformed))
    diffs = [float(original[k]) - float(transformed[k]) for k in keys]
    if not diffs:
        return {"n_pairs": 0, "mean_original": None, "mean_transformed": None,
                "mean_degradation": None, "bootstrap_ci95": [None, None],
                "permutation_p_one_sided_degradation": None, "permutation_p_two_sided": None}
    mean_original = sum(original[k] for k in keys) / len(keys)
    mean_transformed = sum(transformed[k] for k in keys) / len(keys)
    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boot = []
    for _ in range(iterations):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boot.append(sum(sample) / len(sample))
    boot.sort()
    lower = boot[int(0.025 * (len(boot) - 1))]
    upper = boot[int(0.975 * (len(boot) - 1))]
    if all(d == 0 for d in diffs):
        p_one = p_two = 1.0
    else:
        perm = []
        for _ in range(iterations):
            signed = [d if rng.random() < 0.5 else -d for d in diffs]
            perm.append(sum(signed) / len(signed))
        p_one = (1 + sum(v >= observed for v in perm)) / (iterations + 1)
        p_two = (1 + sum(abs(v) >= abs(observed) for v in perm)) / (iterations + 1)
    return {"n_pairs": len(keys), "mean_original": mean_original, "mean_transformed": mean_transformed,
            "mean_degradation": observed, "bootstrap_ci95": [lower, upper],
            "permutation_p_one_sided_degradation": p_one, "permutation_p_two_sided": p_two}


def aggregate(variants: list[dict[str, Any]], preds: dict[str, list[dict[str, Any]]],
              seed: int, level: str = "category") -> dict[str, Any]:
    """Group per (kind, role, dimension, dimension_value) recall + degradation vs
    baseline. ``preds`` maps variant_id -> predicted spans (one model)."""
    # counts[(kind, role, dimension, value)] -> counts
    counts: dict[tuple, dict[str, int]] = defaultdict(_empty_counts)
    # baseline (per entity) + per-(dimension, role) per-entity mean scores (for degradation)
    baseline: dict[str, float] = {}
    entity_role: dict[str, str] = {}
    per_dim: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for v in variants:
        t = v["target"]
        outcome = classify_target(t, preds.get(v["variant_id"], []), level=level)
        dim = t["dimension"]
        role = t.get("role", "")
        entity = f"{v['document_id']}::{t['span_index']}"
        _add(counts[(t["kind"], role, dim, t["dimension_value"])], outcome)
        score = outcome_score(outcome, "full_plus_partial")
        if dim == "baseline":
            baseline[entity] = score
            entity_role[entity] = role
        else:
            per_dim[(dim, role)][entity].append(score)

    rows: list[dict[str, Any]] = []
    for (kind, role, dim, value), c in sorted(counts.items()):
        rows.append({"level": level, "kind": kind, "role": role, "dimension": dim,
                     "dimension_value": value, **c, **_recalls(c)})

    # degradation vs baseline, per (dimension, role) so each role's Δ is its own
    degradation: dict[str, dict[str, Any]] = defaultdict(dict)
    for (dim, role), ent_scores in per_dim.items():
        transformed = {e: sum(s) / len(s) for e, s in ent_scores.items() if s}
        base_for_role = {e: b for e, b in baseline.items() if entity_role.get(e) == role}
        degradation[dim][role] = paired_degradation_stats(base_for_role, transformed, seed=seed + len(dim) + len(role))

    baseline_recall = (sum(baseline.values()) / len(baseline)) if baseline else None
    return {"level": level, "rows": rows, "degradation": dict(degradation),
            "baseline_recall": baseline_recall, "n_targets": len(baseline)}
