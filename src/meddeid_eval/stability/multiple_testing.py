"""Cross-benchmark multiple-testing control for stability degradation."""

from __future__ import annotations

import copy
import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import read_json, write_json

PRESPECIFIED_SCOPES = ("uza", "synthetic", "primary-care")
FDR_ALPHA = 0.05
MIN_PAIRS = 5
MIN_NOTES = 5
P_VALUE_FIELD = "permutation_p_one_sided_degradation"
Q_VALUE_FIELD = "permutation_q_bh_one_sided_degradation"
SIGNIFICANCE_FIELD = "significant_degradation_bh"


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in input order."""
    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Benjamini-Hochberg inputs must be finite values in [0, 1]")
    if not values:
        return []
    count = len(values)
    order = sorted(range(count), key=lambda index: (values[index], index))
    adjusted = [1.0] * count
    running_minimum = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running_minimum = min(running_minimum, values[original_index] * count / rank)
        adjusted[original_index] = min(1.0, running_minimum)
    return adjusted


def _validate_scopes(analyses: Mapping[str, Any], scopes: Sequence[str]) -> None:
    expected = list(scopes)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("prespecified scopes must be non-empty and unique")
    missing = [scope for scope in expected if scope not in analyses]
    unexpected = sorted(set(analyses) - set(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(
            "BH correction requires exactly all prespecified scopes; "
            + "; ".join(details)
        )


def _eligible(cell: Mapping[str, Any], min_pairs: int, min_notes: int) -> bool:
    p_value = cell.get(P_VALUE_FIELD)
    if p_value is None:
        return False
    n_pairs = cell.get("n_pairs")
    n_notes = cell.get("n_clusters")
    if n_pairs is None or n_notes is None:
        raise ValueError(
            "degradation cell has a raw permutation p-value but lacks n_pairs or "
            "n_clusters; rerun `meddeid-eval stability analyze` with note-cluster inference"
        )
    p_value = float(p_value)
    if not 0.0 <= p_value <= 1.0:
        raise ValueError(f"{P_VALUE_FIELD} must be a finite value in [0, 1]")
    return int(n_pairs) >= min_pairs and int(n_notes) >= min_notes


def apply_cross_scope_bh(
    analyses: Mapping[str, dict[str, Any]],
    *,
    prespecified_scopes: Sequence[str] = PRESPECIFIED_SCOPES,
    alpha: float = FDR_ALPHA,
    min_pairs: int = MIN_PAIRS,
    min_notes: int = MIN_NOTES,
) -> dict[str, Any]:
    """Attach one complete cross-scope BH family per model, in place."""
    _validate_scopes(analyses, prespecified_scopes)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    models = sorted({model for analysis in analyses.values() for model in analysis})
    manifest: dict[str, Any] = {
        "method": "Benjamini-Hochberg",
        "alpha": alpha,
        "p_value": "one-sided note-cluster sign-flip permutation",
        "family_definition": (
            "All eligible category-level degradation cells across every prespecified "
            "benchmark scope, corrected separately per model."
        ),
        "prespecified_scopes": list(prespecified_scopes),
        "eligibility": {"min_pairs": min_pairs, "min_contributing_notes": min_notes},
        "models": {},
    }
    for model in models:
        cells: list[tuple[str, str, str, dict[str, Any]]] = []
        for scope in prespecified_scopes:
            category = analyses[scope].get(model, {}).get("category", {})
            for dimension, roles in category.get("degradation", {}).items():
                for role, cell in roles.items():
                    cell[Q_VALUE_FIELD] = None
                    cell[SIGNIFICANCE_FIELD] = False
                    if _eligible(cell, min_pairs, min_notes):
                        cells.append((scope, dimension, role, cell))
        q_values = benjamini_hochberg([cell[P_VALUE_FIELD] for _, _, _, cell in cells])
        records = []
        for (scope, dimension, role, cell), q_value in zip(cells, q_values):
            degradation = cell.get("mean_degradation")
            significant = bool(
                degradation is not None and float(degradation) > 0 and q_value < alpha
            )
            cell[Q_VALUE_FIELD] = q_value
            cell[SIGNIFICANCE_FIELD] = significant
            records.append(
                {
                    "scope": scope,
                    "dimension": dimension,
                    "role": role,
                    "n_pairs": cell.get("n_pairs"),
                    "n_clusters": cell.get("n_clusters"),
                    "mean_degradation": degradation,
                    P_VALUE_FIELD: cell[P_VALUE_FIELD],
                    Q_VALUE_FIELD: q_value,
                    SIGNIFICANCE_FIELD: significant,
                }
            )
        manifest["models"][model] = {
            "family_size": len(records),
            "n_significant": sum(record[SIGNIFICANCE_FIELD] for record in records),
            "cells": records,
        }
    return manifest


def _write_cells_csv(path: Path, manifest: Mapping[str, Any]) -> None:
    fields = [
        "model",
        "scope",
        "dimension",
        "role",
        "n_pairs",
        "n_clusters",
        "mean_degradation",
        P_VALUE_FIELD,
        Q_VALUE_FIELD,
        SIGNIFICANCE_FIELD,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model, result in manifest["models"].items():
            for cell in result["cells"]:
                writer.writerow({"model": model, **cell})


def run_adjust(
    scope_files: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    prespecified_scopes: Sequence[str] = PRESPECIFIED_SCOPES,
) -> dict[str, Any]:
    """Write adjusted copies and an audit manifest without mutating raw analyses."""
    _validate_scopes(scope_files, prespecified_scopes)
    analyses = {
        scope: copy.deepcopy(read_json(scope_files[scope]))
        for scope in prespecified_scopes
    }
    manifest = apply_cross_scope_bh(analyses, prespecified_scopes=prespecified_scopes)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    adjusted_files = {}
    for scope in prespecified_scopes:
        path = destination / f"{scope}.stability_analysis.adjusted.json"
        write_json(path, analyses[scope])
        adjusted_files[scope] = str(path)
    manifest_path = destination / "stability_multiple_testing.json"
    csv_path = destination / "stability_multiple_testing.csv"
    write_json(manifest_path, manifest)
    _write_cells_csv(csv_path, manifest)
    return {
        "manifest": str(manifest_path),
        "csv": str(csv_path),
        "adjusted_analyses": adjusted_files,
        "models": {
            model: {
                "family_size": result["family_size"],
                "n_significant": result["n_significant"],
            }
            for model, result in manifest["models"].items()
        },
    }
