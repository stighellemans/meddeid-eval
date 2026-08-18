from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from meddeid_eval.stability.plots import grouped_bar_data, render_stability_plots


def _aggregate(rows, pooled_rows):
    return {"category": {"rows": rows, "pooled_rows": pooled_rows, "degradation": {}}}


def test_grouped_bar_data_pools_roles_and_preserves_missing_values() -> None:
    per_model = {
        "model-a": _aggregate(
            [
                {
                    "kind": "name",
                    "role": "patient",
                    "dimension": "capitalization",
                    "dimension_value": "lower",
                    "total": 2,
                    "full": 1,
                    "partial": 0,
                },
                {
                    "kind": "name",
                    "role": "caregiver",
                    "dimension": "capitalization",
                    "dimension_value": "lower",
                    "total": 2,
                    "full": 2,
                    "partial": 0,
                },
            ],
            [
                {
                    "kind": "name",
                    "dimension": "capitalization",
                    "dimension_value": "lower",
                    "n_clusters": 3,
                    "bootstrap_ci95": [0.5, 1.0],
                }
            ],
        ),
        "model-b": _aggregate([], []),
    }

    values, data = grouped_bar_data(per_model, "capitalization")

    assert values == ["lower"]
    assert data["model-a"]["lower"]["recall"] == 0.75
    assert data["model-a"]["lower"]["total"] == 4
    assert data["model-b"]["lower"]["recall"] is None


def test_render_stability_plots_writes_vector_and_raster_outputs(tmp_path) -> None:
    rows = [
        {
            "kind": "name",
            "role": "patient",
            "dimension": "baseline",
            "dimension_value": "original",
            "total": 10,
            "full": 9,
            "partial": 0,
        },
        {
            "kind": "name",
            "role": "patient",
            "dimension": "capitalization",
            "dimension_value": "lower",
            "total": 10,
            "full": 8,
            "partial": 0,
        },
    ]
    pooled = [
        {
            "kind": "name",
            "dimension": "capitalization",
            "dimension_value": "lower",
            "n_clusters": 8,
            "bootstrap_ci95": [0.65, 0.95],
        }
    ]
    paths = render_stability_plots(tmp_path, {"model-a": _aggregate(rows, pooled)})
    assert {path.name for path in paths} == {
        "name_capitalization.png",
        "name_capitalization.pdf",
    }
    assert all(path.stat().st_size > 1_000 for path in paths)
