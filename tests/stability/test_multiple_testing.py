from __future__ import annotations

import json

import pytest

from meddeid_eval.stability.multiple_testing import (
    MIN_NOTES,
    P_VALUE_FIELD,
    PRESPECIFIED_SCOPES,
    Q_VALUE_FIELD,
    SIGNIFICANCE_FIELD,
    apply_cross_scope_bh,
    benjamini_hochberg,
    run_adjust,
)


def _analysis(
    p_by_model: dict[str, float], *, n_pairs: int = 10, n_clusters: int = 8
) -> dict:
    return {
        model: {
            "category": {
                "degradation": {
                    "format": {
                        "patient": {
                            "n_pairs": n_pairs,
                            "n_clusters": n_clusters,
                            "mean_degradation": 0.1,
                            P_VALUE_FIELD: p_value,
                        }
                    }
                }
            },
            "label": {
                "degradation": {
                    "format": {
                        "patient": {
                            "n_pairs": 100,
                            "n_clusters": 100,
                            "mean_degradation": 0.2,
                            P_VALUE_FIELD: 0.000001,
                        }
                    }
                }
            },
        }
        for model, p_value in p_by_model.items()
    }


def test_benjamini_hochberg_preserves_order() -> None:
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        benjamini_hochberg([1.1])


def test_cross_scope_bh_uses_a_separate_complete_family_per_model() -> None:
    analyses = {
        "uza": _analysis({"uza": 0.01, "synthetic": 0.001}),
        "synthetic": _analysis({"uza": 0.04, "synthetic": 0.5}),
        "primary-care": _analysis({"uza": 0.03, "synthetic": 0.9}),
    }
    manifest = apply_cross_scope_bh(analyses)
    assert [
        analyses[scope]["uza"]["category"]["degradation"]["format"]["patient"][
            Q_VALUE_FIELD
        ]
        for scope in PRESPECIFIED_SCOPES
    ] == pytest.approx([0.03, 0.04, 0.04])
    assert manifest["models"]["synthetic"]["family_size"] == 3
    assert (
        Q_VALUE_FIELD
        not in analyses["uza"]["uza"]["label"]["degradation"]["format"]["patient"]
    )


def test_underpowered_cell_is_excluded_and_stale_result_cleared() -> None:
    analyses = {scope: _analysis({"model": 0.01}) for scope in PRESPECIFIED_SCOPES}
    cell = analyses["synthetic"]["model"]["category"]["degradation"]["format"][
        "patient"
    ]
    cell["n_clusters"] = MIN_NOTES - 1
    cell[Q_VALUE_FIELD] = 0.001
    cell[SIGNIFICANCE_FIELD] = True
    manifest = apply_cross_scope_bh(analyses)
    assert cell[Q_VALUE_FIELD] is None
    assert cell[SIGNIFICANCE_FIELD] is False
    assert manifest["models"]["model"]["family_size"] == 2


def test_run_adjust_writes_copies_without_overwriting_sources(tmp_path) -> None:
    sources = {}
    original = {}
    for scope in PRESPECIFIED_SCOPES:
        path = tmp_path / scope / "stability_analysis.json"
        path.parent.mkdir()
        path.write_text(json.dumps(_analysis({"model": 0.01})), encoding="utf-8")
        sources[scope] = path
        original[scope] = path.read_text(encoding="utf-8")
    result = run_adjust(sources, tmp_path / "adjusted")
    assert all(
        path.read_text(encoding="utf-8") == original[scope]
        for scope, path in sources.items()
    )
    assert result["models"]["model"] == {"family_size": 3, "n_significant": 3}
