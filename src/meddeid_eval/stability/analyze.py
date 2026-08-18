"""Stage C — turn predictions into per-dimension recall, degradation stats, and a
scannable report (+ optional plots).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import StabilityConfig
from .io import read_jsonl, write_json
from .metrics import _empty_counts, _recalls, aggregate

CSV_FIELDS = [
    "model",
    "level",
    "kind",
    "role",
    "dimension",
    "dimension_value",
    "total",
    "full",
    "partial",
    "missed",
    "full_recall",
    "partial_recall",
    "full_plus_partial_recall",
    "n_clusters",
    "bootstrap_ci95_lower",
    "bootstrap_ci95_upper",
]

MIN_PAIRS = 5
MIN_CLUSTERS = 5


def _pool_by_dimension(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, int]]:
    """Sum counts across dimension_values -> keyed by (kind, role, dimension)."""
    pooled: dict[tuple[str, str, str], dict[str, int]] = defaultdict(_empty_counts)
    for r in rows:
        c = pooled[(r["kind"], r["role"], r["dimension"])]
        for k in ("total", "full", "partial", "missed"):
            c[k] += int(r[k])
    return pooled


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _try_plots(out_dir: Path, per_model: dict[str, dict[str, Any]]) -> list[str]:
    try:
        from .plots import render_stability_plots

        paths = render_stability_plots(out_dir / "plots", per_model)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        return []
    return [str(path) for path in paths]


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _fmt_delta(stats: dict[str, Any]) -> str:
    md = stats.get("mean_degradation")
    if md is None:
        return "n/a"
    p = stats.get("permutation_p_one_sided_degradation")
    star = " *" if (p is not None and p < 0.05 and md > 0) else ""
    n_pairs = stats.get("n_pairs")
    n_clusters = stats.get("n_clusters")
    if (n_pairs is not None and n_pairs < MIN_PAIRS) or (
        n_clusters is not None and n_clusters < MIN_CLUSTERS
    ):
        return f"not estimated (pairs={n_pairs}, notes={n_clusters})"
    suffix = f" (pairs={n_pairs}, notes={n_clusters})"
    return f"{-100 * md:+.1f}pp{star}{suffix}"


def _report_md(
    cfg: StabilityConfig, per_model: dict[str, dict[str, Any]], plots: list[str]
) -> str:
    lines = ["# DEID name & date stability report", ""]
    lines.append(f"- dataset: `{cfg.dataset}`")
    lines.append(f"- models: {', '.join(per_model)}")
    lines.append(f"- seed: {cfg.seed}")
    lines.append("")
    lines.append(
        "Recall = fraction of perturbed target spans still detected (full+partial) as the "
        "correct **category** (Name / Date). Δ vs baseline is percentage-points change from the "
        "unperturbed span; `*` marks a nominal one-sided permutation p < 0.05. "
        "Confirmatory significance requires the cross-scope `adjust` step and q < 0.05."
    )
    lines.append("")
    for model, agg in per_model.items():
        cat = agg["category"]
        lines.append(f"## {model}")
        lines.append(
            f"- baseline recall: {_fmt_pct(cat['baseline_recall'])} over {cat['n_targets']} targets"
        )
        lines.append("")
        pooled = _pool_by_dimension(cat["rows"])
        lines.append(
            "n = perturbed samples scored; degradation intervals resample complete notes "
            "while keeping baseline and perturbed outcomes paired. Cells require at least "
            f"{MIN_PAIRS} paired target spans from {MIN_CLUSTERS} contributing notes."
        )
        lines.append("")
        lines.append("| dimension | role | recall | n | Δ vs baseline (pairs, notes) |")
        lines.append("|---|---|---|---|---|")
        for (kind, role, dim), c in sorted(pooled.items()):
            if dim == "baseline":
                continue
            rec = _recalls(c)["full_plus_partial_recall"]
            deg = cat["degradation"].get(dim, {}).get(role, {})
            lines.append(
                f"| {dim} | {role or '-'} | {_fmt_pct(rec)} | {c['total']} | {_fmt_delta(deg)} |"
            )
        lines.append("")
        # per-value breakdown for the two richest name dimensions
        for dim in ("capitalization", "format"):
            vals = [r for r in cat["rows"] if r["dimension"] == dim]
            if not vals:
                continue
            lines.append(f"### {model} — {dim} breakdown")
            lines.append("| role | value | recall | n |")
            lines.append("|---|---|---|---|")
            for r in sorted(vals, key=lambda r: (r["role"], r["dimension_value"])):
                lines.append(
                    f"| {r['role'] or '-'} | {r['dimension_value']} | {_fmt_pct(r['full_plus_partial_recall'])} | {r['total']} |"
                )
            lines.append("")
    if plots:
        lines.append("## plots")
        for p in (path for path in plots if Path(path).suffix.lower() == ".png"):
            rel = Path(p).name
            lines.append(f"- ![{rel}](plots/{rel})")
        lines.append("")
    return "\n".join(lines)


def run_analyze(cfg: StabilityConfig, only: list[str] | None = None) -> dict[str, Any]:
    variants_path = cfg.output_dir / "variants.jsonl"
    if not variants_path.exists():
        raise FileNotFoundError(f"{variants_path} not found — run `expand` first")
    variants = read_jsonl(variants_path)

    pred_dir = cfg.output_dir / "predictions"
    per_model: dict[str, dict[str, Any]] = {}
    all_csv_rows: list[dict[str, Any]] = []
    model_ids = [m.id for m in cfg.models if (not only or m.id in set(only))]
    for mid in model_ids:
        pf = pred_dir / f"{mid}.jsonl"
        if not pf.exists():
            continue
        preds = {r["variant_id"]: r.get("spans", []) for r in read_jsonl(pf)}
        cat = aggregate(variants, preds, cfg.seed, level="category")
        lbl = aggregate(variants, preds, cfg.seed, level="label")
        per_model[mid] = {"category": cat, "label": lbl}
        for level_agg in (cat, lbl):
            for r in level_agg["rows"]:
                all_csv_rows.append({"model": mid, **r})
    if not per_model:
        raise FileNotFoundError(
            f"no prediction files found under {pred_dir} — run `infer` first"
        )

    _write_csv(cfg.output_dir / "recall_by_dimension.csv", all_csv_rows)
    write_json(
        cfg.output_dir / "stability_analysis.json",
        {
            mid: {"category": a["category"], "label": a["label"]}
            for mid, a in per_model.items()
        },
    )
    plots = _try_plots(cfg.output_dir, per_model)
    report = _report_md(cfg, per_model, plots)
    (cfg.output_dir / "stability_report.md").write_text(report, encoding="utf-8")
    return {
        "models": list(per_model),
        "report": str(cfg.output_dir / "stability_report.md"),
        "csv": str(cfg.output_dir / "recall_by_dimension.csv"),
        "plots": plots,
    }
