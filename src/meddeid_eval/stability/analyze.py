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

CSV_FIELDS = ["model", "level", "kind", "role", "dimension", "dimension_value",
              "total", "full", "partial", "missed", "full_recall", "partial_recall",
              "full_plus_partial_recall"]


def _pool_by_dimension(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, int]]:
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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plots are optional
        return []

    made: list[str] = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models = list(per_model)

    def bars(dimension: str, title: str, fname: str) -> None:
        # dimension_value -> {model: full_plus_partial_recall}
        values: dict[str, dict[str, float]] = defaultdict(dict)
        for model, agg in per_model.items():
            for r in agg["category"]["rows"]:
                if r["dimension"] == dimension:
                    values[str(r["dimension_value"])][model] = r["full_plus_partial_recall"]
        if not values:
            return
        labels = sorted(values)
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), 4.2))
        width = 0.8 / max(1, len(models))
        for mi, model in enumerate(models):
            ys = [values[v].get(model, 0.0) for v in labels]
            xs = [i + mi * width for i in range(len(labels))]
            ax.bar(xs, ys, width=width, label=model)
        ax.set_xticks([i + width * (len(models) - 1) / 2 for i in range(len(labels))])
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("recall (full+partial)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        p = plots_dir / fname
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(str(p))

    def value_shift_line() -> None:
        fig, ax = plt.subplots(figsize=(9, 4))
        drew = False
        for model, agg in per_model.items():
            pts = sorted((int(r["dimension_value"]), r["full_plus_partial_recall"])
                         for r in agg["category"]["rows"] if r["dimension"] == "date_value_shift")
            if pts:
                ax.plot([x for x, _ in pts], [y for _, y in pts], label=model, linewidth=1.4)
                drew = True
        if not drew:
            plt.close(fig)
            return
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("shifted year")
        ax.set_ylabel("recall (full+partial)")
        ax.set_title("Date recall by shifted year")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        p = plots_dir / "date_value_shift.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        made.append(str(p))

    bars("capitalization", "Name recall by capitalization", "name_capitalization.png")
    bars("format", "Name recall by format", "name_format.png")
    bars("date_format", "Date recall by format", "date_format.png")
    value_shift_line()
    return made


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _fmt_delta(stats: dict[str, Any]) -> str:
    md = stats.get("mean_degradation")
    if md is None:
        return "n/a"
    p = stats.get("permutation_p_one_sided_degradation")
    star = " *" if (p is not None and p < 0.05 and md > 0) else ""
    n_pairs = stats.get("n_pairs")
    suffix = f" (n={n_pairs})" if n_pairs is not None else ""
    return f"{-100 * md:+.1f}pp{star}{suffix}"


def _report_md(cfg: StabilityConfig, per_model: dict[str, dict[str, Any]], plots: list[str]) -> str:
    lines = ["# DEID name & date stability report", ""]
    lines.append(f"- dataset: `{cfg.dataset}`")
    lines.append(f"- models: {', '.join(per_model)}")
    lines.append(f"- seed: {cfg.seed}")
    lines.append("")
    lines.append("Recall = fraction of perturbed target spans still detected (full+partial) as the "
                 "correct **category** (Name / Date). Δ vs baseline is percentage-points change from the "
                 "unperturbed span; `*` marks a one-sided permutation p < 0.05 (significant degradation).")
    lines.append("")
    for model, agg in per_model.items():
        cat = agg["category"]
        lines.append(f"## {model}")
        lines.append(f"- baseline recall: {_fmt_pct(cat['baseline_recall'])} over {cat['n_targets']} targets")
        lines.append("")
        pooled = _pool_by_dimension(cat["rows"])
        lines.append("n = perturbed samples scored; n_pairs = distinct target spans "
                     "paired with baseline for the degradation test.")
        lines.append("")
        lines.append("| dimension | role | recall | n | Δ vs baseline (n_pairs) |")
        lines.append("|---|---|---|---|---|")
        for (kind, role, dim), c in sorted(pooled.items()):
            if dim == "baseline":
                continue
            rec = _recalls(c)["full_plus_partial_recall"]
            deg = cat["degradation"].get(dim, {}).get(role, {})
            lines.append(f"| {dim} | {role or '-'} | {_fmt_pct(rec)} | {c['total']} | {_fmt_delta(deg)} |")
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
                lines.append(f"| {r['role'] or '-'} | {r['dimension_value']} | {_fmt_pct(r['full_plus_partial_recall'])} | {r['total']} |")
            lines.append("")
    if plots:
        lines.append("## plots")
        for p in plots:
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
        raise FileNotFoundError(f"no prediction files found under {pred_dir} — run `infer` first")

    _write_csv(cfg.output_dir / "recall_by_dimension.csv", all_csv_rows)
    write_json(cfg.output_dir / "stability_analysis.json",
               {mid: {"category": a["category"], "label": a["label"]} for mid, a in per_model.items()})
    plots = _try_plots(cfg.output_dir, per_model)
    report = _report_md(cfg, per_model, plots)
    (cfg.output_dir / "stability_report.md").write_text(report, encoding="utf-8")
    return {"models": list(per_model), "report": str(cfg.output_dir / "stability_report.md"),
            "csv": str(cfg.output_dir / "recall_by_dimension.csv"), "plots": plots}
