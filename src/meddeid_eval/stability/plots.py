"""Publication-grade, reusable stability plots.

The renderers consume the in-memory result of ``stability.metrics.aggregate``.
They never reinterpret missing values as zero and pool roles by their underlying
counts before drawing category-level figures.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from meddeid_eval.plotting import (
    clean_axes,
    display_label,
    model_colors,
    plotting_style,
    save_figure,
)

MIN_PAIRS = 5
MIN_NOTES = 5

VALUE_ORDER = {
    "capitalization": ["full", "first_only", "lower", "upper"],
    "format": [
        "full_name",
        "first_only",
        "initials_only",
        "first_initials",
        "title_dr",
    ],
    "date_format": [
        "numeric_slash_short",
        "numeric_slash_long",
        "numeric_dash_long",
        "numeric_dot_long",
        "textual_full",
        "textual_abbr_dot",
        "textual_hyphen",
        "weekday_textual",
    ],
}

VALUE_LABELS = {
    "full": "Title case",
    "first_only": "First-cap only",
    "lower": "lowercase",
    "upper": "UPPERCASE",
    "full_name": "First Last",
    "initials_only": "Initials",
    "first_initials": "First + initials",
    "title_dr": "dr. First Last",
    "numeric_slash_short": "12/5/83",
    "numeric_slash_long": "12/05/1983",
    "numeric_dash_long": "12-05-1983",
    "numeric_dot_long": "12.5.1983",
    "textual_full": "12 mei 1983",
    "textual_abbr_dot": "12 mei. 1983",
    "textual_hyphen": "12-mei-1983",
    "weekday_textual": "do 12 mei 1983",
}

DIMENSION_LABELS = {
    "date_value_shift": "Date - year shift",
    "date_format": "Date - written format",
    "age_format": "Age - phrasing",
    "capitalization": "Name case",
    "format": "Name format",
    "name_source": "Name swap",
}


def _ordered_values(dimension: str, available: Iterable[str]) -> list[str]:
    available_set = {str(value) for value in available}
    preferred = [
        value for value in VALUE_ORDER.get(dimension, []) if value in available_set
    ]
    return preferred + sorted(available_set - set(preferred), key=str.casefold)


def _pooled_value(
    category: Mapping[str, Any], dimension: str, value: str
) -> dict[str, Any]:
    rows = [
        row
        for row in category.get("rows", [])
        if row.get("dimension") == dimension
        and str(row.get("dimension_value")) == value
    ]
    total = sum(int(row.get("total", 0)) for row in rows)
    detected = sum(int(row.get("full", 0)) + int(row.get("partial", 0)) for row in rows)
    pooled = next(
        (
            row
            for row in category.get("pooled_rows", [])
            if row.get("dimension") == dimension
            and str(row.get("dimension_value")) == value
        ),
        None,
    )
    recall = detected / total if total else None
    ci = pooled.get("bootstrap_ci95") if pooled else None
    return {
        "recall": recall,
        "total": total,
        "n_clusters": pooled.get("n_clusters") if pooled else None,
        "ci": ci,
    }


def grouped_bar_data(
    per_model: Mapping[str, Mapping[str, Any]], dimension: str
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    """Prepare pooled chart data without overwriting roles or inventing zeros."""
    values = {
        str(row["dimension_value"])
        for aggregate in per_model.values()
        for row in aggregate["category"].get("rows", [])
        if row.get("dimension") == dimension
    }
    ordered = _ordered_values(dimension, values)
    return ordered, {
        model: {
            value: _pooled_value(aggregate["category"], dimension, value)
            for value in ordered
        }
        for model, aggregate in per_model.items()
    }


def _baseline(category: Mapping[str, Any], kind: str) -> float | None:
    rows = [
        row
        for row in category.get("rows", [])
        if row.get("dimension") == "baseline" and row.get("kind") == kind
    ]
    total = sum(int(row.get("total", 0)) for row in rows)
    detected = sum(int(row.get("full", 0)) + int(row.get("partial", 0)) for row in rows)
    return detected / total if total else None


def _bar_plot(
    per_model: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    *,
    dimension: str,
    kind: str,
    title: str,
    stem: str,
    formats: Iterable[str],
    dpi: int,
    names: Mapping[str, str],
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import ticker
    from matplotlib.lines import Line2D

    values, data = grouped_bar_data(per_model, dimension)
    if not values:
        return []
    models = list(per_model)
    colors = model_colors(models)
    x = np.arange(len(values), dtype=float)
    width = min(0.72 / max(len(models), 1), 0.34)
    figure_width = max(6.2, 1.15 * len(values) + 2.2)
    fig, ax = plt.subplots(figsize=(figure_width, 4.5), constrained_layout=True)

    for model_index, model in enumerate(models):
        offset = (model_index - (len(models) - 1) / 2) * width
        for value_index, value in enumerate(values):
            item = data[model][value]
            recall = item["recall"]
            if recall is None:
                ax.text(
                    x[value_index] + offset,
                    0.025,
                    "NA",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#777777",
                    rotation=90,
                )
                continue
            position = x[value_index] + offset
            ax.bar(
                position,
                recall,
                width=width,
                color=colors[model],
                edgecolor="#333333",
                linewidth=0.6,
                zorder=3,
                label=names[model] if value_index == 0 else None,
            )
            ci = item.get("ci")
            top = recall
            if ci and ci[0] is not None and ci[1] is not None:
                low = max(0.0, recall - float(ci[0]))
                high = max(0.0, float(ci[1]) - recall)
                ax.errorbar(
                    position,
                    recall,
                    yerr=[[low], [high]],
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=1.0,
                    capsize=3,
                    zorder=4,
                )
                top = max(top, float(ci[1]))
            ax.text(
                position,
                min(top + 0.018, 1.035),
                f"{100 * recall:.0f}%\n(n={item['total']:,})",
                ha="center",
                va="bottom",
                fontsize=6.8,
                linespacing=0.95,
            )

    for model in models:
        baseline = _baseline(per_model[model]["category"], kind)
        if baseline is not None:
            ax.axhline(
                baseline,
                color=colors[model],
                linestyle=(0, (4, 2)),
                linewidth=1.1,
                alpha=0.8,
                zorder=2,
            )

    ax.set_xticks(
        x, [VALUE_LABELS.get(value, display_label(value)) for value in values]
    )
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Recall (higher is better)")
    ax.set_title(title, loc="left", pad=10)
    clean_axes(ax, grid_axis="y")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="#777777",
            linestyle=(0, (4, 2)),
            label="Unperturbed baseline",
        )
    )
    labels.append("Unperturbed baseline")
    ax.legend(
        handles, labels, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5)
    )
    return save_figure(fig, out_dir, stem, formats=formats, dpi=dpi)


def _date_shift_plot(
    per_model: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    *,
    formats: Iterable[str],
    dpi: int,
    names: Mapping[str, str],
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import ticker
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    models = list(per_model)
    colors = model_colors(models)
    fig, ax = plt.subplots(figsize=(9.2, 4.5), constrained_layout=True)
    lower_limits: list[float] = []
    drew_interval = False
    for model in models:
        rows = [
            row
            for row in per_model[model]["category"].get("rows", [])
            if row.get("dimension") == "date_value_shift"
        ]
        by_year: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_year.setdefault(int(row["dimension_value"]), []).append(row)
        years: list[int] = []
        recalls: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for year in sorted(by_year):
            item = _pooled_value(
                per_model[model]["category"], "date_value_shift", str(year)
            )
            if item["recall"] is None:
                continue
            years.append(year)
            recalls.append(float(item["recall"]))
            ci = item.get("ci")
            lows.append(float(ci[0]) if ci and ci[0] is not None else np.nan)
            highs.append(float(ci[1]) if ci and ci[1] is not None else np.nan)
        if not years:
            continue
        ax.plot(
            years,
            recalls,
            color=colors[model],
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=names[model],
            zorder=3,
        )
        if not np.isnan(lows).all():
            ax.fill_between(
                years,
                lows,
                highs,
                color=colors[model],
                alpha=0.18,
                linewidth=0,
                zorder=2,
            )
            lower_limits.extend(value for value in lows if not np.isnan(value))
            drew_interval = True
        baseline = _baseline(per_model[model]["category"], "date")
        if baseline is not None:
            ax.axhline(
                baseline,
                color=colors[model],
                linestyle=(0, (4, 2)),
                linewidth=1.1,
                alpha=0.75,
                zorder=2,
            )

    if not ax.lines:
        plt.close(fig)
        return []
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Year written into the date")
    ax.set_ylabel("Recall (higher is better)")
    ax.set_title("Date recall under year shifts", loc="left", pad=10)
    ax.set_ylim(max(0.0, min(lower_limits, default=0.75) - 0.04), 1.01)
    clean_axes(ax, grid_axis="both")
    handles = [
        Line2D([0], [0], color=colors[model], marker="o", label=names[model])
        for model in models
    ]
    if drew_interval:
        handles.append(
            Patch(facecolor="#999999", alpha=0.3, label="95% note-cluster bootstrap CI")
        )
    handles.append(
        Line2D(
            [0],
            [0],
            color="#777777",
            linestyle=(0, (4, 2)),
            label="Unperturbed baseline",
        )
    )
    ax.legend(handles=handles, frameon=False, loc="lower left")
    return save_figure(fig, out_dir, "date_value_shift", formats=formats, dpi=dpi)


def _forest_plot(
    per_model: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    *,
    formats: Iterable[str],
    dpi: int,
    names: Mapping[str, str],
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    preferred = [
        ("date_value_shift", "date"),
        ("date_format", "date"),
        ("age_format", "age"),
        ("capitalization", "patient"),
        ("capitalization", "caregiver"),
        ("format", "patient"),
        ("format", "caregiver"),
        ("name_source", "patient"),
        ("name_source", "caregiver"),
    ]
    available = {
        (dimension, role)
        for aggregate in per_model.values()
        for dimension, roles in aggregate["category"].get("degradation", {}).items()
        for role in roles
    }
    rows = [item for item in preferred if item in available]
    rows.extend(sorted(available - set(rows)))
    if not rows:
        return []
    models = list(per_model)
    colors = model_colors(models)
    y = np.arange(len(rows), dtype=float)
    offset = min(0.16, 0.34 / max(len(models), 1))
    fig, ax = plt.subplots(
        figsize=(8.1, max(4.2, 0.48 * len(rows) + 1.8)), constrained_layout=True
    )
    ax.axvline(0, color="#333333", linewidth=1.0, zorder=2)
    any_points = False
    any_q = False
    for model_index, model in enumerate(models):
        ypos = y + (model_index - (len(models) - 1) / 2) * 2 * offset
        for row_index, (dimension, role) in enumerate(rows):
            stats = (
                per_model[model]["category"]
                .get("degradation", {})
                .get(dimension, {})
                .get(role, {})
            )
            mean = stats.get("mean_degradation")
            ci = stats.get("bootstrap_ci95")
            n_pairs = stats.get("n_pairs")
            n_notes = stats.get("n_clusters")
            if (
                mean is None
                or not ci
                or ci[0] is None
                or n_pairs is None
                or n_notes is None
                or int(n_pairs) < MIN_PAIRS
                or int(n_notes) < MIN_NOTES
            ):
                continue
            delta = -100 * float(mean)
            low, high = -100 * float(ci[1]), -100 * float(ci[0])
            ax.errorbar(
                delta,
                ypos[row_index],
                xerr=[[delta - low], [high - delta]],
                fmt="o",
                color=colors[model],
                ecolor=colors[model],
                elinewidth=1.3,
                capsize=3,
                markersize=5.5,
                markeredgecolor="#333333",
                markeredgewidth=0.5,
                zorder=4,
            )
            q_value = stats.get("permutation_q_bh_one_sided_degradation")
            if q_value is not None:
                any_q = True
                if float(mean) > 0 and float(q_value) < 0.05:
                    ax.text(
                        high + 0.5,
                        ypos[row_index],
                        "*",
                        color=colors[model],
                        va="center",
                    )
            any_points = True
    if not any_points:
        plt.close(fig)
        return []
    ax.set_yticks(
        y,
        [
            f"{DIMENSION_LABELS.get(dimension, display_label(dimension))} - {display_label(role)}"
            for dimension, role in rows
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Recall change vs. unperturbed (percentage points)\nnegative is worse; positive is better"
    )
    ax.set_title("Recall change under controlled perturbation", loc="left", pad=10)
    clean_axes(ax, grid_axis="x")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=colors[model],
            markeredgecolor="#333333",
            label=names[model],
        )
        for model in models
    ]
    if any_q:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="$*$",
                linestyle="",
                color="#333333",
                label="BH-adjusted q < 0.05",
            )
        )
    ax.legend(handles=handles, frameon=False, loc="lower right")
    return save_figure(fig, out_dir, "degradation_forest", formats=formats, dpi=dpi)


def render_stability_plots(
    out_dir: str | Path,
    per_model: Mapping[str, Mapping[str, Any]],
    *,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
    display_names: Mapping[str, str] | None = None,
) -> list[Path]:
    """Render the complete stability plot family and return created paths."""
    destination = Path(out_dir)
    names = {
        model: (display_names or {}).get(model, display_label(model))
        for model in per_model
    }
    paths: list[Path] = []
    with plotting_style():
        paths.extend(
            _bar_plot(
                per_model,
                destination,
                dimension="capitalization",
                kind="name",
                title="Name recall by capitalization",
                stem="name_capitalization",
                formats=formats,
                dpi=dpi,
                names=names,
            )
        )
        paths.extend(
            _bar_plot(
                per_model,
                destination,
                dimension="format",
                kind="name",
                title="Name recall by written format",
                stem="name_format",
                formats=formats,
                dpi=dpi,
                names=names,
            )
        )
        paths.extend(
            _bar_plot(
                per_model,
                destination,
                dimension="date_format",
                kind="date",
                title="Date recall by written format",
                stem="date_format",
                formats=formats,
                dpi=dpi,
                names=names,
            )
        )
        paths.extend(
            _date_shift_plot(
                per_model, destination, formats=formats, dpi=dpi, names=names
            )
        )
        paths.extend(
            _forest_plot(per_model, destination, formats=formats, dpi=dpi, names=names)
        )
    return paths
