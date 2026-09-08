"""Comparison plots for one or more ``meddeid-eval score`` artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from meddeid_core.taxonomy import CATEGORIES, ENTITY_LABELS, split_label

from .plotting import (
    clean_axes,
    display_label,
    model_colors,
    plotting_style,
    save_figure,
)

_MISSED_LABEL = "<missed>"
_SPURIOUS_LABEL = "<spurious>"


def _source_name(payload: Mapping[str, Any], index: int) -> str:
    return str(payload.get("run", {}).get("name") or f"System {index + 1}")


def _ordered_labels(labels: Iterable[str]) -> list[str]:
    category_rank = {category: index for index, category in enumerate(CATEGORIES)}
    entity_rank = {label: index for index, label in enumerate(ENTITY_LABELS)}
    return sorted(
        set(labels),
        key=lambda label: (
            category_rank.get(split_label(label)[0], len(category_rank)),
            entity_rank.get(label, len(entity_rank)),
            label.casefold(),
        ),
    )


def _contrast_text(cmap, value: float) -> str:
    red, green, blue = cmap(float(value))[:3]
    channels = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in (red, green, blue)
    ]
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return "white" if luminance < 0.179 else "#111111"


def _confusion_display_label(label: str) -> str:
    if label == _MISSED_LABEL:
        return "Missed"
    if label == _SPURIOUS_LABEL:
        return "Spurious"
    return display_label(label)


def _overview(
    payloads: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import ticker

    colors = model_colors(names)
    specs = [
        ("core_pii_recall", "Core PII covered ↑\nlabel ignored"),
        (
            "exact_label_accuracy_matched",
            "Label accuracy ↑\namong matched spans",
        ),
        ("exact_f1", "Exact spans ↑\nboundary + label"),
        ("non_pii_redaction_rate", "Non-PII redacted ↓\nover-redaction"),
    ]
    fig, axis = plt.subplots(
        figsize=(8.8, max(4.2, 0.38 * len(names) + 3.3)),
        constrained_layout=True,
    )
    y = np.arange(len(specs), dtype=float)
    offsets = (
        np.array([0.0]) if len(names) == 1 else np.linspace(-0.18, 0.18, len(names))
    )
    for source_index, (payload, name) in enumerate(zip(payloads, names)):
        values = [
            float(payload[field]) if payload.get(field) is not None else float("nan")
            for field, _ in specs
        ]
        source_y = y + offsets[source_index]
        for row, (value, row_y) in enumerate(zip(values, source_y)):
            if np.isnan(value):
                axis.text(
                    0.01,
                    row_y,
                    "NA",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#777777",
                )
                continue
            axis.plot(
                [0, value],
                [row_y, row_y],
                color=colors[name],
                linewidth=2.2,
                alpha=0.35,
                solid_capstyle="round",
                zorder=2,
            )
            axis.scatter(
                [value],
                [row_y],
                s=54,
                color=colors[name],
                edgecolor="#222222",
                linewidth=0.7,
                label=name if row == 0 else None,
                zorder=4,
            )
            right_edge = value > 0.91
            axis.annotate(
                f"{100 * value:.1f}%",
                (value, row_y),
                xytext=(-7 if right_edge else 7, 0),
                textcoords="offset points",
                ha="right" if right_edge else "left",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
    axis.set_xlim(-0.025, 1.08)
    axis.set_ylim(-0.55, len(specs) - 0.45)
    axis.set_yticks(y, [label for _, label in specs])
    axis.invert_yaxis()
    axis.set_xlabel("Share of characters or spans")
    axis.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    axis.set_title(
        "Privacy coverage and span accuracy\n"
        "Coverage ignores label; exact spans require the boundary and label to match",
        loc="left",
        pad=12,
    )
    clean_axes(axis, grid_axis="x")
    if len(names) > 1:
        axis.legend(frameon=False, loc="lower right", title="System")
    return save_figure(fig, out_dir, "performance_overview", formats=formats, dpi=dpi)


def _exact_by_label_plot(
    payloads: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors

    by_source = []
    labels: set[str] = set()
    for payload in payloads:
        rows = {
            str(row["label"]): row
            for row in payload.get("details", {}).get("exact_by_label", [])
        }
        by_source.append(rows)
        labels.update(rows)
    if not labels:
        return []
    ordered = _ordered_labels(labels)
    columns = ("exact_precision", "exact_recall", "exact_f1")
    column_names = ("Precision", "Recall", "F1")
    ncols = min(3, len(names))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(5.0, 3.9 * ncols), max(4.0, 0.42 * len(ordered) + 1.9) * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    cmap = colors.LinearSegmentedColormap.from_list(
        "meddeid_exact", ["#CC3311", "#F5F1E6", "#009E73"]
    ).with_extremes(bad="#F0F0F0")
    image = None
    for axis, name, source_rows in zip(axes.flat, names, by_source):
        matrix = np.full((len(ordered), len(columns)), np.nan)
        for row_index, label in enumerate(ordered):
            row = source_rows.get(label, {})
            for column_index, field in enumerate(columns):
                value = row.get(field)
                if value is not None:
                    matrix[row_index, column_index] = float(value)
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=cmap)
        axis.set_title(name, loc="left", pad=8)
        axis.set_xticks(np.arange(len(columns)), column_names)
        axis.set_yticks(
            np.arange(len(ordered)),
            [
                f"{display_label(label)}  (gold={source_rows.get(label, {}).get('gold_spans', 0)}, pred={source_rows.get(label, {}).get('predicted_spans', 0)})"
                for label in ordered
            ],
        )
        axis.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row_index, column_index in zip(*np.where(~np.isnan(matrix))):
            value = float(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                f"{100 * value:.0f}%",
                ha="center",
                va="center",
                fontsize=7.2,
                color=_contrast_text(cmap, value),
            )
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in list(axes.flat)[len(names) :]:
        axis.remove()
    if image is not None:
        colorbar = fig.colorbar(image, ax=list(axes.flat)[: len(names)], shrink=0.7)
        colorbar.set_label("Exact-span score")
    fig.suptitle("Exact-span metrics by primary label", fontweight="bold")
    return save_figure(fig, out_dir, "exact_metrics_by_label", formats=formats, dpi=dpi)


def _subannotation_coverage_matrix(
    payloads: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    by_source = []
    categories: set[str] = set()
    for payload in payloads:
        rows = {
            str(row["subannotation_category"]): row
            for row in payload.get("details", {}).get(
                "recall_by_subannotation_category", []
            )
        }
        by_source.append(rows)
        categories.update(rows)
    if not categories:
        return []
    ordered = sorted(categories, key=str.casefold)
    ncols = min(3, len(names))
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(5.0, 3.7 * ncols), max(4.0, 0.42 * len(ordered) + 1.9) * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, name, source_rows in zip(axes.flat, names, by_source):
        matrix = np.full((len(ordered), 2), np.nan)
        counts = np.zeros((len(ordered), 2), dtype=int)
        for row_index, category in enumerate(ordered):
            row = source_rows.get(category)
            if row is None:
                continue
            matched = int(row["matched_core_pii_chars"])
            total = int(row["total_core_pii_chars"])
            missed = total - matched
            counts[row_index] = (matched, missed)
            if total:
                matrix[row_index] = (matched / total, missed / total)
        image = axis.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Blues")
        axis.set_title(name, loc="left", pad=8)
        axis.set_xticks(np.arange(2), ("Detected", "Missed"))
        axis.set_yticks(
            np.arange(len(ordered)),
            [
                f"{display_label(category)}  (n={source_rows.get(category, {}).get('total_core_pii_chars', 0):,} chars)"
                for category in ordered
            ],
        )
        axis.set_xticks(np.arange(-0.5, 2, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row_index, column_index in zip(*np.where(~np.isnan(matrix))):
            value = float(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                f"{counts[row_index, column_index]:,}\n{100 * value:.0f}%",
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if value > 0.58 else "#111111",
            )
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in list(axes.flat)[len(names) :]:
        axis.remove()
    if image is not None:
        colorbar = fig.colorbar(image, ax=list(axes.flat)[: len(names)], shrink=0.7)
        colorbar.set_label("Share of category characters")
    fig.suptitle(
        "Subannotation coverage (predictions do not classify subcategories)",
        fontweight="bold",
    )
    return save_figure(
        fig, out_dir, "subannotation_coverage_matrix", formats=formats, dpi=dpi
    )


def _recall_heatmap(
    payloads: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    out_dir: Path,
    *,
    detail_key: str,
    row_key: str,
    title: str,
    stem: str,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors, ticker

    by_source = []
    all_rows: set[str] = set()
    denominators: dict[str, int] = {}
    for payload in payloads:
        source_rows = {
            str(row[row_key]): row
            for row in payload.get("details", {}).get(detail_key, [])
        }
        by_source.append(source_rows)
        all_rows.update(source_rows)
        for label, row in source_rows.items():
            denominators[label] = max(
                denominators.get(label, 0), int(row["total_core_pii_chars"])
            )
    if not all_rows:
        return []
    rows = (
        _ordered_labels(all_rows)
        if row_key == "gold_label"
        else sorted(all_rows, key=str.casefold)
    )
    matrix = np.full((len(rows), len(names)), np.nan)
    for column, source_rows in enumerate(by_source):
        for row_index, label in enumerate(rows):
            value = source_rows.get(label, {}).get("core_pii_recall")
            if value is not None:
                matrix[row_index, column] = float(value)
    cmap = colors.LinearSegmentedColormap.from_list(
        "meddeid_recall", ["#CC3311", "#F5F1E6", "#009E73"]
    )
    cmap = cmap.with_extremes(bad="#F0F0F0")
    fig, ax = plt.subplots(
        figsize=(max(6.6, 0.85 * len(names) + 3.4), max(4.2, 0.42 * len(rows) + 1.8)),
        constrained_layout=True,
    )
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(
        np.arange(len(names)), names, rotation=35, ha="right", rotation_mode="anchor"
    )
    ax.set_yticks(
        np.arange(len(rows)),
        [
            f"{display_label(label)}  (n={denominators.get(label, 0):,})"
            for label in rows
        ],
    )
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index, column in zip(*np.where(~np.isnan(matrix))):
        value = matrix[row_index, column]
        ax.text(
            column,
            row_index,
            f"{100 * value:.0f}%",
            ha="center",
            va="center",
            fontsize=7.2,
            color=_contrast_text(cmap, value),
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Core PII recall")
    colorbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title(title, loc="left", pad=10)
    return save_figure(fig, out_dir, stem, formats=formats, dpi=dpi)


def _non_pii_heatmap(
    payloads: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors, ticker

    by_source = []
    labels: set[str] = set()
    for payload in payloads:
        rows = {
            str(row["prediction_label"]): int(row["non_pii_redacted_chars"])
            for row in payload.get("details", {}).get(
                "non_pii_redaction_by_predicted_label", []
            )
        }
        by_source.append(rows)
        labels.update(rows)
    if not labels:
        return []
    ordered = sorted(
        labels,
        key=lambda label: (
            -sum(source.get(label, 0) for source in by_source),
            label.casefold(),
        ),
    )
    matrix = np.array(
        [[source.get(label, 0) for source in by_source] for label in ordered],
        dtype=float,
    )
    maximum = float(matrix.max(initial=0))
    if maximum <= 0:
        return []
    masked = np.ma.masked_equal(matrix, 0)
    norm = colors.LogNorm(vmin=1, vmax=max(2, maximum))
    cmap = plt.get_cmap("YlOrRd").with_extremes(bad="white")
    fig, ax = plt.subplots(
        figsize=(
            max(6.6, 0.85 * len(names) + 3.4),
            max(4.0, 0.42 * len(ordered) + 1.8),
        ),
        constrained_layout=True,
    )
    image = ax.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(
        np.arange(len(names)), names, rotation=35, ha="right", rotation_mode="anchor"
    )
    ax.set_yticks(np.arange(len(ordered)), [display_label(label) for label in ordered])
    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
    ax.grid(which="minor", color="#D9D9D9", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index, column in zip(*np.nonzero(matrix)):
        value = int(matrix[row_index, column])
        ax.text(
            column,
            row_index,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if float(norm(value)) > 0.62 else "#111111",
        )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Non-PII redacted characters (log scale)")
    colorbar.ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_title("Non-PII redactions by predicted label", loc="left", pad=10)
    return save_figure(fig, out_dir, "non_pii_redactions", formats=formats, dpi=dpi)


def _exact_label_confusion_plot(
    payloads: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors

    rows = []
    labels: set[str] = set()
    for payload, name in zip(payloads, names):
        details = payload.get("details", {})
        confusion_rows = details.get("matched_label_confusion") or details.get(
            "exact_label_confusion", []
        )
        for item in confusion_rows:
            gold_label = str(item["gold_label"])
            prediction_label = str(item["prediction_label"])
            count = int(item["spans"])
            if count <= 0:
                continue
            rows.append((name, gold_label, prediction_label, count))
            labels.update((gold_label, prediction_label))
    if not rows:
        return []
    gold_labels = {row[1] for row in rows}
    prediction_labels = {row[2] for row in rows}
    ordinary_labels = (gold_labels | prediction_labels) - {
        _MISSED_LABEL,
        _SPURIOUS_LABEL,
    }
    shared_order = _ordered_labels(ordinary_labels)
    ordered_gold = shared_order + (
        [_SPURIOUS_LABEL] if _SPURIOUS_LABEL in gold_labels else []
    )
    ordered_prediction = shared_order + (
        [_MISSED_LABEL] if _MISSED_LABEL in prediction_labels else []
    )
    gold_index = {label: position for position, label in enumerate(ordered_gold)}
    prediction_index = {
        label: position for position, label in enumerate(ordered_prediction)
    }
    matrices = {
        name: np.zeros((len(ordered_gold), len(ordered_prediction)), dtype=int)
        for name in names
    }
    for name, gold_label, prediction_label, count in rows:
        matrices[name][gold_index[gold_label], prediction_index[prediction_label]] += (
            count
        )
    maximum = max(int(matrix.max(initial=0)) for matrix in matrices.values())
    if maximum <= 0:
        return []
    ncols = min(3, len(names))
    nrows = (len(names) + ncols - 1) // ncols
    side = max(4.2, 0.42 * max(len(ordered_gold), len(ordered_prediction)) + 1.8)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(side * ncols, side * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("Blues").with_extremes(bad="white")
    norm = colors.Normalize(vmin=0, vmax=1)
    image = None
    for axis, name in zip(axes.flat, names):
        matrix = matrices[name]
        row_totals = matrix.sum(axis=1, keepdims=True)
        shares = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        masked = np.ma.masked_equal(shares, 0)
        image = axis.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
        axis.set_title(name, loc="left", pad=8)
        axis.set_xticks(
            np.arange(len(ordered_prediction)),
            [_confusion_display_label(label) for label in ordered_prediction],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_yticks(
            np.arange(len(ordered_gold)),
            [_confusion_display_label(label) for label in ordered_gold],
        )
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Gold label")
        axis.set_xticks(np.arange(-0.5, len(ordered_prediction), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(ordered_gold), 1), minor=True)
        axis.grid(which="minor", color="#D9D9D9", linewidth=0.7)
        axis.tick_params(which="minor", bottom=False, left=False)
        for row_index, column in zip(*np.nonzero(matrix)):
            value = int(matrix[row_index, column])
            share = float(shares[row_index, column])
            axis.text(
                column,
                row_index,
                f"{value:,}\n{100 * share:.0f}%",
                ha="center",
                va="center",
                fontsize=6.8,
                color="white" if share > 0.62 else "#111111",
            )
        for spine in axis.spines.values():
            spine.set_visible(False)
    for axis in list(axes.flat)[len(names) :]:
        axis.remove()
    if image is not None:
        colorbar = fig.colorbar(image, ax=list(axes.flat)[: len(names)], shrink=0.7)
        colorbar.set_label("Share within gold-label outcome")
    fig.suptitle("Matched-span outcomes by primary label", fontweight="bold")
    return save_figure(fig, out_dir, "exact_label_confusion", formats=formats, dpi=dpi)


def _timing_plot(
    payloads: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    out_dir: Path,
    formats: Iterable[str],
    dpi: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from matplotlib.lines import Line2D

    rows = []
    for payload, name in zip(payloads, names):
        run = payload.get("run", {})
        seconds = run.get("seconds")
        if seconds is None or float(seconds) <= 0:
            continue
        rows.append(
            (
                name,
                float(seconds),
                float(payload["core_pii_recall"]),
                str(run.get("device", "unknown")),
            )
        )
    if not rows:
        return []
    devices = list(dict.fromkeys(row[3] for row in rows))
    device_colors = model_colors(devices)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for name, seconds, recall, device in rows:
        ax.scatter(
            seconds,
            recall,
            s=52,
            color=device_colors[device],
            edgecolor="#222222",
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            name,
            (seconds, recall),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.9,
            },
        )
    ax.set_xscale("log")
    ax.set_xlabel("Evaluation time (seconds, log scale)")
    ax.set_ylabel("Core PII recall")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(max(0, min(row[2] for row in rows) - 0.06), 1.02)
    ax.set_title("Accuracy vs. run time", loc="left", pad=10)
    clean_axes(ax, grid_axis="both")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=device_colors[device],
                markeredgecolor="#222222",
                label=display_label(device),
            )
            for device in devices
        ],
        title="Hardware",
        frameon=False,
        loc="lower left",
    )
    return save_figure(fig, out_dir, "accuracy_vs_runtime", formats=formats, dpi=dpi)


def render_comparison_plots(
    payloads: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Render all comparison plots supported by the supplied score artifacts."""
    if not payloads:
        raise ValueError("at least one score artifact is required")
    names = [_source_name(payload, index) for index, payload in enumerate(payloads)]
    if len(set(names)) != len(names):
        raise ValueError("score artifact run names must be unique")
    destination = Path(out_dir)
    paths: list[Path] = []
    with plotting_style():
        paths.extend(_overview(payloads, names, destination, formats, dpi))
        paths.extend(_exact_by_label_plot(payloads, names, destination, formats, dpi))
        paths.extend(
            _recall_heatmap(
                payloads,
                names,
                destination,
                detail_key="recall_by_gold_label",
                row_key="gold_label",
                title="Core PII recall by gold label",
                stem="recall_by_gold_label",
                formats=formats,
                dpi=dpi,
            )
        )
        paths.extend(
            _subannotation_coverage_matrix(payloads, names, destination, formats, dpi)
        )
        paths.extend(_non_pii_heatmap(payloads, names, destination, formats, dpi))
        paths.extend(
            _exact_label_confusion_plot(payloads, names, destination, formats, dpi)
        )
        paths.extend(_timing_plot(payloads, names, destination, formats, dpi))
    return paths
