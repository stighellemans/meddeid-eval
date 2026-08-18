"""Shared, optional plotting conventions for MedDeID packages.

Matplotlib is imported lazily so metric-only installations stay lightweight.
Callers should use :func:`plotting_style` rather than modifying global rcParams.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

MODEL_PALETTE = (
    "#E67924",  # orange
    "#2C7593",  # blue
    "#009E73",  # green
    "#CC79A7",  # purple
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#6B7280",  # grey
)

TYPE_COLORS = {
    "human": "#111111",
    "rule": "#009E73",
    "neural": "#2C7593",
    "generative": "#E67924",
    "unknown": "#6B7280",
}


def display_label(value: str) -> str:
    """Turn a stable machine identifier into a readable fallback label."""
    return str(value).replace("_", " ").replace("-", " ").strip()


def model_colors(model_ids: Iterable[str]) -> dict[str, str]:
    """Return deterministic, colour-blind-friendly colours in input order."""
    return {
        model_id: MODEL_PALETTE[index % len(MODEL_PALETTE)]
        for index, model_id in enumerate(dict.fromkeys(model_ids))
    }


@contextmanager
def plotting_style() -> Iterator[None]:
    """Apply the manuscript-tested visual defaults without leaking global state."""
    import matplotlib as mpl

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.9,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.facecolor": "white",
        }
    ):
        yield


def clean_axes(ax, *, grid_axis: str | None = None) -> None:
    """Use the restrained axis treatment shared by the manuscript figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#DCDCDC", linewidth=0.8)
        ax.set_axisbelow(True)


def save_figure(
    fig,
    out_dir: str | Path,
    stem: str,
    *,
    formats: Iterable[str] = ("png", "pdf"),
    dpi: int = 300,
) -> list[Path]:
    """Save a figure in requested raster/vector formats and close it."""
    import matplotlib.pyplot as plt

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    normalized = list(dict.fromkeys(str(fmt).lower().lstrip(".") for fmt in formats))
    unsupported = sorted(set(normalized) - {"png", "pdf", "svg"})
    if unsupported:
        raise ValueError(f"unsupported plot format(s): {unsupported}")
    for fmt in normalized:
        path = destination / f"{stem}.{fmt}"
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths
