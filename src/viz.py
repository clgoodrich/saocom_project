"""Visualization helpers for generating consistent project figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_STYLE = {
    "figure.figsize": (10, 6),
    "axes.titlesize": "large",
    "axes.labelsize": "medium",
}


def _apply_style(overrides: dict[str, object] | None = None) -> None:
    style = DEFAULT_STYLE.copy()
    if overrides:
        style.update(overrides)
    plt.rcParams.update(style)


def plot_summary_statistics(summary: pd.DataFrame, *, destination: Path, style: dict[str, object] | None = None) -> None:
    """Render a bar chart of summary statistics."""

    _apply_style(style)
    ax = summary.transpose().plot(kind="bar")
    ax.set_title("Summary statistics by feature")
    ax.set_ylabel("Value")
    plt.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=300)
    plt.close()


def save_multiple(figures: Iterable[plt.Figure], directory: Path, *, prefix: str = "figure") -> list[Path]:
    """Persist multiple Matplotlib figures with a shared naming convention."""

    directory.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for index, figure in enumerate(figures, start=1):
        path = directory / f"{prefix}_{index:02d}.png"
        figure.savefig(path, dpi=300)
        saved_paths.append(path)
        plt.close(figure)
    return saved_paths
