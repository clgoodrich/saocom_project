"""High-level orchestration entry point for the SAOCOM validation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import io_utils, viz


@dataclass(slots=True)
class AnalysisConfig:
    """Parameters describing input sources and destinations for analysis outputs."""

    processed_data: Path = io_utils.PROCESSED_DATA_DIR
    figures_dir: Path = io_utils.OUTPUTS_DIR / "figures"


def run(config: AnalysisConfig | None = None) -> None:
    """Run the full analysis pipeline and persist summary outputs."""

    config = config or AnalysisConfig()
    io_utils.ensure_directories((config.processed_data, config.figures_dir))

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(config.processed_data.glob("*.csv")):
        frames.append(io_utils.load_csv(csv_path))

    if not frames:
        raise FileNotFoundError(
            "No processed CSV files were found. Run the cleaning pipeline or place files in data/processed."
        )

    combined = pd.concat(frames, ignore_index=True)
    summary = combined.describe(include="all", datetime_is_numeric=True)
    summary_destination = config.processed_data / "summary_statistics.csv"
    io_utils.save_dataframe(summary, summary_destination)

    viz.plot_summary_statistics(summary, destination=config.figures_dir / "summary_statistics.png")


if __name__ == "__main__":
    run()
