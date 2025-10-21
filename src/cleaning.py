"""Deterministic preprocessing routines for the SAOCOM project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import io_utils


@dataclass(slots=True)
class CleaningConfig:
    """Configuration controlling cleaning steps for the pipeline."""

    source_paths: io_utils.DatasetPaths = io_utils.DatasetPaths()
    output_suffix: str = "clean"


def _list_csv_files(directory: Path) -> Iterable[Path]:
    return sorted(path for path in directory.glob("*.csv") if path.is_file())


def clean_all(*, config: CleaningConfig) -> None:
    """Example cleaning routine that copies CSV inputs to the processed directory."""

    io_utils.ensure_directories(
        (
            config.source_paths.raw,
            config.source_paths.processed,
        )
    )

    for csv_path in _list_csv_files(config.source_paths.raw):
        cleaned = _standardize_columns(io_utils.load_csv(csv_path))
        destination = config.source_paths.processed / f"{csv_path.stem}_{config.output_suffix}.csv"
        io_utils.save_dataframe(cleaned, destination)


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case and strip leading/trailing spaces."""

    frame = frame.copy()
    frame.columns = [column.strip().lower().replace(" ", "_") for column in frame.columns]
    return frame
