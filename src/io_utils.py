"""Utilities for loading and persisting datasets used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(slots=True)
class DatasetPaths:
    """Container describing canonical project directories."""

    raw: Path = RAW_DATA_DIR
    interim: Path = INTERIM_DATA_DIR
    processed: Path = PROCESSED_DATA_DIR
    outputs: Path = OUTPUTS_DIR


def ensure_directories(paths: Iterable[Path] | None = None) -> None:
    """Create the provided directories if they do not already exist."""

    to_create = paths or (RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR)
    for directory in to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_csv(path: Path | str, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Load a CSV file with sensible defaults for the project."""

    return pd.read_csv(path, **read_csv_kwargs)


def save_dataframe(data: pd.DataFrame, destination: Path | str, *, index: bool = False, **to_csv_kwargs: Any) -> None:
    """Persist a DataFrame to disk using UTF-8 encoding by default."""

    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=index, encoding=to_csv_kwargs.pop("encoding", "utf-8"), **to_csv_kwargs)
