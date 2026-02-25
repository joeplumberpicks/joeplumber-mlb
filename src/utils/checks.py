from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def require_files(paths: list[Path], label: str) -> None:
    missing = [str(path.resolve()) for path in paths if not Path(path).exists()]
    if missing:
        missing_str = "\n - ".join([""] + missing)
        raise FileNotFoundError(f"Missing required files for {label}:{missing_str}")


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for {label}: {missing_cols}")


def print_rowcount(name: str, df: pd.DataFrame) -> None:
    print(f"Row count [{name}]: {len(df):,}")
