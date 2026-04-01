"""
I/O utilities for Joe Plumber MLB Engine Layer 1 ingest tables.

Purpose
-------
Provide shared file read/write helpers for normalized ingest outputs.

This module is Layer 1 only:
- file I/O helpers
- directory creation
- lightweight logging helpers

No modeling logic.
No feature engineering.
No target creation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Ensure the parent directory for a file path exists.

    Returns
    -------
    Path
        Normalized Path object.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists.

    Returns
    -------
    Path
        Normalized Path object.
    """
    out_path = Path(path)
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path


def write_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    engine: str = "pyarrow",
    compression: str = "snappy",
    verbose: bool = True,
) -> Path:
    """
    Write a DataFrame to parquet.

    Parameters
    ----------
    df:
        DataFrame to write.
    path:
        Output parquet path.
    index:
        Whether to write the index.
    engine:
        Parquet engine, default 'pyarrow'.
    compression:
        Compression codec, default 'snappy'.
    verbose:
        Whether to print the output path.

    Returns
    -------
    Path
        Written file path.
    """
    out_path = ensure_parent_dir(path)
    df.to_parquet(out_path, index=index, engine=engine, compression=compression)

    if verbose:
        print(f"Writing to: {out_path}")

    return out_path


def read_parquet(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    engine: str = "pyarrow",
) -> pd.DataFrame:
    """
    Read a parquet file into a DataFrame.
    """
    in_path = Path(path)
    return pd.read_parquet(in_path, columns=columns, engine=engine)


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    index: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Write a DataFrame to CSV.

    Returns
    -------
    Path
        Written file path.
    """
    out_path = ensure_parent_dir(path)
    df.to_csv(out_path, index=index)

    if verbose:
        print(f"Writing to: {out_path}")

    return out_path


def read_csv(
    path: str | Path,
    *,
    dtype: dict[str, Any] | None = None,
    parse_dates: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame.
    """
    in_path = Path(path)
    return pd.read_csv(in_path, dtype=dtype, parse_dates=parse_dates)


def path_exists(path: str | Path) -> bool:
    """
    Return True if a path exists.
    """
    return Path(path).exists()


def require_path(path: str | Path) -> Path:
    """
    Require that a path exists, otherwise raise FileNotFoundError.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Required path not found: {in_path}")
    return in_path


def log_section(script_path: str) -> None:
    """
    Print a standard section header for scripts.
    """
    print(f"========== {script_path} =========")


def log_kv(key: str, value: Any) -> None:
    """
    Print a key-value line in a standard format.
    """
    print(f"{key}: {value}")


def log_dataframe_summary(df: pd.DataFrame, label: str) -> None:
    """
    Print a compact DataFrame summary.
    """
    print(f"Row count [{label}]: {len(df):,}")

    if len(df) == 0:
        return

    if "game_pk" in df.columns:
        print(f"Distinct game_pk: {df['game_pk'].nunique(dropna=True):,}")

    if "venue_id" in df.columns:
        print(f"Distinct venue_id: {df['venue_id'].nunique(dropna=True):,}")

    if "game_date" in df.columns:
        print(f"Min game_date: {df['game_date'].min()}")
        print(f"Max game_date: {df['game_date'].max()}")


def write_dataset(
    df: pd.DataFrame,
    path: str | Path,
    *,
    file_format: str = "parquet",
    index: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Path:
    """
    Write a dataset in the requested format.

    Supported formats
    -----------------
    - parquet
    - csv
    """
    fmt = str(file_format).strip().lower()

    if fmt == "parquet":
        return write_parquet(df, path, index=index, verbose=verbose, **kwargs)
    if fmt == "csv":
        return write_csv(df, path, index=index, verbose=verbose)

    raise ValueError(f"Unsupported file_format='{file_format}'. Expected 'parquet' or 'csv'.")


def read_dataset(
    path: str | Path,
    *,
    file_format: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a dataset, inferring format from suffix when not provided.
    """
    in_path = Path(path)

    fmt = file_format
    if fmt is None:
        suffix = in_path.suffix.lower()
        if suffix == ".parquet":
            fmt = "parquet"
        elif suffix == ".csv":
            fmt = "csv"
        else:
            raise ValueError(f"Could not infer file format from suffix: {in_path.suffix}")

    fmt = str(fmt).strip().lower()

    if fmt == "parquet":
        return read_parquet(in_path, **kwargs)
    if fmt == "csv":
        return read_csv(in_path, **kwargs)

    raise ValueError(f"Unsupported file_format='{fmt}'. Expected 'parquet' or 'csv'.")


__all__ = [
    "ensure_parent_dir",
    "ensure_dir",
    "write_parquet",
    "read_parquet",
    "write_csv",
    "read_csv",
    "path_exists",
    "require_path",
    "log_section",
    "log_kv",
    "log_dataframe_summary",
    "write_dataset",
    "read_dataset",
]
