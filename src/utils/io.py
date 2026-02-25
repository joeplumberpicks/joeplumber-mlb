from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def safe_mkdir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def read_parquet(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path.resolve()}")
    return pd.read_parquet(parquet_path)


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    parquet_path = Path(path)
    safe_mkdir(parquet_path.parent)
    df.to_parquet(parquet_path, index=False)


def read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path.resolve()}")
    return pd.read_csv(csv_path)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    csv_path = Path(path)
    safe_mkdir(csv_path.parent)
    df.to_csv(csv_path, index=False)


def write_json(obj: dict[str, Any], path: str | Path) -> None:
    json_path = Path(path)
    safe_mkdir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, sort_keys=True)
