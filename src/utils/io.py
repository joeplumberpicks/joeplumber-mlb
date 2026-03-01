from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


def safe_mkdir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _read_parquet_parts(parquet_path: Path) -> pd.DataFrame:
    part_files = sorted(p for p in parquet_path.rglob("*.parquet") if p.is_file())
    if not part_files:
        raise FileNotFoundError(f"No parquet part files found under dataset directory: {parquet_path.resolve()}")
    frames = [pd.read_parquet(part_path) for part_path in part_files]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _is_arrow_invalid(exc: Exception) -> bool:
    exc_type = type(exc)
    return exc_type.__name__ == "ArrowInvalid" or "pyarrow.lib" in exc_type.__module__


def read_parquet(path: str | Path) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path.resolve()}")

    if parquet_path.is_dir():
        logging.warning("read_parquet fallback: reading parquet dataset directory via parts concat: %s", parquet_path.resolve())
        return _read_parquet_parts(parquet_path)

    try:
        return pd.read_parquet(parquet_path)
    except Exception as exc:  # noqa: BLE001
        if _is_arrow_invalid(exc):
            dataset_dir = parquet_path if parquet_path.is_dir() else (parquet_path if parquet_path.suffix == ".parquet" and parquet_path.is_dir() else None)
            if dataset_dir is None and parquet_path.suffix == ".parquet":
                possible_dir = parquet_path
                if possible_dir.exists() and possible_dir.is_dir():
                    dataset_dir = possible_dir
            if dataset_dir is not None and dataset_dir.exists() and dataset_dir.is_dir():
                logging.warning(
                    "read_parquet ArrowInvalid fallback: reading dataset parts from %s", dataset_dir.resolve()
                )
                return _read_parquet_parts(dataset_dir)

            parent = parquet_path.parent
            if parent.exists() and parent.is_dir() and parquet_path.name.endswith(".parquet"):
                candidate_dir = parent / parquet_path.name
                if candidate_dir.exists() and candidate_dir.is_dir():
                    logging.warning(
                        "read_parquet ArrowInvalid fallback: reading dataset parts from %s", candidate_dir.resolve()
                    )
                    return _read_parquet_parts(candidate_dir)
        raise


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
