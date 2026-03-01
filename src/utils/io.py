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


def _extract_hive_partition_value(path: Path, key: str) -> str | None:
    needle_prefix = f"{key}="
    for part in path.parts:
        if part.startswith(needle_prefix):
            return part[len(needle_prefix):]
    return None


def _apply_filters_to_part_files(
    dataset_dir: Path,
    part_files: list[Path],
    filters: list[tuple[str, str, Any]] | None,
) -> list[Path]:
    if not filters:
        return part_files

    keep = part_files
    for col, op, val in filters:
        if op != "=":
            continue

        partition_dir = dataset_dir / f"{col}={val}"
        if partition_dir.exists() and partition_dir.is_dir():
            keep = [p for p in keep if f"{col}={val}" in str(p)]
        else:
            sval = str(val)
            keep2: list[Path] = []
            for p in keep:
                pv = _extract_hive_partition_value(p, col)
                if pv is None or pv == sval:
                    keep2.append(p)
            keep = keep2
    return keep


def _read_parquet_parts(
    parquet_path: Path,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    part_files = sorted(p for p in parquet_path.rglob("*.parquet") if p.is_file())
    if not part_files:
        raise FileNotFoundError(f"No parquet part files found under dataset directory: {parquet_path.resolve()}")

    part_files = _apply_filters_to_part_files(parquet_path, part_files, filters)
    if not part_files:
        return pd.DataFrame()

    frames = [pd.read_parquet(part_path, columns=columns) for part_path in part_files]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _is_arrow_invalid(exc: Exception) -> bool:
    exc_type = type(exc)
    return exc_type.__name__ == "ArrowInvalid" or "pyarrow.lib" in exc_type.__module__


def read_parquet(
    path: str | Path,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path.resolve()}")

    if parquet_path.is_dir():
        logging.warning(
            "read_parquet fallback: reading parquet dataset directory via parts concat: %s (filters=%s columns=%s)",
            parquet_path.resolve(),
            filters,
            columns,
        )
        return _read_parquet_parts(parquet_path, columns=columns, filters=filters)

    try:
        return pd.read_parquet(parquet_path, columns=columns)
    except Exception as exc:  # noqa: BLE001
        if _is_arrow_invalid(exc):
            parent = parquet_path.parent
            candidate_dirs: list[Path] = []
            if parquet_path.exists() and parquet_path.is_dir():
                candidate_dirs.append(parquet_path)
            stem_dir = parent / parquet_path.stem
            if stem_dir.exists() and stem_dir.is_dir():
                candidate_dirs.append(stem_dir)
            file_named_dir = parent / parquet_path.name
            if file_named_dir.exists() and file_named_dir.is_dir():
                candidate_dirs.append(file_named_dir)

            for d in candidate_dirs:
                logging.warning(
                    "read_parquet ArrowInvalid fallback: reading dataset parts from %s (filters=%s columns=%s)",
                    d.resolve(),
                    filters,
                    columns,
                )
                return _read_parquet_parts(d, columns=columns, filters=filters)
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
