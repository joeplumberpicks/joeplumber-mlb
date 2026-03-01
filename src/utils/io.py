from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd


_MISSING_FIELD_RE = re.compile(r"No match for FieldRef\.Name\(([^)]+)\)")


def safe_mkdir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _extract_hive_partition_value(path: Path, key: str) -> str | None:
    needle_prefix = f"{key}="
    for part in path.parts:
        if part.startswith(needle_prefix):
            return part[len(needle_prefix) :]
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


def _is_arrow_invalid(exc: Exception) -> bool:
    exc_type = type(exc)
    return exc_type.__name__ == "ArrowInvalid" or "pyarrow.lib" in exc_type.__module__


def _drop_missing_field_from_columns(exc: Exception, columns: list[str] | None) -> list[str] | None:
    if not columns:
        return columns

    match = _MISSING_FIELD_RE.search(str(exc))
    if not match:
        return columns

    missing = match.group(1).strip('"\'')
    new_cols = [c for c in columns if c.strip('"\'') != missing]
    if new_cols != columns:
        logging.warning("read_parquet: dropping missing requested column=%s and retrying", missing)
    return new_cols


def _read_single_parquet_with_column_retries(
    parquet_file: Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    cols = list(columns) if columns is not None else None
    while True:
        try:
            return pd.read_parquet(parquet_file, columns=cols)
        except Exception as exc:  # noqa: BLE001
            if not _is_arrow_invalid(exc):
                raise
            new_cols = _drop_missing_field_from_columns(exc, cols)
            if new_cols == cols:
                raise
            cols = new_cols


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

    cols = list(columns) if columns is not None else None
    while True:
        try:
            frames = [_read_single_parquet_with_column_retries(part_path, columns=cols) for part_path in part_files]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True, sort=False)
        except Exception as exc:  # noqa: BLE001
            if not _is_arrow_invalid(exc):
                raise
            new_cols = _drop_missing_field_from_columns(exc, cols)
            if new_cols == cols:
                raise
            cols = new_cols


def read_parquet(
    path: str | Path,
    columns: list[str] | None = None,
    filters: list[tuple[str, str, Any]] | None = None,
) -> pd.DataFrame:
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path.resolve()}")

    if parquet_path.is_dir():
        if columns is None:
            raise ValueError(
                "read_parquet requires columns=... for dataset directories to avoid OOM reads: "
                f"{parquet_path.resolve()}"
            )
        logging.warning(
            "read_parquet fallback: reading parquet dataset directory via parts concat: %s (filters=%s columns=%s)",
            parquet_path.resolve(),
            filters,
            columns,
        )
        return _read_parquet_parts(parquet_path, columns=columns, filters=filters)

    cols = list(columns) if columns is not None else None
    while True:
        try:
            return pd.read_parquet(parquet_path, columns=cols)
        except Exception as exc:  # noqa: BLE001
            if _is_arrow_invalid(exc):
                new_cols = _drop_missing_field_from_columns(exc, cols)
                if new_cols != cols:
                    cols = new_cols
                    continue

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

                for dataset_dir in candidate_dirs:
                    logging.warning(
                        "read_parquet ArrowInvalid fallback: reading dataset parts from %s (filters=%s columns=%s)",
                        dataset_dir.resolve(),
                        filters,
                        cols,
                    )
                    return _read_parquet_parts(dataset_dir, columns=cols, filters=filters)
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
