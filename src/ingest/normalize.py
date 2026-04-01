"""
Shared normalization utilities for Joe Plumber MLB Engine Layer 1 ingest tables.

Purpose
-------
Provide reusable, ingest-safe normalization helpers for:
- column names
- team abbreviations
- strings
- booleans
- numeric coercion
- datetime coercion
- dataframe cleanup

This module is Layer 1 only:
- normalization helpers only
- no modeling logic
- no feature engineering
- no target creation
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


TEAM_ABBREV_MAP: dict[str, str] = {
    "AZ": "ARI",
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CUBS": "CHC",
    "CWS": "CHA",
    "CHW": "CHA",
    "CHA": "CHA",
    "WHITE SOX": "CHA",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KCA",
    "KCR": "KCA",
    "KCA": "KCA",
    "LAA": "LAA",
    "ANA": "LAA",
    "LAD": "LAN",
    "LAN": "LAN",
    "MIA": "MIA",
    "FLA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYY": "NYA",
    "NYA": "NYA",
    "NYYANKEES": "NYA",
    "NYM": "NYN",
    "NYN": "NYN",
    "NYMETS": "NYN",
    "OAK": "OAK",
    "ATH": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SDN",
    "SDP": "SDN",
    "SDN": "SDN",
    "SEA": "SEA",
    "SF": "SFN",
    "SFG": "SFN",
    "SFN": "SFN",
    "STL": "SLN",
    "SL": "SLN",
    "SLN": "SLN",
    "TB": "TBA",
    "TBR": "TBA",
    "TAMPA BAY": "TBA",
    "TBA": "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WAS",
    "WAS": "WAS",
    "NATS": "WAS",
}


TRUE_VALUES = {True, 1, 1.0, "1", "true", "True", "TRUE", "y", "Y", "yes", "YES"}
FALSE_VALUES = {False, 0, 0.0, "0", "false", "False", "FALSE", "n", "N", "no", "NO"}


def snake_case(value: str) -> str:
    """
    Convert a string into normalized snake_case column format.
    """
    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column names to snake_case.
    """
    out = df.copy()
    out.columns = [snake_case(col) for col in out.columns]
    return out


def strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip leading/trailing whitespace from string/object columns.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return out


def empty_strings_to_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert blank strings to pandas NA.
    """
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].replace(r"^\s*$", pd.NA, regex=True)
    return out


def normalize_text_value(value: Any) -> Any:
    """
    Normalize a single text value by stripping whitespace.
    """
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, str):
        value = value.strip()
        return value if value else pd.NA
    return value


def normalize_text_series(series: pd.Series) -> pd.Series:
    """
    Normalize a text series by stripping whitespace and blanking empty strings.
    """
    return series.map(normalize_text_value).astype("string")


def normalize_team_abbreviation(value: Any) -> Any:
    """
    Normalize a team abbreviation to the canonical Joe Plumber form.
    """
    if value is None or pd.isna(value):
        return pd.NA

    raw = str(value).strip().upper()
    raw = re.sub(r"[^A-Z]", "", raw)

    if raw in TEAM_ABBREV_MAP:
        return TEAM_ABBREV_MAP[raw]

    return raw if raw else pd.NA


def normalize_team_series(series: pd.Series) -> pd.Series:
    """
    Normalize a series of team abbreviations.
    """
    return series.map(normalize_team_abbreviation).astype("string")


def to_nullable_int(series: pd.Series) -> pd.Series:
    """
    Convert a series to pandas nullable Int64.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def to_nullable_float(series: pd.Series) -> pd.Series:
    """
    Convert a series to numeric float.
    """
    return pd.to_numeric(series, errors="coerce")


def to_string(series: pd.Series) -> pd.Series:
    """
    Convert a series to pandas string dtype.
    """
    return series.astype("string")


def to_boolean(series: pd.Series, default: bool | None = None) -> pd.Series:
    """
    Convert common truthy/falsy values to pandas nullable boolean.
    """
    s = series.copy()

    if default is not None:
        s = s.fillna(default)

    def _map_value(x: Any) -> Any:
        if pd.isna(x):
            return pd.NA
        if x in TRUE_VALUES:
            return True
        if x in FALSE_VALUES:
            return False
        return pd.NA

    return s.map(_map_value).astype("boolean")


def to_datetime_utc(series: pd.Series) -> pd.Series:
    """
    Convert a series to timezone-aware UTC datetimes.
    """
    return pd.to_datetime(series, utc=True, errors="coerce")


def to_date_string(series: pd.Series) -> pd.Series:
    """
    Convert a series to YYYY-MM-DD string dates.
    """
    return pd.to_datetime(series, errors="coerce").dt.date.astype("string")


def coalesce_columns(df: pd.DataFrame, candidates: list[str], default: Any = None) -> pd.Series:
    """
    Return the first available column from candidates, otherwise a default-valued series.
    """
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def deduplicate_on_key(
    df: pd.DataFrame,
    subset: list[str],
    keep: str = "last",
    sort_by: list[str] | None = None,
) -> pd.DataFrame:
    """
    Deduplicate a DataFrame on a given key.

    Parameters
    ----------
    df:
        DataFrame to deduplicate.
    subset:
        Columns that define uniqueness.
    keep:
        Which duplicate to keep, passed to pandas drop_duplicates.
    sort_by:
        Optional sort columns to apply before deduplication.
    """
    out = df.copy()

    if sort_by:
        existing_sort = [c for c in sort_by if c in out.columns]
        if existing_sort:
            out = out.sort_values(existing_sort, kind="stable")

    existing_subset = [c for c in subset if c in out.columns]
    if not existing_subset:
        return out.reset_index(drop=True)

    return out.drop_duplicates(subset=existing_subset, keep=keep).reset_index(drop=True)


def standardize_dataframe(
    df: pd.DataFrame,
    *,
    normalize_colnames: bool = True,
    strip_text: bool = True,
    blank_to_na: bool = True,
) -> pd.DataFrame:
    """
    Apply a standard Layer 1 cleanup pipeline.

    Steps:
    - normalize column names
    - strip string whitespace
    - convert blank strings to NA
    """
    out = df.copy()

    if normalize_colnames:
        out = normalize_columns(out)
    if strip_text:
        out = strip_strings(out)
    if blank_to_na:
        out = empty_strings_to_na(out)

    return out


__all__ = [
    "TEAM_ABBREV_MAP",
    "snake_case",
    "normalize_columns",
    "strip_strings",
    "empty_strings_to_na",
    "normalize_text_value",
    "normalize_text_series",
    "normalize_team_abbreviation",
    "normalize_team_series",
    "to_nullable_int",
    "to_nullable_float",
    "to_string",
    "to_boolean",
    "to_datetime_utc",
    "to_date_string",
    "coalesce_columns",
    "deduplicate_on_key",
    "standardize_dataframe",
]
