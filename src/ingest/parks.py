"""
Park reference ingestion for Joe Plumber MLB Engine.

Purpose
-------
Normalize park / venue reference records into a clean Layer 1 reference table.

This module is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering
- no target creation
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _as_dataframe(records: pd.DataFrame | list[dict[str, Any]] | None) -> pd.DataFrame:
    """Convert supported record inputs into a DataFrame."""
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, list):
        return pd.DataFrame(records)
    raise TypeError(f"Unsupported records type: {type(records)!r}")


def _coalesce_series(
    df: pd.DataFrame,
    candidates: list[str],
    default: Any = None,
) -> pd.Series:
    """Return the first available column from candidates, else a default series."""
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _to_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _to_nullable_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_string(series: pd.Series) -> pd.Series:
    return series.astype("string")


def _to_boolean(series: pd.Series, default: bool | None = None) -> pd.Series:
    s = series.copy()
    if default is not None:
        s = s.fillna(default)

    true_values = {True, 1, 1.0, "1", "true", "True", "TRUE", "y", "Y", "yes", "YES"}
    false_values = {False, 0, 0.0, "0", "false", "False", "FALSE", "n", "N", "no", "NO"}

    def _map_value(x: Any) -> Any:
        if pd.isna(x):
            return pd.NA
        if x in true_values:
            return True
        if x in false_values:
            return False
        return pd.NA

    return s.map(_map_value).astype("boolean")


def _empty_parks_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "venue_id",
            "venue_name",
            "team",
            "team_name",
            "city",
            "state",
            "country",
            "surface_type",
            "roof_type",
            "is_dome",
            "is_retractable",
            "altitude_ft",
            "latitude",
            "longitude",
            "time_zone",
            "weather_station_id",
            "park_factor_runs",
            "park_factor_hr",
            "park_factor_hits",
            "park_factor_2b",
            "park_factor_3b",
            "park_factor_bb",
            "park_source",
            "park_source_season",
            "source",
            "source_pull_ts",
        ]
    )


def normalize_park_records(
    records: pd.DataFrame | list[dict[str, Any]] | None,
    source: str | None = None,
) -> pd.DataFrame:
    """
    Normalize raw park / venue records into a one-row-per-venue table.
    """
    df = _as_dataframe(records)

    if df.empty:
        return _empty_parks_df()

    out = pd.DataFrame(
        {
            "venue_id": _to_nullable_int(
                _coalesce_series(df, ["venue_id", "park_id", "stadium_id", "id"])
            ),
            "venue_name": _to_string(
                _coalesce_series(df, ["venue_name", "park_name", "stadium_name", "name"])
            ),
            "team": _to_string(
                _coalesce_series(df, ["team", "team_abbr", "home_team"])
            ),
            "team_name": _to_string(
                _coalesce_series(df, ["team_name", "home_team_name"])
            ),
            "city": _to_string(_coalesce_series(df, ["city"])),
            "state": _to_string(_coalesce_series(df, ["state", "state_code", "province"])),
            "country": _to_string(_coalesce_series(df, ["country"], default="USA")),
            "surface_type": _to_string(
                _coalesce_series(df, ["surface_type", "surface", "turf_type"])
            ),
            "roof_type": _to_string(_coalesce_series(df, ["roof_type", "roof"])),
            "is_dome": _to_boolean(_coalesce_series(df, ["is_dome", "dome_flag"])),
            "is_retractable": _to_boolean(
                _coalesce_series(df, ["is_retractable", "retractable_roof_flag"])
            ),
            "altitude_ft": _to_nullable_float(
                _coalesce_series(df, ["altitude_ft", "elevation_ft", "altitude"])
            ),
            "latitude": _to_nullable_float(_coalesce_series(df, ["latitude", "lat"])),
            "longitude": _to_nullable_float(_coalesce_series(df, ["longitude", "lon", "lng"])),
            "time_zone": _to_string(_coalesce_series(df, ["time_zone", "timezone"])),
            "weather_station_id": _to_string(
                _coalesce_series(df, ["weather_station_id", "metar_station_id", "station_id"])
            ),
            "park_factor_runs": _to_nullable_float(
                _coalesce_series(df, ["park_factor_runs", "pf_runs"])
            ),
            "park_factor_hr": _to_nullable_float(
                _coalesce_series(df, ["park_factor_hr", "pf_hr"])
            ),
            "park_factor_hits": _to_nullable_float(
                _coalesce_series(df, ["park_factor_hits", "pf_hits"])
            ),
            "park_factor_2b": _to_nullable_float(
                _coalesce_series(df, ["park_factor_2b", "pf_2b"])
            ),
            "park_factor_3b": _to_nullable_float(
                _coalesce_series(df, ["park_factor_3b", "pf_3b"])
            ),
            "park_factor_bb": _to_nullable_float(
                _coalesce_series(df, ["park_factor_bb", "pf_bb"])
            ),
            "park_source": _to_string(
                _coalesce_series(df, ["park_source"], default=source if source is not None else "unknown")
            ),
            "park_source_season": _to_nullable_int(
                _coalesce_series(df, ["park_source_season", "season"])
            ),
            "source": _to_string(
                _coalesce_series(df, ["source"], default=source if source is not None else "unknown")
            ),
            "source_pull_ts": pd.to_datetime(
                _coalesce_series(df, ["source_pull_ts"], default=pd.Timestamp.utcnow()),
                utc=True,
                errors="coerce",
            ),
        }
    )

    roof_lower = out["roof_type"].str.lower()

    missing_is_dome = out["is_dome"].isna()
    out.loc[missing_is_dome, "is_dome"] = roof_lower.loc[missing_is_dome].isin(
        ["dome", "fixed", "fixed_roof", "closed"]
    )

    missing_is_retractable = out["is_retractable"].isna()
    out.loc[missing_is_retractable, "is_retractable"] = roof_lower.loc[missing_is_retractable].isin(
        ["retractable", "retractable_roof", "retractable roof"]
    )

    out["is_dome"] = out["is_dome"].astype("boolean")
    out["is_retractable"] = out["is_retractable"].astype("boolean")

    out = out.sort_values(
        ["team", "venue_id", "venue_name"],
        kind="stable",
    ).reset_index(drop=True)

    return out


def validate_parks_reference(df: pd.DataFrame) -> None:
    required_columns = {"venue_id", "venue_name"}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"parks validation failed; missing required columns: {missing}")

    if df["venue_id"].isna().any():
        raise ValueError(
            f"parks validation failed; null venue_id count={int(df['venue_id'].isna().sum())}"
        )

    dupes = int(df.duplicated(subset=["venue_id"]).sum())
    if dupes:
        raise ValueError(f"parks validation failed; duplicate venue_id count={dupes}")


def summarize_parks_reference(df: pd.DataFrame, label: str = "parks") -> None:
    row_count = len(df)
    distinct_venues = df["venue_id"].nunique(dropna=True) if "venue_id" in df.columns else 0

    print(f"Row count [{label}]: {row_count:,}")
    print(f"Distinct venue_id: {distinct_venues:,}")

    for col in ["venue_id", "venue_name", "team", "weather_station_id"]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")

    if "venue_id" in df.columns:
        print(f"Duplicates on [venue_id]: {int(df.duplicated(subset=['venue_id']).sum()):,}")


def build_parks_reference(
    records: pd.DataFrame | list[dict[str, Any]] | None,
    source: str = "unknown",
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    df = normalize_park_records(records=records, source=source)

    if validate and not df.empty:
        validate_parks_reference(df)

    if verbose:
        summarize_parks_reference(df, label="parks")

    return df