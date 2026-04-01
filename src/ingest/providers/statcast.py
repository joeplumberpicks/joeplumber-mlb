"""
Statcast Data Provider

Purpose
-------
Handles raw data pulls from Statcast via pybaseball.

This is a LOW-LEVEL provider:
- No normalization contracts
- No schema enforcement
- No feature logic
- No modeling logic

Returns raw or lightly standardized DataFrames.

Used by:
- plate_appearances.py
- future pitch-level ingest
- future batter/pitcher event pulls
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd

try:
    from pybaseball import statcast, statcast_batter, statcast_pitcher
except ImportError:  # pragma: no cover
    statcast = None
    statcast_batter = None
    statcast_pitcher = None


def _require_pybaseball() -> None:
    """Raise a helpful error if pybaseball is not installed."""
    if statcast is None or statcast_batter is None or statcast_pitcher is None:
        raise ImportError(
            "pybaseball is required for src.ingest.providers.statcast. "
            "Install it with: pip install pybaseball"
        )


def _coerce_iso_date(value: str | date | datetime) -> str:
    """Convert supported date-like values to YYYY-MM-DD."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _standardize_statcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply light provider-level cleanup to a Statcast DataFrame.

    This is intentionally minimal:
    - snake-safe-ish lowercase columns are left as-is
    - game_date coerced to date string
    - numeric ids lightly coerced where common
    """
    out = df.copy()

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype("string")

    for col in ["game_pk", "batter", "pitcher"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out


def fetch_statcast(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch raw Statcast data for a date range.

    Parameters
    ----------
    start_date:
        Inclusive start date.
    end_date:
        Inclusive end date.
    verbose:
        Whether to print a basic summary.

    Returns
    -------
    pandas.DataFrame
        Raw Statcast event/pitch data from pybaseball.
    """
    _require_pybaseball()

    start_date_str = _coerce_iso_date(start_date)
    end_date_str = _coerce_iso_date(end_date)

    df = statcast(start_dt=start_date_str, end_dt=end_date_str)
    df = _standardize_statcast_dataframe(df)

    if verbose:
        print(f"Row count [statcast_{start_date_str}_{end_date_str}]: {len(df):,}")

    return df


def fetch_statcast_batter(
    batter_id: int,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch raw Statcast data for a single batter over a date range.

    Parameters
    ----------
    batter_id:
        MLBAM batter id.
    start_date:
        Inclusive start date.
    end_date:
        Inclusive end date.
    verbose:
        Whether to print a basic summary.

    Returns
    -------
    pandas.DataFrame
        Raw batter-specific Statcast data.
    """
    _require_pybaseball()

    start_date_str = _coerce_iso_date(start_date)
    end_date_str = _coerce_iso_date(end_date)

    df = statcast_batter(start_dt=start_date_str, end_dt=end_date_str, player_id=int(batter_id))
    df = _standardize_statcast_dataframe(df)

    if verbose:
        print(f"Row count [statcast_batter_{batter_id}_{start_date_str}_{end_date_str}]: {len(df):,}")

    return df


def fetch_statcast_pitcher(
    pitcher_id: int,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch raw Statcast data for a single pitcher over a date range.

    Parameters
    ----------
    pitcher_id:
        MLBAM pitcher id.
    start_date:
        Inclusive start date.
    end_date:
        Inclusive end date.
    verbose:
        Whether to print a basic summary.

    Returns
    -------
    pandas.DataFrame
        Raw pitcher-specific Statcast data.
    """
    _require_pybaseball()

    start_date_str = _coerce_iso_date(start_date)
    end_date_str = _coerce_iso_date(end_date)

    df = statcast_pitcher(start_dt=start_date_str, end_dt=end_date_str, player_id=int(pitcher_id))
    df = _standardize_statcast_dataframe(df)

    if verbose:
        print(f"Row count [statcast_pitcher_{pitcher_id}_{start_date_str}_{end_date_str}]: {len(df):,}")

    return df


def extract_plate_appearances_from_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightly flatten Statcast pitch/event data into one-row-per-plate-appearance.

    Notes
    -----
    This is still provider-layer logic, not your normalized Layer 1 schema.
    It groups by game + at_bat_number and keeps the last pitch/event row
    as the PA result row.

    Returns
    -------
    pandas.DataFrame
        Lightly flattened one-row-per-PA DataFrame.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "game_date",
                "batter_id",
                "pitcher_id",
                "inning",
                "inning_topbot",
                "pa_index",
                "event_type",
                "event_text",
                "rbi",
                "outs_when_up",
                "balls",
                "strikes",
                "batting_team",
                "fielding_team",
            ]
        )

    required = {"game_pk", "at_bat_number"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "extract_plate_appearances_from_statcast requires columns: "
            f"{missing}. Available columns sample: {list(df.columns)[:25]}"
        )

    work = df.copy()

    sort_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"] if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols, kind="stable")

    pa_df = (
        work.groupby(["game_pk", "at_bat_number"], dropna=False, as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    out = pd.DataFrame(
        {
            "game_pk": pd.to_numeric(pa_df.get("game_pk"), errors="coerce").astype("Int64"),
            "game_date": pd.to_datetime(pa_df.get("game_date"), errors="coerce").dt.date.astype("string")
            if "game_date" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="string"),
            "batter_id": pd.to_numeric(pa_df.get("batter"), errors="coerce").astype("Int64")
            if "batter" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="Int64"),
            "pitcher_id": pd.to_numeric(pa_df.get("pitcher"), errors="coerce").astype("Int64")
            if "pitcher" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="Int64"),
            "inning": pd.to_numeric(pa_df.get("inning"), errors="coerce").astype("Int64")
            if "inning" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="Int64"),
            "inning_topbot": pa_df.get("inning_topbot", pd.Series(pd.NA, index=pa_df.index)).astype("string"),
            "pa_index": pd.to_numeric(pa_df.get("at_bat_number"), errors="coerce").astype("Int64"),
            "event_type": pa_df.get("events", pd.Series(pd.NA, index=pa_df.index)).astype("string"),
            "event_text": pa_df.get("description", pd.Series(pd.NA, index=pa_df.index)).astype("string"),
            "rbi": pd.to_numeric(pa_df.get("rbi"), errors="coerce").astype("Int64")
            if "rbi" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="Int64"),
            "outs_when_up": pd.to_numeric(pa_df.get("outs_when_up"), errors="coerce").astype("Int64")
            if "outs_when_up" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="Int64"),
            "balls": pd.to_numeric(pa_df.get("balls"), errors="coerce").astype("Int64")
            if "balls" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="Int64"),
            "strikes": pd.to_numeric(pa_df.get("strikes"), errors="coerce").astype("Int64")
            if "strikes" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="Int64"),
            "batting_team": pa_df.get("batting_team", pd.Series(pd.NA, index=pa_df.index)).astype("string")
            if "batting_team" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="string"),
            "fielding_team": pa_df.get("fielding_team", pd.Series(pd.NA, index=pa_df.index)).astype("string")
            if "fielding_team" in pa_df.columns
            else pd.Series(pd.NA, index=pa_df.index, dtype="string"),
        }
    )

    return out.sort_values(["game_date", "game_pk", "pa_index"], kind="stable").reset_index(drop=True)


def test_connection(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> bool:
    """
    Simple health check to confirm Statcast access is working.
    """
    try:
        df = fetch_statcast(start_date=start_date, end_date=end_date, verbose=False)
        return isinstance(df, pd.DataFrame)
    except Exception:
        return False


__all__ = [
    "fetch_statcast",
    "fetch_statcast_batter",
    "fetch_statcast_pitcher",
    "extract_plate_appearances_from_statcast",
    "test_connection",
]
