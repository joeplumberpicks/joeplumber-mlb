"""
Lineup ingestion for Joe Plumber MLB Engine.

Purpose
-------
Normalize probable/projected/confirmed lineups and starting pitchers into clean
Layer 1 ingest tables.

This module is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering
- no target creation

Design
------
This file is provider-agnostic at the normalization layer.

Public functions
----------------
- build_projected_lineups(...)
- build_confirmed_lineups(...)
- build_starting_pitchers(...)

Input format
------------
These builders expect a list[dict] or DataFrame with lineup records already
pulled from a source such as Rotowire, Fangraphs, or another provider.

The source payload can be mapped into the normalized contract through the
normalization helpers below.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


def _safe_get(dct: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Safely walk nested dictionaries."""
    cur: Any = dct
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _as_dataframe(records: pd.DataFrame | list[dict[str, Any]] | None) -> pd.DataFrame:
    """Convert supported record inputs into a DataFrame."""
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, list):
        return pd.DataFrame(records)
    raise TypeError(f"Unsupported records type: {type(records)!r}")


def _empty_lineups_df() -> pd.DataFrame:
    """Return empty normalized lineup DataFrame."""
    return pd.DataFrame(
        columns=[
            "game_pk",
            "game_date",
            "season",
            "team",
            "opponent",
            "is_home",
            "lineup_status",
            "source",
            "source_pull_ts",
            "batting_order",
            "player_id",
            "player_name",
            "handedness_bat",
            "handedness_throw",
            "position",
            "is_starting_lineup",
        ]
    )


def _empty_starting_pitchers_df() -> pd.DataFrame:
    """Return empty normalized starting pitchers DataFrame."""
    return pd.DataFrame(
        columns=[
            "game_pk",
            "game_date",
            "season",
            "team",
            "opponent",
            "is_home",
            "pitcher_id",
            "pitcher_name",
            "throws",
            "starter_status",
            "source",
            "source_pull_ts",
        ]
    )


def normalize_lineup_records(
    records: pd.DataFrame | list[dict[str, Any]] | None,
    lineup_status: str,
    source: str | None = None,
) -> pd.DataFrame:
    """
    Normalize lineup records into a clean one-row-per-player lineup table.

    Expected normalized output columns
    ----------------------------------
    - game_pk
    - game_date
    - season
    - team
    - opponent
    - is_home
    - lineup_status
    - source
    - source_pull_ts
    - batting_order
    - player_id
    - player_name
    - handedness_bat
    - handedness_throw
    - position
    - is_starting_lineup

    Notes
    -----
    This function is tolerant of multiple possible source column names.
    """
    df = _as_dataframe(records)

    if df.empty:
        return _empty_lineups_df()

    out = pd.DataFrame(
        {
            "game_pk": pd.to_numeric(
                df.get("game_pk", df.get("game_id")),
                errors="coerce",
            ).astype("Int64"),
            "game_date": pd.to_datetime(
                df.get("game_date", df.get("date")),
                errors="coerce",
            ).dt.date.astype("string"),
            "season": pd.to_numeric(
                df.get("season"),
                errors="coerce",
            ).astype("Int64"),
            "team": df.get("team", df.get("team_abbr")).astype("string"),
            "opponent": df.get("opponent", df.get("opp", df.get("opponent_abbr"))).astype("string"),
            "is_home": (
                df.get("is_home", df.get("home_flag", False))
                if "is_home" in df.columns or "home_flag" in df.columns
                else False
            ),
            "lineup_status": lineup_status,
            "source": df.get("source", source if source is not None else "unknown"),
            "source_pull_ts": pd.to_datetime(
                df.get("source_pull_ts", pd.Timestamp.utcnow()),
                utc=True,
                errors="coerce",
            ),
            "batting_order": pd.to_numeric(
                df.get("batting_order", df.get("lineup_slot", df.get("order"))),
                errors="coerce",
            ).astype("Int64"),
            "player_id": pd.to_numeric(
                df.get("player_id", df.get("batter_id", df.get("mlbam_id"))),
                errors="coerce",
            ).astype("Int64"),
            "player_name": df.get("player_name", df.get("name", df.get("batter_name"))).astype("string"),
            "handedness_bat": (
    df["handedness_bat"]
    if "handedness_bat" in df.columns
    else df["bats"]
    if "bats" in df.columns
    else pd.Series(pd.NA, index=df.index)
).astype("string"),
            "handedness_throw": (
    df["handedness_throw"]
    if "handedness_throw" in df.columns
    else df["throws"]
    if "throws" in df.columns
    else pd.Series(pd.NA, index=df.index)
).astype("string"),
            "position": df.get("position", df.get("pos")).astype("string"),
            "is_starting_lineup": (
                df.get("is_starting_lineup", True)
                if "is_starting_lineup" in df.columns
                else True
            ),
        }
    )

    out["is_home"] = pd.Series(out["is_home"]).astype("boolean")
    out["is_starting_lineup"] = pd.Series(out["is_starting_lineup"]).astype("boolean")

    out = out.sort_values(
        ["game_date", "game_pk", "team", "batting_order", "player_name"],
        kind="stable",
    ).reset_index(drop=True)

    return out


def normalize_starting_pitcher_records(
    records: pd.DataFrame | list[dict[str, Any]] | None,
    starter_status: str,
    source: str | None = None,
) -> pd.DataFrame:
    """
    Normalize starting pitcher records into a clean one-row-per-team table.
    """
    df = _as_dataframe(records)

    if df.empty:
        return _empty_starting_pitchers_df()

    out = pd.DataFrame(
        {
            "game_pk": pd.to_numeric(
                df.get("game_pk", df.get("game_id")),
                errors="coerce",
            ).astype("Int64"),
            "game_date": pd.to_datetime(
                df.get("game_date", df.get("date")),
                errors="coerce",
            ).dt.date.astype("string"),
            "season": pd.to_numeric(
                df.get("season"),
                errors="coerce",
            ).astype("Int64"),
            "team": df.get("team", df.get("team_abbr")).astype("string"),
            "opponent": df.get("opponent", df.get("opp", df.get("opponent_abbr"))).astype("string"),
            "is_home": (
                df.get("is_home", df.get("home_flag", False))
                if "is_home" in df.columns or "home_flag" in df.columns
                else False
            ),
            "pitcher_id": pd.to_numeric(
                df.get("pitcher_id", df.get("player_id", df.get("mlbam_id"))),
                errors="coerce",
            ).astype("Int64"),
            "pitcher_name": df.get("pitcher_name", df.get("player_name", df.get("name"))).astype("string"),
            "throws": df.get("throws", df.get("handedness_throw")).astype("string"),
            "starter_status": starter_status,
            "source": df.get("source", source if source is not None else "unknown"),
            "source_pull_ts": pd.to_datetime(
                df.get("source_pull_ts", pd.Timestamp.utcnow()),
                utc=True,
                errors="coerce",
            ),
        }
    )

    out["is_home"] = pd.Series(out["is_home"]).astype("boolean")

    out = out.sort_values(
        ["game_date", "game_pk", "team", "pitcher_name"],
        kind="stable",
    ).reset_index(drop=True)

    return out


def validate_lineups(df: pd.DataFrame) -> None:
    """
    Validate normalized lineup output.

    Raises
    ------
    ValueError
        If required columns are missing or key constraints fail.
    """
    required_columns = {
        "game_pk",
        "game_date",
        "season",
        "team",
        "lineup_status",
        "batting_order",
        "player_name",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"lineups validation failed; missing required columns: {missing}")

    if df["game_pk"].isna().any():
        raise ValueError(f"lineups validation failed; null game_pk count={int(df['game_pk'].isna().sum())}")

    dupes = int(df.duplicated(subset=["game_pk", "team", "lineup_status", "batting_order"]).sum())
    if dupes:
        raise ValueError(
            "lineups validation failed; duplicate key count="
            f"{dupes} on ['game_pk', 'team', 'lineup_status', 'batting_order']"
        )


def validate_starting_pitchers(df: pd.DataFrame) -> None:
    """
    Validate normalized starting pitcher output.
    """
    required_columns = {
        "game_pk",
        "game_date",
        "season",
        "team",
        "starter_status",
        "pitcher_name",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"starting pitchers validation failed; missing required columns: {missing}")

    if df["game_pk"].isna().any():
        raise ValueError(
            f"starting pitchers validation failed; null game_pk count={int(df['game_pk'].isna().sum())}"
        )

    dupes = int(df.duplicated(subset=["game_pk", "team", "starter_status"]).sum())
    if dupes:
        raise ValueError(
            "starting pitchers validation failed; duplicate key count="
            f"{dupes} on ['game_pk', 'team', 'starter_status']"
        )


def summarize_lineups(df: pd.DataFrame, label: str = "lineups") -> None:
    """Print a compact ingest summary."""
    row_count = len(df)
    distinct_games = df["game_pk"].nunique(dropna=True) if "game_pk" in df.columns else 0
    min_date = df["game_date"].min() if "game_date" in df.columns and row_count else None
    max_date = df["game_date"].max() if "game_date" in df.columns and row_count else None

    print(f"Row count [{label}]: {row_count:,}")
    print(f"Distinct game_pk: {distinct_games:,}")
    print(f"Min game_date: {min_date}")
    print(f"Max game_date: {max_date}")

    for col in ["game_pk", "team", "batting_order", "player_name"]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")

    if {"game_pk", "team", "lineup_status", "batting_order"}.issubset(df.columns):
        print(
            "Duplicates on ['game_pk', 'team', 'lineup_status', 'batting_order']: "
            f"{int(df.duplicated(subset=['game_pk', 'team', 'lineup_status', 'batting_order']).sum()):,}"
        )


def summarize_starting_pitchers(df: pd.DataFrame, label: str = "starting_pitchers") -> None:
    """Print a compact ingest summary."""
    row_count = len(df)
    distinct_games = df["game_pk"].nunique(dropna=True) if "game_pk" in df.columns else 0
    min_date = df["game_date"].min() if "game_date" in df.columns and row_count else None
    max_date = df["game_date"].max() if "game_date" in df.columns and row_count else None

    print(f"Row count [{label}]: {row_count:,}")
    print(f"Distinct game_pk: {distinct_games:,}")
    print(f"Min game_date: {min_date}")
    print(f"Max game_date: {max_date}")

    for col in ["game_pk", "team", "pitcher_name"]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")

    if {"game_pk", "team", "starter_status"}.issubset(df.columns):
        print(
            "Duplicates on ['game_pk', 'team', 'starter_status']: "
            f"{int(df.duplicated(subset=['game_pk', 'team', 'starter_status']).sum()):,}"
        )


def build_projected_lineups(
    records: pd.DataFrame | list[dict[str, Any]] | None,
    source: str = "unknown",
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build normalized projected lineup table.
    """
    df = normalize_lineup_records(
        records=records,
        lineup_status="projected",
        source=source,
    )

    if validate and not df.empty:
        validate_lineups(df)

    if verbose:
        summarize_lineups(df, label="projected_lineups")

    return df


def build_confirmed_lineups(
    records: pd.DataFrame | list[dict[str, Any]] | None,
    source: str = "unknown",
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build normalized confirmed lineup table.
    """
    df = normalize_lineup_records(
        records=records,
        lineup_status="confirmed",
        source=source,
    )

    if validate and not df.empty:
        validate_lineups(df)

    if verbose:
        summarize_lineups(df, label="confirmed_lineups")

    return df


def build_starting_pitchers(
    records: pd.DataFrame | list[dict[str, Any]] | None,
    starter_status: str = "probable",
    source: str = "unknown",
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build normalized starting pitchers table.
    """
    df = normalize_starting_pitcher_records(
        records=records,
        starter_status=starter_status,
        source=source,
    )

    if validate and not df.empty:
        validate_starting_pitchers(df)

    if verbose:
        summarize_starting_pitchers(df, label="starting_pitchers")

    return df
