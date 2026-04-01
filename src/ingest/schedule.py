"""
Schedule ingestion for Joe Plumber MLB Engine.

Purpose
-------
Pull MLB schedule data and normalize it into a clean one-row-per-game table.

This module is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering
- no target creation

Primary public function
-----------------------
build_schedule_games(...)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd
import requests

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
DEFAULT_GAME_TYPES = ("R", "S")
REQUEST_TIMEOUT = 30


@dataclass(frozen=True)
class ScheduleIngestConfig:
    """Configuration for schedule ingestion."""

    season: int
    start_date: str | None = None
    end_date: str | None = None
    game_types: tuple[str, ...] = DEFAULT_GAME_TYPES
    sport_id: int = 1
    hydrate: str = "team,linescore,flags,seriesStatus,venue"
    request_timeout: int = REQUEST_TIMEOUT


def _coerce_iso_date(value: str | date | datetime | None) -> str | None:
    """Convert supported date-like values to YYYY-MM-DD."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _coerce_game_type_param(game_types: Iterable[str] | None) -> str:
    """Convert iterable of game type codes into MLB API query format."""
    values = tuple(gt.strip().upper() for gt in (game_types or DEFAULT_GAME_TYPES) if str(gt).strip())
    if not values:
        values = DEFAULT_GAME_TYPES
    return ",".join(values)


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


def fetch_schedule_json(
    season: int,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    game_types: Iterable[str] = DEFAULT_GAME_TYPES,
    sport_id: int = 1,
    hydrate: str = "team,linescore,flags,seriesStatus,venue",
    request_timeout: int = REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """
    Fetch raw MLB schedule JSON from StatsAPI.

    Parameters
    ----------
    season:
        MLB season year.
    start_date, end_date:
        Optional date bounds. If omitted, the API returns the season schedule.
    game_types:
        Iterable of MLB game type codes. Default includes regular season and spring.
    sport_id:
        MLB sport id, default 1.
    hydrate:
        Optional MLB API hydrate string.
    request_timeout:
        Timeout in seconds.

    Returns
    -------
    dict
        Raw JSON response from the MLB StatsAPI.
    """
    params: dict[str, Any] = {
        "sportId": sport_id,
        "season": season,
        "gameTypes": _coerce_game_type_param(game_types),
        "hydrate": hydrate,
    }

    start_date_iso = _coerce_iso_date(start_date)
    end_date_iso = _coerce_iso_date(end_date)

    if start_date_iso:
        params["startDate"] = start_date_iso
    if end_date_iso:
        params["endDate"] = end_date_iso

    response = requests.get(MLB_SCHEDULE_URL, params=params, timeout=request_timeout)
    response.raise_for_status()
    return response.json()


def normalize_schedule_json(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """
    Normalize raw MLB schedule JSON into a one-row-per-game DataFrame.

    Output columns are intentionally ingest-safe and modeling-agnostic.
    """
    rows: list[dict[str, Any]] = []

    for date_block in payload.get("dates", []):
        game_date = date_block.get("date")

        for game in date_block.get("games", []):
            away_team = _safe_get(game, "teams", "away", "team", "abbreviation")
            home_team = _safe_get(game, "teams", "home", "team", "abbreviation")
            away_team_name = _safe_get(game, "teams", "away", "team", "name")
            home_team_name = _safe_get(game, "teams", "home", "team", "name")
            venue_id = _safe_get(game, "venue", "id")
            venue_name = _safe_get(game, "venue", "name")

            game_datetime_utc_raw = game.get("gameDate")
            game_datetime_utc = pd.to_datetime(game_datetime_utc_raw, utc=True, errors="coerce")
            game_datetime_et = game_datetime_utc.tz_convert("America/New_York") if pd.notna(game_datetime_utc) else pd.NaT

            status_abstract = _safe_get(game, "status", "abstractGameState")
            status_detailed = _safe_get(game, "status", "detailedState")
            coded_game_state = _safe_get(game, "status", "codedGameState")
            series_description = game.get("seriesDescription")
            doubleheader = game.get("doubleHeader")
            game_number = game.get("gameNumber")
            day_night = game.get("dayNight")

            row = {
                "game_pk": game.get("gamePk"),
                "game_date": game_date,
                "season": season,
                "game_type": game.get("gameType"),
                "status": status_abstract,
                "status_detailed": status_detailed,
                "coded_game_state": coded_game_state,
                "scheduled_start_time_utc": game_datetime_utc.isoformat() if pd.notna(game_datetime_utc) else None,
                "scheduled_start_time_et": game_datetime_et.isoformat() if pd.notna(game_datetime_et) else None,
                "away_team": away_team,
                "home_team": home_team,
                "away_team_name": away_team_name,
                "home_team_name": home_team_name,
                "away_team_id": _safe_get(game, "teams", "away", "team", "id"),
                "home_team_id": _safe_get(game, "teams", "home", "team", "id"),
                "venue_id": venue_id,
                "venue_name": venue_name,
                "doubleheader_flag": str(doubleheader).upper() == "Y",
                "game_number": game_number,
                "series_description": series_description,
                "day_night": day_night,
                "scheduled_innings": game.get("scheduledInnings"),
                "if_necessary": bool(game.get("ifNecessary", False)),
                "if_necessary_description": game.get("ifNecessaryDescription"),
                "resume_date": game.get("resumeDate"),
                "resume_game_date": game.get("resumeGameDate"),
                "public_facing": bool(game.get("publicFacing", True)),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "game_date",
                "season",
                "game_type",
                "status",
                "status_detailed",
                "coded_game_state",
                "scheduled_start_time_utc",
                "scheduled_start_time_et",
                "away_team",
                "home_team",
                "away_team_name",
                "home_team_name",
                "away_team_id",
                "home_team_id",
                "venue_id",
                "venue_name",
                "doubleheader_flag",
                "game_number",
                "series_description",
                "day_night",
                "scheduled_innings",
                "if_necessary",
                "if_necessary_description",
                "resume_date",
                "resume_game_date",
                "public_facing",
            ]
        )

    # Standard dtypes / ordering
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype("string")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["venue_id"] = pd.to_numeric(df["venue_id"], errors="coerce").astype("Int64")
    df["away_team_id"] = pd.to_numeric(df["away_team_id"], errors="coerce").astype("Int64")
    df["home_team_id"] = pd.to_numeric(df["home_team_id"], errors="coerce").astype("Int64")
    df["game_number"] = pd.to_numeric(df["game_number"], errors="coerce").astype("Int64")
    df["scheduled_innings"] = pd.to_numeric(df["scheduled_innings"], errors="coerce").astype("Int64")
    df["doubleheader_flag"] = df["doubleheader_flag"].fillna(False).astype(bool)
    df["if_necessary"] = df["if_necessary"].fillna(False).astype(bool)
    df["public_facing"] = df["public_facing"].fillna(True).astype(bool)

    sort_cols = ["game_date", "scheduled_start_time_et", "game_pk"]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return df


def validate_schedule_games(df: pd.DataFrame) -> None:
    """
    Validate normalized schedule output.

    Raises
    ------
    ValueError
        If required columns are missing or key constraints fail.
    """
    required_columns = {
        "game_pk",
        "game_date",
        "season",
        "away_team",
        "home_team",
        "venue_id",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"schedule validation failed; missing required columns: {missing}")

    if df["game_pk"].isna().any():
        bad = int(df["game_pk"].isna().sum())
        raise ValueError(f"schedule validation failed; null game_pk count={bad}")

    dupes = int(df.duplicated(subset=["game_pk"]).sum())
    if dupes:
        raise ValueError(f"schedule validation failed; duplicate game_pk count={dupes}")


def summarize_schedule_games(df: pd.DataFrame, label: str = "games_schedule") -> None:
    """Print a compact ingest summary."""
    row_count = len(df)
    distinct_games = df["game_pk"].nunique(dropna=True) if "game_pk" in df.columns else 0
    min_date = df["game_date"].min() if "game_date" in df.columns and row_count else None
    max_date = df["game_date"].max() if "game_date" in df.columns and row_count else None

    print(f"Row count [{label}]: {row_count:,}")
    print(f"Distinct game_pk: {distinct_games:,}")
    print(f"Min game_date: {min_date}")
    print(f"Max game_date: {max_date}")

    for col in ["game_pk", "game_date", "away_team", "home_team", "venue_id"]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")

    if "game_pk" in df.columns:
        print(f"Duplicates on [game_pk]: {int(df.duplicated(subset=['game_pk']).sum()):,}")


def build_schedule_games(
    season: int,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    game_types: Iterable[str] = DEFAULT_GAME_TYPES,
    sport_id: int = 1,
    hydrate: str = "team,linescore,flags,seriesStatus,venue",
    request_timeout: int = REQUEST_TIMEOUT,
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch, normalize, and validate MLB schedule games.

    Returns
    -------
    pandas.DataFrame
        Clean normalized schedule table.
    """
    payload = fetch_schedule_json(
        season=season,
        start_date=start_date,
        end_date=end_date,
        game_types=game_types,
        sport_id=sport_id,
        hydrate=hydrate,
        request_timeout=request_timeout,
    )
    df = normalize_schedule_json(payload=payload, season=season)

    if validate:
        validate_schedule_games(df)

    if verbose:
        label_parts = [str(season)]
        if start_date:
            label_parts.append(_coerce_iso_date(start_date) or "")
        if end_date and end_date != start_date:
            label_parts.append(_coerce_iso_date(end_date) or "")
        summarize_schedule_games(df, label=f"games_schedule_{'_'.join([p for p in label_parts if p])}")

    return df
