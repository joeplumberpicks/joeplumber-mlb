"""
Weather ingestion for Joe Plumber MLB Engine.

Purpose
-------
Pull game-level weather from MLB schedule/game metadata and normalize it into a
clean one-row-per-game weather table.

This module is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering
- no target creation

Primary public function
-----------------------
build_weather_games(...)
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd
import requests

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
DEFAULT_GAME_TYPES = ("R", "S")
REQUEST_TIMEOUT = 30


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


def _extract_temperature_f(weather_raw: str | None) -> float | None:
    """
    Extract temperature in Fahrenheit from free-text weather string.

    Example inputs:
    - "72 degrees, Sunny"
    - "68 degrees, Partly Cloudy"
    """
    if not weather_raw:
        return None
    match = re.search(r"(-?\d+)\s*degrees", str(weather_raw), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _extract_weather_condition(weather_raw: str | None) -> str | None:
    """
    Extract a normalized weather condition string from free-text weather.

    Example:
    - "72 degrees, Sunny" -> "Sunny"
    """
    if not weather_raw:
        return None
    parts = [p.strip() for p in str(weather_raw).split(",") if p.strip()]
    if len(parts) >= 2:
        return ", ".join(parts[1:]).strip() or None
    return None


def _extract_wind_mph(wind_raw: str | None) -> float | None:
    """
    Extract numeric wind speed from free-text wind string.

    Example inputs:
    - "10 mph, Out To LF"
    - "7 mph, In From CF"
    - "0 mph, None"
    """
    if not wind_raw:
        return None
    match = re.search(r"(-?\d+)\s*mph", str(wind_raw), flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _extract_wind_direction_text(wind_raw: str | None) -> str | None:
    """
    Extract raw wind direction descriptor from free-text wind string.

    Example:
    - "10 mph, Out To LF" -> "Out To LF"
    """
    if not wind_raw:
        return None
    parts = [p.strip() for p in str(wind_raw).split(",") if p.strip()]
    if len(parts) >= 2:
        return ", ".join(parts[1:]).strip() or None
    return None


def _normalize_wind_flags(wind_direction_text: str | None) -> tuple[float, float, float]:
    """
    Convert wind direction text into simple directional flags.

    Returns
    -------
    tuple
        (weather_wind_out, weather_wind_in, weather_crosswind)
    """
    if not wind_direction_text:
        return 0.0, 0.0, 0.0

    value = str(wind_direction_text).strip().lower()

    if value in {"none", "calm"}:
        return 0.0, 0.0, 0.0

    is_out = any(token in value for token in ["out to", "out toward", "blowing out"])
    is_in = any(token in value for token in ["in from", "blowing in"])
    is_cross = any(
        token in value
        for token in ["left to right", "right to left", "from lf", "from rf", "toward lf", "toward rf"]
    )

    if is_out:
        return 1.0, 0.0, 0.0
    if is_in:
        return 0.0, 1.0, 0.0
    if is_cross:
        return 0.0, 0.0, 1.0

    return 0.0, 0.0, 0.0


def _infer_roof_status(weather_raw: str | None, wind_raw: str | None) -> str | None:
    """
    Infer a simple roof status from available weather text.

    This is best-effort ingest metadata only, not a modeling feature.
    """
    combined = " ".join([str(x) for x in [weather_raw, wind_raw] if x]).lower()
    if not combined:
        return None

    if "roof closed" in combined or "closed roof" in combined:
        return "closed"
    if "roof open" in combined or "open roof" in combined:
        return "open"
    if "dome" in combined:
        return "closed"

    return None


def fetch_weather_json(
    season: int,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    game_types: Iterable[str] = DEFAULT_GAME_TYPES,
    sport_id: int = 1,
    hydrate: str = "weather,venue,linescore,team",
    request_timeout: int = REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """
    Fetch raw MLB schedule JSON including weather-related fields.
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


def normalize_weather_json(payload: dict[str, Any], season: int) -> pd.DataFrame:
    """
    Normalize raw MLB schedule JSON into a one-row-per-game weather DataFrame.
    """
    rows: list[dict[str, Any]] = []

    for date_block in payload.get("dates", []):
        fallback_game_date = date_block.get("date")

        for game in date_block.get("games", []):
            weather_raw = game.get("weather")
            wind_raw = game.get("wind")

            wind_direction_text = _extract_wind_direction_text(wind_raw)
            weather_wind_out, weather_wind_in, weather_crosswind = _normalize_wind_flags(wind_direction_text)

            row = {
                "game_pk": game.get("gamePk"),
                "game_date": game.get("officialDate") or fallback_game_date,
                "season": season,
                "home_team": _safe_get(game, "teams", "home", "team", "abbreviation"),
                "away_team": _safe_get(game, "teams", "away", "team", "abbreviation"),
                "venue_id": _safe_get(game, "venue", "id"),
                "venue_name": _safe_get(game, "venue", "name"),
                "temperature": _extract_temperature_f(weather_raw),
                "temperature_f": _extract_temperature_f(weather_raw),
                "wind_speed": _extract_wind_mph(wind_raw),
                "wind_mph": _extract_wind_mph(wind_raw),
                "wind_direction": wind_raw,
                "wind_dir": wind_direction_text,
                "weather_condition": _extract_weather_condition(weather_raw),
                "weather_raw": weather_raw,
                "wind_raw": wind_raw,
                "roof_status": _infer_roof_status(weather_raw, wind_raw),
                "humidity": None,
                "precipitation_risk": None,
                "weather_wind_out": weather_wind_out,
                "weather_wind_in": weather_wind_in,
                "weather_crosswind": weather_crosswind,
                "weather_source": "mlb_statsapi_schedule",
                "weather_pull_ts": pd.Timestamp.utcnow().isoformat(),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "game_date",
                "season",
                "home_team",
                "away_team",
                "venue_id",
                "venue_name",
                "temperature",
                "temperature_f",
                "wind_speed",
                "wind_mph",
                "wind_direction",
                "wind_dir",
                "weather_condition",
                "weather_raw",
                "wind_raw",
                "roof_status",
                "humidity",
                "precipitation_risk",
                "weather_wind_out",
                "weather_wind_in",
                "weather_crosswind",
                "weather_source",
                "weather_pull_ts",
            ]
        )

    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["venue_id"] = pd.to_numeric(df["venue_id"], errors="coerce").astype("Int64")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype("string")

    float_cols = [
        "temperature",
        "temperature_f",
        "wind_speed",
        "wind_mph",
        "humidity",
        "precipitation_risk",
        "weather_wind_out",
        "weather_wind_in",
        "weather_crosswind",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = ["game_date", "game_pk"]
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return df


def validate_weather_games(df: pd.DataFrame) -> None:
    """
    Validate normalized weather output.

    Raises
    ------
    ValueError
        If required columns are missing or key constraints fail.
    """
    required_columns = {
        "game_pk",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "venue_id",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"weather validation failed; missing required columns: {missing}")

    if df["game_pk"].isna().any():
        bad = int(df["game_pk"].isna().sum())
        raise ValueError(f"weather validation failed; null game_pk count={bad}")

    dupes = int(df.duplicated(subset=["game_pk"]).sum())
    if dupes:
        raise ValueError(f"weather validation failed; duplicate game_pk count={dupes}")


def summarize_weather_games(df: pd.DataFrame, label: str = "weather_game") -> None:
    """Print a compact ingest summary."""
    row_count = len(df)
    distinct_games = df["game_pk"].nunique(dropna=True) if "game_pk" in df.columns else 0
    min_date = df["game_date"].min() if "game_date" in df.columns and row_count else None
    max_date = df["game_date"].max() if "game_date" in df.columns and row_count else None

    print(f"Row count [{label}]: {row_count:,}")
    print(f"Distinct game_pk: {distinct_games:,}")
    print(f"Min game_date: {min_date}")
    print(f"Max game_date: {max_date}")

    for col in ["game_pk", "game_date", "temperature_f", "wind_mph", "wind_dir"]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")

    if "game_pk" in df.columns:
        print(f"Duplicates on [game_pk]: {int(df.duplicated(subset=['game_pk']).sum()):,}")


def build_weather_games(
    season: int,
    start_date: str | date | datetime | None = None,
    end_date: str | date | datetime | None = None,
    game_types: Iterable[str] = DEFAULT_GAME_TYPES,
    sport_id: int = 1,
    hydrate: str = "weather,venue,linescore,team",
    request_timeout: int = REQUEST_TIMEOUT,
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch, normalize, and validate MLB weather data.

    Returns
    -------
    pandas.DataFrame
        Clean normalized weather table.
    """
    payload = fetch_weather_json(
        season=season,
        start_date=start_date,
        end_date=end_date,
        game_types=game_types,
        sport_id=sport_id,
        hydrate=hydrate,
        request_timeout=request_timeout,
    )
    df = normalize_weather_json(payload=payload, season=season)

    if validate:
        validate_weather_games(df)

    if verbose:
        label_parts = [str(season)]
        if start_date:
            label_parts.append(_coerce_iso_date(start_date) or "")
        if end_date and end_date != start_date:
            label_parts.append(_coerce_iso_date(end_date) or "")
        summarize_weather_games(df, label=f"weather_game_{'_'.join([p for p in label_parts if p])}")

    return df
