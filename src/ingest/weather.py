"""
Weather ingestion for Joe Plumber MLB Engine using NOAA/AWC METAR data.

Purpose
-------
Pull game-level weather from the official Aviation Weather Center METAR API
and normalize it into a clean one-row-per-game weather table.

This module is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering for scoring
- no target creation

Notes
-----
- Requires a mapping from MLB venue/team to nearby METAR station.
- Designed for current / recent slates. METAR API history is limited.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd
import requests

METAR_API_URL = "https://aviationweather.gov/api/data/metar"
REQUEST_TIMEOUT = 30

# Supports both canonical Joe Plumber abbreviations and common live-feed abbreviations.
TEAM_TO_METAR_STATION: dict[str, str] = {
    "ARI": "KPHX",
    "AZ": "KPHX",
    "ATL": "KATL",
    "BAL": "KBWI",
    "BOS": "KBOS",
    "CHC": "KORD",
    "CHA": "KMDW",
    "CHW": "KMDW",
    "CWS": "KMDW",
    "CIN": "KLUK",
    "CLE": "KCLE",
    "COL": "KDEN",
    "DET": "KDTW",
    "HOU": "KIAH",
    "KCA": "KMCI",
    "KC": "KMCI",
    "KCR": "KMCI",
    "LAA": "KSNA",
    "LAD": "KLAX",
    "LAN": "KLAX",
    "MIA": "KMIA",
    "MIL": "KMKE",
    "MIN": "KMSP",
    "NYA": "KLGA",
    "NYY": "KLGA",
    "NYN": "KLGA",
    "NYM": "KLGA",
    "OAK": "KOAK",
    "ATH": "KOAK",
    "PHI": "KPHL",
    "PIT": "KPIT",
    "SD": "KSAN",
    "SDN": "KSAN",
    "SDP": "KSAN",
    "SEA": "KSEA",
    "SF": "KSFO",
    "SFN": "KSFO",
    "SFG": "KSFO",
    "SLN": "KSTL",
    "STL": "KSTL",
    "TBA": "KTPA",
    "TB": "KTPA",
    "TBR": "KTPA",
    "TEX": "KDFW",
    "TOR": "CYYZ",
    "WAS": "KDCA",
    "WSH": "KDCA",
}


def _coerce_iso_date(value: str | date | datetime | None) -> str | None:
    """Convert supported date-like values to YYYY-MM-DD."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


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


def _to_float(value: Any) -> float | None:
    """Coerce value to float if possible."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    """Coerce value to int if possible."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _normalize_wind_flags(wind_dir_degrees: float | None) -> tuple[float, float, float]:
    """
    Convert wind direction degrees into coarse field-direction flags.

    This is intentionally simple and ingest-safe.
    Assumes:
    - out = blowing toward outfield roughly 315°..45°
    - in = blowing in from center roughly 135°..225°
    - cross = everything else
    """
    if wind_dir_degrees is None:
        return 0.0, 0.0, 0.0

    deg = wind_dir_degrees % 360

    if deg >= 315 or deg <= 45:
        return 1.0, 0.0, 0.0
    if 135 <= deg <= 225:
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def _infer_roof_status(home_team: str | None) -> str | None:
    """
    Best-effort roof status from team / park type.

    This remains ingest metadata only.
    """
    retractable_or_dome = {
        "ARI",
        "AZ",
        "HOU",
        "MIA",
        "MIL",
        "SEA",
        "TBA",
        "TB",
        "TOR",
        "TEX",
    }
    if home_team in retractable_or_dome:
        return "unknown_retractable"
    return "open_air"


def fetch_metar_json(
    station_ids: Iterable[str],
    hours: float = 3.0,
    request_timeout: int = REQUEST_TIMEOUT,
) -> list[dict[str, Any]]:
    """
    Fetch METAR observations for one or more stations.

    Parameters
    ----------
    station_ids:
        ICAO station ids, e.g. ["KATL", "KBOS"].
    hours:
        Lookback window in hours.
    request_timeout:
        Timeout in seconds.

    Returns
    -------
    list[dict[str, Any]]
        Raw METAR records from the API.
    """
    ids = sorted({str(x).strip().upper() for x in station_ids if str(x).strip()})
    if not ids:
        return []

    params = {
        "ids": ",".join(ids),
        "format": "json",
        "hours": str(hours),
    }

    response = requests.get(METAR_API_URL, params=params, timeout=request_timeout)
    response.raise_for_status()

    payload = response.json()
    if isinstance(payload, list):
        return payload
    return []


def normalize_metar_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize raw METAR records into station-level weather observations.
    """
    rows: list[dict[str, Any]] = []

    for rec in records:
        station_id = (
            rec.get("icaoId")
            or rec.get("station_id")
            or rec.get("station")
            or rec.get("id")
        )

        observed_at = (
            rec.get("obsTime")
            or rec.get("observationTime")
            or rec.get("observation_time")
        )

        temp_c = _to_float(
            rec.get("temp")
            if rec.get("temp") is not None
            else rec.get("tempC")
        )
        dewpoint_c = _to_float(
            rec.get("dewp")
            if rec.get("dewp") is not None
            else rec.get("dewpoint")
        )

        wind_dir_deg = _to_float(
            rec.get("wdir")
            if rec.get("wdir") is not None
            else rec.get("windDir")
        )
        wind_speed_kt = _to_float(
            rec.get("wspd")
            if rec.get("wspd") is not None
            else rec.get("windSpeed")
        )
        wind_gust_kt = _to_float(
            rec.get("wgst")
            if rec.get("wgst") is not None
            else rec.get("windGust")
        )

        pressure_hpa = _to_float(
            rec.get("altim")
            if rec.get("altim") is not None
            else rec.get("seaLevelPressure")
        )

        visibility_mi = _to_float(
            rec.get("visib")
            if rec.get("visib") is not None
            else rec.get("visibility")
        )

        temperature_f = (temp_c * 9.0 / 5.0 + 32.0) if temp_c is not None else None
        dewpoint_f = (dewpoint_c * 9.0 / 5.0 + 32.0) if dewpoint_c is not None else None
        wind_mph = (wind_speed_kt * 1.15078) if wind_speed_kt is not None else None
        wind_gust_mph = (wind_gust_kt * 1.15078) if wind_gust_kt is not None else None

        weather_wind_out, weather_wind_in, weather_crosswind = _normalize_wind_flags(wind_dir_deg)

        row = {
            "station_id": station_id,
            "observed_at_utc": pd.to_datetime(observed_at, utc=True, errors="coerce"),
            "raw_metar": rec.get("rawOb") or rec.get("raw_text") or rec.get("rawText"),
            "flight_category": rec.get("flight_category"),
            "weather_condition": rec.get("wxString") or rec.get("wx"),
            "temperature_c": temp_c,
            "temperature_f": temperature_f,
            "dewpoint_c": dewpoint_c,
            "dewpoint_f": dewpoint_f,
            "wind_dir_degrees": wind_dir_deg,
            "wind_speed_kt": wind_speed_kt,
            "wind_speed_mph": wind_mph,
            "wind_gust_kt": wind_gust_kt,
            "wind_gust_mph": wind_gust_mph,
            "pressure_hpa": pressure_hpa,
            "visibility_miles": visibility_mi,
            "ceiling_ft": _to_int(rec.get("ceil")),
            "humidity": None,
            "precipitation_mm": None,
            "weather_wind_out": weather_wind_out,
            "weather_wind_in": weather_wind_in,
            "weather_crosswind": weather_crosswind,
            "weather_source": "awc_metar",
            "weather_pull_ts": pd.Timestamp.utcnow(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "station_id",
                "observed_at_utc",
                "raw_metar",
                "flight_category",
                "weather_condition",
                "temperature_c",
                "temperature_f",
                "dewpoint_c",
                "dewpoint_f",
                "wind_dir_degrees",
                "wind_speed_kt",
                "wind_speed_mph",
                "wind_gust_kt",
                "wind_gust_mph",
                "pressure_hpa",
                "visibility_miles",
                "ceiling_ft",
                "humidity",
                "precipitation_mm",
                "weather_wind_out",
                "weather_wind_in",
                "weather_crosswind",
                "weather_source",
                "weather_pull_ts",
            ]
        )

    df = df.sort_values(["station_id", "observed_at_utc"], kind="stable").reset_index(drop=True)
    return df


def build_weather_games(
    schedule_df: pd.DataFrame,
    station_map: dict[str, str] | None = None,
    hours: float = 3.0,
    request_timeout: int = REQUEST_TIMEOUT,
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build one-row-per-game weather table by joining games to nearby METAR stations.

    Parameters
    ----------
    schedule_df:
        Normalized schedule DataFrame. Must include:
        - game_pk
        - game_date
        - season
        - home_team
        - away_team
        - venue_id
        - venue_name
    station_map:
        Optional override mapping of team abbreviation -> METAR station.
    hours:
        Lookback window for most recent METAR.
    request_timeout:
        Timeout in seconds.
    validate:
        Whether to validate output.
    verbose:
        Whether to print summary.

    Returns
    -------
    pandas.DataFrame
        One row per game with normalized weather fields.
    """
    required_cols = {
        "game_pk",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "venue_id",
        "venue_name",
    }
    missing = sorted(required_cols.difference(schedule_df.columns))
    if missing:
        raise ValueError(f"build_weather_games requires schedule columns: missing={missing}")

    station_lookup = dict(TEAM_TO_METAR_STATION)
    if station_map:
        station_lookup.update({str(k): str(v).upper() for k, v in station_map.items()})

    work = schedule_df.copy()
    work["station_id"] = work["home_team"].map(station_lookup)

    station_ids = [x for x in work["station_id"].dropna().astype(str).unique().tolist() if x]
    metar_records = fetch_metar_json(
        station_ids=station_ids,
        hours=hours,
        request_timeout=request_timeout,
    )
    station_weather = normalize_metar_records(metar_records)

    if not station_weather.empty:
        station_weather = (
            station_weather.sort_values(["station_id", "observed_at_utc"], kind="stable")
            .drop_duplicates(subset=["station_id"], keep="last")
            .reset_index(drop=True)
        )

    df = work.merge(
        station_weather,
        on="station_id",
        how="left",
        validate="m:1",
    )

    df["roof_status"] = df["home_team"].map(_infer_roof_status)

    out = pd.DataFrame(
        {
            "game_pk": pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64"),
            "game_date": pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype("string"),
            "season": pd.to_numeric(df["season"], errors="coerce").astype("Int64"),
            "home_team": df["home_team"].astype("string"),
            "away_team": df["away_team"].astype("string"),
            "venue_id": pd.to_numeric(df["venue_id"], errors="coerce").astype("Int64"),
            "venue_name": df["venue_name"].astype("string"),
            "station_id": df["station_id"].astype("string"),
            "observed_at_utc": pd.to_datetime(df["observed_at_utc"], utc=True, errors="coerce"),
            "temperature_f": pd.to_numeric(df["temperature_f"], errors="coerce"),
            "temperature_c": pd.to_numeric(df["temperature_c"], errors="coerce"),
            "dewpoint_f": pd.to_numeric(df["dewpoint_f"], errors="coerce"),
            "dewpoint_c": pd.to_numeric(df["dewpoint_c"], errors="coerce"),
            "wind_mph": pd.to_numeric(df["wind_speed_mph"], errors="coerce"),
            "wind_kt": pd.to_numeric(df["wind_speed_kt"], errors="coerce"),
            "wind_gust_mph": pd.to_numeric(df["wind_gust_mph"], errors="coerce"),
            "wind_gust_kt": pd.to_numeric(df["wind_gust_kt"], errors="coerce"),
            "wind_direction_deg": pd.to_numeric(df["wind_dir_degrees"], errors="coerce"),
            "wind_dir_text": df["wind_dir_degrees"].apply(
                lambda x: None if pd.isna(x) else f"{int(float(x))}°"
            ).astype("string"),
            "pressure_hpa": pd.to_numeric(df["pressure_hpa"], errors="coerce"),
            "visibility_miles": pd.to_numeric(df["visibility_miles"], errors="coerce"),
            "ceiling_ft": pd.to_numeric(df["ceiling_ft"], errors="coerce").astype("Int64"),
            "weather_condition": df["weather_condition"].astype("string"),
            "flight_category": df["flight_category"].astype("string"),
            "humidity": pd.to_numeric(df["humidity"], errors="coerce"),
            "precipitation_mm": pd.to_numeric(df["precipitation_mm"], errors="coerce"),
            "roof_status": df["roof_status"].astype("string"),
            "weather_wind_out": pd.to_numeric(df["weather_wind_out"], errors="coerce"),
            "weather_wind_in": pd.to_numeric(df["weather_wind_in"], errors="coerce"),
            "weather_crosswind": pd.to_numeric(df["weather_crosswind"], errors="coerce"),
            "raw_metar": df["raw_metar"].astype("string"),
            "weather_source": df["weather_source"].fillna("awc_metar").astype("string"),
            "weather_pull_ts": pd.to_datetime(df["weather_pull_ts"], utc=True, errors="coerce"),
        }
    )

    out = out.sort_values(["game_date", "game_pk"], kind="stable").reset_index(drop=True)

    if validate:
        validate_weather_games(out)

    if verbose:
        summarize_weather_games(out)

    return out


def validate_weather_games(df: pd.DataFrame) -> None:
    """
    Validate normalized weather output.
    """
    required_columns = {
        "game_pk",
        "game_date",
        "season",
        "home_team",
        "away_team",
        "venue_id",
        "station_id",
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
    """
    Print compact ingest summary.
    """
    row_count = len(df)
    distinct_games = df["game_pk"].nunique(dropna=True) if "game_pk" in df.columns else 0
    min_date = df["game_date"].min() if "game_date" in df.columns and row_count else None
    max_date = df["game_date"].max() if "game_date" in df.columns and row_count else None

    print(f"Row count [{label}]: {row_count:,}")
    print(f"Distinct game_pk: {distinct_games:,}")
    print(f"Min game_date: {min_date}")
    print(f"Max game_date: {max_date}")

    for col in ["game_pk", "station_id", "temperature_f", "wind_mph", "wind_direction_deg"]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")

    if "game_pk" in df.columns:
        print(f"Duplicates on [game_pk]: {int(df.duplicated(subset=['game_pk']).sum()):,}")
