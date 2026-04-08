"""
Build unified game spine for Joe Plumber MLB Engine.

Purpose
-------
Create one clean pregame row per game by joining Layer 1 ingest outputs.

This module is Layer 2 only:
- joins raw/normalized ingest tables
- standardizes one-row-per-game structure
- adds lineup/starter availability indicators

This module does NOT:
- build rolling features
- create targets
- do model scoring
- include same-day outcome leakage
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _copy_or_empty(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a copy of a DataFrame or an empty DataFrame."""
    if df is None:
        return pd.DataFrame()
    return df.copy()


def _safe_series(df: pd.DataFrame, col: str, default: Any = pd.NA) -> pd.Series:
    """Return a column if present, else a default-valued series."""
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype("boolean")


def _to_str(series: pd.Series) -> pd.Series:
    return series.astype("string")


def _normalize_team_abbr(team: object) -> object:
    if team is None or pd.isna(team):
        return pd.NA
    s = str(team).strip().upper()
    aliases = {
        "AZ": "ARI",
        "ARZ": "ARI",
        "ATH": "OAK",
        "OAKLAND": "OAK",
        "CWS": "CHW",
        "CHIWS": "CHW",
        "KCR": "KC",
        "KAN": "KC",
        "SDP": "SD",
        "SFG": "SF",
        "TBR": "TB",
        "TAM": "TB",
        "WAS": "WSH",
    }
    return aliases.get(s, s)


def _fill_false_bool(series: pd.Series) -> pd.Series:
    return series.astype("boolean").fillna(False)


def _prep_schedule(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare schedule table as the base of the spine."""
    df = schedule_df.copy()

    required = ["game_pk", "game_date", "season", "away_team", "home_team", "venue_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"schedule_df missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "game_pk": _to_int(df["game_pk"]),
            "game_date": pd.to_datetime(df["game_date"], errors="coerce").dt.date.astype("string"),
            "season": _to_int(df["season"]),
            "game_type": _to_str(_safe_series(df, "game_type")),
            "status": _to_str(_safe_series(df, "status")),
            "status_detailed": _to_str(_safe_series(df, "status_detailed")),
            "scheduled_start_time_utc": _to_str(_safe_series(df, "scheduled_start_time_utc")),
            "scheduled_start_time_et": _to_str(_safe_series(df, "scheduled_start_time_et")),
            "away_team": _to_str(df["away_team"]),
            "home_team": _to_str(df["home_team"]),
            "away_team_id": _to_int(_safe_series(df, "away_team_id")),
            "home_team_id": _to_int(_safe_series(df, "home_team_id")),
            "away_team_name": _to_str(_safe_series(df, "away_team_name")),
            "home_team_name": _to_str(_safe_series(df, "home_team_name")),
            "venue_id": _to_int(df["venue_id"]),
            "venue_name": _to_str(_safe_series(df, "venue_name")),
            "doubleheader_flag": _to_bool(_safe_series(df, "doubleheader_flag", False)),
            "game_number": _to_int(_safe_series(df, "game_number")),
            "series_description": _to_str(_safe_series(df, "series_description")),
            "day_night": _to_str(_safe_series(df, "day_night")),
        }
    )

    dupes = int(out.duplicated(subset=["game_pk"]).sum())
    if dupes:
        raise ValueError(f"schedule base has duplicate game_pk rows: {dupes}")

    return out.sort_values(["game_date", "scheduled_start_time_et", "game_pk"], kind="stable").reset_index(drop=True)


def _prep_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare game metadata for left join into the spine."""
    if games_df.empty:
        return pd.DataFrame(columns=["game_pk"])

    df = games_df.copy()

    out = pd.DataFrame(
        {
            "game_pk": _to_int(_safe_series(df, "game_pk")),
            "game_status": _to_str(_safe_series(df, "status")),
            "game_status_detailed": _to_str(_safe_series(df, "status_detailed")),
            "away_score": _to_int(_safe_series(df, "away_score")),
            "home_score": _to_int(_safe_series(df, "home_score")),
            "home_win": _to_bool(_safe_series(df, "home_win")),
            "innings": _to_int(_safe_series(df, "innings")),
            "start_time_utc": _to_str(_safe_series(df, "start_time_utc")),
            "start_time_et": _to_str(_safe_series(df, "start_time_et")),
        }
    )

    out = out.drop_duplicates(subset=["game_pk"], keep="last").reset_index(drop=True)
    return out


def _prep_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare weather for left join into the spine."""
    if weather_df.empty:
        return pd.DataFrame(columns=["game_pk"])

    df = weather_df.copy()

    wind_deg_col = "wind_direction_deg" if "wind_direction_deg" in df.columns else "wind_dir_degrees"

    out = pd.DataFrame(
        {
            "game_pk": _to_int(_safe_series(df, "game_pk")),
            "weather_station_id": _to_str(_safe_series(df, "station_id")),
            "weather_observed_at_utc": pd.to_datetime(_safe_series(df, "observed_at_utc"), utc=True, errors="coerce"),
            "temperature_f": _to_float(_safe_series(df, "temperature_f")),
            "temperature_c": _to_float(_safe_series(df, "temperature_c")),
            "dewpoint_f": _to_float(_safe_series(df, "dewpoint_f")),
            "dewpoint_c": _to_float(_safe_series(df, "dewpoint_c")),
            "wind_mph": _to_float(_safe_series(df, "wind_mph")),
            "wind_kt": _to_float(_safe_series(df, "wind_kt")),
            "wind_gust_mph": _to_float(_safe_series(df, "wind_gust_mph")),
            "wind_gust_kt": _to_float(_safe_series(df, "wind_gust_kt")),
            "wind_direction_deg": _to_float(_safe_series(df, wind_deg_col)),
            "wind_dir_text": _to_str(_safe_series(df, "wind_dir_text")),
            "pressure_hpa": _to_float(_safe_series(df, "pressure_hpa")),
            "visibility_miles": _to_float(_safe_series(df, "visibility_miles")),
            "ceiling_ft": _to_int(_safe_series(df, "ceiling_ft")),
            "weather_condition": _to_str(_safe_series(df, "weather_condition")),
            "flight_category": _to_str(_safe_series(df, "flight_category")),
            "humidity": _to_float(_safe_series(df, "humidity")),
            "precipitation_mm": _to_float(_safe_series(df, "precipitation_mm")),
            "roof_status_weather": _to_str(_safe_series(df, "roof_status")),
            "weather_wind_out": _to_float(_safe_series(df, "weather_wind_out")),
            "weather_wind_in": _to_float(_safe_series(df, "weather_wind_in")),
            "weather_crosswind": _to_float(_safe_series(df, "weather_crosswind")),
            "weather_source": _to_str(_safe_series(df, "weather_source")),
        }
    )

    out = out.drop_duplicates(subset=["game_pk"], keep="last").reset_index(drop=True)
    return out


def _prep_parks(parks_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare parks reference for left join into the spine."""
    if parks_df.empty:
        return pd.DataFrame(columns=["venue_id"])

    df = parks_df.copy()

    out = pd.DataFrame(
        {
            "venue_id": _to_int(_safe_series(df, "venue_id")),
            "park_team": _to_str(_safe_series(df, "team")),
            "park_team_name": _to_str(_safe_series(df, "team_name")),
            "park_city": _to_str(_safe_series(df, "city")),
            "park_state": _to_str(_safe_series(df, "state")),
            "park_country": _to_str(_safe_series(df, "country")),
            "surface_type": _to_str(_safe_series(df, "surface_type")),
            "roof_type": _to_str(_safe_series(df, "roof_type")),
            "is_dome": _to_bool(_safe_series(df, "is_dome")),
            "is_retractable": _to_bool(_safe_series(df, "is_retractable")),
            "altitude_ft": _to_float(_safe_series(df, "altitude_ft")),
            "park_latitude": _to_float(_safe_series(df, "latitude")),
            "park_longitude": _to_float(_safe_series(df, "longitude")),
            "park_time_zone": _to_str(_safe_series(df, "time_zone")),
            "park_weather_station_id": _to_str(_safe_series(df, "weather_station_id")),
            "park_factor_runs": _to_float(_safe_series(df, "park_factor_runs")),
            "park_factor_hr": _to_float(_safe_series(df, "park_factor_hr")),
            "park_factor_hits": _to_float(_safe_series(df, "park_factor_hits")),
            "park_factor_2b": _to_float(_safe_series(df, "park_factor_2b")),
            "park_factor_3b": _to_float(_safe_series(df, "park_factor_3b")),
            "park_factor_bb": _to_float(_safe_series(df, "park_factor_bb")),
        }
    )

    out = out.drop_duplicates(subset=["venue_id"], keep="last").reset_index(drop=True)
    return out


def _summarize_lineups(lineups_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Summarize lineups to one row per game/team.
    Expected input: projected or confirmed lineups.
    """
    if lineups_df.empty:
        return pd.DataFrame(columns=["game_pk", "team"])

    df = lineups_df.copy()

    work = pd.DataFrame(
        {
            "game_pk": _to_int(_safe_series(df, "game_pk")),
            "team": _to_str(_safe_series(df, "team")),
            "lineup_status": _to_str(_safe_series(df, "lineup_status")),
            "batting_order": _to_int(_safe_series(df, "batting_order")),
            "player_id": _to_int(_safe_series(df, "player_id")),
            "player_name": _to_str(_safe_series(df, "player_name")),
            "is_starting_lineup": _to_bool(_safe_series(df, "is_starting_lineup", True)),
        }
    )

    grouped = (
        work.groupby(["game_pk", "team"], dropna=False)
        .agg(
            lineup_status=("lineup_status", "last"),
            lineup_spots=("batting_order", lambda s: int(s.notna().sum())),
            distinct_players=("player_id", lambda s: int(s.dropna().nunique()) if "Int64" in str(s.dtype) else int(s.astype("string").dropna().nunique())),
            first_batter=("player_name", "first"),
            last_batter=("player_name", "last"),
        )
        .reset_index()
    )

    grouped[f"{prefix}_lineup_status"] = grouped.pop("lineup_status").astype("string")
    grouped[f"{prefix}_lineup_spots"] = pd.to_numeric(grouped.pop("lineup_spots"), errors="coerce").astype("Int64")
    grouped[f"{prefix}_distinct_players"] = pd.to_numeric(grouped.pop("distinct_players"), errors="coerce").astype("Int64")
    grouped[f"{prefix}_first_batter"] = grouped.pop("first_batter").astype("string")
    grouped[f"{prefix}_last_batter"] = grouped.pop("last_batter").astype("string")
    grouped[f"{prefix}_lineup_found"] = True

    return grouped


def _summarize_starters(starters_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize starting pitchers to one row per game/team."""
    if starters_df.empty:
        return pd.DataFrame(columns=["game_pk", "team"])

    df = starters_df.copy()

    out = pd.DataFrame(
        {
            "game_pk": _to_int(_safe_series(df, "game_pk")),
            "team": _to_str(_safe_series(df, "team")),
            "starter_status": _to_str(_safe_series(df, "starter_status")),
            "starter_pitcher_id": _to_int(_safe_series(df, "pitcher_id")),
            "starter_pitcher_name": _to_str(_safe_series(df, "pitcher_name")),
            "starter_throws": _to_str(_safe_series(df, "throws")),
        }
    )

    out = out.drop_duplicates(subset=["game_pk", "team"], keep="last").reset_index(drop=True)
    out["starter_found"] = True
    return out


def _join_team_side(
    spine_df: pd.DataFrame,
    side_df: pd.DataFrame,
    side_team_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Join a game/team summary table twice-friendly with prefixed output columns.
    """
    if side_df.empty:
        return spine_df

    df = side_df.copy()
    rename_map = {c: f"{prefix}_{c}" for c in df.columns if c not in ["game_pk", side_team_col]}
    df = df.rename(columns=rename_map)

    merged = spine_df.merge(
        df,
        left_on=["game_pk", f"{prefix}_team"],
        right_on=["game_pk", side_team_col],
        how="left",
        validate="1:1",
    ).drop(columns=[side_team_col])

    return merged


def validate_model_spine_game(df: pd.DataFrame) -> None:
    """Validate final game spine output."""
    required = {
        "game_pk",
        "game_date",
        "season",
        "away_team",
        "home_team",
        "venue_id",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"model spine missing required columns: {missing}")

    null_game_pk = int(df["game_pk"].isna().sum())
    if null_game_pk:
        raise ValueError(f"model spine has null game_pk rows: {null_game_pk}")

    dupes = int(df.duplicated(subset=["game_pk"]).sum())
    if dupes:
        raise ValueError(f"model spine has duplicate game_pk rows: {dupes}")


def summarize_model_spine_game(df: pd.DataFrame, label: str = "model_spine_game") -> None:
    """Print compact summary for the game spine."""
    print(f"Row count [{label}]: {len(df):,}")
    print(f"Distinct game_pk: {df['game_pk'].nunique(dropna=True):,}")
    if len(df):
        print(f"Min game_date: {df['game_date'].min()}")
        print(f"Max game_date: {df['game_date'].max()}")

    for col in [
        "venue_id",
        "weather_station_id",
        "away_starter_pitcher_id",
        "home_starter_pitcher_id",
        "away_projected_lineup_found",
        "home_projected_lineup_found",
        "away_confirmed_lineup_found",
        "home_confirmed_lineup_found",
    ]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")

    if "away_starter_found" in df.columns and "home_starter_found" in df.columns:
        starter_pct = ((_fill_false_bool(df["away_starter_found"]) & _fill_false_bool(df["home_starter_found"])).mean()) * 100
        print(f"Pct with both starters: {starter_pct:.2f}")

    if "away_projected_lineup_found" in df.columns and "home_projected_lineup_found" in df.columns:
        lineup_pct = ((_fill_false_bool(df["away_projected_lineup_found"]) & _fill_false_bool(df["home_projected_lineup_found"])).mean()) * 100
        print(f"Pct with both projected lineups: {lineup_pct:.2f}")


def build_model_spine_game(
    schedule_df: pd.DataFrame,
    games_df: pd.DataFrame | None = None,
    weather_df: pd.DataFrame | None = None,
    parks_df: pd.DataFrame | None = None,
    projected_lineups_df: pd.DataFrame | None = None,
    confirmed_lineups_df: pd.DataFrame | None = None,
    starting_pitchers_df: pd.DataFrame | None = None,
    *,
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build one-row-per-game unified pregame spine.

    Parameters
    ----------
    schedule_df:
        Required normalized schedule table.
    games_df:
        Optional normalized games metadata table.
    weather_df:
        Optional normalized weather table.
    parks_df:
        Optional normalized parks reference table.
    projected_lineups_df:
        Optional normalized projected lineups table.
    confirmed_lineups_df:
        Optional normalized confirmed lineups table.
    starting_pitchers_df:
        Optional normalized starting pitchers table.

    Returns
    -------
    pandas.DataFrame
        Unified game spine.
    """
    base = _prep_schedule(schedule_df)

    # side keys for joining helpers
    base["away_team"] = base["away_team"].astype("string")
    base["home_team"] = base["home_team"].astype("string")
    base["away_team_norm"] = base["away_team"].map(_normalize_team_abbr).astype("string")
    base["home_team_norm"] = base["home_team"].map(_normalize_team_abbr).astype("string")

    games = _prep_games(_copy_or_empty(games_df))
    weather = _prep_weather(_copy_or_empty(weather_df))
    parks = _prep_parks(_copy_or_empty(parks_df))

    proj = _summarize_lineups(_copy_or_empty(projected_lineups_df), prefix="projected")
    conf = _summarize_lineups(_copy_or_empty(confirmed_lineups_df), prefix="confirmed")
    starters = _summarize_starters(_copy_or_empty(starting_pitchers_df))

    if not proj.empty:
        proj["team_norm"] = proj["team"].map(_normalize_team_abbr).astype("string")
    if not conf.empty:
        conf["team_norm"] = conf["team"].map(_normalize_team_abbr).astype("string")
    if not starters.empty:
        starters["team_norm"] = starters["team"].map(_normalize_team_abbr).astype("string")

    spine = base.merge(games, on="game_pk", how="left", validate="1:1")
    spine = spine.merge(weather, on="game_pk", how="left", validate="1:1")
    spine = spine.merge(parks, on="venue_id", how="left", validate="m:1")

    # team-side summaries
    away_proj = proj.rename(columns={"team": "away_team", "team_norm": "away_team_norm"}) if not proj.empty else proj
    home_proj = proj.rename(columns={"team": "home_team", "team_norm": "home_team_norm"}) if not proj.empty else proj
    away_conf = conf.rename(columns={"team": "away_team", "team_norm": "away_team_norm"}) if not conf.empty else conf
    home_conf = conf.rename(columns={"team": "home_team", "team_norm": "home_team_norm"}) if not conf.empty else conf
    away_starters = starters.rename(columns={"team": "away_team", "team_norm": "away_team_norm"}) if not starters.empty else starters
    home_starters = starters.rename(columns={"team": "home_team", "team_norm": "home_team_norm"}) if not starters.empty else starters

    if not away_proj.empty:
        away_proj = away_proj.rename(
            columns={
                "projected_lineup_status": "away_projected_lineup_status",
                "projected_lineup_spots": "away_projected_lineup_spots",
                "projected_distinct_players": "away_projected_distinct_players",
                "projected_first_batter": "away_projected_first_batter",
                "projected_last_batter": "away_projected_last_batter",
                "projected_lineup_found": "away_projected_lineup_found",
            }
        )
        spine = spine.merge(
            away_proj,
            on=["game_pk", "away_team_norm"],
            how="left",
            validate="1:1",
        ).drop(columns=["away_team"], errors="ignore")

    if not home_proj.empty:
        home_proj = home_proj.rename(
            columns={
                "projected_lineup_status": "home_projected_lineup_status",
                "projected_lineup_spots": "home_projected_lineup_spots",
                "projected_distinct_players": "home_projected_distinct_players",
                "projected_first_batter": "home_projected_first_batter",
                "projected_last_batter": "home_projected_last_batter",
                "projected_lineup_found": "home_projected_lineup_found",
            }
        )
        spine = spine.merge(
            home_proj,
            on=["game_pk", "home_team_norm"],
            how="left",
            validate="1:1",
        ).drop(columns=["home_team"], errors="ignore")

    if not away_conf.empty:
        away_conf = away_conf.rename(
            columns={
                "confirmed_lineup_status": "away_confirmed_lineup_status",
                "confirmed_lineup_spots": "away_confirmed_lineup_spots",
                "confirmed_distinct_players": "away_confirmed_distinct_players",
                "confirmed_first_batter": "away_confirmed_first_batter",
                "confirmed_last_batter": "away_confirmed_last_batter",
                "confirmed_lineup_found": "away_confirmed_lineup_found",
            }
        )
        spine = spine.merge(
            away_conf,
            on=["game_pk", "away_team_norm"],
            how="left",
            validate="1:1",
        ).drop(columns=["away_team"], errors="ignore")

    if not home_conf.empty:
        home_conf = home_conf.rename(
            columns={
                "confirmed_lineup_status": "home_confirmed_lineup_status",
                "confirmed_lineup_spots": "home_confirmed_lineup_spots",
                "confirmed_distinct_players": "home_confirmed_distinct_players",
                "confirmed_first_batter": "home_confirmed_first_batter",
                "confirmed_last_batter": "home_confirmed_last_batter",
                "confirmed_lineup_found": "home_confirmed_lineup_found",
            }
        )
        spine = spine.merge(
            home_conf,
            on=["game_pk", "home_team_norm"],
            how="left",
            validate="1:1",
        ).drop(columns=["home_team"], errors="ignore")

    if not away_starters.empty:
        away_starters = away_starters.rename(
            columns={
                "starter_status": "away_starter_status",
                "starter_pitcher_id": "away_starter_pitcher_id",
                "starter_pitcher_name": "away_starter_pitcher_name",
                "starter_throws": "away_starter_throws",
                "starter_found": "away_starter_found",
            }
        )
        spine = spine.merge(
            away_starters,
            on=["game_pk", "away_team_norm"],
            how="left",
            validate="1:1",
        ).drop(columns=["away_team"], errors="ignore")

    if not home_starters.empty:
        home_starters = home_starters.rename(
            columns={
                "starter_status": "home_starter_status",
                "starter_pitcher_id": "home_starter_pitcher_id",
                "starter_pitcher_name": "home_starter_pitcher_name",
                "starter_throws": "home_starter_throws",
                "starter_found": "home_starter_found",
            }
        )
        spine = spine.merge(
            home_starters,
            on=["game_pk", "home_team_norm"],
            how="left",
            validate="1:1",
        ).drop(columns=["home_team"], errors="ignore")

    # restore raw team columns from base if merge dropped them
    if "away_team_x" in spine.columns:
        spine["away_team"] = spine["away_team_x"]
        spine = spine.drop(columns=[c for c in ["away_team_x", "away_team_y"] if c in spine.columns])
    elif "away_team" not in spine.columns:
        spine["away_team"] = base["away_team"]

    if "home_team_x" in spine.columns:
        spine["home_team"] = spine["home_team_x"]
        spine = spine.drop(columns=[c for c in ["home_team_x", "home_team_y"] if c in spine.columns])
    elif "home_team" not in spine.columns:
        spine["home_team"] = base["home_team"]

    # final convenience flags
    if "away_projected_lineup_found" not in spine.columns:
        spine["away_projected_lineup_found"] = pd.Series(False, index=spine.index, dtype="boolean")
    if "home_projected_lineup_found" not in spine.columns:
        spine["home_projected_lineup_found"] = pd.Series(False, index=spine.index, dtype="boolean")
    if "away_confirmed_lineup_found" not in spine.columns:
        spine["away_confirmed_lineup_found"] = pd.Series(False, index=spine.index, dtype="boolean")
    if "home_confirmed_lineup_found" not in spine.columns:
        spine["home_confirmed_lineup_found"] = pd.Series(False, index=spine.index, dtype="boolean")
    if "away_starter_found" not in spine.columns:
        spine["away_starter_found"] = pd.Series(False, index=spine.index, dtype="boolean")
    if "home_starter_found" not in spine.columns:
        spine["home_starter_found"] = pd.Series(False, index=spine.index, dtype="boolean")

    spine["lineups_projected_both_found"] = (_fill_false_bool(spine["away_projected_lineup_found"]) & _fill_false_bool(spine["home_projected_lineup_found"])).astype("boolean")
    spine["lineups_confirmed_both_found"] = (_fill_false_bool(spine["away_confirmed_lineup_found"]) & _fill_false_bool(spine["home_confirmed_lineup_found"])).astype("boolean")
    spine["starters_both_found"] = (_fill_false_bool(spine["away_starter_found"]) & _fill_false_bool(spine["home_starter_found"])).astype("boolean")

    spine = spine.drop(columns=["away_team_norm", "home_team_norm"], errors="ignore")
    spine = spine.sort_values(["game_date", "scheduled_start_time_et", "game_pk"], kind="stable").reset_index(drop=True)

    if validate:
        validate_model_spine_game(spine)
    if verbose:
        summarize_model_spine_game(spine)

    return spine