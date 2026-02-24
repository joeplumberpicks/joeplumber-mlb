from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount, require_columns
from src.utils.io import read_parquet, write_parquet

GAMES_COLUMNS = [
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "home_sp_id",
    "away_sp_id",
    "park_id",
    "season",
]
PA_COLUMNS = ["game_pk", "pa_id", "batter_id", "pitcher_id", "event_type", "season"]
WEATHER_COLUMNS = ["game_pk", "temperature_f", "wind_mph", "wind_dir", "season"]
PARK_COLUMNS = ["park_id", "park_name", "park_factor", "season"]


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalize_pa_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.copy()

    if "batter_id" not in out.columns and "batter" in out.columns:
        out["batter_id"] = out["batter"]
    if "pitcher_id" not in out.columns and "pitcher" in out.columns:
        out["pitcher_id"] = out["pitcher"]

    if "pa_id" not in out.columns:
        if "at_bat_number" in out.columns and "game_pk" in out.columns:
            out["pa_id"] = out["game_pk"].astype(str) + "-" + out["at_bat_number"].astype(str)
        else:
            out["pa_id"] = pd.RangeIndex(start=0, stop=len(out), step=1).astype(str)

    if "season" not in out.columns:
        out["season"] = season

    if "event_type" not in out.columns and "events" in out.columns:
        out["event_type"] = out["events"]

    for col in PA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out


def _normalize_parks_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.copy()

    if "park_id" not in out.columns and "venue_id" in out.columns:
        out["park_id"] = out["venue_id"]
    if "venue_id" not in out.columns and "park_id" in out.columns:
        out["venue_id"] = out["park_id"]

    if "season" not in out.columns:
        out["season"] = season

    for col in PARK_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out


def load_or_placeholder(raw_path: Path, columns: list[str], label: str, season: int) -> pd.DataFrame:
    if raw_path.exists():
        df = read_parquet(raw_path)
        if label == "plate_appearances":
            df = _normalize_pa_df(df, season)
            print_rowcount("plate_appearances_normalized", df)
        if label == "parks":
            df = _normalize_parks_df(df, season)
            print_rowcount("parks_normalized", df)
        require_columns(df, columns, label)
        print_rowcount(label, df)
        return df

    logging.warning("Missing raw input for %s season %s. Creating empty placeholder: %s", label, season, raw_path)
    df = _empty_df(columns)
    print_rowcount(label, df)
    return df


def build_spine_for_season(season: int, dirs: dict[str, Path], force: bool = False) -> dict[str, Path]:
    processed_by_season = dirs["processed_dir"] / "by_season"
    raw_by_season = dirs["raw_dir"] / "by_season"
    processed_by_season.mkdir(parents=True, exist_ok=True)

    raw_games = raw_by_season / f"games_{season}.parquet"
    raw_pa = raw_by_season / f"pa_{season}.parquet"
    raw_weather = raw_by_season / f"weather_game_{season}.parquet"
    raw_parks = raw_by_season / f"parks_{season}.parquet"

    out_games = processed_by_season / f"games_{season}.parquet"
    out_pa = processed_by_season / f"pa_{season}.parquet"
    out_weather = processed_by_season / f"weather_game_{season}.parquet"
    out_parks = processed_by_season / f"parks_{season}.parquet"

    if out_games.exists() and not force:
        logging.info("Season outputs already exist and force=False; reloading processed tables for season %s", season)
        games_df = read_parquet(out_games)
        pa_df = read_parquet(out_pa)
        weather_df = read_parquet(out_weather)
        parks_df = read_parquet(out_parks)
    else:
        games_df = load_or_placeholder(raw_games, GAMES_COLUMNS, "games", season)
        pa_df = load_or_placeholder(raw_pa, PA_COLUMNS, "plate_appearances", season)
        weather_df = load_or_placeholder(raw_weather, WEATHER_COLUMNS, "weather", season)
        parks_df = load_or_placeholder(raw_parks, PARK_COLUMNS, "parks", season)

        if not games_df.empty and "season" not in games_df:
            games_df["season"] = season

        for df in [games_df, pa_df, weather_df, parks_df]:
            if "season" in df.columns and df["season"].isna().any():
                df["season"] = df["season"].fillna(season)

        pa_df = pa_df[PA_COLUMNS].copy()
        print_rowcount("plate_appearances_processed", pa_df)

        print(f"Writing to: {out_games.resolve()}")
        write_parquet(games_df, out_games)
        print(f"Writing to: {out_pa.resolve()}")
        write_parquet(pa_df, out_pa)
        print(f"Writing to: {out_weather.resolve()}")
        write_parquet(weather_df, out_weather)
        print(f"Writing to: {out_parks.resolve()}")
        write_parquet(parks_df, out_parks)

    return {"games": out_games, "pa": out_pa, "weather": out_weather, "parks": out_parks}


def _apply_park_venue_mapping(model_spine: pd.DataFrame, processed_by_season: Path, seasons: list[int]) -> pd.DataFrame:
    if "park_id" not in model_spine.columns:
        model_spine["park_id"] = pd.NA
    if "venue_id" not in model_spine.columns:
        model_spine["venue_id"] = pd.NA

    parks_frames: list[pd.DataFrame] = []
    for season in seasons:
        parks_path = processed_by_season / f"parks_{season}.parquet"
        if parks_path.exists():
            parks_df = read_parquet(parks_path)
            parks_df = _normalize_parks_df(parks_df, season)
            parks_frames.append(parks_df)

    if not parks_frames:
        return model_spine

    parks_all = pd.concat(parks_frames, ignore_index=True).drop_duplicates()
    venue_to_park = {}
    park_to_venue = {}

    if "venue_id" in parks_all.columns and "park_id" in parks_all.columns:
        for _, row in parks_all.dropna(subset=["venue_id", "park_id"]).iterrows():
            venue_to_park[row["venue_id"]] = row["park_id"]
            park_to_venue[row["park_id"]] = row["venue_id"]

    park_missing = model_spine["park_id"].isna() & model_spine["venue_id"].notna()
    model_spine.loc[park_missing, "park_id"] = model_spine.loc[park_missing, "venue_id"].map(venue_to_park)

    venue_missing = model_spine["venue_id"].isna() & model_spine["park_id"].notna()
    model_spine.loc[venue_missing, "venue_id"] = model_spine.loc[venue_missing, "park_id"].map(park_to_venue)

    return model_spine


def build_model_spine(dirs: dict[str, Path], seasons: list[int]) -> Path:
    processed_by_season = dirs["processed_dir"] / "by_season"
    model_spine_path = dirs["processed_dir"] / "model_spine_game.parquet"

    frames: list[pd.DataFrame] = []
    for season in seasons:
        season_path = processed_by_season / f"games_{season}.parquet"
        if season_path.exists():
            df = read_parquet(season_path)
            if "season" not in df.columns:
                df["season"] = season
            frames.append(df)

    if frames:
        model_spine = pd.concat(frames, ignore_index=True)
    else:
        model_spine = _empty_df(GAMES_COLUMNS)

    model_spine = _apply_park_venue_mapping(model_spine, processed_by_season, seasons)

    keep_cols = [
        "game_pk",
        "game_date",
        "home_team",
        "away_team",
        "home_sp_id",
        "away_sp_id",
        "park_id",
        "venue_id",
        "season",
    ]
    for col in keep_cols:
        if col not in model_spine.columns:
            model_spine[col] = pd.NA
    model_spine = model_spine[keep_cols]

    print_rowcount("model_spine_game", model_spine)
    print(f"Writing to: {model_spine_path.resolve()}")
    write_parquet(model_spine, model_spine_path)
    return model_spine_path
