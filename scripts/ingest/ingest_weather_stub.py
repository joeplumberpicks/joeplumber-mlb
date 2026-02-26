from __future__ import annotations

"""Weather ingest stub that emits one row per game."""

from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount, require_files
from src.utils.io import read_parquet, write_parquet

WEATHER_COLUMNS = ["game_pk", "temperature_f", "wind_mph", "wind_dir", "season"]


def write_weather_stub_for_games(dirs: dict[str, Path], season: int) -> Path:
    """Write per-game weather placeholder rows so downstream joins remain stable."""
    raw_by_season = dirs["raw_dir"] / "by_season"
    games_path = raw_by_season / f"games_{season}.parquet"
    require_files([games_path], f"weather_stub_games_{season}")

    games_df = read_parquet(games_path)
    if "game_pk" not in games_df.columns:
        games_df["game_pk"] = pd.NA

    weather_df = games_df[["game_pk"]].drop_duplicates().copy()
    weather_df["temperature_f"] = pd.NA
    weather_df["wind_mph"] = pd.NA
    weather_df["wind_dir"] = pd.NA
    weather_df["season"] = season
    weather_df = weather_df[WEATHER_COLUMNS]

    out_path = raw_by_season / f"weather_game_{season}.parquet"
    print_rowcount(f"weather_game_{season}", weather_df)
    print(f"Writing to: {out_path.resolve()}")
    write_parquet(weather_df, out_path)
    return out_path
