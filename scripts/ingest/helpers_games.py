from __future__ import annotations

"""Build game-level raw table from plate appearance raw data."""

from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount, require_files
from src.utils.io import read_parquet, write_parquet


REQUIRED_GAME_COLUMNS = [
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "season",
    "home_sp_id",
    "away_sp_id",
    "park_id",
]


def build_games_from_pa(dirs: dict[str, Path], season: int) -> Path:
    """Build one row per game_pk from raw PA table."""
    raw_by_season = dirs["raw_dir"] / "by_season"
    pa_path = raw_by_season / f"pa_{season}.parquet"
    require_files([pa_path], f"build_games_from_pa_{season}")

    pa_df = read_parquet(pa_path)
    for col in ["game_pk", "game_date", "home_team", "away_team"]:
        if col not in pa_df.columns:
            pa_df[col] = pd.NA

    games_df = (
        pa_df[["game_pk", "game_date", "home_team", "away_team"]]
        .dropna(subset=["game_pk"])
        .drop_duplicates(subset=["game_pk"]) 
        .copy()
    )
    games_df["game_date"] = pd.to_datetime(games_df["game_date"], errors="coerce").dt.date
    games_df["season"] = season
    games_df["home_sp_id"] = pd.NA
    games_df["away_sp_id"] = pd.NA
    games_df["park_id"] = pd.NA

    for col in REQUIRED_GAME_COLUMNS:
        if col not in games_df.columns:
            games_df[col] = pd.NA

    games_df = games_df[REQUIRED_GAME_COLUMNS]

    out_path = raw_by_season / f"games_{season}.parquet"
    print_rowcount(f"games_{season}", games_df)
    print(f"Writing to: {out_path.resolve()}")
    write_parquet(games_df, out_path)
    return out_path
