#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import statsapi

from src.utils.drive import resolve_data_dirs
from src.utils.io import load_config, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build foundational MLB spine tables.")
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2024")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    return parser.parse_args()


def fetch_schedule_games(season: int, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    schedule = statsapi.schedule(start_date=start_date, end_date=end_date, season=season)

    records: list[dict] = []
    for game in schedule:
        records.append(
            {
                "game_pk": game.get("game_id"),
                "season": season,
                "game_date": game.get("game_date"),
                "status": game.get("status"),
                "venue_id": game.get("venue_id"),
                "venue_name": game.get("venue_name"),
                "home_team_id": game.get("home_id"),
                "home_team_name": game.get("home_name"),
                "away_team_id": game.get("away_id"),
                "away_team_name": game.get("away_name"),
            }
        )

    games = pd.DataFrame.from_records(records)
    if games.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "season",
                "game_date",
                "status",
                "venue_id",
                "venue_name",
                "home_team_id",
                "home_team_name",
                "away_team_id",
                "away_team_name",
            ]
        )

    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    return games


def dedupe_games(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return games.copy()

    sorted_games = games.sort_values(
        by=["game_pk", "game_date", "status", "home_team_id", "away_team_id"],
        na_position="last",
        kind="mergesort",
    )
    deduped = sorted_games.drop_duplicates(subset=["game_pk"], keep="first")
    return deduped.reset_index(drop=True)


def build_parks(games: pd.DataFrame) -> pd.DataFrame:
    if games.empty:
        return pd.DataFrame(columns=["venue_id", "venue_name"])

    parks = (
        games.loc[:, ["venue_id", "venue_name"]]
        .dropna(subset=["venue_id"])
        .sort_values(by=["venue_id", "venue_name"], kind="mergesort")
        .drop_duplicates(subset=["venue_id"], keep="first")
        .reset_index(drop=True)
    )
    return parks


def build_pa_stub(games: pd.DataFrame) -> pd.DataFrame:
    pa_columns = ["game_pk", "batting_team_id", "batting_team_name", "plate_appearances"]
    if games.empty:
        return pd.DataFrame(columns=pa_columns)

    pa = pd.DataFrame(
        {
            "game_pk": games["game_pk"],
            "batting_team_id": pd.Series([pd.NA] * len(games), dtype="Int64"),
            "batting_team_name": pd.Series([pd.NA] * len(games), dtype="string"),
            "plate_appearances": pd.Series([pd.NA] * len(games), dtype="Int64"),
        }
    )
    return pa[pa_columns]


def build_weather_game_stub(games: pd.DataFrame) -> pd.DataFrame:
    weather_columns = ["game_pk", "temp_f", "wind_mph", "wind_dir", "conditions"]
    if games.empty:
        return pd.DataFrame(columns=weather_columns)

    weather = pd.DataFrame(
        {
            "game_pk": games["game_pk"],
            "temp_f": pd.Series([np.nan] * len(games), dtype="float64"),
            "wind_mph": pd.Series([np.nan] * len(games), dtype="float64"),
            "wind_dir": pd.Series([pd.NA] * len(games), dtype="string"),
            "conditions": pd.Series([pd.NA] * len(games), dtype="string"),
        }
    )
    return weather[weather_columns]


def validate_games(games: pd.DataFrame) -> None:
    required = {
        "game_pk",
        "season",
        "game_date",
        "status",
        "venue_id",
        "venue_name",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games missing required columns: {sorted(missing)}")

    if games["game_pk"].isna().any():
        raise ValueError("games contains null game_pk values")

    if games["game_pk"].duplicated().any():
        raise ValueError("games contains duplicate game_pk values after dedupe")


def validate_parks(parks: pd.DataFrame) -> None:
    required = {"venue_id", "venue_name"}
    missing = required - set(parks.columns)
    if missing:
        raise ValueError(f"parks missing required columns: {sorted(missing)}")

    if parks["venue_id"].duplicated().any():
        raise ValueError("parks contains duplicate venue_id values")


def validate_pa(pa: pd.DataFrame) -> None:
    required = {"game_pk", "batting_team_id", "batting_team_name", "plate_appearances"}
    missing = required - set(pa.columns)
    if missing:
        raise ValueError(f"pa missing required columns: {sorted(missing)}")


def validate_weather_game(weather_game: pd.DataFrame) -> None:
    required = {"game_pk", "temp_f", "wind_mph", "wind_dir", "conditions"}
    missing = required - set(weather_game.columns)
    if missing:
        raise ValueError(f"weather_game missing required columns: {sorted(missing)}")


def main() -> None:
    args = parse_args()

    config = load_config()
    dirs = resolve_data_dirs(config)
    processed_dir = dirs["processed"]

    games = fetch_schedule_games(args.season, args.start, args.end)
    games = dedupe_games(games)
    validate_games(games)

    parks = build_parks(games)
    pa = build_pa_stub(games)
    weather_game = build_weather_game_stub(games)

    validate_parks(parks)
    validate_pa(pa)
    validate_weather_game(weather_game)

    print(f"games rows: {len(games)}")
    print(f"parks rows: {len(parks)}")
    print(f"pa rows: {len(pa)}")
    print(f"weather_game rows: {len(weather_game)}")

    write_parquet(games, processed_dir / "games.parquet")
    write_parquet(pa, processed_dir / "pa.parquet")
    write_parquet(weather_game, processed_dir / "weather_game.parquet")
    write_parquet(parks, processed_dir / "parks.parquet")

    print(f"Wrote spine parquet files to: {processed_dir}")


if __name__ == "__main__":
    main()
