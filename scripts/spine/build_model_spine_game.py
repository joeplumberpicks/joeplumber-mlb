#!/usr/bin/env python3
"""
Build unified game spine for Joe Plumber MLB Engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingest.io import log_kv, log_section, read_dataset, write_parquet
from src.spine import build_model_spine_game
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model_spine_game.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def _read_if_exists(path: Path):
    return read_dataset(path) if path.exists() else None


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/spine/build_model_spine_game.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    raw_live_dir = Path(dirs["raw_dir"]) / "live"
    processed_live_dir = Path(dirs["processed_dir"]) / "live"
    processed_by_season_dir = Path(dirs["processed_dir"]) / "by_season"

    if args.date:
        schedule_path = raw_live_dir / f"games_schedule_{args.season}_{args.date}.parquet"
        games_path = raw_live_dir / f"games_{args.season}_{args.date}.parquet"
        weather_path = raw_live_dir / f"weather_game_{args.season}_{args.date}.parquet"
        proj_path = raw_live_dir / f"projected_lineups_{args.season}_{args.date}.parquet"
        conf_path = raw_live_dir / f"confirmed_lineups_{args.season}_{args.date}.parquet"
        starters_path = raw_live_dir / f"starting_pitchers_{args.season}_{args.date}.parquet"
        parks_path = Path(dirs["processed_dir"]) / "parks.parquet"

        out_path = processed_live_dir / f"model_spine_game_{args.season}_{args.date}.parquet"
    else:
        schedule_path = processed_by_season_dir / f"games_schedule_{args.season}.parquet"
        games_path = processed_by_season_dir / f"games_{args.season}.parquet"
        weather_path = processed_by_season_dir / f"weather_game_{args.season}.parquet"
        proj_path = processed_by_season_dir / f"projected_lineups_{args.season}.parquet"
        conf_path = processed_by_season_dir / f"confirmed_lineups_{args.season}.parquet"
        starters_path = processed_by_season_dir / f"starting_pitchers_{args.season}.parquet"
        parks_path = Path(dirs["processed_dir"]) / "parks.parquet"

        out_path = processed_by_season_dir / f"model_spine_game_{args.season}.parquet"

    schedule_df = read_dataset(schedule_path)
    games_df = _read_if_exists(games_path)
    weather_df = _read_if_exists(weather_path)
    projected_df = _read_if_exists(proj_path)
    confirmed_df = _read_if_exists(conf_path)
    starters_df = _read_if_exists(starters_path)
    parks_df = _read_if_exists(parks_path)

    spine_df = build_model_spine_game(
        schedule_df=schedule_df,
        games_df=games_df,
        weather_df=weather_df,
        parks_df=parks_df,
        projected_lineups_df=projected_df,
        confirmed_lineups_df=confirmed_df,
        starting_pitchers_df=starters_df,
        validate=True,
        verbose=True,
    )

    write_parquet(spine_df, out_path)


if __name__ == "__main__":
    main()
