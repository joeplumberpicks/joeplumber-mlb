#!/usr/bin/env python3
"""
Ingest normalized MLB game weather.

Purpose
-------
Build game-level weather for a date range or full season and write normalized
weather outputs to the data lake.

This script is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering
- no target creation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingest.io import log_kv, log_section, write_parquet
from src.ingest.schedule import build_schedule_games
from src.ingest.weather import build_weather_games
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest normalized MLB weather.")
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2026")
    parser.add_argument("--start-date", type=str, default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="Optional end date YYYY-MM-DD")
    parser.add_argument(
        "--game-types",
        nargs="+",
        default=["R", "S"],
        help="MLB game types, e.g. R S",
    )
    parser.add_argument(
        "--sport-id",
        type=int,
        default=1,
        help="MLB sport id, default 1",
    )
    parser.add_argument(
        "--metar-hours",
        type=float,
        default=3.0,
        help="METAR lookback window in hours, default 3.0",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project config",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for runner compatibility; currently does not alter behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/ingest/ingest_weather_games.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    log_kv("data_root", dirs["data_root"])
    log_kv("raw_dir", dirs["raw_dir"])
    log_kv("processed_dir", dirs["processed_dir"])
    log_kv("reference_dir", dirs["reference_dir"])

    schedule_df = build_schedule_games(
        season=args.season,
        start_date=args.start_date,
        end_date=args.end_date,
        game_types=args.game_types,
        sport_id=args.sport_id,
        validate=True,
        verbose=True,
    )

    df = build_weather_games(
        schedule_df=schedule_df,
        hours=args.metar_hours,
        validate=True,
        verbose=True,
    )

    raw_live_dir = Path(dirs["raw_dir"]) / "live"
    processed_by_season_dir = Path(dirs["processed_dir"]) / "by_season"

    latest_out = raw_live_dir / f"weather_game_{args.season}.parquet"
    write_parquet(df, latest_out)

    if args.start_date and (args.end_date is None or args.end_date == args.start_date):
        dated_out = raw_live_dir / f"weather_game_{args.season}_{args.start_date}.parquet"
        write_parquet(df, dated_out)
    elif args.start_date and args.end_date:
        dated_out = raw_live_dir / f"weather_game_{args.season}_{args.start_date}_{args.end_date}.parquet"
        write_parquet(df, dated_out)
    else:
        season_out = processed_by_season_dir / f"weather_game_{args.season}.parquet"
        write_parquet(df, season_out)


if __name__ == "__main__":
    main()
