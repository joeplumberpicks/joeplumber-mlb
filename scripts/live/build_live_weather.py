from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingest.ingest_weather_game import write_weather_for_live_date
from scripts.reference.build_weather_game import build_weather_game_table, normalize_weather_frame
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical date-scoped live weather artifact for game-level daily engines.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--skip-master", action="store_true", help="Skip rebuild of processed/weather_game.parquet.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_live_weather.log")
    log_header("scripts/live/build_live_weather.py", repo_root, config_path, dirs)

    raw_out = write_weather_for_live_date(dirs=dirs, season=args.season, date=args.date)
    raw_df = read_parquet(raw_out)
    norm_df = normalize_weather_frame(raw_df, fallback_season=args.season)
    processed_live_out = dirs["processed_dir"] / "live" / f"weather_game_{args.season}_{args.date}.parquet"
    write_parquet(norm_df, processed_live_out)
    print(f"Row count [weather_game_{args.season}_{args.date}_processed_live]: {len(norm_df):,}")
    print(f"canonical_weather_out={processed_live_out.resolve()}")

    if not args.skip_master:
        master_path, _ = build_weather_game_table(
            dirs=dirs,
            season_start=args.season,
            season_end=args.season,
            date=None,
        )
        print(f"weather_master_out={master_path}")


if __name__ == "__main__":
    main()
