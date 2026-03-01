from __future__ import annotations

"""Run raw ingest flow for one season."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingest.build_parks_reference import build_parks_reference
from scripts.ingest.helpers_games import build_games_from_pa
from scripts.ingest.ingest_statcast_pa import ingest_statcast_pa
from scripts.ingest.ingest_weather_stub import write_weather_stub_for_games
from scripts.ingest.run_ingest_parks import _fetch_venues, _venues_to_df
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest raw Statcast season data into Drive-rooted lake.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "run_ingest_season.log")
    log_header("scripts/ingest/run_ingest_season.py", repo_root, config_path, dirs)
    print(
        f"Args: season={args.season}, start={args.start}, end={args.end}, "
        f"chunk_days={args.chunk_days}, force={args.force}"
    )

    ingest_statcast_pa(
        dirs=dirs,
        season=args.season,
        start=args.start,
        end=args.end,
        chunk_days=args.chunk_days,
        force=args.force,
    )
    build_games_from_pa(dirs=dirs, season=args.season)
    venues = _fetch_venues()
    parks_df = _venues_to_df(venues, args.season)
    parks_path = dirs["raw_dir"] / "by_season" / f"parks_{args.season}.parquet"
    print(f"Row count [parks_{args.season}]: {len(parks_df):,}")
    print(f"Writing to: {parks_path.resolve()}")
    from src.utils.io import write_parquet

    write_parquet(parks_df, parks_path)
    if len(parks_df) == 0:
        raise RuntimeError("Parks ingest returned 0 rows during season ingest.")
    build_parks_reference(dirs=dirs, season=args.season, repo_root=repo_root)
    write_weather_stub_for_games(dirs=dirs, season=args.season)


if __name__ == "__main__":
    main()
