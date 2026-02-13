#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.build_hitter_game_targets import build_and_write_hitter_game_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hitter-game targets table.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--events", type=str, default="data/processed/events_pa.parquet")
    p.add_argument("--games", type=str, default="data/processed/games.parquet")
    p.add_argument("--output", type=str, default=None, help="default data/processed/targets/targets_hitter_game_{season}.parquet")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("build_hitter_game_targets")
    output = Path(args.output) if args.output else Path(f"data/processed/targets/targets_hitter_game_{args.season}.parquet")

    build_and_write_hitter_game_targets(
        season=args.season,
        events_path=Path(args.events) if args.events else None,
        games_path=Path(args.games) if args.games else None,
        output_path=output,
        start=args.start,
        end=args.end,
        logger=logger,
    )


if __name__ == "__main__":
    main()
