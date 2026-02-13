#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.build_game_targets import build_and_write_game_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standardized game-level targets table.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--games", type=str, default="data/processed/games.parquet")
    parser.add_argument("--game-runs", type=str, default=None)
    parser.add_argument("--events", type=str, default="data/processed/events_pa.parquet")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("build_game_targets")

    games_path = Path(args.games)
    game_runs_path = (
        Path(args.game_runs)
        if args.game_runs is not None
        else (Path("data/processed/game_runs.parquet") if Path("data/processed/game_runs.parquet").exists() else None)
    )
    events_path = Path(args.events) if args.events else None
    output_path = Path(args.output) if args.output else Path(
        f"data/processed/targets/targets_game_{args.season}.parquet"
    )

    build_and_write_game_targets(
        season=args.season,
        games_path=games_path,
        output_path=output_path,
        game_runs_path=game_runs_path,
        events_path=events_path,
        start=args.start,
        end=args.end,
        logger=logger,
    )


if __name__ == "__main__":
    main()
