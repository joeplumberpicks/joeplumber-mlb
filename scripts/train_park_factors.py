#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.parks.train_park_factors import run_smoke_test, train_park_factors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train leakage-safe park factors and write game-level park effects.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--park-game", type=str, default=None)
    parser.add_argument("--targets-game", type=str, default=None)
    parser.add_argument("--targets-hitter", type=str, default=None)
    parser.add_argument("--events", type=str, default="data/processed/events_pa.parquet")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    log = logging.getLogger("train_park_factors")

    if args.smoke_test:
        run_smoke_test(log)
        return

    park_game_path = Path(args.park_game) if args.park_game else Path(f"data/processed/park_game_{args.season}.parquet")
    targets_game_path = (
        Path(args.targets_game) if args.targets_game else Path(f"data/processed/targets/targets_game_{args.season}.parquet")
    )
    targets_hitter_path = (
        Path(args.targets_hitter)
        if args.targets_hitter
        else Path(f"data/processed/targets/targets_hitter_game_{args.season}.parquet")
    )
    output_path = (
        Path(args.output) if args.output else Path(f"data/processed/park_factors_game_{args.season}.parquet")
    )

    train_park_factors(
        season=args.season,
        park_game_path=park_game_path,
        targets_game_path=targets_game_path,
        targets_hitter_path=targets_hitter_path,
        events_path=Path(args.events),
        output_path=output_path,
        start=args.start,
        end=args.end,
        logger=log,
    )


if __name__ == "__main__":
    main()
