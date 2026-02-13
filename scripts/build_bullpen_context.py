#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.bullpen_context import build_and_write_bullpen_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bullpen / relief context features per team-game.")
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2024")
    parser.add_argument("--start", type=str, default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Optional end date YYYY-MM-DD")
    parser.add_argument(
        "--spine",
        type=str,
        default="data/processed/model_spine_game.parquet",
        help="Canonical spine parquet path",
    )
    parser.add_argument(
        "--pitcher-game",
        type=str,
        default=None,
        help="Optional pitcher/team pitching game parquet override path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/processed/bullpen_game_{season}.parquet)",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("build_bullpen_context")

    spine_path = Path(args.spine)
    pitcher_game_path = Path(args.pitcher_game) if args.pitcher_game else None
    output_path = Path(args.output) if args.output else Path(f"data/processed/bullpen_game_{args.season}.parquet")

    build_and_write_bullpen_context(
        season=args.season,
        spine_path=spine_path,
        output_path=output_path,
        start=args.start,
        end=args.end,
        pitcher_game_path=pitcher_game_path,
        logger=logger,
    )


if __name__ == "__main__":
    main()
