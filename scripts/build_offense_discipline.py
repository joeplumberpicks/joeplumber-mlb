#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.offense_discipline import build_and_write_offense_discipline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offense discipline rolling features from Statcast pitches.")
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2024")
    parser.add_argument("--start", type=str, default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Optional end date YYYY-MM-DD")
    parser.add_argument(
        "--pitches",
        type=str,
        default=None,
        help="Pitches parquet path (default: data/raw/statcast/pitches_{season}.parquet)",
    )
    parser.add_argument(
        "--games",
        type=str,
        default=None,
        help="Optional games parquet path for batting-team inference (default: data/processed/games.parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/processed/offense_discipline_team_{season}.parquet)",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("build_offense_discipline")

    pitches_path = Path(args.pitches) if args.pitches else Path(f"data/raw/statcast/pitches_{args.season}.parquet")
    games_path = Path(args.games) if args.games else Path("data/processed/games.parquet")
    output_path = Path(args.output) if args.output else Path(
        f"data/processed/offense_discipline_team_{args.season}.parquet"
    )

    build_and_write_offense_discipline(
        season=args.season,
        pitches_path=pitches_path,
        games_path=games_path if games_path.exists() else None,
        output_path=output_path,
        start=args.start,
        end=args.end,
        logger=logger,
    )


if __name__ == "__main__":
    main()
