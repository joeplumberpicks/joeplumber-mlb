#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.pitcher_game_features import build_and_write_pitcher_game_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build starter-level pitcher game feature table.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--spine", type=str, default="data/processed/model_spine_game.parquet")
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="default: data/processed/statcast/statcast_game_context_{season}.parquet",
    )
    parser.add_argument(
        "--offense",
        type=str,
        default=None,
        help="default: data/processed/offense_discipline_team_{season}.parquet",
    )
    parser.add_argument(
        "--bullpen",
        type=str,
        default=None,
        help="default: data/processed/bullpen_game_{season}.parquet",
    )
    parser.add_argument("--pitcher-rolling", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="default: data/processed/model_features_pitcher_game_{season}.parquet",
    )
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("build_pitcher_game_features")

    context = Path(args.context) if args.context else Path(f"data/processed/statcast/statcast_game_context_{args.season}.parquet")
    offense = Path(args.offense) if args.offense else Path(f"data/processed/offense_discipline_team_{args.season}.parquet")
    bullpen = Path(args.bullpen) if args.bullpen else Path(f"data/processed/bullpen_game_{args.season}.parquet")
    output = Path(args.output) if args.output else Path(f"data/processed/model_features_pitcher_game_{args.season}.parquet")

    build_and_write_pitcher_game_features(
        season=args.season,
        spine_path=Path(args.spine),
        context_path=context,
        offense_path=offense if offense.exists() else None,
        bullpen_path=bullpen if bullpen.exists() else None,
        pitcher_rolling_path=Path(args.pitcher_rolling) if args.pitcher_rolling else None,
        output_path=output,
        start=args.start,
        end=args.end,
        allow_partial=args.allow_partial,
        logger=logger,
    )


if __name__ == "__main__":
    main()
