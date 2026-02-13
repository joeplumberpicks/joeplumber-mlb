#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.parks.build_park_game import build_park_game
from src.utils.paths import get_data_root, processed_dir, reference_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic game-level park reference table.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--games", type=str, default=str(processed_dir() / "games.parquet"))
    parser.add_argument("--stadiums", type=str, default=str(reference_dir() / "mlb_stadiums.csv"))
    parser.add_argument("--overrides", type=str, default=str(reference_dir() / "park_overrides.csv"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-games", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    print(f"Using data root: {get_data_root()}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    output = Path(args.output) if args.output else (processed_dir() / f"park_game_{args.season}.parquet")
    build_park_game(
        season=args.season,
        start=args.start,
        end=args.end,
        games_path=Path(args.games),
        stadiums_path=Path(args.stadiums),
        overrides_path=Path(args.overrides),
        output_path=output,
        allow_partial=args.allow_partial,
        max_games=args.max_games,
        logger=logging.getLogger("build_park_game"),
    )


if __name__ == "__main__":
    main()
