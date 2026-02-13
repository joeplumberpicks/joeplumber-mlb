#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.paths import get_data_root, processed_dir, reference_dir
from src.weather.build_weather_game import build_weather_game, run_smoke_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build game-level weather table from hourly provider data.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--games", type=str, default=str(processed_dir() / "games.parquet"))
    parser.add_argument("--spine", type=str, default=str(processed_dir() / "model_spine_game.parquet"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--provider", type=str, default="visualcrossing")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--overrides", type=str, default=str(reference_dir() / "park_overrides.csv"))
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--include-spring-training", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(f"Using data root: {get_data_root()}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    if args.smoke_test:
        run_smoke_test(
            season=args.season,
            games_path=Path(args.games),
            spine_path=Path(args.spine),
            logger=logging.getLogger("build_weather_game"),
        )
        return

    output = Path(args.output) if args.output else (processed_dir() / f"weather_game_{args.season}.parquet")
    build_weather_game(
        season=args.season,
        start=args.start,
        end=args.end,
        games_path=Path(args.games),
        spine_path=Path(args.spine) if args.spine else None,
        out_path=output,
        provider=args.provider,
        max_games=args.max_games,
        allow_partial=args.allow_partial,
        include_spring_training=args.include_spring_training,
        overrides_path=Path(args.overrides),
        logger=logging.getLogger("build_weather_game"),
    )


if __name__ == "__main__":
    main()
