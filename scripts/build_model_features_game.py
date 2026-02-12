#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.merge_game_features import build_and_write_model_features_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged game-level model features table.")
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
        "--context",
        type=str,
        default=None,
        help="Statcast game context path (default: data/processed/statcast/statcast_game_context_{season}.parquet)",
    )
    parser.add_argument(
        "--bullpen",
        type=str,
        default=None,
        help="Bullpen context path (default when --with-bullpen: data/processed/bullpen_game_{season}.parquet)",
    )
    parser.add_argument(
        "--offense",
        type=str,
        default=None,
        help="Offense discipline path (default when --with-offense: data/processed/offense_discipline_team_{season}.parquet)",
    )
    parser.add_argument(
        "--pitches",
        type=str,
        default=None,
        help="Optional pitches parquet path for pitcher-hand fallback lookup (default: data/raw/statcast/pitches_{season}.parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/processed/model_features_game_{season}.parquet)",
    )
    parser.add_argument("--allow-partial", action="store_true", help="Allow <90% starter-context join coverage.")
    parser.add_argument("--with-bullpen", action="store_true", help="Enable optional bullpen context merge.")
    parser.add_argument("--with-offense", action="store_true", help="Enable optional offense discipline merge.")
    parser.add_argument("--with-weather", action="store_true", help="Enable weather_game merge.")
    parser.add_argument("--with-weather-factors", action="store_true", help="Enable weather_factors_game merge.")
    parser.add_argument("--weather", type=str, default=None, help="Weather game path override.")
    parser.add_argument("--weather-factors", type=str, default=None, help="Weather factors game path override.")
    parser.add_argument("--with-park", action="store_true", help="Enable park_game merge.")
    parser.add_argument("--park", type=str, default=None, help="Park game path override.")
    parser.add_argument("--with-park-factors", action="store_true", help="Enable park_factors_game merge.")
    parser.add_argument("--park-factors", type=str, default=None, help="Park factors path override.")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("build_model_features_game")

    spine_path = Path(args.spine)
    context_path = Path(args.context) if args.context else Path(
        f"data/processed/statcast/statcast_game_context_{args.season}.parquet"
    )
    output_path = Path(args.output) if args.output else Path(
        f"data/processed/model_features_game_{args.season}.parquet"
    )

    bullpen_path: Path | None = None
    if args.with_bullpen:
        bullpen_path = Path(args.bullpen) if args.bullpen else Path(f"data/processed/bullpen_game_{args.season}.parquet")
        if not bullpen_path.exists():
            raise FileNotFoundError(
                "--with-bullpen was set but bullpen parquet is missing: "
                f"{bullpen_path}. Build it with scripts/build_bullpen_context.py first."
            )

    offense_path: Path | None = None
    if args.with_offense:
        offense_path = Path(args.offense) if args.offense else Path(
            f"data/processed/offense_discipline_team_{args.season}.parquet"
        )
        if not offense_path.exists():
            raise FileNotFoundError(
                "--with-offense was set but offense discipline parquet is missing: "
                f"{offense_path}. Build it with scripts/build_offense_discipline.py first."
            )

    pitches_path = Path(args.pitches) if args.pitches else Path(f"data/raw/statcast/pitches_{args.season}.parquet")

    weather_path: Path | None = None
    if args.with_weather:
        weather_path = Path(args.weather) if args.weather else Path(f"data/processed/weather_game_{args.season}.parquet")
        if not weather_path.exists():
            raise FileNotFoundError(
                "--with-weather was set but weather game parquet is missing: "
                f"{weather_path}. Build it with scripts/build_weather_game.py first."
            )

    weather_factors_path: Path | None = None
    if args.with_weather_factors:
        weather_factors_path = (
            Path(args.weather_factors)
            if args.weather_factors
            else Path(f"data/processed/weather_factors_game_{args.season}.parquet")
        )
        if not weather_factors_path.exists():
            raise FileNotFoundError(
                "--with-weather-factors was set but weather factors parquet is missing: "
                f"{weather_factors_path}. Build it with scripts/train_weather_factors.py first."
            )


    park_path: Path | None = None
    if args.with_park:
        park_path = Path(args.park) if args.park else Path(f"data/processed/park_game_{args.season}.parquet")
        if not park_path.exists():
            raise FileNotFoundError(
                "--with-park was set but park_game parquet is missing: "
                f"{park_path}. Build it with scripts/build_park_game.py first."
            )

    park_factors_path: Path | None = None
    if args.with_park_factors:
        park_factors_path = (
            Path(args.park_factors)
            if args.park_factors
            else Path(f"data/processed/park_factors_game_{args.season}.parquet")
        )
        if not park_factors_path.exists():
            raise FileNotFoundError(
                "--with-park-factors was set but park_factors parquet is missing: "
                f"{park_factors_path}. Build it with scripts/train_park_factors.py first."
            )

    build_and_write_model_features_game(
        season=args.season,
        spine_path=spine_path,
        context_path=context_path,
        output_path=output_path,
        bullpen_path=bullpen_path,
        offense_path=offense_path,
        weather_game_path=weather_path,
        weather_factors_path=weather_factors_path,
        park_game_path=park_path,
        park_factors_path=park_factors_path,
        pitches_path=pitches_path if pitches_path.exists() else None,
        start=args.start,
        end=args.end,
        allow_partial=args.allow_partial,
        logger=logger,
    )


if __name__ == "__main__":
    main()
