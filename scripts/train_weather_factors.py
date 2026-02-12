#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.weather.train_weather_factors import run_smoke_test, train_weather_factors_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weather-factor models (HR/Totals/YRFI) and produce game-level deltas.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--features-weather", type=str, default=None)
    parser.add_argument("--targets-game", type=str, default=None)
    parser.add_argument("--events", type=str, default="data/processed/events_pa.parquet")
    parser.add_argument("--hitter-targets", type=str, default=None)
    parser.add_argument("--model-out", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--provider", type=str, default="visualcrossing")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    log = logging.getLogger("train_weather_factors")

    if args.smoke_test:
        run_smoke_test(log)
        return

    weather = Path(args.features_weather) if args.features_weather else Path(f"data/processed/weather_game_{args.season}.parquet")
    targets = Path(args.targets_game) if args.targets_game else Path(f"data/processed/targets/targets_game_{args.season}.parquet")
    model_out = Path(args.model_out) if args.model_out else Path(f"data/models/weather_factors_{args.season}")
    output = Path(args.output) if args.output else Path(f"data/processed/weather_factors_game_{args.season}.parquet")
    hitter_targets = (
        Path(args.hitter_targets)
        if args.hitter_targets
        else Path(f"data/processed/targets/targets_hitter_game_{args.season}.parquet")
    )

    train_weather_factors_from_paths(
        season=args.season,
        weather_path=weather,
        targets_game_path=targets,
        model_dir=model_out,
        output_path=output,
        events_path=Path(args.events),
        hitter_targets_path=hitter_targets,
        provider=args.provider,
        force=args.force,
        logger=log,
    )


if __name__ == "__main__":
    main()
