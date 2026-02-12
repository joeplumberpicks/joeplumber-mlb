#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train_moneyline import run_smoke_test, train_moneyline_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Moneyline (home win) model from game features + targets.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Features parquet path (default: data/processed/model_features_game_{season}.parquet)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Targets parquet path (default: data/processed/targets/targets_game_{season}.parquet)",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="Model directory (default: data/models/moneyline_{season}/)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny synthetic smoke test for the numpy fallback trainer and exit",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing model artifacts")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("train_moneyline")

    if args.smoke_test:
        run_smoke_test(logger)
        return

    features_path = Path(args.features) if args.features else Path(
        f"data/processed/model_features_game_{args.season}.parquet"
    )
    targets_path = Path(args.targets) if args.targets else Path(
        f"data/processed/targets/targets_game_{args.season}.parquet"
    )
    model_dir = Path(args.model_out) if args.model_out else Path(f"data/models/moneyline_{args.season}")

    train_moneyline_from_paths(
        season=args.season,
        features_path=features_path,
        targets_path=targets_path,
        model_dir=model_dir,
        force=args.force,
        logger=logger,
    )


if __name__ == "__main__":
    main()
