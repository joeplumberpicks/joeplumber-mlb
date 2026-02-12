#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train_pitcher_ks import run_smoke_test, train_pitcher_ks_from_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pitcher strikeout regression model.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--features", type=str, default=None, help="default data/processed/model_features_pitcher_game_{season}.parquet")
    p.add_argument("--targets", type=str, default=None, help="default data/processed/targets/targets_pitcher_game_{season}.parquet")
    p.add_argument("--model-out", type=str, default=None, help="default data/models/pitcher_ks_{season}/")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("train_pitcher_ks")

    if args.smoke_test:
        run_smoke_test(logger)
        return

    features = Path(args.features) if args.features else Path(f"data/processed/model_features_pitcher_game_{args.season}.parquet")
    targets = Path(args.targets) if args.targets else Path(f"data/processed/targets/targets_pitcher_game_{args.season}.parquet")
    model_out = Path(args.model_out) if args.model_out else Path(f"data/models/pitcher_ks_{args.season}")

    train_pitcher_ks_from_paths(
        season=args.season,
        features_path=features,
        targets_path=targets,
        model_dir=model_out,
        force=args.force,
        logger=logger,
    )


if __name__ == "__main__":
    main()
