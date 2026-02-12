#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.train_hitter_hr import run_smoke_test, train_hitter_hr_from_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hitter HR model.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--features", type=str, default=None, help="default data/processed/model_features_hitter_game_{season}.parquet")
    p.add_argument("--targets", type=str, default=None, help="default data/processed/targets/targets_hitter_game_{season}.parquet")
    p.add_argument("--model-out", type=str, default=None, help="default data/models/hitter_hr_{season}/")
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    a = parse_args()
    logger = logging.getLogger("train_hitter_hr")

    if a.smoke_test:
        run_smoke_test(logger)
        return

    features = Path(a.features) if a.features else Path(f"data/processed/model_features_hitter_game_{a.season}.parquet")
    targets = Path(a.targets) if a.targets else Path(f"data/processed/targets/targets_hitter_game_{a.season}.parquet")
    model_out = Path(a.model_out) if a.model_out else Path(f"data/models/hitter_hr_{a.season}")

    train_hitter_hr_from_paths(
        season=a.season,
        features_path=features,
        targets_path=targets,
        model_dir=model_out,
        force=a.force,
        logger=logger,
    )


if __name__ == "__main__":
    main()
