#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.infer_hitter_hit1p import infer_hitter_hit1p_from_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hitter 1+ hit model inference.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--features", type=str, default=None, help="default data/processed/model_features_hitter_game_{season}.parquet")
    p.add_argument("--model-dir", type=str, default=None, help="default data/models/hitter_hit1p_{season}/")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    a = parse_args()
    logger = logging.getLogger("infer_hitter_hit1p")

    features = Path(a.features) if a.features else Path(f"data/processed/model_features_hitter_game_{a.season}.parquet")
    model_dir = Path(a.model_dir) if a.model_dir else Path(f"data/models/hitter_hit1p_{a.season}")
    output = Path(f"data/outputs/hitter_hit1p_preds_{a.season}.csv")

    infer_hitter_hit1p_from_paths(
        season=a.season,
        features_path=features,
        model_dir=model_dir,
        output_path=output,
        start=a.start,
        end=a.end,
        logger=logger,
    )


if __name__ == "__main__":
    main()
