#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.infer_totals import infer_totals_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Totals inference from trained model artifacts.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Features parquet path (default: data/processed/model_features_game_{season}.parquet)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model artifact directory (default: data/models/totals_{season}/)",
    )
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("infer_totals")

    features_path = Path(args.features) if args.features else Path(
        f"data/processed/model_features_game_{args.season}.parquet"
    )
    model_dir = Path(args.model_dir) if args.model_dir else Path(f"data/models/totals_{args.season}")
    output_path = Path(f"data/outputs/totals_preds_{args.season}.csv")

    infer_totals_from_paths(
        season=args.season,
        features_path=features_path,
        model_dir=model_dir,
        output_path=output_path,
        start=args.start,
        end=args.end,
        logger=logger,
    )


if __name__ == "__main__":
    main()
