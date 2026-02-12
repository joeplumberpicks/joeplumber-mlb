#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.hitter_pitchtype import build_and_write_hitter_pitchtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Statcast hitter pitch-type features by pitcher handedness."
    )
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2024")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional input parquet path (default: data/raw/statcast/pitches_{season}.parquet)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output parquet path (default: data/processed/statcast/hitter_pitchtype_{season}.parquet)",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("build_hitter_pitchtype")

    input_path = Path(args.input) if args.input else Path(f"data/raw/statcast/pitches_{args.season}.parquet")
    output_path = Path(args.output) if args.output else Path(
        f"data/processed/statcast/hitter_pitchtype_{args.season}.parquet"
    )

    build_and_write_hitter_pitchtype(
        season=args.season,
        input_path=input_path,
        output_path=output_path,
        logger=logger,
    )


if __name__ == "__main__":
    main()
