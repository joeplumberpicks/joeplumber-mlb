#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.statcast_ingestion import (
    normalize_statcast_pitches,
    pull_statcast_pitches,
    resolve_date_window,
    validate_statcast_pitches,
)
from src.utils.drive import resolve_data_dirs
from src.utils.io import load_config, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull pitch-by-pitch Statcast data and save parquet output."
    )
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2024")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--chunk-days", type=int, default=7, help="Date chunk size for pulls")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per chunk")
    parser.add_argument("--backoff-seconds", type=float, default=2.0, help="Exponential retry backoff base")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    setup_logging()
    logger = logging.getLogger("pull_statcast_pitches")
    args = parse_args()

    start, end = resolve_date_window(args.season, args.start, args.end)
    logger.info("Starting Statcast ingestion for season=%s window=%s..%s", args.season, start, end)

    raw_df = pull_statcast_pitches(
        start,
        end,
        chunk_days=args.chunk_days,
        max_retries=args.max_retries,
        backoff_seconds=args.backoff_seconds,
        logger=logger,
    )
    normalized = normalize_statcast_pitches(raw_df, season=args.season)

    summary = validate_statcast_pitches(normalized)
    logger.info("Validation summary")
    logger.info("rows=%d", summary.rows)
    logger.info("unique_batters=%d", summary.unique_batters)
    logger.info("unique_pitchers=%d", summary.unique_pitchers)
    logger.info("null_rate_pitch_type=%.4f", summary.null_rate_pitch_type)
    logger.info("null_rate_plate_x=%.4f", summary.null_rate_plate_x)
    logger.info("null_rate_plate_z=%.4f", summary.null_rate_plate_z)

    config = load_config()
    data_dirs = resolve_data_dirs(config)
    output_path = data_dirs["raw"] / "statcast" / f"pitches_{args.season}.parquet"
    write_parquet(normalized, output_path)

    logger.info("Wrote Statcast pitch data to %s", output_path)


if __name__ == "__main__":
    main()
