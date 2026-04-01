#!/usr/bin/env python3
"""
Ingest normalized MLB plate appearances.

Purpose
-------
Read provider-level plate appearance / event records and write normalized
plate appearances to the data lake.

This script is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering
- no target creation

Notes
-----
This runner is intentionally input-driven. It expects provider-level records
from a CSV or parquet file that were pulled separately from a source such as
MLB StatsAPI or Statcast.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingest.io import (
    log_kv,
    log_section,
    read_dataset,
    write_parquet,
)
from src.ingest.plate_appearances import build_plate_appearances
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest normalized MLB plate appearances.")
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2026")
    parser.add_argument("--date", type=str, default=None, help="Optional game date YYYY-MM-DD")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (csv/parquet) containing provider-level PA/event records.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="unknown",
        help="Source label, e.g. mlb_statsapi or statcast.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/project.yaml",
        help="Path to project config",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Accepted for runner compatibility; currently does not alter behavior.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/ingest/ingest_plate_appearances.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    log_kv("data_root", dirs["data_root"])
    log_kv("raw_dir", dirs["raw_dir"])
    log_kv("processed_dir", dirs["processed_dir"])
    log_kv("reference_dir", dirs["reference_dir"])

    records = read_dataset(args.input)

    df = build_plate_appearances(
        records=records,
        source=args.source,
        validate=True,
        verbose=True,
    )

    raw_live_dir = Path(dirs["raw_dir"]) / "live"
    processed_by_season_dir = Path(dirs["processed_dir"]) / "by_season"

    latest_out = raw_live_dir / f"plate_appearances_{args.season}.parquet"
    write_parquet(df, latest_out)

    if args.date:
        dated_out = raw_live_dir / f"plate_appearances_{args.season}_{args.date}.parquet"
        write_parquet(df, dated_out)
    else:
        season_out = processed_by_season_dir / f"plate_appearances_{args.season}.parquet"
        write_parquet(df, season_out)


if __name__ == "__main__":
    main()
