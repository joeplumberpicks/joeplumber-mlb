#!/usr/bin/env python3
"""
Ingest normalized MLB park reference data.

Purpose
-------
Read provider-level park / venue records and write normalized park reference
data to the data lake.

This script is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering
- no target creation

Notes
-----
This runner is intentionally input-driven. It expects provider-level records
from a CSV or parquet file that were pulled separately from a source such as
MLB StatsAPI, FanGraphs park factors, or a maintained reference file.
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
from src.ingest.parks import build_parks_reference
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest normalized MLB park reference data.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file (csv/parquet) containing provider-level park / venue records.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="unknown",
        help="Source label, e.g. mlb_statsapi or fangraphs.",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Optional season label for dated output naming.",
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

    log_section("scripts/ingest/ingest_parks.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    log_kv("data_root", dirs["data_root"])
    log_kv("raw_dir", dirs["raw_dir"])
    log_kv("processed_dir", dirs["processed_dir"])
    log_kv("reference_dir", dirs["reference_dir"])

    records = read_dataset(args.input)

    df = build_parks_reference(
        records=records,
        source=args.source,
        validate=True,
        verbose=True,
    )

    reference_dir = Path(dirs["reference_dir"])
    processed_dir = Path(dirs["processed_dir"])

    latest_ref_out = reference_dir / "parks.parquet"
    write_parquet(df, latest_ref_out)

    latest_processed_out = processed_dir / "parks.parquet"
    write_parquet(df, latest_processed_out)

    if args.season is not None:
        season_ref_out = reference_dir / f"parks_{args.season}.parquet"
        write_parquet(df, season_ref_out)


if __name__ == "__main__":
    main()
