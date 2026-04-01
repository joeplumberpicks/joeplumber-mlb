#!/usr/bin/env python3
"""
Ingest normalized MLB lineups and starting pitchers.

Purpose
-------
Read provider-level lineup / starter records and write normalized projected
lineups, confirmed lineups, and starting pitchers to the data lake.

This script is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering
- no target creation

Notes
-----
This runner is intentionally input-driven. It expects provider-level records
from a CSV or parquet file that were pulled separately from a source such as
Rotowire or FanGraphs.

Typical usage:
- projected lineups from Rotowire/FanGraphs
- confirmed lineups from Rotowire/another source
- probable/confirmed starters from the same or separate file
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
from src.ingest.lineups import (
    build_confirmed_lineups,
    build_projected_lineups,
    build_starting_pitchers,
)
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest normalized MLB lineups and starters.")
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2026")
    parser.add_argument("--date", type=str, required=True, help="Slate date YYYY-MM-DD")

    parser.add_argument(
        "--projected-lineups-input",
        type=str,
        default=None,
        help="Optional input file (csv/parquet) for projected lineup records.",
    )
    parser.add_argument(
        "--confirmed-lineups-input",
        type=str,
        default=None,
        help="Optional input file (csv/parquet) for confirmed lineup records.",
    )
    parser.add_argument(
        "--starting-pitchers-input",
        type=str,
        default=None,
        help="Optional input file (csv/parquet) for starting pitcher records.",
    )

    parser.add_argument(
        "--projected-source",
        type=str,
        default="unknown",
        help="Source label for projected lineups, e.g. rotowire or fangraphs.",
    )
    parser.add_argument(
        "--confirmed-source",
        type=str,
        default="unknown",
        help="Source label for confirmed lineups.",
    )
    parser.add_argument(
        "--starters-source",
        type=str,
        default="unknown",
        help="Source label for starting pitchers.",
    )
    parser.add_argument(
        "--starter-status",
        type=str,
        default="probable",
        help="Starter status label, e.g. probable or confirmed.",
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


def _read_optional_input(path_str: str | None):
    if not path_str:
        return None
    return read_dataset(path_str)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/ingest/ingest_lineups.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    log_kv("data_root", dirs["data_root"])
    log_kv("raw_dir", dirs["raw_dir"])
    log_kv("processed_dir", dirs["processed_dir"])
    log_kv("reference_dir", dirs["reference_dir"])

    projected_records = _read_optional_input(args.projected_lineups_input)
    confirmed_records = _read_optional_input(args.confirmed_lineups_input)
    starter_records = _read_optional_input(args.starting_pitchers_input)

    projected_df = build_projected_lineups(
        records=projected_records,
        source=args.projected_source,
        validate=True,
        verbose=True,
    )
    confirmed_df = build_confirmed_lineups(
        records=confirmed_records,
        source=args.confirmed_source,
        validate=True,
        verbose=True,
    )
    starters_df = build_starting_pitchers(
        records=starter_records,
        starter_status=args.starter_status,
        source=args.starters_source,
        validate=True,
        verbose=True,
    )

    raw_live_dir = Path(dirs["raw_dir"]) / "live"

    projected_latest_out = raw_live_dir / f"projected_lineups_{args.season}.parquet"
    projected_dated_out = raw_live_dir / f"projected_lineups_{args.season}_{args.date}.parquet"

    confirmed_latest_out = raw_live_dir / f"confirmed_lineups_{args.season}.parquet"
    confirmed_dated_out = raw_live_dir / f"confirmed_lineups_{args.season}_{args.date}.parquet"

    starters_latest_out = raw_live_dir / f"starting_pitchers_{args.season}.parquet"
    starters_dated_out = raw_live_dir / f"starting_pitchers_{args.season}_{args.date}.parquet"

    write_parquet(projected_df, projected_latest_out)
    write_parquet(projected_df, projected_dated_out)

    write_parquet(confirmed_df, confirmed_latest_out)
    write_parquet(confirmed_df, confirmed_dated_out)

    write_parquet(starters_df, starters_latest_out)
    write_parquet(starters_df, starters_dated_out)


if __name__ == "__main__":
    main()
