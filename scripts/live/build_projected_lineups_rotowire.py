#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.ingest.io import log_kv, log_section, write_parquet
from src.ingest.lineups import build_projected_lineups
from src.providers.rotowire import extract_lineups
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs

from scripts.live._rotowire_live import (
    combine_provider_tables,
    enrich_provider_rows_with_schedule,
    load_schedule_for_date,
    print_join_quality,
    pull_rotowire_tables,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build projected lineups from Rotowire.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/live/build_projected_lineups_rotowire.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    raw_live_dir = Path(dirs["raw_dir"]) / "live"

    schedule_df = load_schedule_for_date(raw_live_dir, args.season, args.date)

    rotowire_cfg = config.get("rotowire", {})
    url = str(rotowire_cfg.get("projected_lineups_url", "")).strip()
    request_timeout = int(rotowire_cfg.get("request_timeout", 30))

    tables = pull_rotowire_tables(url=url, request_timeout=request_timeout)
    provider_df = combine_provider_tables(
        tables,
        extract_lineups,
        status_label="projected",
    )
    provider_df = enrich_provider_rows_with_schedule(provider_df, schedule_df, args.date)

    out_df = build_projected_lineups(
        records=provider_df,
        source="rotowire",
        validate=True,
        verbose=True,
    )

    print_join_quality(out_df, "projected_lineups_joined")

    latest_out = raw_live_dir / f"projected_lineups_{args.season}.parquet"
    dated_out = raw_live_dir / f"projected_lineups_{args.season}_{args.date}.parquet"

    write_parquet(out_df, latest_out)
    write_parquet(out_df, dated_out)


if __name__ == "__main__":
    main()
