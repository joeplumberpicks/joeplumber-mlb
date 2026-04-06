#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingest.io import log_kv, log_section, write_parquet
from src.ingest.lineups import build_projected_lineups
from src.providers import fangraphs
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs

from scripts.live._live_lineup_helpers import (
    enrich_with_schedule,
    load_schedule_for_date,
    print_quality,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build projected lineups from FanGraphs.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _pull_projected_table(config: dict) -> pd.DataFrame:
    fg_cfg = config.get("fangraphs", {})
    url = str(fg_cfg.get("projected_lineups_url", "")).strip()
    request_timeout = int(fg_cfg.get("request_timeout", 30))

    if not url:
        raise ValueError("Missing fangraphs.projected_lineups_url in configs/project.yaml")

    if ".csv" in url.lower():
        return fangraphs.read_csv_export(
            url=url,
            request_timeout=request_timeout,
            verbose=True,
        )

    tables = fangraphs.read_html_tables(
        url=url,
        request_timeout=request_timeout,
        verbose=True,
    )

    frames = []
    for idx, table in enumerate(tables):
        try:
            extracted = fangraphs.extract_projected_lineups(table)
        except Exception:
            continue

        if extracted is None or extracted.empty:
            continue

        extracted = extracted.copy()
        extracted["source_table_idx"] = idx
        frames.append(extracted)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/live/build_projected_lineups_fangraphs.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    raw_live_dir = Path(dirs["raw_dir"]) / "live"
    schedule_df = load_schedule_for_date(raw_live_dir, args.season, args.date)

    provider_df = _pull_projected_table(config)
    provider_df = enrich_with_schedule(provider_df, schedule_df, args.date)

    out_df = build_projected_lineups(
        records=provider_df,
        source="fangraphs",
        validate=True,
        verbose=True,
    )

    print_quality(out_df, "projected_lineups_out")

    latest_out = raw_live_dir / f"projected_lineups_{args.season}.parquet"
    dated_out = raw_live_dir / f"projected_lineups_{args.season}_{args.date}.parquet"

    write_parquet(out_df, latest_out)
    write_parquet(out_df, dated_out)

    print(f"projected_lineups_out={dated_out}")


if __name__ == "__main__":
    main()
