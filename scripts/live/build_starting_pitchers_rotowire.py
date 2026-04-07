#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingest.id_resolution import resolve_starting_pitcher_ids
from src.ingest.io import log_kv, log_section, write_parquet
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs

from scripts.live._live_lineup_helpers import (
    enrich_with_schedule,
    load_schedule_for_date,
    print_quality,
)
from src.providers import rotowire


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build starting pitchers from Rotowire.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _pull_starters(config: dict) -> pd.DataFrame:
    rw_cfg = config.get("rotowire", {})
    url = str(rw_cfg.get("starting_pitchers_url", "")).strip()
    request_timeout = int(rw_cfg.get("request_timeout", 30))

    if not url:
        raise ValueError("Missing rotowire.starting_pitchers_url in configs/project.yaml")

    tables = rotowire.read_html_tables(
        url=url,
        request_timeout=request_timeout,
        verbose=True,
    )

    frames = []
    for idx, table in enumerate(tables):
        try:
            extracted = rotowire.extract_starting_pitchers(table, starter_status="probable")
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


def _coerce_final_schema(df: pd.DataFrame, season: int, slate_date: str) -> pd.DataFrame:
    out = df.copy()

    if out.empty:
        return pd.DataFrame(
            columns=[
                "game_pk", "game_date", "season", "team", "opponent", "is_home",
                "pitcher_id", "pitcher_name", "throws", "starter_status",
                "source", "source_pull_ts", "rotowire_id",
                "pitcher_id_resolution_method",
            ]
        )

    out["game_date"] = pd.to_datetime(out.get("game_date", slate_date), errors="coerce").dt.date.astype("string")
    out["game_date"] = out["game_date"].fillna(slate_date)
    out["season"] = season

    if "pitcher_id" in out.columns:
        out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")
    else:
        out["pitcher_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    if "pitcher_name" not in out.columns:
        out["pitcher_name"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["pitcher_name"] = out["pitcher_name"].astype("string").str.strip()

    if "rotowire_id" in out.columns:
        out["rotowire_id"] = pd.to_numeric(out["rotowire_id"], errors="coerce").astype("Int64")
    else:
        out["rotowire_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    out["throws"] = out.get("throws", pd.Series(pd.NA, index=out.index, dtype="string"))
    out["starter_status"] = out.get("starter_status", "probable")
    out["source"] = out.get("source", "rotowire")
    out["source_pull_ts"] = pd.Timestamp.utcnow().isoformat()

    keep = [
        "game_pk", "game_date", "season", "team", "opponent", "is_home",
        "pitcher_id", "pitcher_name", "throws", "starter_status",
        "source", "source_pull_ts", "rotowire_id",
    ]

    for c in keep:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[keep].copy()
    out = out.dropna(subset=["team", "pitcher_name"], how="any").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/live/build_starting_pitchers_rotowire.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    raw_live_dir = Path(dirs["raw_dir"]) / "live"
    raw_live_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(dirs["processed_dir"])

    schedule_df = load_schedule_for_date(raw_live_dir, args.season, args.date)

    provider_df = _pull_starters(config)
    debug_raw = raw_live_dir / f"DEBUG_starting_pitchers_provider_{args.season}_{args.date}.parquet"
    provider_df.to_parquet(debug_raw, index=False)

    enriched_df = enrich_with_schedule(provider_df, schedule_df, args.date)
    debug_enriched = raw_live_dir / f"DEBUG_starting_pitchers_enriched_{args.season}_{args.date}.parquet"
    enriched_df.to_parquet(debug_enriched, index=False)

    out_df = _coerce_final_schema(enriched_df, args.season, args.date)
    out_df = resolve_starting_pitcher_ids(out_df, processed_dir)

    print_quality(out_df, "starting_pitchers_out")

    latest_out = raw_live_dir / f"starting_pitchers_{args.season}.parquet"
    dated_out = raw_live_dir / f"starting_pitchers_{args.season}_{args.date}.parquet"

    write_parquet(out_df, latest_out)
    write_parquet(out_df, dated_out)

    print(f"starting_pitchers_out={dated_out}")
    print(f"debug_provider_out={debug_raw}")
    print(f"debug_enriched_out={debug_enriched}")


if __name__ == "__main__":
    main()