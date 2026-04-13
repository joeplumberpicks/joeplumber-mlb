#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from calendar import monthrange
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.providers.statcast import fetch_statcast, extract_plate_appearances_from_statcast
from src.ingest.plate_appearances import build_plate_appearances
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild historical plate appearance files from Statcast."
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_month_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    chunks: list[tuple[str, str]] = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)

    while cur <= end:
        last_day = monthrange(cur.year, cur.month)[1]

        chunk_start = max(cur, start)
        chunk_end = min(
            pd.Timestamp(year=cur.year, month=cur.month, day=last_day),
            end,
        )

        chunks.append(
            (
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
        )

        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1)
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1)

    return chunks


def dedupe_pas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"game_pk", "pa_index"}.issubset(out.columns):
        sort_cols = [
            c
            for c in ["game_date", "game_pk", "inning", "inning_topbot", "pa_index"]
            if c in out.columns
        ]

        out = (
            out.sort_values(sort_cols, kind="stable")
            .drop_duplicates(["game_pk", "pa_index"], keep="last")
            .reset_index(drop=True)
        )

    return out


def main() -> None:
    args = parse_args()

    config = load_config(REPO_ROOT / args.config)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    out_dir = processed_dir / "by_season"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"pa_{args.season}.parquet"

    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing file: {out_path}")
        return

    start_date = f"{args.season}-03-01"
    end_date = f"{args.season}-11-30"

    month_chunks = build_month_chunks(start_date, end_date)

    all_parts: list[pd.DataFrame] = []

    print("========================================")
    print("JOE PLUMBER HISTORICAL PA REBUILD")
    print("========================================")
    print(f"season={args.season}")
    print(f"chunks={len(month_chunks)}")
    print("")

    for chunk_start, chunk_end in month_chunks:
        print(f"Pulling Statcast {chunk_start} -> {chunk_end}")

        raw_df = fetch_statcast(
            start_date=chunk_start,
            end_date=chunk_end,
            verbose=True,
        )

        if raw_df.empty:
            print("No rows returned.")
            continue

        pa_like = extract_plate_appearances_from_statcast(raw_df)

        if pa_like.empty:
            print("No PA rows after extraction.")
            continue

        normalized = build_plate_appearances(
            records=pa_like,
            source="statcast",
            validate=True,
            verbose=True,
        )

        normalized["season"] = args.season

        normalized = dedupe_pas(normalized)

        all_parts.append(normalized)

        print(f"normalized_rows={len(normalized):,}")
        print("")

    if not all_parts:
        raise RuntimeError(f"No PA rows built for season {args.season}")

    season_pa = pd.concat(all_parts, ignore_index=True)

    season_pa = dedupe_pas(season_pa)

    if "game_date" in season_pa.columns:
        season_pa["game_date"] = pd.to_datetime(
            season_pa["game_date"],
            errors="coerce",
        )

    season_pa = season_pa.sort_values(
        [
            c
            for c in ["game_date", "game_pk", "inning", "inning_topbot", "pa_index"]
            if c in season_pa.columns
        ],
        kind="stable",
    ).reset_index(drop=True)

    season_pa.to_parquet(out_path, index=False)

    print("")
    print("========================================")
    print("PA REBUILD COMPLETE")
    print("========================================")
    print(f"season={args.season}")
    print(f"rows={len(season_pa):,}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()
