%%bash
cd /content/joeplumber-mlb

mkdir -p scripts/ingest

cat > scripts/ingest/rebuild_historical_pa_from_statcast.py <<'PY'
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def month_chunks(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    chunks = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1)

    while cur <= end:
        last_day = monthrange(cur.year, cur.month)[1]
        chunk_start = max(cur, start)
        chunk_end = min(pd.Timestamp(year=cur.year, month=cur.month, day=last_day), end)

        chunks.append((
            chunk_start.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d")
        ))

        if cur.month == 12:
            cur = pd.Timestamp(year=cur.year + 1, month=1, day=1)
        else:
            cur = pd.Timestamp(year=cur.year, month=cur.month + 1, day=1)

    return chunks


def main():
    args = parse_args()

    config = load_config(REPO_ROOT / args.config)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    out_dir = Path(dirs["processed_dir"]) / "by_season"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"pa_{args.season}.parquet"

    if out_path.exists() and not args.overwrite:
        print(f"Skipping existing: {out_path}")
        return

    all_parts = []

    for start_date, end_date in month_chunks(f"{args.season}-03-01", f"{args.season}-11-30"):
        print(f"Pulling {start_date} -> {end_date}")

        raw_df = fetch_statcast(start_date, end_date, verbose=True)

        if raw_df.empty:
            continue

        pa_like = extract_plate_appearances_from_statcast(raw_df)

        if pa_like.empty:
            continue

        normalized = build_plate_appearances(
            records=pa_like,
            source="statcast",
            validate=True,
            verbose=True,
        )

        normalized["season"] = args.season
        all_parts.append(normalized)

    if not all_parts:
        raise RuntimeError(f"No PA data found for season {args.season}")

    season_pa = pd.concat(all_parts, ignore_index=True)

    if {"game_pk", "pa_index"}.issubset(season_pa.columns):
        season_pa = (
            season_pa.sort_values(["game_date", "game_pk", "pa_index"])
            .drop_duplicates(["game_pk", "pa_index"], keep="last")
            .reset_index(drop=True)
        )

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
PY

chmod +x scripts/ingest/rebuild_historical_pa_from_statcast.py
