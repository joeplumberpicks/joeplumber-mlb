#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.build_nrfi_features import build_nrfi_features
from src.features.build_moneyline_features import build_moneyline_features
from src.features.build_hr_features import build_hr_features
from src.features.build_rbi_features import build_rbi_features
from src.features.build_tb2_features import build_tb2_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily feature views.")
    parser.add_argument("--season", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="/content/drive/MyDrive/joeplumber-mlb/data")
    return parser.parse_args()


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    season = str(args.season)
    date_str = args.date

    processed_live = data_root / "processed" / "live"
    raw_live = data_root / "raw" / "live"

    spine_path = processed_live / f"model_spine_game_{season}_{date_str}.parquet"
    lineups_path = raw_live / f"confirmed_lineups_{season}_{date_str}.parquet"
    batter_roll_path = data_root / "processed" / "batter_statcast_rolling.parquet"
    pitcher_roll_path = data_root / "processed" / "pitcher_statcast_rolling.parquet"

    print(f"=== BUILD DAILY FEATURE VIEWS :: {date_str} ===")
    print(f"Loading spine: {spine_path}")
    print(f"Loading lineups: {lineups_path}")
    print(f"Loading batter rollings: {batter_roll_path}")
    print(f"Loading pitcher rollings: {pitcher_roll_path}")

    _require(spine_path)
    _require(lineups_path)
    _require(batter_roll_path)
    _require(pitcher_roll_path)

    spine = pd.read_parquet(spine_path)
    lineups = pd.read_parquet(lineups_path)
    batter_roll = pd.read_parquet(batter_roll_path)
    pitcher_roll = pd.read_parquet(pitcher_roll_path)

    nrfi = build_nrfi_features(spine, lineups, batter_roll, pitcher_roll)
    ml = build_moneyline_features(spine, lineups, batter_roll, pitcher_roll)
    hr = build_hr_features(spine, lineups, batter_roll, pitcher_roll)
    rbi = build_rbi_features(spine, lineups, batter_roll, pitcher_roll)
    tb2 = build_tb2_features(spine, lineups, batter_roll, pitcher_roll)

    out_nrfi = processed_live / f"nrfi_features_{season}_{date_str}.parquet"
    out_ml = processed_live / f"ml_features_{season}_{date_str}.parquet"
    out_hr = processed_live / f"hr_features_{season}_{date_str}.parquet"
    out_rbi = processed_live / f"rbi_features_{season}_{date_str}.parquet"
    out_tb2 = processed_live / f"tb2_features_{season}_{date_str}.parquet"

    nrfi.to_parquet(out_nrfi, index=False)
    ml.to_parquet(out_ml, index=False)
    hr.to_parquet(out_hr, index=False)
    rbi.to_parquet(out_rbi, index=False)
    tb2.to_parquet(out_tb2, index=False)

    print("✅ daily feature views built")
    print(f"NRFI rows: {len(nrfi):,}")
    print(f"Moneyline rows: {len(ml):,}")
    print(f"HR rows: {len(hr):,}")
    print(f"RBI rows: {len(rbi):,}")
    print(f"2+ Bases rows: {len(tb2):,}")
    print(f"out_nrfi={out_nrfi}")
    print(f"out_ml={out_ml}")
    print(f"out_hr={out_hr}")
    print(f"out_rbi={out_rbi}")
    print(f"out_tb2={out_tb2}")


if __name__ == "__main__":
    main()