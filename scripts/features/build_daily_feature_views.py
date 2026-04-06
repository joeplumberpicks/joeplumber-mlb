#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.features.build_hr_features import build_hr_features
from src.features.build_moneyline_features import build_moneyline_features
from src.features.build_nrfi_features import build_nrfi_features
from src.features.build_rbi_features import build_rbi_features
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def get_today_date() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily feature views for NRFI, Moneyline, HR, RBI.")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--season", type=str, default="2026")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config((repo_root / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    raw_dir = Path(dirs["raw_dir"])

    date_str = args.date or get_today_date()
    season = str(args.season)

    print(f"=== BUILD DAILY FEATURE VIEWS :: {date_str} ===")

    spine_path = processed_dir / "live" / f"model_spine_game_{season}_{date_str}.parquet"
    lineup_candidates = [
        raw_dir / "live" / f"confirmed_lineups_{season}_{date_str}.parquet",
        raw_dir / "live" / f"projected_lineups_{season}_{date_str}.parquet",
    ]
    batter_roll_path = processed_dir / "batter_statcast_rolling.parquet"
    pitcher_roll_path = processed_dir / "pitcher_statcast_rolling.parquet"

    _require(spine_path)
    _require(batter_roll_path)
    _require(pitcher_roll_path)

    lineup_path = None
    for p in lineup_candidates:
        if p.exists():
            lineup_path = p
            break

    if lineup_path is None:
        raise FileNotFoundError(
            f"Missing lineup parquet. Checked: {[str(p) for p in lineup_candidates]}"
        )

    print(f"Loading spine: {spine_path}")
    print(f"Loading lineups: {lineup_path}")
    print(f"Loading batter rollings: {batter_roll_path}")
    print(f"Loading pitcher rollings: {pitcher_roll_path}")

    spine = pd.read_parquet(spine_path)
    lineups = pd.read_parquet(lineup_path)
    batter_roll = pd.read_parquet(batter_roll_path)
    pitcher_roll = pd.read_parquet(pitcher_roll_path)

    nrfi = build_nrfi_features(spine, lineups, batter_roll, pitcher_roll)
    ml = build_moneyline_features(spine, lineups, batter_roll, pitcher_roll)
    hr = build_hr_features(spine, lineups, batter_roll, pitcher_roll)
    rbi = build_rbi_features(spine, lineups, batter_roll, pitcher_roll)

    out_nrfi = processed_dir / "live" / f"nrfi_features_{season}_{date_str}.parquet"
    out_ml = processed_dir / "live" / f"ml_features_{season}_{date_str}.parquet"
    out_hr = processed_dir / "live" / f"hr_features_{season}_{date_str}.parquet"
    out_rbi = processed_dir / "live" / f"rbi_features_{season}_{date_str}.parquet"

    nrfi.to_parquet(out_nrfi, index=False)
    ml.to_parquet(out_ml, index=False)
    hr.to_parquet(out_hr, index=False)
    rbi.to_parquet(out_rbi, index=False)

    print("✅ daily feature views built")
    print(f"NRFI rows: {len(nrfi):,}")
    print(f"Moneyline rows: {len(ml):,}")
    print(f"HR rows: {len(hr):,}")
    print(f"RBI rows: {len(rbi):,}")
    print(f"out_nrfi={out_nrfi}")
    print(f"out_ml={out_ml}")
    print(f"out_hr={out_hr}")
    print(f"out_rbi={out_rbi}")


if __name__ == "__main__":
    main()