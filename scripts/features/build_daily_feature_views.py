import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

from src.features.build_nrfi_features import build_nrfi_features
from src.features.build_moneyline_features import build_moneyline_features
from src.features.build_hr_features import build_hr_features
from src.features.build_rbi_features import build_rbi_features

DATA = Path("/content/drive/MyDrive/joeplumber-mlb/data")


def get_today_date():
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--season", type=str, default="2026")
    args = parser.parse_args()

    DATE = args.date if args.date else get_today_date()
    SEASON = args.season

    print(f"=== BUILD DAILY FEATURE VIEWS :: {DATE} ===")

    # Load inputs
    spine_path = DATA / f"processed/live/model_spine_game_{SEASON}_{DATE}.parquet"
    lineup_path = DATA / f"raw/live/projected_lineups_{SEASON}_{DATE}.parquet"

    print(f"Loading spine: {spine_path}")
    print(f"Loading lineups: {lineup_path}")

    spine = pd.read_parquet(spine_path)
    lineups = pd.read_parquet(lineup_path)

    bat = pd.read_parquet(DATA / "processed/batter_statcast_rolling.parquet")
    pit = pd.read_parquet(DATA / "processed/pitcher_statcast_rolling.parquet")

    # Build features
    nrfi = build_nrfi_features(spine, pit, bat)
    ml = build_moneyline_features(spine, pit)
    hr = build_hr_features(lineups, bat, pit)
    rbi = build_rbi_features(lineups, bat, pit)

    # Output paths
    out_nrfi = DATA / f"processed/live/nrfi_features_{SEASON}_{DATE}.parquet"
    out_ml = DATA / f"processed/live/ml_features_{SEASON}_{DATE}.parquet"
    out_hr = DATA / f"processed/live/hr_features_{SEASON}_{DATE}.parquet"
    out_rbi = DATA / f"processed/live/rbi_features_{SEASON}_{DATE}.parquet"

    nrfi.to_parquet(out_nrfi, index=False)
    ml.to_parquet(out_ml, index=False)
    hr.to_parquet(out_hr, index=False)
    rbi.to_parquet(out_rbi, index=False)

    print("✅ daily feature views built")
    print(f"NRFI rows: {len(nrfi)}")
    print(f"ML rows: {len(ml)}")
    print(f"HR rows: {len(hr)}")
    print(f"RBI rows: {len(rbi)}")


if __name__ == "__main__":
    main()
