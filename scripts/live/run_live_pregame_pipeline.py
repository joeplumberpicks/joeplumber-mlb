#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live pregame pipeline.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"\nRUNNING: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    print("========== scripts/live/run_live_pregame_pipeline.py =========")
    print(f"repo_root: {repo_root}")
    print(f"config_path: {repo_root / args.config}")

    py = sys.executable
    config_arg = ["--config", args.config]
    force_arg = ["--force"] if args.force else []

    _run(
        [
            py,
            "scripts/ingest/run_ingest_day.py",
            "--season",
            str(args.season),
            "--date",
            args.date,
            *config_arg,
            *force_arg,
        ]
    )

    spine_path = Path("/content/drive/MyDrive/joeplumber-mlb/data/processed/live") / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(f"Missing spine output: {spine_path}")

    df = pd.read_parquet(spine_path).copy()

    print(f"\nRow count [model_spine_game_{args.season}_{args.date}]: {len(df):,}")
    print(f"Distinct game_pk: {df['game_pk'].nunique() if 'game_pk' in df.columns else 0}")

    if "away_starter_found" in df.columns and "home_starter_found" in df.columns:
        starter_pct = ((df["away_starter_found"].fillna(False) & df["home_starter_found"].fillna(False)).mean()) * 100
        print(f"pct_with_starters={starter_pct:.2f}")

    if "away_projected_lineup_found" in df.columns and "home_projected_lineup_found" in df.columns:
        projected_pct = ((df["away_projected_lineup_found"].fillna(False) & df["home_projected_lineup_found"].fillna(False)).mean()) * 100
        print(f"pct_with_projected_lineups={projected_pct:.2f}")

    if "away_confirmed_lineup_found" in df.columns and "home_confirmed_lineup_found" in df.columns:
        confirmed_pct = ((df["away_confirmed_lineup_found"].fillna(False) & df["home_confirmed_lineup_found"].fillna(False)).mean()) * 100
        print(f"pct_with_confirmed_lineups={confirmed_pct:.2f}")

    for c in ["temperature_f", "wind_mph", "weather_wind_out", "weather_wind_in", "weather_crosswind"]:
        if c in df.columns:
            print(f"Nulls [{c}]: {int(df[c].isna().sum())}")

    print(f"daily_out={spine_path}")


if __name__ == "__main__":
    main()