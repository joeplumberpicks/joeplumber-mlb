#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def get_today_date() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily NRFI board.")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--season", type=str, default="2026")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def sigmoid_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0).clip(-20, 20)
    return x.map(lambda v: 1.0 / (1.0 + math.exp(-float(v))))


def add_weighted_feature(score: pd.Series, df: pd.DataFrame, col: str, weight: float) -> pd.Series:
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        score = score + (vals * weight)
    return score


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config((repo_root / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    outputs_dir = Path(dirs["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    date_str = args.date or get_today_date()
    season = str(args.season)

    feat_path = processed_dir / "live" / f"nrfi_features_{season}_{date_str}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")

    df = pd.read_parquet(feat_path).copy()

    print(f"=== JOE PLUMBER NRFI RUN :: {date_str} ===")
    print(f"Loading features: {feat_path}")
    print(f"Row count [nrfi_features]: {len(df):,}")

    score = pd.Series(0.0, index=df.index, dtype="float64")

    # Starter-driven core
    for col, wt in {
        "home_sp_k_rate_roll7": 0.90,
        "away_sp_k_rate_roll7": 0.90,
        "home_sp_bb_rate_roll7": -0.75,
        "away_sp_bb_rate_roll7": -0.75,
        "home_sp_hr_rate_roll7": -0.85,
        "away_sp_hr_rate_roll7": -0.85,
        "home_sp_barrel_rate_roll7": -0.45,
        "away_sp_barrel_rate_roll7": -0.45,
        "home_sp_hardhit_rate_roll7": -0.35,
        "away_sp_hardhit_rate_roll7": -0.35,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Top-3 lineup pressure
    for col, wt in {
        "home_top3_hardhit_rate_roll7": -0.55,
        "away_top3_hardhit_rate_roll7": -0.55,
        "home_top3_barrel_rate_roll7": -0.65,
        "away_top3_barrel_rate_roll7": -0.65,
        "home_top3_ev_mean_roll7": -0.015,
        "away_top3_ev_mean_roll7": -0.015,
        "home_top3_bb_rate_roll7": -0.30,
        "away_top3_bb_rate_roll7": -0.30,
        "home_top3_k_rate_roll7": 0.25,
        "away_top3_k_rate_roll7": 0.25,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Park / weather environment
    for col, wt in {
        "weather_wind_out": -0.20,
        "weather_wind_in": 0.12,
        "weather_crosswind": 0.03,
        "temperature_f": -0.003,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    df["nrfi_score_raw"] = score
    df["nrfi_prob"] = sigmoid_series(score)
    df["yrfi_prob"] = 1.0 - df["nrfi_prob"]
    df["pick"] = df["nrfi_prob"].ge(0.5).map({True: "NRFI", False: "YRFI"})

    # Confidence bucket
    df["confidence"] = pd.cut(
        df["nrfi_prob"],
        bins=[0.0, 0.52, 0.56, 0.60, 0.65, 1.0],
        labels=["C", "B-", "B+", "A", "A+"],
        include_lowest=True,
    )

    keep = [
        c for c in [
            "game_date",
            "away_team",
            "home_team",
            "nrfi_score_raw",
            "nrfi_prob",
            "yrfi_prob",
            "pick",
            "confidence",
        ]
        if c in df.columns
    ]

    board = df[keep].sort_values("nrfi_prob", ascending=False).reset_index(drop=True)

    out_csv = outputs_dir / f"nrfi_board_{season}_{date_str}.csv"
    out_parquet = outputs_dir / f"nrfi_board_{season}_{date_str}.parquet"

    board.to_csv(out_csv, index=False)
    board.to_parquet(out_parquet, index=False)

    print(f"✅ NRFI board built: {out_csv}")
    print(f"✅ NRFI board parquet: {out_parquet}")
    print(board.head(15).to_string(index=False))


if __name__ == "__main__":
    main()