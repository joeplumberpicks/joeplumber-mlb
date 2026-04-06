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
    parser = argparse.ArgumentParser(description="Run daily Moneyline board.")
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

    feat_path = processed_dir / "live" / f"ml_features_{season}_{date_str}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")

    df = pd.read_parquet(feat_path).copy()

    print(f"=== JOE PLUMBER MONEYLINE RUN :: {date_str} ===")
    print(f"Loading features: {feat_path}")
    print(f"Row count [ml_features]: {len(df):,}")

    score = pd.Series(0.0, index=df.index, dtype="float64")

    # Starter edge
    for col, wt in {
        "home_sp_k_rate_roll15": 1.10,
        "away_sp_k_rate_roll15": -1.10,
        "home_sp_bb_rate_roll15": -0.80,
        "away_sp_bb_rate_roll15": 0.80,
        "home_sp_hr_rate_roll15": -0.90,
        "away_sp_hr_rate_roll15": 0.90,
        "home_sp_barrel_rate_roll15": -0.60,
        "away_sp_barrel_rate_roll15": 0.60,
        "home_sp_hardhit_rate_roll15": -0.45,
        "away_sp_hardhit_rate_roll15": 0.45,
        "home_sp_runs_rate_roll15": -0.85,
        "away_sp_runs_rate_roll15": 0.85,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Explicit diff features if present
    for col, wt in {
        "sp_diff_k_rate_roll7": 0.70,
        "sp_diff_bb_rate_roll7": -0.55,
        "sp_diff_hr_rate_roll7": -0.65,
        "sp_diff_runs_rate_roll7": -0.75,
        "sp_diff_barrel_rate_roll7": -0.35,
        "sp_diff_hardhit_rate_roll7": -0.25,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Team offense edge
    for col, wt in {
        "home_team_rbi_roll7": 0.18,
        "away_team_rbi_roll7": -0.18,
        "home_team_tb_roll7": 0.22,
        "away_team_tb_roll7": -0.22,
        "home_team_barrel_rate_roll7": 0.55,
        "away_team_barrel_rate_roll7": -0.55,
        "home_team_hardhit_rate_roll7": 0.45,
        "away_team_hardhit_rate_roll7": -0.45,
        "home_team_ev_mean_roll7": 0.015,
        "away_team_ev_mean_roll7": -0.015,
        "home_team_bb_rate_roll7": 0.20,
        "away_team_bb_rate_roll7": -0.20,
        "home_team_k_rate_roll7": -0.15,
        "away_team_k_rate_roll7": 0.15,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Offensive diff features if present
    for col, wt in {
        "off_diff_rbi_roll7": 0.20,
        "off_diff_tb_roll7": 0.24,
        "off_diff_barrel_rate_roll7": 0.60,
        "off_diff_hardhit_rate_roll7": 0.50,
        "off_diff_ev_mean_roll7": 0.018,
        "off_diff_bb_rate_roll7": 0.22,
        "off_diff_k_rate_roll7": -0.18,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Environment
    for col, wt in {
        "weather_wind_out": -0.05,
        "weather_wind_in": 0.03,
        "temperature_f": -0.001,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    df["home_win_score_raw"] = score
    df["home_win_prob"] = sigmoid_series(score)
    df["away_win_prob"] = 1.0 - df["home_win_prob"]
    df["pick"] = df["home_win_prob"].ge(0.5).map({True: "HOME", False: "AWAY"})

    df["confidence"] = pd.cut(
        df["home_win_prob"].where(df["home_win_prob"] >= 0.5, 1.0 - df["home_win_prob"]),
        bins=[0.0, 0.53, 0.57, 0.62, 0.68, 1.0],
        labels=["C", "B-", "B+", "A", "A+"],
        include_lowest=True,
    )

    keep = [
        c for c in [
            "game_date",
            "away_team",
            "home_team",
            "home_win_score_raw",
            "home_win_prob",
            "away_win_prob",
            "pick",
            "confidence",
        ]
        if c in df.columns
    ]

    board = df[keep].sort_values("home_win_prob", ascending=False).reset_index(drop=True)

    out_csv = outputs_dir / f"moneyline_board_{season}_{date_str}.csv"
    out_parquet = outputs_dir / f"moneyline_board_{season}_{date_str}.parquet"

    board.to_csv(out_csv, index=False)
    board.to_parquet(out_parquet, index=False)

    print(f"✅ Moneyline board built: {out_csv}")
    print(f"✅ Moneyline board parquet: {out_parquet}")
    print(board.head(15).to_string(index=False))


if __name__ == "__main__":
    main()