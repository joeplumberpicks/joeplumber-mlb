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
    parser = argparse.ArgumentParser(description="Run daily No Homerun Game board.")
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
        score = score + vals * weight
    return score


def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    std = x.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=x.index, dtype="float64")
    return (x - x.mean()) / std


def confidence_from_prob(prob: pd.Series) -> pd.Series:
    return pd.cut(
        prob,
        bins=[0.0, 0.52, 0.56, 0.61, 0.67, 1.0],
        labels=["C", "B-", "B+", "A", "A+"],
        include_lowest=True,
    )


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

    feat_path = processed_dir / "live" / f"no_hr_game_features_{season}_{date_str}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")

    df = pd.read_parquet(feat_path).copy()

    print(f"=== JOE PLUMBER NO HR GAME RUN :: {date_str} ===")
    print(f"Loading features: {feat_path}")
    print(f"Row count [no_hr_game_features]: {len(df):,}")

    # Positive score = more likely no HR game
    score = pd.Series(0.0, index=df.index, dtype="float64")

    # Pitcher suppression
    for col, wt in {
        "game_sp_k_sum_roll15": 0.30,
        "game_sp_hr_allowed_sum_roll15": -1.05,
        "game_sp_barrel_allowed_sum_roll15": -0.85,
        "game_sp_hardhit_allowed_sum_roll15": -0.55,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Lineup power pressure
    for col, wt in {
        "game_lineup_hr_roll15_sum": -0.95,
        "game_lineup_barrel_rate_roll15_sum": -0.85,
        "game_lineup_hardhit_rate_roll15_sum": -0.60,
        "game_lineup_tb_roll15_sum": -0.30,
        "game_lineup_ev_mean_roll15_sum": -0.012,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Environment
    for col, wt in {
        "weather_wind_out": -0.20,
        "weather_wind_in": 0.12,
        "temperature_f": -0.003,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Incomplete-data penalties
    for col, penalty in {
        "missing_home_starter": -0.18,
        "missing_away_starter": -0.18,
        "missing_home_lineup_core": -0.12,
        "missing_away_lineup_core": -0.12,
    }.items():
        if col in df.columns:
            mask = df[col].fillna(False).astype(bool)
            score.loc[mask] = score.loc[mask] + penalty

    score_z = zscore_series(score)
    df["no_hr_game_score_raw"] = score_z
    df["p_no_hr_game"] = sigmoid_series(score_z * 1.00)
    df["p_yes_hr_game"] = 1.0 - df["p_no_hr_game"]
    df["pick"] = df["p_no_hr_game"].ge(0.50).map({True: "NO_HR", False: "HR_YES"})
    df["confidence"] = confidence_from_prob(df[["p_no_hr_game", "p_yes_hr_game"]].max(axis=1))

    keep = [
        c for c in [
            "game_date",
            "away_team",
            "home_team",
            "no_hr_game_score_raw",
            "p_no_hr_game",
            "p_yes_hr_game",
            "pick",
            "confidence",
        ]
        if c in df.columns
    ]

    board = df[keep].sort_values("p_no_hr_game", ascending=False).reset_index(drop=True)

    out_csv = outputs_dir / f"no_hr_game_board_{season}_{date_str}.csv"
    out_parquet = outputs_dir / f"no_hr_game_board_{season}_{date_str}.parquet"

    board.to_csv(out_csv, index=False)
    board.to_parquet(out_parquet, index=False)

    print(f"✅ No HR Game board built: {out_csv}")
    print(f"✅ No HR Game board parquet: {out_parquet}")
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()