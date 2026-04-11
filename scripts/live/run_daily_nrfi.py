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
        score = score + vals * weight
    return score


def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    std = x.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=x.index, dtype="float64")
    z = (x - x.mean()) / std
    return z.clip(-2.5, 2.5)


def confidence_from_prob(prob: pd.Series) -> pd.Series:
    return pd.cut(
        prob,
        bins=[0.0, 0.52, 0.55, 0.58, 0.61, 1.0],
        labels=["C", "B-", "B+", "A", "A+"],
        include_lowest=True,
    )


def shrink_to_half(prob: pd.Series, factor: float = 0.58) -> pd.Series:
    """
    Pull probabilities back toward 0.50 to avoid over-aggressive tails.
    factor < 1.0 = more conservative.
    """
    prob = pd.to_numeric(prob, errors="coerce").fillna(0.5)
    return 0.5 + (prob - 0.5) * factor


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

    # Core scoring
    for col, wt in {
        "home_sp_k_rate_roll7": 0.55,
        "away_sp_k_rate_roll7": 0.55,
        "home_sp_bb_rate_roll7": -0.45,
        "away_sp_bb_rate_roll7": -0.45,
        "home_sp_hr_rate_roll7": -0.50,
        "away_sp_hr_rate_roll7": -0.50,
        "home_sp_barrel_rate_allowed_roll7": -0.45,
        "away_sp_barrel_rate_allowed_roll7": -0.45,
        "home_sp_hardhit_rate_allowed_roll7": -0.35,
        "away_sp_hardhit_rate_allowed_roll7": -0.35,
        "home_top3_bb_rate_roll15": -0.25,
        "away_top3_bb_rate_roll15": -0.25,
        "home_top3_barrel_rate_roll15": -0.30,
        "away_top3_barrel_rate_roll15": -0.30,
        "home_top3_hardhit_rate_roll15": -0.25,
        "away_top3_hardhit_rate_roll15": -0.25,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Environment - lighter than HR model
    for col, wt in {
        "weather_wind_out": -0.08,
        "weather_wind_in": 0.04,
        "temperature_f": -0.0015,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Missing starter penalty
    missing_starter = pd.Series(False, index=df.index)
    for c in ["home_starter_pitcher_id", "away_starter_pitcher_id", "home_sp_id", "away_sp_id"]:
        if c in df.columns:
            missing_starter = missing_starter | df[c].isna()

    score.loc[missing_starter] = score.loc[missing_starter] - 0.20

    # Raw score + slate-normalized score
    df["nrfi_score_raw_unscaled"] = score
    score_z = zscore_series(score)
    df["nrfi_score_raw"] = score_z

    # Less aggressive logistic transform
    nrfi_prob_raw = sigmoid_series(score_z * 0.72)

    # Shrink toward 50% to reduce fake 70-80% YRFI spikes
    nrfi_prob = shrink_to_half(nrfi_prob_raw, factor=0.58)

    # Reasonable cap range until proper calibration is built
    nrfi_prob = nrfi_prob.clip(0.32, 0.68)

    df["nrfi_prob"] = nrfi_prob
    df["yrfi_prob"] = 1.0 - df["nrfi_prob"]
    df["pick"] = df["nrfi_prob"].ge(0.50).map({True: "NRFI", False: "YRFI"})

    strongest_side_prob = df[["nrfi_prob", "yrfi_prob"]].max(axis=1)
    df["confidence"] = confidence_from_prob(strongest_side_prob)

    # Sort by strongest edge, not just highest NRFI
    df["edge_strength"] = (strongest_side_prob - 0.50).abs()

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
            "edge_strength",
        ]
        if c in df.columns
    ]

    board = (
        df[keep]
        .sort_values(["edge_strength", "nrfi_prob"], ascending=[False, False])
        .reset_index(drop=True)
    )

    if "edge_strength" in board.columns:
        board = board.drop(columns=["edge_strength"])

    out_csv = outputs_dir / f"nrfi_board_{season}_{date_str}.csv"
    out_parquet = outputs_dir / f"nrfi_board_{season}_{date_str}.parquet"

    board.to_csv(out_csv, index=False)
    board.to_parquet(out_parquet, index=False)

    print(f"✅ NRFI board built: {out_csv}")
    print(f"✅ NRFI board parquet: {out_parquet}")
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()