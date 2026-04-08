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
    parser = argparse.ArgumentParser(description="Run daily RBI board.")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--season", type=str, default="2026")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def add_weighted_feature(score: pd.Series, df: pd.DataFrame, col: str, weight: float) -> pd.Series:
    if col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        score = score + (vals * weight)
    return score


def sigmoid_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0).clip(-20, 20)
    return x.map(lambda v: 1.0 / (1.0 + math.exp(-float(v))))


def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    std = x.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=x.index, dtype="float64")
    return (x - x.mean()) / std


def confidence_from_prob(prob: pd.Series) -> pd.Series:
    return pd.cut(
        prob,
        bins=[0.0, 0.54, 0.60, 0.66, 0.72, 1.0],
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

    feat_path = processed_dir / "live" / f"rbi_features_{season}_{date_str}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")

    df = pd.read_parquet(feat_path).copy()

    print(f"=== JOE PLUMBER RBI RUN :: {date_str} ===")
    print(f"Loading features: {feat_path}")
    print(f"Row count [rbi_features]: {len(df):,}")

    score = pd.Series(0.0, index=df.index, dtype="float64")

    for col, wt in {
        "rbi_roll15": 0.95,
        "rbi_roll30": 0.60,
        "tb_roll15": 0.28,
        "tb_roll30": 0.18,
        "hardhit_rate_roll15": 0.45,
        "barrel_rate_roll15": 0.35,
        "bb_rate_roll15": 0.16,
        "k_rate_roll15": -0.08,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    for col, wt in {
        "opp_bb_rate_roll15": 0.30,
        "opp_hr_allowed_roll15": 0.28,
        "opp_hardhit_rate_allowed_roll15": 0.22,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    for col, wt in {
        "rbi_walk_pressure_roll15": 1.20,
        "rbi_power_pressure_roll15": 1.00,
        "rbi_contact_pressure_roll15": 0.90,
        "rbi_team_onbase_pressure": 0.70,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    for col, wt in {
        "team_ctx_bb_rate_roll15": 0.30,
        "team_ctx_tb_roll15": 0.20,
        "team_ctx_rbi_roll15": 0.16,
        "team_ctx_hardhit_rate_roll15": 0.18,
        "team_ctx_barrel_rate_roll15": 0.12,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    if "lineup_weight" in df.columns:
        score = score * pd.to_numeric(df["lineup_weight"], errors="coerce").fillna(1.0)

    for col, wt in {
        "weather_wind_out": 0.06,
        "weather_wind_in": -0.03,
        "temperature_f": 0.002,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    if "tie_break_noise" in df.columns:
        score = score + pd.to_numeric(df["tie_break_noise"], errors="coerce").fillna(0.0)

    missing_core = pd.Series(False, index=df.index)
    for c in ["player_id", "opp_pitcher_id"]:
        if c in df.columns:
            missing_core = missing_core | df[c].isna()

    score.loc[missing_core] = score.loc[missing_core] - 0.20

    score_z = zscore_series(score)
    df["rbi_score_raw"] = score_z
    df["p_rbi"] = sigmoid_series(score_z * 0.90)
    df["confidence"] = confidence_from_prob(df["p_rbi"])

    batter_name_col = _pick_col(df, ["batter_name", "player_name", "name"])
    team_col = _pick_col(df, ["team", "team_abbr", "batting_team"])
    opp_col = _pick_col(df, ["opponent"])

    keep = [
        c for c in [
            "game_date",
            team_col,
            opp_col,
            batter_name_col,
            "rbi_score_raw",
            "p_rbi",
            "confidence",
        ]
        if c is not None and c in df.columns
    ]

    board = df[keep].sort_values("p_rbi", ascending=False).reset_index(drop=True)
    board["rank"] = range(1, len(board) + 1)
    board = board[["rank"] + [c for c in board.columns if c != "rank"]]

    out_csv = outputs_dir / f"rbi_board_{season}_{date_str}.csv"
    out_parquet = outputs_dir / f"rbi_board_{season}_{date_str}.parquet"

    board.to_csv(out_csv, index=False)
    board.to_parquet(out_parquet, index=False)

    print(f"✅ RBI board built: {out_csv}")
    print(f"✅ RBI board parquet: {out_parquet}")
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()