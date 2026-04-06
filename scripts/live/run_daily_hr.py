#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def get_today_date() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily HR board.")
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

    feat_path = processed_dir / "live" / f"hr_features_{season}_{date_str}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")

    df = pd.read_parquet(feat_path).copy()

    print(f"=== JOE PLUMBER HR RUN :: {date_str} ===")
    print(f"Loading features: {feat_path}")
    print(f"Row count [hr_features]: {len(df):,}")

    score = pd.Series(0.0, index=df.index, dtype="float64")

    # Hitter form / power
    for col, wt in {
        "hr_roll15": 0.45,
        "hr_roll30": 0.35,
        "barrel_rate_roll15": 1.35,
        "barrel_rate_roll30": 0.95,
        "hardhit_rate_roll15": 0.95,
        "hardhit_rate_roll30": 0.65,
        "ev_mean_roll15": 0.025,
        "ev_mean_roll30": 0.018,
        "la_mean_roll15": 0.020,
        "tb_roll15": 0.12,
        "k_rate_roll15": -0.18,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Pitcher vulnerability
    for col, wt in {
        "opp_hr_allowed_roll15": 0.85,
        "opp_hr_allowed_roll30": 0.65,
        "opp_barrel_rate_allowed_roll15": 1.00,
        "opp_barrel_rate_allowed_roll30": 0.70,
        "opp_hardhit_rate_allowed_roll15": 0.70,
        "opp_hardhit_rate_allowed_roll30": 0.45,
        "opp_ev_mean_roll15": 0.015,
        "opp_bb_rate_roll15": 0.12,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Matchup edge features from builder
    for col, wt in {
        "matchup_barrel_edge_roll15": 0.85,
        "matchup_hardhit_edge_roll15": 0.60,
        "matchup_ev_edge_roll15": 0.018,
        "matchup_hr_pressure_roll15": 0.40,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Opportunity / lineup
    lineup_slot_col = _pick_col(df, ["lineup_slot", "batting_order", "order_spot", "slot"])
    if lineup_slot_col is not None:
        slot_num = pd.to_numeric(df[lineup_slot_col], errors="coerce")
        score = score + slot_num.map({
            1: 0.18,
            2: 0.22,
            3: 0.35,
            4: 0.42,
            5: 0.30,
            6: 0.15,
        }).fillna(0.0)

    # Environment
    for col, wt in {
        "weather_wind_out": 0.12,
        "weather_wind_in": -0.08,
        "temperature_f": 0.004,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    df["hr_score_raw"] = score
    df["p_hr"] = sigmoid_series(score)

    df["confidence"] = pd.cut(
        df["p_hr"],
        bins=[0.0, 0.06, 0.09, 0.12, 0.16, 1.0],
        labels=["C", "B-", "B+", "A", "A+"],
        include_lowest=True,
    )

    batter_name_col = _pick_col(df, ["batter_name", "player_name", "name"])
    team_col = _pick_col(df, ["team", "team_abbr", "batting_team"])
    opp_col = _pick_col(df, ["opponent"])

    keep = [
        c for c in [
            "game_date",
            team_col,
            opp_col,
            batter_name_col,
            "hr_score_raw",
            "p_hr",
            "confidence",
        ]
        if c is not None and c in df.columns
    ]

    board = df[keep].sort_values("p_hr", ascending=False).reset_index(drop=True)
    board["rank"] = range(1, len(board) + 1)

    cols = ["rank"] + [c for c in board.columns if c != "rank"]
    board = board[cols]

    out_csv = outputs_dir / f"hr_board_{season}_{date_str}.csv"
    out_parquet = outputs_dir / f"hr_board_{season}_{date_str}.parquet"

    board.to_csv(out_csv, index=False)
    board.to_parquet(out_parquet, index=False)

    print(f"✅ HR board built: {out_csv}")
    print(f"✅ HR board parquet: {out_parquet}")
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()