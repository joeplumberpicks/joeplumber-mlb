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

    # Hitter production form
    for col, wt in {
        "rbi_roll15": 1.10,
        "rbi_roll30": 0.75,
        "tb_roll15": 0.35,
        "tb_roll30": 0.22,
        "hardhit_rate_roll15": 0.75,
        "hardhit_rate_roll30": 0.45,
        "barrel_rate_roll15": 0.55,
        "barrel_rate_roll30": 0.35,
        "bb_rate_roll15": 0.18,
        "k_rate_roll15": -0.15,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Opposing pitcher run allowance / baserunner pressure
    for col, wt in {
        "opp_bb_rate_roll15": 0.30,
        "opp_bb_rate_roll30": 0.18,
        "opp_hr_allowed_roll15": 0.35,
        "opp_hr_allowed_roll30": 0.22,
        "opp_runs_rate_roll15": 0.45,
        "opp_runs_rate_roll30": 0.28,
        "opp_hardhit_rate_allowed_roll15": 0.30,
        "opp_barrel_rate_allowed_roll15": 0.22,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Team context / lineup support
    for col, wt in {
        "team_ctx_bb_rate_roll15": 0.32,
        "team_ctx_bb_rate_roll7": 0.22,
        "team_ctx_tb_roll15": 0.22,
        "team_ctx_rbi_roll15": 0.18,
        "team_ctx_hardhit_rate_roll15": 0.28,
        "team_ctx_barrel_rate_roll15": 0.18,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Builder-generated matchup pressure
    for col, wt in {
        "rbi_walk_pressure_roll15": 0.28,
        "rbi_power_pressure_roll15": 0.35,
        "rbi_contact_pressure_roll15": 0.26,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    # Lineup slot bonus
    lineup_slot_col = _pick_col(df, ["lineup_slot", "batting_order", "order_spot", "slot"])
    if lineup_slot_col is not None:
        slot_num = pd.to_numeric(df[lineup_slot_col], errors="coerce")
        score = score + slot_num.map({
            1: 0.10,
            2: 0.20,
            3: 0.45,
            4: 0.55,
            5: 0.35,
            6: 0.15,
        }).fillna(0.0)

    # Environment
    for col, wt in {
        "weather_wind_out": 0.08,
        "weather_wind_in": -0.04,
        "temperature_f": 0.002,
    }.items():
        score = add_weighted_feature(score, df, col, wt)

    df["rbi_score_raw"] = score
    df["p_rbi"] = sigmoid_series(score)

    df["confidence"] = pd.cut(
        df["p_rbi"],
        bins=[0.0, 0.18, 0.24, 0.31, 0.40, 1.0],
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
            "rbi_score_raw",
            "p_rbi",
            "confidence",
        ]
        if c is not None and c in df.columns
    ]

    board = df[keep].sort_values("p_rbi", ascending=False).reset_index(drop=True)
    board["rank"] = range(1, len(board) + 1)

    cols = ["rank"] + [c for c in board.columns if c != "rank"]
    board = board[cols]

    out_csv = outputs_dir / f"rbi_board_{season}_{date_str}.csv"
    out_parquet = outputs_dir / f"rbi_board_{season}_{date_str}.parquet"

    board.to_csv(out_csv, index=False)
    board.to_parquet(out_parquet, index=False)

    print(f"✅ RBI board built: {out_csv}")
    print(f"✅ RBI board parquet: {out_parquet}")
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()