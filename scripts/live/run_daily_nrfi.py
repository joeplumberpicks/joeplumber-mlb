#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _sigmoid(x: pd.Series) -> pd.Series:
    return 1.0 / (1.0 + (-x).apply(pd.np.exp))


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config((repo_root / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    outputs_dir = Path(dirs["outputs_dir"])

    date_str = args.date or get_today_date()
    season = str(args.season)

    feat_path = processed_dir / "live" / f"nrfi_features_{season}_{date_str}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")

    df = pd.read_parquet(feat_path).copy()

    score = pd.Series(0.0, index=df.index)

    feature_weights = {
        "home_sp_k_rate_roll7": 0.90,
        "away_sp_k_rate_roll7": 0.90,
        "home_sp_bb_rate_roll7": -0.75,
        "away_sp_bb_rate_roll7": -0.75,
        "home_sp_hr_rate_roll7": -0.85,
        "away_sp_hr_rate_roll7": -0.85,
        "home_top3_hardhit_rate_roll7": -0.55,
        "away_top3_hardhit_rate_roll7": -0.55,
        "home_top3_barrel_rate_roll7": -0.65,
        "away_top3_barrel_rate_roll7": -0.65,
    }

    for col, wt in feature_weights.items():
        if col in df.columns:
            score = score + wt * pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # weather/park leans if present
    for col, wt in {
        "weather_wind_out": -0.20,
        "temperature_f": -0.003,
        "weather_wind_in": 0.12,
    }.items():
        if col in df.columns:
            score = score + wt * pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["nrfi_score_raw"] = score
    df["nrfi_prob"] = 1.0 / (1.0 + pd.Series((-score).clip(-20, 20)).map(lambda x: __import__("math").exp(x)))
    df["yrfi_prob"] = 1.0 - df["nrfi_prob"]
    df["pick"] = df["nrfi_prob"].ge(0.5).map({True: "NRFI", False: "YRFI"})

    keep = [c for c in ["game_date", "away_team", "home_team", "nrfi_prob", "yrfi_prob", "pick"] if c in df.columns]
    board = df[keep].sort_values("nrfi_prob", ascending=False).reset_index(drop=True)

    out_path = outputs_dir / f"nrfi_board_{season}_{date_str}.csv"
    board.to_csv(out_path, index=False)

    print(f"✅ NRFI board built: {out_path}")
    print(board.head(15).to_string(index=False))


if __name__ == "__main__":
    main()