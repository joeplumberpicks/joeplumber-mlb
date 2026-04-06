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


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config((repo_root / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    outputs_dir = Path(dirs["outputs_dir"])

    date_str = args.date or get_today_date()
    season = str(args.season)

    feat_path = processed_dir / "live" / f"hr_features_{season}_{date_str}.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing: {feat_path}")

    df = pd.read_parquet(feat_path).copy()

    score = pd.Series(0.0, index=df.index)

    weights = {
        "barrel_rate_roll15": 1.35,
        "hardhit_rate_roll15": 0.95,
        "ev_mean_roll15": 0.025,
        "la_mean_roll15": 0.020,
        "hr_roll15": 0.30,
        "hr_allowed_roll15": 0.85,
        "barrel_rate_allowed_roll15": 1.00,
        "hardhit_rate_allowed_roll15": 0.70,
    }

    for col, wt in weights.items():
        if col in df.columns:
            score = score + wt * pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col, wt in {"weather_wind_out": 0.12, "temperature_f": 0.004}.items():
        if col in df.columns:
            score = score + wt * pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["hr_score"] = score

    batter_name_col = _pick_col(df, ["batter_name", "player_name", "name"])
    keep = [c for c in ["game_date", "team", "opponent", batter_name_col, "hr_score"] if c is not None and c in df.columns]

    board = df[keep].sort_values("hr_score", ascending=False).reset_index(drop=True)

    out_path = outputs_dir / f"hr_board_{season}_{date_str}.csv"
    board.to_csv(out_path, index=False)

    print(f"✅ HR board built: {out_path}")
    print(board.head(25).to_string(index=False))


if __name__ == "__main__":
    main()