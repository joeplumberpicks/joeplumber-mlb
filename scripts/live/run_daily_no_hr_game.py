#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
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


def sigmoid_series(x: pd.Series | np.ndarray) -> pd.Series:
    x = pd.Series(x, dtype="float64")
    return 1.0 / (1.0 + np.exp(-x.clip(-20, 20)))


def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    std = x.std(ddof=0)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(0.0, index=x.index, dtype="float64")
    return (x - x.mean()) / std


def confidence_from_prob(prob: pd.Series) -> pd.Series:
    return pd.cut(
        prob,
        bins=[0.0, 0.52, 0.58, 0.66, 0.75, 1.0],
        labels=["C", "B-", "B+", "A", "A+"],
        include_lowest=True,
    )


def first_numeric(df: pd.DataFrame, candidates: list[str], default: float | None = None) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    found = False
    for c in candidates:
        if c in df.columns:
            found = True
            out = out.combine_first(pd.to_numeric(df[c], errors="coerce"))
    if not found and default is not None:
        out = pd.Series(default, index=df.index, dtype="float64")
    if default is not None:
        out = out.fillna(default)
    return out


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

    # ------------------------------------------------------------------
    # PRIMARY MODEL INPUTS
    # Trust the feature builder if it already produced game-level HR expectations.
    # ------------------------------------------------------------------
    expected_hr_neutral = first_numeric(df, ["expected_hr_neutral"], default=np.nan)
    expected_hr_today = first_numeric(df, ["expected_hr_today"], default=np.nan)
    env_hr_delta = first_numeric(df, ["env_hr_delta"], default=np.nan)
    expected_hr_today_adj = first_numeric(df, ["expected_hr_today_adj"], default=np.nan)
    p_no_hr_game_est = first_numeric(df, ["p_no_hr_game_est"], default=np.nan)
    p_yes_hr_game_est = first_numeric(df, ["p_yes_hr_game_est"], default=np.nan)

    # lineup completeness helpers
    home_lineup_count = first_numeric(df, ["home_lineup_count"], default=9.0)
    away_lineup_count = first_numeric(df, ["away_lineup_count"], default=9.0)

    missing_home_starter = (
        df["home_starter_pitcher_id"].isna().astype(float)
        if "home_starter_pitcher_id" in df.columns else pd.Series(0.0, index=df.index)
    )
    missing_away_starter = (
        df["away_starter_pitcher_id"].isna().astype(float)
        if "away_starter_pitcher_id" in df.columns else pd.Series(0.0, index=df.index)
    )

    # ------------------------------------------------------------------
    # FALLBACK REPAIR
    # If expected_hr_today_adj wasn't built upstream, create it here.
    # ------------------------------------------------------------------
    if expected_hr_neutral.isna().all():
        expected_hr_neutral = pd.Series(0.90, index=df.index, dtype="float64")

    if expected_hr_today.isna().all():
        expected_hr_today = expected_hr_neutral.copy()

    if env_hr_delta.isna().all():
        env_hr_delta = expected_hr_today - expected_hr_neutral

    if expected_hr_today_adj.isna().all():
        lineup_penalty = (
            (home_lineup_count < 7).astype(float) * 0.08
            + (away_lineup_count < 7).astype(float) * 0.08
        )
        starter_penalty = missing_home_starter * 0.05 + missing_away_starter * 0.05
        expected_hr_today_adj = (expected_hr_today + lineup_penalty + starter_penalty).clip(lower=0.02)

    expected_hr_neutral = expected_hr_neutral.fillna(expected_hr_neutral.median() if expected_hr_neutral.notna().any() else 0.90)
    expected_hr_today = expected_hr_today.fillna(expected_hr_today.median() if expected_hr_today.notna().any() else 0.95)
    env_hr_delta = env_hr_delta.fillna(expected_hr_today - expected_hr_neutral)
    expected_hr_today_adj = expected_hr_today_adj.fillna(expected_hr_today_adj.median() if expected_hr_today_adj.notna().any() else 0.95)

    # ------------------------------------------------------------------
    # PROBABILITY LAYER
    # Preferred: use Poisson estimate from feature build.
    # If not present, compute from adjusted expected HR.
    # ------------------------------------------------------------------
    if p_no_hr_game_est.isna().all():
        p_no_hr_game_est = np.exp(-expected_hr_today_adj.clip(0.02, 6.0))
    else:
        p_no_hr_game_est = p_no_hr_game_est.fillna(np.exp(-expected_hr_today_adj.clip(0.02, 6.0)))

    if p_yes_hr_game_est.isna().all():
        p_yes_hr_game_est = 1.0 - p_no_hr_game_est
    else:
        p_yes_hr_game_est = p_yes_hr_game_est.fillna(1.0 - p_no_hr_game_est)

    # ------------------------------------------------------------------
    # Optional light calibration layer
    # Keeps probabilities from being too extreme while preserving ranking.
    # ------------------------------------------------------------------
    score_base = -expected_hr_today_adj
    score_z = zscore_series(score_base)

    calibrated_no_hr = (
        0.78 * p_no_hr_game_est
        + 0.22 * sigmoid_series(score_z * 0.90)
    ).clip(0.02, 0.98)

    df["expected_hr_neutral"] = expected_hr_neutral
    df["expected_hr_today"] = expected_hr_today
    df["env_hr_delta"] = env_hr_delta
    df["expected_hr_today_adj"] = expected_hr_today_adj
    df["no_hr_game_score_raw"] = score_base
    df["p_no_hr_game"] = calibrated_no_hr
    df["p_yes_hr_game"] = 1.0 - df["p_no_hr_game"]
    df["pick"] = df["p_no_hr_game"].ge(0.50).map({True: "NO_HR", False: "HR_YES"})
    df["confidence"] = confidence_from_prob(df[["p_no_hr_game", "p_yes_hr_game"]].max(axis=1))

    keep = [
        c for c in [
            "game_date",
            "away_team",
            "home_team",
            "expected_hr_neutral",
            "expected_hr_today",
            "env_hr_delta",
            "expected_hr_today_adj",
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