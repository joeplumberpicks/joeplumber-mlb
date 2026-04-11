#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
    parser = argparse.ArgumentParser(description="Run daily HR board.")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--season", type=str, default="2026")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def logistic(x: pd.Series | np.ndarray) -> pd.Series:
    x = pd.Series(x)
    return 1.0 / (1.0 + np.exp(-x.clip(-20, 20)))


def slate_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    std = float(s.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - float(s.mean())) / std).clip(-3.0, 3.0)


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def first_non_null_numeric(df: pd.DataFrame, candidates: list[str], default: float | None = None) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    found = False
    for c in candidates:
        if c in df.columns:
            found = True
            out = out.combine_first(pd.to_numeric(df[c], errors="coerce"))
    if not found and default is not None:
        out = pd.Series(default, index=df.index, dtype=float)
    if default is not None:
        out = out.fillna(default)
    return out


def fill_with_median(series: pd.Series, fallback: float) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    non_na = s.dropna()
    if non_na.empty:
        return s.fillna(fallback)
    return s.fillna(float(non_na.median()))


def series_null_pct(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").isna().mean()) * 100.0


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
    out_csv = outputs_dir / f"hr_board_{season}_{date_str}.csv"
    out_parquet = outputs_dir / f"hr_board_{season}_{date_str}.parquet"

    print(f"=== JOE PLUMBER HR RUN :: {date_str} ===")
    print(f"Loading features: {feat_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing features file: {feat_path}")

    df = pd.read_parquet(feat_path).copy()
    print(f"Row count [hr_features]: {len(df):,}")
    if df.empty:
        raise ValueError("HR features file is empty.")

    game_date_col = pick_col(df, ["game_date"])
    team_col = pick_col(df, ["team"], required=True)
    opp_col = pick_col(df, ["opponent"])
    player_col = pick_col(df, ["player_name", "batter_name"], required=True)

    barrel_rate = first_non_null_numeric(df, ["barrel_rate"])
    iso = first_non_null_numeric(df, ["iso"])
    hr_per_pa = first_non_null_numeric(df, ["hr_per_pa"])
    hard_hit_rate = first_non_null_numeric(df, ["hard_hit_rate"])
    flyball = first_non_null_numeric(df, ["flyball"])
    pulled_air = first_non_null_numeric(df, ["pulled_air"])
    ev = first_non_null_numeric(df, ["ev"])
    la = first_non_null_numeric(df, ["la"])
    tb_per_pa = first_non_null_numeric(df, ["tb_per_pa"])
    bb_rate = first_non_null_numeric(df, ["bb_rate"])
    k_rate = first_non_null_numeric(df, ["k_rate"])
    lineup_spot = first_non_null_numeric(df, ["lineup_spot"], default=6.0)
    lineup_weight = first_non_null_numeric(df, ["lineup_weight"], default=1.0)

    pitcher_hr9 = first_non_null_numeric(df, ["opp_pitcher_hr9"])
    pitcher_barrel = first_non_null_numeric(df, ["opp_pitcher_barrel_rate"])
    pitcher_hard_hit = first_non_null_numeric(df, ["opp_pitcher_hard_hit_rate"])
    pitcher_bb = first_non_null_numeric(df, ["opp_pitcher_bb_rate"])

    temperature_f = first_non_null_numeric(df, ["temperature_f"], default=72.0)
    env_temp_boost = first_non_null_numeric(df, ["env_temp_boost"])
    env_wind_out_effect = first_non_null_numeric(df, ["env_wind_out_effect"])
    env_wind_in_effect = first_non_null_numeric(df, ["env_wind_in_effect"])
    env_crosswind_effect = first_non_null_numeric(df, ["env_crosswind_effect"])
    park_hr_factor = first_non_null_numeric(df, ["park_hr_factor"], default=1.0)
    hr_weather_delta = first_non_null_numeric(df, ["hr_weather_delta"], default=0.0)

    print("\n=== HR FEATURE AUDIT ===")
    print(f"barrel_rate: null_pct={series_null_pct(barrel_rate):.2f}%")
    print(f"iso: null_pct={series_null_pct(iso):.2f}%")
    print(f"hr_per_pa: null_pct={series_null_pct(hr_per_pa):.2f}%")
    print(f"hard_hit_rate: null_pct={series_null_pct(hard_hit_rate):.2f}%")
    print(f"flyball: null_pct={series_null_pct(flyball):.2f}%")
    print(f"pulled_air: null_pct={series_null_pct(pulled_air):.2f}%")
    print(f"ev: null_pct={series_null_pct(ev):.2f}%")
    print(f"la: null_pct={series_null_pct(la):.2f}%")
    print(f"lineup_spot: null_pct={series_null_pct(lineup_spot):.2f}%")

    barrel_rate = fill_with_median(barrel_rate, 0.055)
    iso = fill_with_median(iso, 0.155)
    hr_per_pa = fill_with_median(hr_per_pa, 0.025)
    hard_hit_rate = fill_with_median(hard_hit_rate, 0.38)
    flyball = fill_with_median(flyball, 0.35)
    pulled_air = fill_with_median(pulled_air, 0.10)
    ev = fill_with_median(ev, 89.0)
    la = fill_with_median(la, 13.0)
    tb_per_pa = fill_with_median(tb_per_pa, 0.45)
    bb_rate = fill_with_median(bb_rate, 0.08)
    k_rate = fill_with_median(k_rate, 0.22)

    pitcher_hr9 = fill_with_median(pitcher_hr9, 1.10)
    pitcher_barrel = fill_with_median(pitcher_barrel, 0.08)
    pitcher_hard_hit = fill_with_median(pitcher_hard_hit, 0.40)
    pitcher_bb = fill_with_median(pitcher_bb, 0.08)

    env_temp_boost = fill_with_median(env_temp_boost, 0.0)
    env_wind_out_effect = fill_with_median(env_wind_out_effect, 0.0)
    env_wind_in_effect = fill_with_median(env_wind_in_effect, 0.0)
    env_crosswind_effect = fill_with_median(env_crosswind_effect, 0.0)
    park_hr_factor = fill_with_median(park_hr_factor, 1.0)
    hr_weather_delta = fill_with_median(hr_weather_delta, 0.0)

    power_flag = (
        (barrel_rate >= 0.055) |
        (iso >= 0.150) |
        (hr_per_pa >= 0.022)
    ).astype(int)

    power_penalty = np.where(power_flag == 0, 0.92, 1.00)

    contact_only_hr = (
        barrel_rate * 2.20 +
        iso * 1.50 +
        hr_per_pa * 1.80 +
        hard_hit_rate * 0.90 +
        ((ev - 88.0).clip(lower=0.0) * 0.020) +
        ((la - 12.0).clip(-10, 18) * 0.010) +
        pitcher_hr9 * 0.35 +
        pitcher_barrel * 0.55 +
        pitcher_hard_hit * 0.40 +
        pitcher_bb * 0.06
    )

    cpw_delta = hr_weather_delta

    score = contact_only_hr + cpw_delta

    score = np.where(lineup_spot <= 2, score * 1.02, score)
    score = np.where(lineup_spot == 3, score * 1.05, score)
    score = np.where(lineup_spot == 4, score * 1.07, score)
    score = np.where(lineup_spot == 5, score * 1.03, score)
    score = np.where(lineup_spot >= 7, score * 0.92, score)
    score = np.where(lineup_spot >= 8, score * 0.84, score)
    score = np.where(lineup_spot >= 9, score * 0.76, score)

    score = score * lineup_weight.clip(lower=0.80, upper=1.15)
    score = score * power_penalty

    weak_power_penalty = (
        (barrel_rate < 0.050).astype(float) * 0.08 +
        (iso < 0.140).astype(float) * 0.06 +
        (hard_hit_rate < 0.34).astype(float) * 0.05
    )
    elite_boost = (
        (barrel_rate >= 0.12).astype(float) * 0.18 +
        (iso >= 0.25).astype(float) * 0.14 +
        (hr_per_pa >= 0.050).astype(float) * 0.14
    )
    score = score - weak_power_penalty + elite_boost

    df["hr_score_raw"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    df["z_score"] = slate_zscore(df["hr_score_raw"])

    df["p_hr"] = logistic(-2.05 + (df["z_score"] * 0.72)).clip(0.05, 0.24)

    df["confidence"] = pd.cut(
        df["z_score"],
        bins=[-10, -1.0, -0.20, 0.45, 1.10, 1.80, 10],
        labels=["F", "D", "C", "B", "A", "A+"],
        include_lowest=True,
    ).astype(str)

    if "tie_break_noise" in df.columns:
        noise = pd.to_numeric(df["tie_break_noise"], errors="coerce").fillna(0.0) / 1000.0
    else:
        noise = (df[player_col].astype(str) + "|" + df[team_col].astype(str)).map(
            lambda x: (hash(x) % 1000) / 1_000_000.0
        ).astype(float)

    df["sort_score"] = df["z_score"] + noise

    board = pd.DataFrame({
        "game_date": df[game_date_col] if game_date_col in df.columns else date_str,
        "team": df[team_col],
        "opponent": df[opp_col] if opp_col in df.columns else "",
        "player_name": df[player_col],
        "hr_score_raw": df["hr_score_raw"],
        "confidence": df["confidence"],
        "p_hr": df["p_hr"],
        "sort_score": df["sort_score"],
    })

    board = board.sort_values(["sort_score", "p_hr"], ascending=[False, False]).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    board = board.drop(columns=["sort_score"])
    board_top = board.head(25).copy()

    board_top.to_csv(out_csv, index=False)
    board_top.to_parquet(out_parquet, index=False)

    print(f"✅ HR board built: {out_csv}")
    print(f"✅ HR board parquet: {out_parquet}")
    print(board_top.to_string(index=False))


if __name__ == "__main__":
    main()