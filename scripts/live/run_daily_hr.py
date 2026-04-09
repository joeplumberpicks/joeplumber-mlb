#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_data_root(config: dict) -> Path:
    drive_root = config.get("drive_data_root", "joeplumber-mlb/data")
    return Path("/content/drive/MyDrive") / drive_root


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = False) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {list(candidates)}")
    return None


def coalesce_numeric(
    df: pd.DataFrame,
    candidates: Iterable[str],
    default: float | None = None,
) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if default is not None:
                s = s.fillna(default)
            return s
    if default is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.Series(default, index=df.index, dtype=float)


def coalesce_text(df: pd.DataFrame, candidates: Iterable[str], default: str = "") -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def logistic(x: pd.Series | np.ndarray) -> pd.Series:
    x = pd.Series(x)
    return 1.0 / (1.0 + np.exp(-x.clip(-20, 20)))


def slate_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    std = float(s.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    z = (s - float(s.mean())) / std
    return z.clip(-3.0, 3.0)


def fill_with_median(series: pd.Series, fallback: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    non_na = s.dropna()
    if non_na.empty:
        return s.fillna(fallback)
    return s.fillna(float(non_na.median()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    args = parser.parse_args()

    print(f"=== JOE PLUMBER HR RUN :: {args.date} ===")

    config = load_config(args.config)
    data_root = resolve_data_root(config)
    processed_live = data_root / "processed" / "live"
    outputs_dir = data_root / "outputs"
    ensure_dir(outputs_dir)

    features_path = processed_live / f"hr_features_{args.season}_{args.date}.parquet"
    out_csv = outputs_dir / f"hr_board_{args.season}_{args.date}.csv"
    out_parquet = outputs_dir / f"hr_board_{args.season}_{args.date}.parquet"

    print(f"Loading features: {features_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    df = pd.read_parquet(features_path).copy()
    print(f"Row count [hr_features]: {len(df):,}")
    if df.empty:
        raise ValueError("HR features file is empty.")

    game_date_col = pick_col(df, ["game_date"], required=False)
    team_col = pick_col(df, ["team", "bat_team", "offense_team"], required=True)
    opp_col = pick_col(df, ["opponent", "opp_team", "pitching_team"], required=False)
    player_col = pick_col(df, ["player_name", "batter_name", "name"], required=True)

    print("\n=== HR FEATURE AUDIT ===")
    audit_targets = {
        "barrel_rate": ["bat_barrel_rate_roll30", "bat_barrel_rate_roll15"],
        "iso": ["bat_iso_roll30", "bat_iso_roll15"],
        "hr_per_pa": ["bat_hr_per_pa_roll30", "bat_hr_per_pa_roll15"],
        "hard_hit_rate": ["bat_hard_hit_rate_roll30", "bat_hard_hit_rate_roll15"],
        "lineup_spot": ["batting_order"],
    }
    for label, candidates in audit_targets.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            print(f"❌ MISSING COLUMN: {label}")
        else:
            null_pct = float(df[found].isna().mean()) * 100.0
            print(f"{label} -> {found}: null_pct={null_pct:.2f}%")

    # =========================
    # TRUE HR FEATURE MAPPING
    # =========================

    barrel_rate = coalesce_numeric(df, [
        "bat_barrel_rate_roll30",
        "bat_barrel_rate_roll15",
    ])

    iso = coalesce_numeric(df, [
        "bat_iso_roll30",
        "bat_iso_roll15",
    ])

    hr_per_pa = coalesce_numeric(df, [
        "bat_hr_per_pa_roll30",
        "bat_hr_per_pa_roll15",
    ])

    hard_hit_rate = coalesce_numeric(df, [
        "bat_hard_hit_rate_roll30",
        "bat_hard_hit_rate_roll15",
    ])

    flyball = coalesce_numeric(df, [
        "bat_fb_rate_roll30",
        "bat_fb_rate_roll15",
    ])

    pulled_air = coalesce_numeric(df, [
        "bat_pulled_air_rate_roll30",
        "bat_pulled_air_rate_roll15",
    ])

    ev = coalesce_numeric(df, [
        "bat_avg_ev_roll30",
        "bat_avg_ev_roll15",
    ])

    la = coalesce_numeric(df, [
        "bat_avg_la_roll30",
        "bat_avg_la_roll15",
    ])

    pitcher_hr9 = coalesce_numeric(df, [
        "opp_pit_hr9_roll30",
        "opp_pit_hr9_roll15",
    ])

    pitcher_barrel = coalesce_numeric(df, [
        "opp_pit_barrel_rate_roll30",
        "opp_pit_barrel_rate_roll15",
    ])

    pitcher_hard_hit = coalesce_numeric(df, [
        "opp_pit_hard_hit_rate_roll30",
        "opp_pit_hard_hit_rate_roll15",
    ])

    pitcher_bb = coalesce_numeric(df, [
        "opp_pit_bb_rate_roll30",
        "opp_pit_bb_rate_roll15",
    ])

    temp_f = coalesce_numeric(df, ["temperature_f"], default=72.0)
    wind_out = coalesce_numeric(df, ["weather_wind_out"], default=0.0)
    lineup_spot = coalesce_numeric(df, ["batting_order"], default=6.0)

    base_score = coalesce_numeric(df, ["hr_score_raw", "score_raw", "score"], default=0.0)

    # hard validation
    if barrel_rate.isna().all():
        raise ValueError("🚨 barrel_rate missing — HR model invalid")
    if iso.isna().all():
        raise ValueError("🚨 iso missing — HR model invalid")

    # fill missing values
    barrel_rate = fill_with_median(barrel_rate, 0.06)
    iso = fill_with_median(iso, 0.16)
    hr_per_pa = fill_with_median(hr_per_pa, 0.025)
    hard_hit_rate = fill_with_median(hard_hit_rate, 0.38)
    flyball = fill_with_median(flyball, 0.35)
    pulled_air = fill_with_median(pulled_air, 0.10)
    ev = fill_with_median(ev, 89.0)
    la = fill_with_median(la, 13.0)

    pitcher_hr9 = fill_with_median(pitcher_hr9, 1.10)
    pitcher_barrel = fill_with_median(pitcher_barrel, 0.08)
    pitcher_hard_hit = fill_with_median(pitcher_hard_hit, 0.40)
    pitcher_bb = fill_with_median(pitcher_bb, 0.08)

    temp_f = fill_with_median(temp_f, 72.0)
    wind_out = fill_with_median(wind_out, 0.0)
    lineup_spot = fill_with_median(lineup_spot, 6.0)
    base_score = pd.to_numeric(base_score, errors="coerce").fillna(0.0)

    # soft power gate
    power_flag = (
        (barrel_rate >= 0.06) |
        (iso >= 0.160) |
        (hr_per_pa >= 0.025)
    ).astype(int)

    power_penalty = np.where(power_flag == 0, 0.88, 1.00)

    # power spine
    hr_power_score = (
        barrel_rate * 0.45 +
        iso * 0.30 +
        hard_hit_rate * 0.25
    )

    hitter_shape = (
        barrel_rate * 2.00 +
        iso * 1.35 +
        hr_per_pa * 1.80 +
        hard_hit_rate * 0.80 +
        flyball * 0.50 +
        pulled_air * 0.40
    )

    contact_quality = (
        (ev - 88.0) * 0.020 +
        (la - 12.0).clip(-10, 18) * 0.010
    )

    matchup = (
        pitcher_hr9 * 0.28 +
        pitcher_hard_hit * 0.45 +
        pitcher_barrel * 0.60 +
        pitcher_bb * 0.08
    )

    environment = (
        ((temp_f - 70.0) / 15.0) * 0.06 +
        wind_out * 0.10
    )

    score = base_score * 0.25
    score += hitter_shape
    score += contact_quality
    score += matchup
    score += environment
    score += hr_power_score * 1.25
    score *= power_penalty

    # lineup adjustment
    score = np.where(lineup_spot <= 2, score * 1.03, score)
    score = np.where(lineup_spot == 3, score * 1.06, score)
    score = np.where(lineup_spot == 4, score * 1.08, score)
    score = np.where(lineup_spot == 5, score * 1.03, score)
    score = np.where(lineup_spot >= 7, score * 0.90, score)
    score = np.where(lineup_spot >= 8, score * 0.80, score)
    score = np.where(lineup_spot >= 9, score * 0.72, score)

    weak_power_penalty = (
        (barrel_rate < 0.055).astype(float) * 0.10 +
        (iso < 0.145).astype(float) * 0.08 +
        (hard_hit_rate < 0.34).astype(float) * 0.06
    )
    score = score - weak_power_penalty

    elite_boost = (
        (barrel_rate >= 0.12).astype(float) * 0.18 +
        (iso >= 0.250).astype(float) * 0.14 +
        (hr_per_pa >= 0.050).astype(float) * 0.14
    )
    score = score + elite_boost

    df["hr_score_raw"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    df["z_score"] = slate_zscore(df["hr_score_raw"])

    # ranking vs calibration separated
    df["p_hr"] = logistic(-2.35 + (df["z_score"] * 0.72))

    df["confidence"] = pd.cut(
        df["z_score"],
        bins=[-10, -1.0, -0.25, 0.5, 1.25, 2.0, 10],
        labels=["F", "D", "C", "B", "A", "A+"],
        include_lowest=True,
    ).astype(str)

    if "tie_break_noise" in df.columns:
        noise = pd.to_numeric(df["tie_break_noise"], errors="coerce").fillna(0.0) / 1000.0
    else:
        name_for_noise = coalesce_text(df, [player_col], default="")
        team_for_noise = coalesce_text(df, [team_col], default="")
        noise = (name_for_noise + "|" + team_for_noise).map(
            lambda x: (hash(x) % 1000) / 1_000_000.0
        ).astype(float)

    df["sort_score"] = df["z_score"] + noise

    board = pd.DataFrame({
        "game_date": coalesce_text(df, [game_date_col] if game_date_col else [], default=args.date),
        "team": coalesce_text(df, [team_col]),
        "opponent": coalesce_text(df, [opp_col] if opp_col else [], default=""),
        "player_name": coalesce_text(df, [player_col]),
        "hr_score_raw": df["hr_score_raw"],
        "p_hr": df["p_hr"],
        "confidence": df["confidence"],
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