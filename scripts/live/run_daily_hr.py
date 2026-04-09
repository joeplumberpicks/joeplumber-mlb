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


def coalesce_numeric(df: pd.DataFrame, candidates: Iterable[str], default: float = 0.0) -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(default)
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

    # -----------------------------
    # Column resolution
    # -----------------------------
    game_date_col = pick_col(df, ["game_date"], required=False)
    team_col = pick_col(df, ["team", "bat_team", "offense_team"], required=True)
    opp_col = pick_col(df, ["opponent", "opp_team", "pitching_team"], required=False)
    player_col = pick_col(df, ["player_name", "batter_name", "name"], required=True)

    lineup_col = pick_col(
        df,
        ["lineup_spot", "batting_order", "order_spot", "confirmed_batting_order", "projected_batting_order"],
        required=False,
    )

    # Core power columns
    barrel_rate = coalesce_numeric(df, ["barrel_rate", "barrel_rate_roll30", "bat_barrel_rate_roll30"], default=0.0)
    iso = coalesce_numeric(df, ["iso", "iso_roll30", "bat_iso_roll30"], default=0.0)
    hr_per_pa = coalesce_numeric(df, ["hr_per_pa", "hr_per_pa_roll30", "bat_hr_per_pa_roll30"], default=0.0)
    hard_hit_rate = coalesce_numeric(df, ["hard_hit_rate", "hard_hit_rate_roll30", "bat_hard_hit_rate_roll30"], default=0.0)

    # Context / matchup columns
    ev = coalesce_numeric(df, ["avg_exit_velocity", "avg_ev", "exit_velocity_roll30", "bat_avg_ev_roll30"], default=0.0)
    la = coalesce_numeric(df, ["avg_launch_angle", "launch_angle_roll30", "bat_avg_la_roll30"], default=0.0)
    flyball = coalesce_numeric(df, ["fly_ball_rate", "fb_rate", "flyball_rate_roll30", "bat_fb_rate_roll30"], default=0.0)
    pulled_air = coalesce_numeric(df, ["pulled_air_rate", "pull_air_rate", "bat_pulled_air_rate_roll30"], default=0.0)

    pitcher_hr9 = coalesce_numeric(df, ["pitcher_hr9", "pit_hr9_roll30", "opp_pitcher_hr9"], default=0.0)
    pitcher_hard_hit = coalesce_numeric(df, ["pitcher_hard_hit_rate", "pit_hard_hit_rate_roll30"], default=0.0)
    pitcher_barrel = coalesce_numeric(df, ["pitcher_barrel_rate", "pit_barrel_rate_roll30"], default=0.0)

    park_hr = coalesce_numeric(df, ["park_hr_factor", "hr_park_factor", "park_factor_hr"], default=1.0)
    weather_hr = coalesce_numeric(df, ["weather_hr_boost", "hr_weather_boost"], default=0.0)
    temp_f = coalesce_numeric(df, ["temperature_f", "temp_f"], default=72.0)
    wind_out = coalesce_numeric(df, ["weather_wind_out", "wind_out"], default=0.0)

    prior_raw = coalesce_numeric(df, ["hr_score_raw", "score_raw", "score"], default=0.0)

    # -----------------------------
    # Hard power filter
    # -----------------------------
    hard_filter = (
        (barrel_rate >= 0.08) |
        (iso >= 0.180) |
        (hr_per_pa >= 0.040)
    )

    filtered = df.loc[hard_filter].copy()

    if filtered.empty:
        relaxed_filter = (
            (barrel_rate >= 0.06) |
            (iso >= 0.160) |
            (hr_per_pa >= 0.030)
        )
        filtered = df.loc[relaxed_filter].copy()

    if filtered.empty:
        filtered = df.copy()

    idx = filtered.index

    barrel_rate = barrel_rate.loc[idx]
    iso = iso.loc[idx]
    hr_per_pa = hr_per_pa.loc[idx]
    hard_hit_rate = hard_hit_rate.loc[idx]

    ev = ev.loc[idx]
    la = la.loc[idx]
    flyball = flyball.loc[idx]
    pulled_air = pulled_air.loc[idx]

    pitcher_hr9 = pitcher_hr9.loc[idx]
    pitcher_hard_hit = pitcher_hard_hit.loc[idx]
    pitcher_barrel = pitcher_barrel.loc[idx]

    park_hr = park_hr.loc[idx]
    weather_hr = weather_hr.loc[idx]
    temp_f = temp_f.loc[idx]
    wind_out = wind_out.loc[idx]

    prior_raw = prior_raw.loc[idx]

    # -----------------------------
    # Base score build
    # -----------------------------
    hr_power_score = (
        barrel_rate * 0.40 +
        iso * 0.30 +
        hard_hit_rate * 0.30
    )

    hitter_shape = (
        barrel_rate * 2.4 +
        iso * 1.8 +
        hr_per_pa * 2.0 +
        hard_hit_rate * 1.2 +
        flyball * 0.9 +
        pulled_air * 0.7
    )

    contact_quality = (
        (ev - 88.0) * 0.035 +
        (la - 12.0).clip(-10, 18) * 0.015
    )

    matchup = (
        pitcher_hr9 * 0.45 +
        pitcher_hard_hit * 0.80 +
        pitcher_barrel * 1.10
    )

    environment = (
        (park_hr - 1.0) * 1.25 +
        weather_hr * 0.80 +
        ((temp_f - 70.0) / 15.0) * 0.12 +
        wind_out * 0.18
    )

    score = (
        prior_raw * 0.35 +
        hitter_shape +
        contact_quality +
        matchup +
        environment
    )

    # HR power boost
    score = score + (hr_power_score * 2.0)

    # -----------------------------
    # Lineup penalty
    # -----------------------------
    lineup_spot = pd.Series(5, index=filtered.index, dtype=float)
    if lineup_col is not None:
        lineup_spot = pd.to_numeric(filtered[lineup_col], errors="coerce").fillna(5.0)

    score = np.where(lineup_spot <= 2, score * 1.08, score)
    score = np.where(lineup_spot == 3, score * 1.06, score)
    score = np.where(lineup_spot == 4, score * 1.04, score)
    score = np.where(lineup_spot == 5, score * 1.01, score)

    score = np.where(lineup_spot >= 7, score * 0.75, score)
    score = np.where(lineup_spot >= 8, score * 0.60, score)

    # -----------------------------
    # Additional guardrails
    # -----------------------------
    weak_power_penalty = (
        (barrel_rate < 0.075).astype(float) * 0.35 +
        (iso < 0.170).astype(float) * 0.25 +
        (hard_hit_rate < 0.38).astype(float) * 0.20
    )
    score = score - weak_power_penalty

    elite_boost = (
        (barrel_rate >= 0.12).astype(float) * 0.25 +
        (iso >= 0.240).astype(float) * 0.20 +
        (hr_per_pa >= 0.055).astype(float) * 0.20
    )
    score = score + elite_boost

    # -----------------------------
    # Slate normalization + confidence
    # -----------------------------
    filtered["hr_score_raw"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    filtered["z_score"] = slate_zscore(filtered["hr_score_raw"])

    filtered["confidence"] = pd.cut(
        filtered["z_score"],
        bins=[-10, -1, 0, 1, 1.8, 2.5, 10],
        labels=["F", "D", "C", "B", "A", "A+"],
        include_lowest=True,
    ).astype(str)

    filtered["p_hr"] = logistic(-3.10 + (filtered["z_score"] * 0.95))

    name_for_noise = coalesce_text(filtered, [player_col], default="")
    team_for_noise = coalesce_text(filtered, [team_col], default="")
    noise_key = name_for_noise + "|" + team_for_noise
    noise = noise_key.map(lambda x: (hash(x) % 1000) / 1_000_000.0).astype(float)
    filtered["sort_score"] = filtered["hr_score_raw"] + noise

    # -----------------------------
    # Final board
    # -----------------------------
    keep_cols = {
        "game_date": coalesce_text(filtered, [game_date_col] if game_date_col else [], default=args.date),
        "team": coalesce_text(filtered, [team_col]),
        "opponent": coalesce_text(filtered, [opp_col] if opp_col else [], default=""),
        "player_name": coalesce_text(filtered, [player_col]),
        "hr_score_raw": filtered["hr_score_raw"],
        "p_hr": filtered["p_hr"],
        "confidence": filtered["confidence"],
        "sort_score": filtered["sort_score"],  # <-- FIX
    }

    board = pd.DataFrame(keep_cols).copy()
    board = board.sort_values(["sort_score", "p_hr"], ascending=[False, False]).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))

    # Remove helper column after sorting
    board = board.drop(columns=["sort_score"])

    board_top = board.head(25).copy()

    board_top.to_csv(out_csv, index=False)
    board_top.to_parquet(out_parquet, index=False)

    print(f"✅ HR board built: {out_csv}")
    print(f"✅ HR board parquet: {out_parquet}")
    print(board_top.to_string(index=False))


if __name__ == "__main__":
    main()