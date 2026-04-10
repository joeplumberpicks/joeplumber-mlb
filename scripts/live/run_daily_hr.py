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


def coalesce_text(df: pd.DataFrame, candidates: Iterable[str], default: str = "") -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].fillna(default).astype(str)
    return pd.Series(default, index=df.index, dtype="object")


def first_non_null_numeric(df: pd.DataFrame, candidates: Iterable[str], default: float | None = None) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    found_any = False
    for c in candidates:
        if c in df.columns:
            found_any = True
            s = pd.to_numeric(df[c], errors="coerce")
            out = out.combine_first(s)
    if default is not None:
        out = out.fillna(default)
    if not found_any and default is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return out


def fill_with_median(series: pd.Series, fallback: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    non_na = s.dropna()
    if non_na.empty:
        return s.fillna(fallback)
    return s.fillna(float(non_na.median()))


def logistic(x: pd.Series | np.ndarray) -> pd.Series:
    x = pd.Series(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x.clip(-20, 20)))


def slate_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    std = float(s.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=s.index)
    z = (s - float(s.mean())) / std
    return z.clip(-3.0, 3.0)


def series_null_pct(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").isna().mean()) * 100.0


def clamp_prob_by_rank(p: pd.Series, rank_index: pd.Index) -> pd.Series:
    """
    Soft cap probability shape so top ranks do not all plateau.
    rank_index is 0-based after sorting descending.
    """
    p = pd.Series(p, index=rank_index, dtype=float).copy()

    # soft descending caps by rank bucket
    caps = []
    for i in rank_index:
        r = int(i) + 1
        if r <= 3:
            caps.append(0.24)
        elif r <= 5:
            caps.append(0.21)
        elif r <= 10:
            caps.append(0.18)
        elif r <= 15:
            caps.append(0.15)
        elif r <= 25:
            caps.append(0.12)
        else:
            caps.append(0.10)
    caps = pd.Series(caps, index=rank_index, dtype=float)

    p = p.clip(lower=0.03)
    p = np.minimum(p, caps)

    # enforce slight monotone decay after sort
    vals = p.values.astype(float)
    for i in range(1, len(vals)):
        vals[i] = min(vals[i], vals[i - 1] - 0.0005 if vals[i - 1] > 0.031 else vals[i])
        vals[i] = max(vals[i], 0.03)
    return pd.Series(vals, index=rank_index)


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

    game_date_col = pick_col(df, ["game_date"])
    team_col = pick_col(df, ["team"], required=True)
    opp_col = pick_col(df, ["opponent"])
    player_col = pick_col(df, ["player_name"], required=True)

    # ------------------------------------------------------------------
    # FEATURE MAP
    # ------------------------------------------------------------------
    barrel_rate = first_non_null_numeric(df, [
        "bat_barrel_rate_roll30",
        "bat_barrel_rate_roll15",
        "bat_barrel_rate_roll7",
        "bat_barrel_rate_roll3",
    ])

    iso = first_non_null_numeric(df, [
        "bat_iso_roll30",
        "bat_iso_roll15",
        "bat_iso_roll7",
        "bat_iso_roll3",
    ])

    hr_per_pa = first_non_null_numeric(df, [
        "bat_hr_per_pa_roll30",
        "bat_hr_per_pa_roll15",
        "bat_hr_per_pa_roll7",
        "bat_hr_per_pa_roll3",
    ])

    hard_hit_rate = first_non_null_numeric(df, [
        "bat_hard_hit_rate_roll30",
        "bat_hard_hit_rate_roll15",
        "bat_hard_hit_rate_roll7",
        "bat_hard_hit_rate_roll3",
    ])

    flyball = first_non_null_numeric(df, [
        "bat_fb_rate_roll30",
        "bat_fb_rate_roll15",
        "bat_fb_rate_roll7",
        "bat_fb_rate_roll3",
    ])

    pulled_air = first_non_null_numeric(df, [
        "bat_pulled_air_rate_roll30",
        "bat_pulled_air_rate_roll15",
        "bat_pulled_air_rate_roll7",
        "bat_pulled_air_rate_roll3",
    ])

    ev = first_non_null_numeric(df, [
        "bat_avg_ev_roll30",
        "bat_avg_ev_roll15",
        "bat_avg_ev_roll7",
        "bat_avg_ev_roll3",
    ])

    la = first_non_null_numeric(df, [
        "bat_avg_la_roll30",
        "bat_avg_la_roll15",
        "bat_avg_la_roll7",
        "bat_avg_la_roll3",
    ])

    tb_per_pa = first_non_null_numeric(df, [
        "bat_tb_per_pa_roll30",
        "bat_tb_per_pa_roll15",
        "bat_tb_per_pa_roll7",
        "bat_tb_per_pa_roll3",
    ])

    pitcher_hr9 = first_non_null_numeric(df, [
        "opp_pit_hr9_roll30",
        "opp_pit_hr9_roll15",
        "opp_pit_hr9_roll7",
        "opp_pit_hr9_roll3",
    ])

    pitcher_barrel = first_non_null_numeric(df, [
        "opp_pit_barrel_rate_roll30",
        "opp_pit_barrel_rate_roll15",
        "opp_pit_barrel_rate_roll7",
        "opp_pit_barrel_rate_roll3",
    ])

    pitcher_hard_hit = first_non_null_numeric(df, [
        "opp_pit_hard_hit_rate_roll30",
        "opp_pit_hard_hit_rate_roll15",
        "opp_pit_hard_hit_rate_roll7",
        "opp_pit_hard_hit_rate_roll3",
    ])

    pitcher_bb = first_non_null_numeric(df, [
        "opp_pit_bb_rate_roll30",
        "opp_pit_bb_rate_roll15",
        "opp_pit_bb_rate_roll7",
        "opp_pit_bb_rate_roll3",
    ])

    temp_f = first_non_null_numeric(df, ["temperature_f"], default=72.0)
    wind_out = first_non_null_numeric(df, ["weather_wind_out"], default=0.0)
    lineup_spot = first_non_null_numeric(df, ["batting_order"], default=6.0)
    lineup_weight = first_non_null_numeric(df, ["lineup_weight"], default=1.0)
    base_score = first_non_null_numeric(df, ["hr_score_raw", "score_raw", "score"], default=0.0)

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

    # ------------------------------------------------------------------
    # SOFT REPAIRS FOR SPARSE EARLY-SEASON DATA
    # ------------------------------------------------------------------
    if hard_hit_rate.isna().all() and ev.notna().any():
        hard_hit_rate = ((ev - 80.0) / 25.0).clip(lower=0.20, upper=0.65)

    if barrel_rate.isna().all():
        barrel_rate = (
            tb_per_pa.fillna(np.nan) * 0.12
            + ((ev - 88.0).clip(lower=0) * 0.0020)
            + ((la - 12.0).clip(lower=0) * 0.0010)
        )

    if iso.isna().all():
        iso = (
            tb_per_pa.fillna(np.nan) * 0.58
            + flyball.fillna(np.nan) * 0.05
            + pulled_air.fillna(np.nan) * 0.04
        )

    if hr_per_pa.isna().all():
        hr_per_pa = (
            barrel_rate.fillna(np.nan) * 0.18
            + iso.fillna(np.nan) * 0.05
            + flyball.fillna(np.nan) * 0.01
        )

    if ev.isna().all():
        ev = pd.Series(np.nan, index=df.index, dtype=float)
    if la.isna().all():
        la = pd.Series(np.nan, index=df.index, dtype=float)

    barrel_rate = fill_with_median(barrel_rate, 0.055)
    iso = fill_with_median(iso, 0.155)
    hr_per_pa = fill_with_median(hr_per_pa, 0.025)
    hard_hit_rate = fill_with_median(hard_hit_rate, 0.38)
    flyball = fill_with_median(flyball, 0.35)
    pulled_air = fill_with_median(pulled_air, 0.10)
    ev = fill_with_median(ev, 89.0)
    la = fill_with_median(la, 13.0)
    tb_per_pa = fill_with_median(tb_per_pa, 0.45)

    pitcher_hr9 = fill_with_median(pitcher_hr9, 1.10)
    pitcher_barrel = fill_with_median(pitcher_barrel, 0.08)
    pitcher_hard_hit = fill_with_median(pitcher_hard_hit, 0.40)
    pitcher_bb = fill_with_median(pitcher_bb, 0.08)

    temp_f = fill_with_median(temp_f, 72.0)
    wind_out = fill_with_median(wind_out, 0.0)
    lineup_spot = fill_with_median(lineup_spot, 6.0)
    lineup_weight = fill_with_median(lineup_weight, 1.0)
    base_score = pd.to_numeric(base_score, errors="coerce").fillna(0.0)

    # ------------------------------------------------------------------
    # HARD / SOFT POWER FILTERS
    # ------------------------------------------------------------------
    hard_power_flag = (
        (barrel_rate >= 0.08) |
        (iso >= 0.180) |
        (hr_per_pa >= 0.040)
    ).astype(int)

    soft_power_flag = (
        (barrel_rate >= 0.055) |
        (iso >= 0.150) |
        (hr_per_pa >= 0.022)
    ).astype(int)

    # Do not wipe out the pool, but penalize weak profiles
    score_penalty = (
        (soft_power_flag == 0).astype(float) * 0.16
        + (hard_power_flag == 0).astype(float) * 0.08
    )

    hr_power_score = (
        barrel_rate * 0.40 +
        iso * 0.30 +
        hard_hit_rate * 0.30
    )

    hitter_shape = (
        barrel_rate * 2.20 +
        iso * 1.55 +
        hr_per_pa * 1.95 +
        hard_hit_rate * 0.85 +
        flyball * 0.40 +
        pulled_air * 0.34
    )

    contact_quality = (
        (ev - 88.0) * 0.022 +
        (la - 12.0).clip(-10, 18) * 0.010
    )

    matchup = (
        pitcher_hr9 * 0.30 +
        pitcher_hard_hit * 0.42 +
        pitcher_barrel * 0.60 +
        pitcher_bb * 0.05
    )

    environment = (
        ((temp_f - 70.0) / 15.0) * 0.05 +
        wind_out * 0.09
    )

    score = base_score * 0.16
    score += hitter_shape
    score += contact_quality
    score += matchup
    score += environment
    score += hr_power_score * 1.35
    score -= score_penalty

    # lineup penalty / boost
    score = np.where(lineup_spot <= 2, score * 1.01, score)
    score = np.where(lineup_spot == 3, score * 1.05, score)
    score = np.where(lineup_spot == 4, score * 1.07, score)
    score = np.where(lineup_spot == 5, score * 1.03, score)
    score = np.where(lineup_spot >= 7, score * 0.75, score)
    score = np.where(lineup_spot >= 8, score * 0.60, score)

    score *= lineup_weight.clip(lower=0.80, upper=1.15)

    # additional weak-power penalty
    weak_power_penalty = (
        (barrel_rate < 0.050).astype(float) * 0.10 +
        (iso < 0.140).astype(float) * 0.08 +
        (hard_hit_rate < 0.34).astype(float) * 0.06
    )
    score -= weak_power_penalty

    # elite separation
    elite_boost = (
        (barrel_rate >= 0.12).astype(float) * 0.18 +
        (iso >= 0.25).astype(float) * 0.16 +
        (hr_per_pa >= 0.050).astype(float) * 0.14
    )
    score += elite_boost

    df["hr_score_raw"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    df["z_score"] = slate_zscore(df["hr_score_raw"])

    # ------------------------------------------------------------------
    # PROBABILITY CALIBRATION
    # ------------------------------------------------------------------
    # Use softer mapping so ranking survives but probability isn't inflated
    raw_p = logistic(-2.25 + (df["z_score"] * 0.82))
    df["p_hr_raw"] = raw_p

    # confidence widening
    df["confidence"] = pd.cut(
        df["z_score"],
        bins=[-10, -1.0, 0.0, 1.0, 1.8, 2.5, 10],
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
        "z_score": df["z_score"],
        "p_hr_raw": df["p_hr_raw"],
        "confidence": df["confidence"],
        "sort_score": df["sort_score"],
    })

    board = board.sort_values(["sort_score", "p_hr_raw"], ascending=[False, False]).reset_index(drop=True)

    # apply probability shape after ranking
    board["p_hr"] = clamp_prob_by_rank(board["p_hr_raw"], board.index)

    board.insert(0, "rank", np.arange(1, len(board) + 1))
    board = board.drop(columns=["sort_score", "p_hr_raw", "z_score"])
    board_top = board.head(25).copy()

    board_top.to_csv(out_csv, index=False)
    board_top.to_parquet(out_parquet, index=False)

    print(f"✅ HR board built: {out_csv}")
    print(f"✅ HR board parquet: {out_parquet}")
    print(board_top.to_string(index=False))


if __name__ == "__main__":
    main()