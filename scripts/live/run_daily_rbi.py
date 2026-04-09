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
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            out = out.combine_first(s)
    if default is not None:
        out = out.fillna(default)
    return out


def fill_with_median(series: pd.Series, fallback: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    non_na = s.dropna()
    if non_na.empty:
        return s.fillna(fallback)
    return s.fillna(float(non_na.median()))


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


def series_null_pct(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").isna().mean()) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    args = parser.parse_args()

    print(f"=== JOE PLUMBER RBI RUN :: {args.date} ===")

    config = load_config(args.config)
    data_root = resolve_data_root(config)
    processed_live = data_root / "processed" / "live"
    outputs_dir = data_root / "outputs"
    ensure_dir(outputs_dir)

    features_path = processed_live / f"rbi_features_{args.season}_{args.date}.parquet"
    out_csv = outputs_dir / f"rbi_board_{args.season}_{args.date}.csv"
    out_parquet = outputs_dir / f"rbi_board_{args.season}_{args.date}.parquet"

    print(f"Loading features: {features_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    df = pd.read_parquet(features_path).copy()
    print(f"Row count [rbi_features]: {len(df):,}")
    if df.empty:
        raise ValueError("RBI features file is empty.")

    game_date_col = pick_col(df, ["game_date"], required=False)
    team_col = pick_col(df, ["team"], required=True)
    opp_col = pick_col(df, ["opponent"], required=False)
    player_col = pick_col(df, ["player_name"], required=True)

    hits = first_non_null_numeric(df, [
        "bat_hits_roll30", "bat_hits_roll15", "bat_hits_roll7", "bat_hits_roll3"
    ])
    tb_per_pa = first_non_null_numeric(df, [
        "bat_tb_per_pa_roll30", "bat_tb_per_pa_roll15", "bat_tb_per_pa_roll7", "bat_tb_per_pa_roll3"
    ])
    hr_per_pa = first_non_null_numeric(df, [
        "bat_hr_per_pa_roll30", "bat_hr_per_pa_roll15", "bat_hr_per_pa_roll7", "bat_hr_per_pa_roll3"
    ])
    bb_rate = first_non_null_numeric(df, [
        "bat_bb_rate_roll30", "bat_bb_rate_roll15", "bat_bb_rate_roll7", "bat_bb_rate_roll3"
    ])
    hard_hit_rate = first_non_null_numeric(df, [
        "bat_hard_hit_rate_roll30", "bat_hard_hit_rate_roll15", "bat_hard_hit_rate_roll7", "bat_hard_hit_rate_roll3"
    ])
    barrel_rate = first_non_null_numeric(df, [
        "bat_barrel_rate_roll30", "bat_barrel_rate_roll15", "bat_barrel_rate_roll7", "bat_barrel_rate_roll3"
    ])
    iso = first_non_null_numeric(df, [
        "bat_iso_roll30", "bat_iso_roll15", "bat_iso_roll7", "bat_iso_roll3"
    ])
    lineup_spot = first_non_null_numeric(df, ["batting_order"], default=6.0)
    lineup_weight = first_non_null_numeric(df, ["lineup_weight"], default=1.0)

    opp_hr9 = first_non_null_numeric(df, [
        "opp_pit_hr9_roll30", "opp_pit_hr9_roll15", "opp_pit_hr9_roll7", "opp_pit_hr9_roll3"
    ])
    opp_hard_hit = first_non_null_numeric(df, [
        "opp_pit_hard_hit_rate_roll30", "opp_pit_hard_hit_rate_roll15", "opp_pit_hard_hit_rate_roll7", "opp_pit_hard_hit_rate_roll3"
    ])
    opp_bb = first_non_null_numeric(df, [
        "opp_pit_bb_rate_roll30", "opp_pit_bb_rate_roll15", "opp_pit_bb_rate_roll7", "opp_pit_bb_rate_roll3"
    ])
    temp_f = first_non_null_numeric(df, ["temperature_f"], default=72.0)
    wind_out = first_non_null_numeric(df, ["weather_wind_out"], default=0.0)
    base_score = first_non_null_numeric(df, ["rbi_score_raw", "score_raw", "score"], default=0.0)

    print("\n=== RBI FEATURE AUDIT ===")
    print(f"hits: null_pct={series_null_pct(hits):.2f}%")
    print(f"tb_per_pa: null_pct={series_null_pct(tb_per_pa):.2f}%")
    print(f"hr_per_pa: null_pct={series_null_pct(hr_per_pa):.2f}%")
    print(f"bb_rate: null_pct={series_null_pct(bb_rate):.2f}%")
    print(f"hard_hit_rate: null_pct={series_null_pct(hard_hit_rate):.2f}%")
    print(f"barrel_rate: null_pct={series_null_pct(barrel_rate):.2f}%")
    print(f"iso: null_pct={series_null_pct(iso):.2f}%")
    print(f"lineup_spot: null_pct={series_null_pct(lineup_spot):.2f}%")

    # proxy repair
    if hits.isna().all():
        hits = tb_per_pa * 1.55
    if iso.isna().all():
        iso = tb_per_pa * 0.55
    if hard_hit_rate.isna().all():
        hard_hit_rate = (tb_per_pa * 0.50).clip(lower=0.28, upper=0.60)
    if barrel_rate.isna().all():
        barrel_rate = (hr_per_pa * 2.40).clip(lower=0.03, upper=0.18)

    hits = fill_with_median(hits, 1.0)
    tb_per_pa = fill_with_median(tb_per_pa, 0.45)
    hr_per_pa = fill_with_median(hr_per_pa, 0.025)
    bb_rate = fill_with_median(bb_rate, 0.08)
    hard_hit_rate = fill_with_median(hard_hit_rate, 0.38)
    barrel_rate = fill_with_median(barrel_rate, 0.055)
    iso = fill_with_median(iso, 0.155)
    lineup_spot = fill_with_median(lineup_spot, 6.0)
    lineup_weight = fill_with_median(lineup_weight, 1.0)
    opp_hr9 = fill_with_median(opp_hr9, 1.10)
    opp_hard_hit = fill_with_median(opp_hard_hit, 0.40)
    opp_bb = fill_with_median(opp_bb, 0.08)
    temp_f = fill_with_median(temp_f, 72.0)
    wind_out = fill_with_median(wind_out, 0.0)
    base_score = pd.to_numeric(base_score, errors="coerce").fillna(0.0)

    # RBI scoring
    rbi_opportunity = (
        hits * 0.55 +
        tb_per_pa * 1.35 +
        hr_per_pa * 1.10 +
        iso * 0.75 +
        hard_hit_rate * 0.55 +
        barrel_rate * 0.55
    )

    matchup = (
        opp_hr9 * 0.18 +
        opp_hard_hit * 0.34 +
        opp_bb * 0.12
    )

    environment = (
        ((temp_f - 70.0) / 15.0) * 0.04 +
        wind_out * 0.06
    )

    lineup_bonus = np.where(lineup_spot <= 2, 0.90, 1.00)
    lineup_bonus = np.where(lineup_spot == 3, 1.18, lineup_bonus)
    lineup_bonus = np.where(lineup_spot == 4, 1.22, lineup_bonus)
    lineup_bonus = np.where(lineup_spot == 5, 1.10, lineup_bonus)
    lineup_bonus = np.where(lineup_spot >= 7, 0.88, lineup_bonus)
    lineup_bonus = np.where(lineup_spot >= 8, 0.78, lineup_bonus)
    lineup_bonus = np.where(lineup_spot >= 9, 0.70, lineup_bonus)

    score = (
        base_score * 0.12 +
        rbi_opportunity +
        matchup +
        environment
    )
    score *= lineup_bonus
    score *= lineup_weight.clip(lower=0.80, upper=1.15)

    df["rbi_score_raw"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    df["z_score"] = slate_zscore(df["rbi_score_raw"])
    df["p_rbi"] = logistic(-0.90 + (df["z_score"] * 0.90))

    df["confidence"] = pd.cut(
        df["z_score"],
        bins=[-10, -1.0, -0.20, 0.45, 1.10, 1.80, 10],
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
        "rbi_score_raw": df["rbi_score_raw"],
        "p_rbi": df["p_rbi"],
        "confidence": df["confidence"],
        "sort_score": df["sort_score"],
    })

    board = board.sort_values(["sort_score", "p_rbi"], ascending=[False, False]).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    board = board.drop(columns=["sort_score"])
    board_top = board.head(25).copy()

    board_top.to_csv(out_csv, index=False)
    board_top.to_parquet(out_parquet, index=False)

    print(f"✅ RBI board built: {out_csv}")
    print(f"✅ RBI board parquet: {out_parquet}")
    print(board_top.to_string(index=False))


if __name__ == "__main__":
    main()