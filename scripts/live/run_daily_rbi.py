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
    team_col = pick_col(df, ["team", "bat_team", "offense_team"], required=True)
    opp_col = pick_col(df, ["opponent", "opp_team", "pitching_team"], required=False)
    player_col = pick_col(df, ["player_name", "batter_name", "name"], required=True)
    lineup_col = pick_col(
        df,
        ["lineup_spot", "batting_order", "order_spot", "confirmed_batting_order", "projected_batting_order"],
        required=False,
    )

    # Audit / fallback core RBI columns
    hits = coalesce_numeric(df, ["hits_per_pa", "bat_hits_per_pa_roll30", "hit_rate", "bat_ba_roll30"], default=np.nan)
    tb_per_pa = coalesce_numeric(df, ["tb_per_pa", "bat_tb_per_pa_roll30"], default=np.nan)
    hr_per_pa = coalesce_numeric(df, ["hr_per_pa", "bat_hr_per_pa_roll30"], default=np.nan)
    bb_rate = coalesce_numeric(df, ["bb_rate", "bat_bb_rate_roll30"], default=np.nan)
    hard_hit_rate = coalesce_numeric(df, ["hard_hit_rate", "bat_hard_hit_rate_roll30"], default=np.nan)
    barrel_rate = coalesce_numeric(df, ["barrel_rate", "bat_barrel_rate_roll30"], default=np.nan)
    iso = coalesce_numeric(df, ["iso", "bat_iso_roll30"], default=np.nan)

    park_rbi = coalesce_numeric(df, ["park_rbi_factor", "park_run_factor", "park_factor_runs"], default=1.0)
    weather_rbi = coalesce_numeric(df, ["weather_rbi_boost", "weather_run_boost"], default=0.0)

    opp_hr9 = coalesce_numeric(df, ["pitcher_hr9", "pit_hr9_roll30", "opp_pitcher_hr9"], default=0.0)
    opp_hard_hit = coalesce_numeric(df, ["pitcher_hard_hit_rate", "pit_hard_hit_rate_roll30"], default=0.0)
    opp_barrel = coalesce_numeric(df, ["pitcher_barrel_rate", "pit_barrel_rate_roll30"], default=0.0)
    opp_bb = coalesce_numeric(df, ["pitcher_bb_rate", "pit_bb_rate_roll30"], default=0.0)

    base_score = coalesce_numeric(df, ["rbi_score_raw", "score_raw", "score"], default=0.0)

    lineup_spot = pd.Series(5, index=df.index, dtype=float)
    if lineup_col is not None:
        lineup_spot = pd.to_numeric(df[lineup_col], errors="coerce").fillna(5.0)

    print("\n=== RBI FEATURE AUDIT ===")
    for name, s in {
        "hits": hits,
        "tb_per_pa": tb_per_pa,
        "hr_per_pa": hr_per_pa,
        "bb_rate": bb_rate,
        "hard_hit_rate": hard_hit_rate,
        "barrel_rate": barrel_rate,
        "iso": iso,
        "lineup_spot": lineup_spot,
    }.items():
        print(f"{name}: null_pct={float(pd.Series(s).isna().mean()) * 100:.2f}%")

    # Median fill, not zero fill
    for s_name in ["hits", "tb_per_pa", "hr_per_pa", "bb_rate", "hard_hit_rate", "barrel_rate", "iso"]:
        s = locals()[s_name]
        med = float(pd.Series(s).median()) if not pd.Series(s).dropna().empty else 0.0
        locals()[s_name] = pd.Series(s).fillna(med)

    # RBI opportunity logic
    contact_prod = (
        hits * 0.30 +
        tb_per_pa * 0.30 +
        hr_per_pa * 0.20 +
        hard_hit_rate * 0.10 +
        barrel_rate * 0.10
    )

    run_env = (
        (park_rbi - 1.0) * 0.90 +
        weather_rbi * 0.60
    )

    pitcher_exploit = (
        opp_hr9 * 0.25 +
        opp_hard_hit * 0.50 +
        opp_barrel * 0.70 +
        opp_bb * 0.20
    )

    # Lineup multiplier: RBI needs top/middle order
    lineup_mult = np.where(lineup_spot <= 2, 0.92, 1.00)
    lineup_mult = np.where(lineup_spot == 3, 1.10, lineup_mult)
    lineup_mult = np.where(lineup_spot == 4, 1.14, lineup_mult)
    lineup_mult = np.where(lineup_spot == 5, 1.08, lineup_mult)
    lineup_mult = np.where(lineup_spot == 6, 0.96, lineup_mult)
    lineup_mult = np.where(lineup_spot >= 7, 0.82, lineup_mult)
    lineup_mult = np.where(lineup_spot >= 8, 0.68, lineup_mult)

    # Soft weak-bat penalty
    weak_penalty = (
        (tb_per_pa < pd.Series(tb_per_pa).median()).astype(float) * 0.10 +
        (hard_hit_rate < pd.Series(hard_hit_rate).median()).astype(float) * 0.08
    )

    score = base_score * 0.35
    score += contact_prod * 1.20
    score += pitcher_exploit
    score += run_env
    score -= weak_penalty
    score *= lineup_mult

    df["rbi_score_raw"] = pd.to_numeric(score, errors="coerce").fillna(0.0)
    df["z_score"] = slate_zscore(df["rbi_score_raw"])

    # Separate probability calibration from raw ranking
    df["p_rbi"] = logistic(-0.15 + (df["z_score"] * 0.80))

    df["confidence"] = pd.cut(
        df["z_score"],
        bins=[-10, -1.0, -0.25, 0.5, 1.25, 2.0, 10],
        labels=["F", "D", "C", "B", "A", "A+"],
        include_lowest=True,
    ).astype(str)

    name_for_noise = coalesce_text(df, [player_col], default="")
    team_for_noise = coalesce_text(df, [team_col], default="")
    noise = (name_for_noise + "|" + team_for_noise).map(lambda x: (hash(x) % 1000) / 1_000_000.0).astype(float)
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