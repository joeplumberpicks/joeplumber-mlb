from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.pitch_rules import (
    CONTACT_DESCRIPTIONS,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
    pitch_group,
)
from src.utils.checks import print_rowcount, require_files
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header


EVENT_TO_HIT = {"single": 1, "double": 2, "triple": 3, "home_run": 4}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build batter/pitcher game logs from pitch-level Statcast data.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--chunk-days", type=int, default=14)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _in_zone(df: pd.DataFrame) -> pd.Series:
    zone_based = df.get("zone", pd.Series(index=df.index, dtype="float64")).isin(list(range(1, 10)))
    if {"plate_x", "plate_z", "sz_top", "sz_bot"}.issubset(df.columns):
        denom = (df["sz_top"] - df["sz_bot"]).replace(0, np.nan)
        nz = (df["plate_z"] - df["sz_bot"]) / denom
        loc_based = df["plate_x"].between(-0.83, 0.83, inclusive="both") & nz.between(0, 1, inclusive="both")
        return zone_based | loc_based.fillna(False)
    return zone_based


def _prepare_pitch_features(pa_df: pd.DataFrame) -> pd.DataFrame:
    df = pa_df.copy()
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
    df["description"] = df.get("description", pd.Series(index=df.index, dtype="object")).fillna("")
    df["events"] = df.get("events", pd.Series(index=df.index, dtype="object"))
    df["event_type"] = df.get("event_type", df["events"])
    df["pitch_group"] = df.get("pitch_type", pd.Series(index=df.index, dtype="object")).map(pitch_group)

    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    df["is_contact"] = df["description"].isin(CONTACT_DESCRIPTIONS).astype(int)
    df["in_zone"] = _in_zone(df).astype(int)
    df["is_chase"] = ((df["is_swing"] == 1) & (df["in_zone"] == 0)).astype(int)

    df["is_k"] = df["events"].isin(["strikeout", "strikeout_double_play"]).astype(int)
    df["is_bb"] = df["events"].isin(["walk", "intent_walk"]).astype(int)
    df["is_hbp"] = df["events"].eq("hit_by_pitch").astype(int)
    df["is_hr"] = df["events"].eq("home_run").astype(int)
    df["is_hit"] = df["events"].isin(EVENT_TO_HIT.keys()).astype(int)

    if {"plate_z", "sz_bot", "sz_top"}.issubset(df.columns):
        denom = (df["sz_top"] - df["sz_bot"]).replace(0, np.nan)
        df["nz"] = (df["plate_z"] - df["sz_bot"]) / denom
    else:
        df["nz"] = np.nan

    if "plate_x" in df.columns:
        df["zone_bucket"] = pd.cut(df["plate_x"], bins=[-99, -0.5, 0.5, 99], labels=["inside", "middle", "outside"])
    else:
        df["zone_bucket"] = "unknown"

    return df


def _agg_rates(df: pd.DataFrame, grp_cols: list[str], id_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=grp_cols)

    src = df.copy()
    if id_col not in src.columns:
        src[id_col] = 1
    for col in ["is_swing", "is_whiff", "is_contact", "in_zone", "is_chase", "is_k", "is_bb", "is_hbp", "is_hr", "is_hit"]:
        if col not in src.columns:
            src[col] = 0
    for col in ["release_speed", "release_spin_rate", "launch_speed"]:
        if col not in src.columns:
            src[col] = pd.NA

    g = src.groupby(grp_cols, dropna=False)
    out = g.agg(
        pitches=(id_col, "count"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        contacts=("is_contact", "sum"),
        in_zone_pitches=("in_zone", "sum"),
        chases=("is_chase", "sum"),
        k=("is_k", "max"),
        bb=("is_bb", "max"),
        hbp=("is_hbp", "max"),
        hr=("is_hr", "max"),
        h=("is_hit", "max"),
        release_speed_mean=("release_speed", "mean"),
        release_spin_rate_mean=("release_spin_rate", "mean"),
        launch_speed_mean=("launch_speed", "mean"),
        launch_speed_max=("launch_speed", "max"),
    ).reset_index()
    out["swing_rate"] = out["swings"] / out["pitches"].replace(0, np.nan)
    out["zone_swing_rate"] = out["swings"] / out["in_zone_pitches"].replace(0, np.nan)
    out["chase_rate"] = out["chases"] / out["swings"].replace(0, np.nan)
    out["whiff_rate"] = out["whiffs"] / out["swings"].replace(0, np.nan)
    out["contact_rate"] = out["contacts"] / out["swings"].replace(0, np.nan)
    return out


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_game_logs_from_pitches.log")
    log_header("scripts/build_game_logs_from_pitches.py", repo_root, config_path, dirs)

    pa_path = dirs["processed_dir"] / "by_season" / f"pa_{args.season}.parquet"
    games_path = dirs["processed_dir"] / "by_season" / f"games_{args.season}.parquet"
    require_files([pa_path, games_path], f"build_game_logs_from_pitches_{args.season}")

    pa_df = read_parquet(pa_path)
    games_df = read_parquet(games_path)
    df = _prepare_pitch_features(pa_df)

    game_cols = [c for c in ["game_pk", "game_date", "home_team", "away_team", "park_id", "venue_id", "canonical_park_key"] if c in games_df.columns]
    game_map = games_df[game_cols].drop_duplicates(subset=["game_pk"]) if "game_pk" in games_df.columns else pd.DataFrame(columns=game_cols)
    if "game_pk" in df.columns and not game_map.empty:
        df = df.merge(game_map, on="game_pk", how="left", suffixes=("", "_g"))

    if args.start:
        df = df[df["game_date"] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df["game_date"] <= pd.to_datetime(args.end)]

    batter_keys = [c for c in ["game_pk", "game_date", "batter", "home_team", "away_team", "park_id", "canonical_park_key"] if c in df.columns]
    pitcher_keys = [c for c in ["game_pk", "game_date", "pitcher", "home_team", "away_team", "park_id", "canonical_park_key"] if c in df.columns]

    batter_game = _agg_rates(df, batter_keys, "pitcher")
    pitcher_game = _agg_rates(df, pitcher_keys, "batter")

    for src, key_col in [(batter_game, "batter"), (pitcher_game, "pitcher")]:
        if key_col in src.columns:
            grp = src.groupby([key_col, "game_date"], dropna=False)["pitches"].sum().reset_index(name="_tmp")
            _ = grp

    matchup_cols = [c for c in ["game_pk", "game_date", "batter", "pitcher", "pitch_group", "zone_bucket"] if c in df.columns]
    matchup = (
        df.groupby(matchup_cols, dropna=False)
        .agg(pitches=("description", "count"), swings=("is_swing", "sum"), whiffs=("is_whiff", "sum"))
        .reset_index()
    )

    out_dir = dirs["processed_dir"] / "by_season"
    batter_path = out_dir / f"batter_game_{args.season}.parquet"
    pitcher_path = out_dir / f"pitcher_game_{args.season}.parquet"
    matchup_path = out_dir / f"pitch_agg_matchup_{args.season}.parquet"

    print_rowcount(f"batter_game_{args.season}", batter_game)
    print(f"Writing to: {batter_path.resolve()}")
    write_parquet(batter_game, batter_path)

    print_rowcount(f"pitcher_game_{args.season}", pitcher_game)
    print(f"Writing to: {pitcher_path.resolve()}")
    write_parquet(pitcher_game, pitcher_path)

    print_rowcount(f"pitch_agg_matchup_{args.season}", matchup)
    print(f"Writing to: {matchup_path.resolve()}")
    write_parquet(matchup, matchup_path)


if __name__ == "__main__":
    main()
