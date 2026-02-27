from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.pitch_rules import CONTACT_DESCRIPTIONS, SWING_DESCRIPTIONS, WHIFF_DESCRIPTIONS, pitch_group
from src.utils.checks import print_rowcount, require_files
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header

EVENT_TO_HIT = {"single", "double", "triple", "home_run"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build batter/pitcher game logs from pitch-level Statcast data.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--chunk-days", type=int, default=14)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _resolve_pitch_source(dirs: dict[str, Path], season: int) -> tuple[Path, pd.DataFrame]:
    processed_pitches = dirs["processed_dir"] / "by_season" / f"pitches_{season}.parquet"
    raw_pa = dirs["raw_dir"] / "by_season" / f"pa_{season}.parquet"

    if processed_pitches.exists():
        df = read_parquet(processed_pitches)
        logging.info("Using pitch-level source: %s", processed_pitches.resolve())
        return processed_pitches, df

    if raw_pa.exists():
        df = read_parquet(raw_pa)
        if "pitch_type" in df.columns:
            logging.info("Using pitch-level source fallback: %s", raw_pa.resolve())
            return raw_pa, df
        raise ValueError(
            f"Raw PA file exists but is not pitch-level (missing 'pitch_type'): {raw_pa.resolve()}"
        )

    raise FileNotFoundError(
        "No pitch-level source found. Expected one of: "
        f"{processed_pitches.resolve()} or {raw_pa.resolve()}"
    )


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "batter_id" not in out.columns and "batter" in out.columns:
        out["batter_id"] = out["batter"]
    if "pitcher_id" not in out.columns and "pitcher" in out.columns:
        out["pitcher_id"] = out["pitcher"]

    for col in ["batter_id", "pitcher_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def _in_zone(df: pd.DataFrame) -> pd.Series:
    zone_col = pd.to_numeric(df.get("zone", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    zone_based = zone_col.isin(list(range(1, 10)))
    if {"plate_x", "plate_z", "sz_top", "sz_bot"}.issubset(df.columns):
        denom = (pd.to_numeric(df["sz_top"], errors="coerce") - pd.to_numeric(df["sz_bot"], errors="coerce")).replace(0, np.nan)
        nz = (pd.to_numeric(df["plate_z"], errors="coerce") - pd.to_numeric(df["sz_bot"], errors="coerce")) / denom
        loc_based = pd.to_numeric(df["plate_x"], errors="coerce").between(-0.83, 0.83, inclusive="both") & nz.between(
            0, 1, inclusive="both"
        )
        return zone_based | loc_based.fillna(False)
    return zone_based


def _prepare_pitch_features(pitches_df: pd.DataFrame) -> pd.DataFrame:
    df = _normalize_ids(pitches_df)
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
    df["is_hit"] = df["events"].isin(EVENT_TO_HIT).astype(int)

    return df


def _merge_game_context(df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    game_cols = [c for c in ["game_pk", "game_date", "home_team", "away_team", "park_id", "park_name", "canonical_park_key"] if c in games_df.columns]
    game_map = games_df[game_cols].drop_duplicates(subset=["game_pk"]) if "game_pk" in game_cols else pd.DataFrame(columns=game_cols)
    if game_map.empty:
        return df
    out = df.merge(game_map, on="game_pk", how="left", suffixes=("", "_game"))
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce")
    return out


def _agg_game(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    base_group = ["game_pk", id_col]
    context = [c for c in ["game_date", "home_team", "away_team", "park_id", "park_name", "canonical_park_key"] if c in df.columns]
    group_cols = base_group + context

    numeric_for_mean = [c for c in ["release_speed", "release_spin_rate", "launch_speed", "launch_angle"] if c in df.columns]
    for col in numeric_for_mean:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    agg_spec: dict[str, tuple[str, str]] = {
        "pitches": ("description", "count"),
        "swings": ("is_swing", "sum"),
        "whiffs": ("is_whiff", "sum"),
        "contacts": ("is_contact", "sum"),
        "in_zone_pitches": ("in_zone", "sum"),
        "chases": ("is_chase", "sum"),
        "k": ("is_k", "max"),
        "bb": ("is_bb", "max"),
        "hbp": ("is_hbp", "max"),
        "hr": ("is_hr", "max"),
        "h": ("is_hit", "max"),
    }
    for col in numeric_for_mean:
        agg_spec[f"{col}_mean"] = (col, "mean")
        agg_spec[f"{col}_max"] = (col, "max")

    out = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
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

    games_path = dirs["processed_dir"] / "by_season" / f"games_{args.season}.parquet"
    require_files([games_path], f"build_game_logs_from_pitches_{args.season}")

    src_path, pitches_df = _resolve_pitch_source(dirs, args.season)
    print(f"Using pitch-level source: {src_path.resolve()}")
    games_df = read_parquet(games_path)

    df = _prepare_pitch_features(pitches_df)
    df = _merge_game_context(df, games_df)

    if args.start:
        df = df[df["game_date"] >= pd.to_datetime(args.start)]
    if args.end:
        df = df[df["game_date"] <= pd.to_datetime(args.end)]

    batter_required = ["game_pk", "batter_id"]
    pitcher_required = ["game_pk", "pitcher_id"]
    missing_b = [c for c in batter_required if c not in df.columns]
    missing_p = [c for c in pitcher_required if c not in df.columns]
    if missing_b or missing_p:
        raise ValueError(f"Missing required id columns for game-log build. missing_batter={missing_b}, missing_pitcher={missing_p}")

    batter_df = df.dropna(subset=["game_pk", "batter_id"]).copy()
    pitcher_df = df.dropna(subset=["game_pk", "pitcher_id"]).copy()
    batter_game = _agg_game(batter_df, "batter_id")
    pitcher_game = _agg_game(pitcher_df, "pitcher_id")

    games_rows = len(games_df)
    if games_rows > 0:
        if len(batter_game) <= games_rows:
            raise RuntimeError(
                f"Unexpected grain: batter_game rows ({len(batter_game)}) should be > games rows ({games_rows})."
            )
        if len(pitcher_game) < games_rows:
            raise RuntimeError(
                f"Unexpected grain: pitcher_game rows ({len(pitcher_game)}) should be >= games rows ({games_rows})."
            )

    matchup_cols = [c for c in ["game_pk", "batter_id", "pitcher_id", "pitch_group"] if c in df.columns]
    matchup = df.groupby(matchup_cols, dropna=False).agg(pitches=("description", "count"), swings=("is_swing", "sum"), whiffs=("is_whiff", "sum")).reset_index()

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
