from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def build_pitcher_game_features(dirs: dict[str, Path], season: int) -> Path:
    processed_dir = Path(dirs["processed_dir"])
    marts_dir = Path(dirs["marts_dir"])
    marts_by_season_dir = marts_dir / "by_season"
    marts_by_season_dir.mkdir(parents=True, exist_ok=True)

    # We already build a correct pitcher-game grain mart at:
    #   marts/by_season/pitcher_props_features_{season}.parquet (keys: game_pk, pitcher_id)
    src_path = marts_by_season_dir / f"pitcher_props_features_{season}.parquet"
    if not src_path.exists():
        raise FileNotFoundError(f"Expected pitcher props mart not found: {src_path}")

    df = pd.read_parquet(src_path).copy()
    if df.empty:
        raise ValueError(f"pitcher_props_features is empty for season={season}: {src_path}")

    if "game_pk" not in df.columns or "pitcher_id" not in df.columns:
        raise ValueError("pitcher_props_features must include game_pk and pitcher_id")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["game_pk", "pitcher_id"]).copy()

    req_targets = ["target_k", "target_outs", "target_er", "target_bb"]
    missing = [c for c in req_targets if c not in df.columns]
    if missing:
        raise ValueError(f"pitcher_props_features missing required target columns: {missing}")

    df = df[df["target_k"].notna()].copy()

    out_path = marts_by_season_dir / f"pitcher_game_features_{season}.parquet"
    logging.info("pitcher_game_features rows=%s path=%s", len(df), out_path)
    df.to_parquet(out_path, index=False)
    return out_path
