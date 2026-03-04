from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def build_hitter_batter_features(dirs: dict[str, Path], season: int) -> Path:
    """
    Build batter-game grain features for hitter props markets.
    """
    processed_dir = Path(dirs["processed_dir"])
    marts_dir = Path(dirs["marts_dir"])
    marts_by_season_dir = marts_dir / "by_season"
    marts_by_season_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT:
    # We already build a correct batter-game grain mart at:
    #   marts/by_season/hitter_props_features_{season}.parquet (keys: game_pk, batter_id)
    # This function should output a modeling-friendly view of that dataset.
    src_path = marts_by_season_dir / f"hitter_props_features_{season}.parquet"
    if not src_path.exists():
        raise FileNotFoundError(f"Expected hitter props mart not found: {src_path}")

    df = pd.read_parquet(src_path).copy()
    if df.empty:
        raise ValueError(f"hitter_props_features is empty for season={season}: {src_path}")

    # ensure key dtypes
    if "game_pk" not in df.columns or "batter_id" not in df.columns:
        raise ValueError("hitter_props_features must include game_pk and batter_id")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["batter_id"] = pd.to_numeric(df["batter_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["game_pk", "batter_id"]).copy()

    # ensure required targets exist
    req_targets = ["target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]
    missing = [c for c in req_targets if c not in df.columns]
    if missing:
        raise ValueError(f"hitter_props_features missing required target columns: {missing}")

    # Keep labeled rows (should be all; but be safe)
    df = df[df["target_hit1p"].notna()].copy()

    out_path = marts_by_season_dir / f"hitter_batter_features_{season}.parquet"
    logging.info("hitter_batter_features rows=%s path=%s", len(df), out_path)
    df.to_parquet(out_path, index=False)
    return out_path
