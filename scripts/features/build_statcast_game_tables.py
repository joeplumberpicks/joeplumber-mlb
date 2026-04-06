#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.statcast_game_aggregations import build_batter_game, build_pitcher_game
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-season statcast game tables.")
    parser.add_argument("--seasons", nargs="+", type=int, required=True, help="Example: --seasons 2025 2026")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config((repo_root / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    by_season_dir = processed_dir / "by_season"

    pa_frames: list[pd.DataFrame] = []

    for season in args.seasons:
        pa_path = by_season_dir / f"pa_{season}.parquet"
        pa_df = _read_if_exists(pa_path)
        if pa_df is None or pa_df.empty:
            print(f"Missing or empty: {pa_path}")
            continue
        pa_frames.append(pa_df)

    if not pa_frames:
        raise FileNotFoundError("No seasonal PA files were found.")

    pa = pd.concat(pa_frames, ignore_index=True, sort=False)

    # normalize expected helper cols where possible
    rename_map = {}
    if "launch_speed" not in pa.columns and "release_speed" in pa.columns:
        rename_map["release_speed"] = "launch_speed"
    if rename_map:
        pa = pa.rename(columns=rename_map)

    for col in ["is_hit", "is_hr", "is_rbi", "is_bb", "is_so", "is_barrel", "is_hard_hit"]:
        if col not in pa.columns:
            pa[col] = 0

    if "total_bases" not in pa.columns:
        pa["total_bases"] = (
            pa.get("is_1b", 0).fillna(0).astype(int)
            + 2 * pa.get("is_2b", 0).fillna(0).astype(int)
            + 3 * pa.get("is_3b", 0).fillna(0).astype(int)
            + 4 * pa.get("is_hr", 0).fillna(0).astype(int)
        )

    batter = build_batter_game(pa)
    pitcher = build_pitcher_game(pa)

    batter_out = processed_dir / "batter_game_statcast.parquet"
    pitcher_out = processed_dir / "pitcher_game_statcast.parquet"

    batter.to_parquet(batter_out, index=False)
    pitcher.to_parquet(pitcher_out, index=False)

    print("✅ statcast game tables built")
    print(f"batter_rows={len(batter):,}")
    print(f"pitcher_rows={len(pitcher):,}")
    print(f"batter_out={batter_out}")
    print(f"pitcher_out={pitcher_out}")


if __name__ == "__main__":
    main()