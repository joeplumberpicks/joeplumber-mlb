#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.rolling import apply_rolling
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-season statcast rolling tables.")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config((repo_root / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])

    batter_path = processed_dir / "batter_game_statcast.parquet"
    pitcher_path = processed_dir / "pitcher_game_statcast.parquet"

    if not batter_path.exists():
        raise FileNotFoundError(f"Missing: {batter_path}")
    if not pitcher_path.exists():
        raise FileNotFoundError(f"Missing: {pitcher_path}")

    bat = pd.read_parquet(batter_path)
    pit = pd.read_parquet(pitcher_path)

    if "game_date" in bat.columns:
        bat["game_date"] = pd.to_datetime(bat["game_date"], errors="coerce")
    if "game_date" in pit.columns:
        pit["game_date"] = pd.to_datetime(pit["game_date"], errors="coerce")

    bat_cols = [c for c in ["hr", "rbi", "tb", "barrel_rate", "hardhit_rate", "ev_mean", "la_mean", "k_rate", "bb_rate"] if c in bat.columns]
    pit_cols = [c for c in ["hr_allowed", "barrel_rate_allowed", "hardhit_rate_allowed", "k_rate", "bb_rate", "ev_mean"] if c in pit.columns]

    bat = apply_rolling(
        bat,
        group_col="batter_id",
        date_col="game_date",
        cols=bat_cols,
        windows=(3, 7, 15, 30),
        shift=1,
    )

    pit = apply_rolling(
        pit,
        group_col="pitcher_id",
        date_col="game_date",
        cols=pit_cols,
        windows=(3, 7, 15, 30),
        shift=1,
    )

    bat_out = processed_dir / "batter_statcast_rolling.parquet"
    pit_out = processed_dir / "pitcher_statcast_rolling.parquet"

    bat.to_parquet(bat_out, index=False)
    pit.to_parquet(pit_out, index=False)

    print("✅ cross-season statcast rolling built")
    print(f"batter_rows={len(bat):,}")
    print(f"pitcher_rows={len(pit):,}")
    print(f"bat_out={bat_out}")
    print(f"pit_out={pit_out}")


if __name__ == "__main__":
    main()