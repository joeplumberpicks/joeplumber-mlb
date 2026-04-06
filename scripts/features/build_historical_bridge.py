#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingest.io import log_kv, log_section, read_dataset, write_parquet
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified historical bridge parquets across seasons.")
    parser.add_argument("--seasons", nargs="+", type=int, required=True, help="Example: --seasons 2025 2026")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return read_dataset(path)
    return None


def _concat_frames(frames: list[pd.DataFrame], label: str) -> pd.DataFrame:
    if not frames:
        print(f"Row count [{label}]: 0")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if "season" in out.columns:
        out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")

    sort_cols = [c for c in ["game_date", "game_pk", "pa_index"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    print(f"Row count [{label}]: {len(out):,}")
    if "game_pk" in out.columns:
        print(f"Distinct game_pk [{label}]: {out['game_pk'].nunique(dropna=True):,}")
    if "game_date" in out.columns and not out.empty:
        print(f"Min game_date [{label}]: {out['game_date'].min()}")
        print(f"Max game_date [{label}]: {out['game_date'].max()}")

    return out


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/features/build_historical_bridge.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)
    log_kv("seasons", args.seasons)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    by_season_dir = processed_dir / "by_season"

    games_frames: list[pd.DataFrame] = []
    pa_frames: list[pd.DataFrame] = []
    weather_frames: list[pd.DataFrame] = []
    spine_frames: list[pd.DataFrame] = []

    for season in args.seasons:
        games_path = by_season_dir / f"games_{season}.parquet"
        pa_path = by_season_dir / f"pa_{season}.parquet"
        weather_path = by_season_dir / f"weather_game_{season}.parquet"
        spine_path = by_season_dir / f"model_spine_game_{season}.parquet"

        games_df = _read_if_exists(games_path)
        if games_df is not None and not games_df.empty:
            games_frames.append(games_df)
        else:
            print(f"Missing or empty: {games_path}")

        pa_df = _read_if_exists(pa_path)
        if pa_df is not None and not pa_df.empty:
            pa_frames.append(pa_df)
        else:
            print(f"Missing or empty: {pa_path}")

        weather_df = _read_if_exists(weather_path)
        if weather_df is not None and not weather_df.empty:
            weather_frames.append(weather_df)
        else:
            print(f"Missing or empty: {weather_path}")

        spine_df = _read_if_exists(spine_path)
        if spine_df is not None and not spine_df.empty:
            spine_frames.append(spine_df)
        else:
            print(f"Missing or empty: {spine_path}")

    games = _concat_frames(games_frames, "games")
    pa = _concat_frames(pa_frames, "pa")
    weather = _concat_frames(weather_frames, "weather_game")
    spine = _concat_frames(spine_frames, "model_spine_game")

    if not games.empty:
        write_parquet(games, processed_dir / "games.parquet")
    if not pa.empty:
        write_parquet(pa, processed_dir / "pa.parquet")
    if not weather.empty:
        write_parquet(weather, processed_dir / "weather_game.parquet")
    if not spine.empty:
        write_parquet(spine, processed_dir / "model_spine_game.parquet")

    print("")
    print("historical_bridge_out:")
    print(processed_dir / "games.parquet")
    print(processed_dir / "pa.parquet")
    print(processed_dir / "weather_game.parquet")
    print(processed_dir / "model_spine_game.parquet")


if __name__ == "__main__":
    main()
