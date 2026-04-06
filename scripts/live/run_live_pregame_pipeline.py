#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.ingest.io import log_kv, log_section, read_dataset
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Joe Plumber live pregame pipeline.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str], repo_root: Path) -> None:
    print("")
    print("RUNNING:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _pct(series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.fillna(False).astype("boolean").mean() * 100.0)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/live/run_live_pregame_pipeline.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    py = sys.executable

    _run(
        [
            py,
            "scripts/ingest/run_ingest_day.py",
            "--season",
            str(args.season),
            "--date",
            args.date,
            "--config",
            args.config,
            "--force",
        ],
        repo_root,
    )

    processed_live_dir = Path(dirs["processed_dir"]) / "live"
    spine_path = processed_live_dir / f"model_spine_game_{args.season}_{args.date}.parquet"

    if not spine_path.exists():
        raise FileNotFoundError(f"Missing spine output: {spine_path}")

    df = read_dataset(spine_path)

    print("")
    print(f"Row count [model_spine_game_{args.season}_{args.date}]: {len(df):,}")
    print(f"Distinct game_pk: {df['game_pk'].nunique(dropna=True):,}")

    if "starters_both_found" in df.columns:
        print(f"pct_with_starters={_pct(df['starters_both_found']):.2f}")

    if "lineups_projected_both_found" in df.columns:
        print(f"pct_with_projected_lineups={_pct(df['lineups_projected_both_found']):.2f}")

    if "lineups_confirmed_both_found" in df.columns:
        print(f"pct_with_confirmed_lineups={_pct(df['lineups_confirmed_both_found']):.2f}")

    for col in ["temperature_f", "wind_mph", "weather_wind_out", "weather_wind_in", "weather_crosswind"]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")

    print(f"daily_out={spine_path}")


if __name__ == "__main__":
    main()
