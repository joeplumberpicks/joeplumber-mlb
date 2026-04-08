#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-day live ingest pipeline.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print(f"\nRUNNING: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _copy_confirmed_to_projected(raw_live_dir: Path, season: int, date_str: str) -> None:
    confirmed_latest = raw_live_dir / f"confirmed_lineups_{season}.parquet"
    confirmed_dated = raw_live_dir / f"confirmed_lineups_{season}_{date_str}.parquet"
    projected_latest = raw_live_dir / f"projected_lineups_{season}.parquet"
    projected_dated = raw_live_dir / f"projected_lineups_{season}_{date_str}.parquet"

    if confirmed_latest.exists():
        shutil.copy2(confirmed_latest, projected_latest)
        print(f"Copied fallback: {confirmed_latest} -> {projected_latest}")

    if confirmed_dated.exists():
        shutil.copy2(confirmed_dated, projected_dated)
        print(f"Copied fallback: {confirmed_dated} -> {projected_dated}")


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    print("========== scripts/ingest/run_ingest_day.py =========")
    print(f"repo_root: {repo_root}")
    print(f"config_path: {repo_root / args.config}")

    py = sys.executable
    config_arg = ["--config", args.config]
    force_arg = ["--force"] if args.force else []

    # Core ingests
    _run(
        [
            py,
            "scripts/ingest/ingest_schedule_games.py",
            "--season",
            str(args.season),
            "--start-date",
            args.date,
            "--end-date",
            args.date,
            *config_arg,
            *force_arg,
        ]
    )

    _run(
        [
            py,
            "scripts/ingest/ingest_game_metadata.py",
            "--season",
            str(args.season),
            "--start-date",
            args.date,
            "--end-date",
            args.date,
            *config_arg,
            *force_arg,
        ]
    )

    _run(
        [
            py,
            "scripts/ingest/ingest_weather_games.py",
            "--season",
            str(args.season),
            "--start-date",
            args.date,
            "--end-date",
            args.date,
            *config_arg,
            *force_arg,
        ]
    )

    # Confirmed lineups from Rotowire
    _run(
        [
            py,
            "scripts/live/build_confirmed_lineups_rotowire.py",
            "--season",
            str(args.season),
            "--date",
            args.date,
            *config_arg,
            *force_arg,
        ]
    )

    raw_live_dir = Path("/content/drive/MyDrive/joeplumber-mlb/data/raw/live")
    _copy_confirmed_to_projected(raw_live_dir, args.season, args.date)

    # Starters: MLB primary, Rotowire fallback
    mlb_ok = True
    try:
        _run(
            [
                py,
                "scripts/live/build_starting_pitchers_mlb.py",
                "--season",
                str(args.season),
                "--date",
                args.date,
                *config_arg,
                *force_arg,
            ]
        )
    except Exception as exc:
        mlb_ok = False
        print(f"MLB starters failed, falling back to Rotowire: {exc}")

    if not mlb_ok:
        _run(
            [
                py,
                "scripts/live/build_starting_pitchers_rotowire.py",
                "--season",
                str(args.season),
                "--date",
                args.date,
                *config_arg,
                *force_arg,
            ]
        )

    # Build spine
    _run(
        [
            py,
            "scripts/spine/build_model_spine_game.py",
            "--season",
            str(args.season),
            "--date",
            args.date,
            *config_arg,
        ]
    )

    daily_out = Path("/content/drive/MyDrive/joeplumber-mlb/data/processed/live") / f"model_spine_game_{args.season}_{args.date}.parquet"
    print(f"\ndaily_out={daily_out}")


if __name__ == "__main__":
    main()