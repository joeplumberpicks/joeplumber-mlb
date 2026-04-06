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
    parser = argparse.ArgumentParser(description="Run Joe Plumber live daily ingest.")
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


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/ingest/run_ingest_day.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    raw_live_dir = Path(dirs["raw_dir"]) / "live"
    processed_live_dir = Path(dirs["processed_dir"]) / "live"

    py = sys.executable

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
            "--config",
            args.config,
            "--force",
        ],
        repo_root,
    )

    schedule_path = raw_live_dir / f"games_schedule_{args.season}_{args.date}.parquet"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Expected schedule output not found: {schedule_path}")

    schedule_df = read_dataset(schedule_path)
    if schedule_df.empty:
        raise ValueError(f"Schedule is empty for {args.date}")

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
            "--config",
            args.config,
            "--force",
        ],
        repo_root,
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
            "--config",
            args.config,
            "--force",
        ],
        repo_root,
    )

    _run(
        [
            py,
            "scripts/live/build_projected_lineups_fangraphs.py",
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

    _run(
        [
            py,
            "scripts/live/build_confirmed_lineups_rotowire.py",
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

    _run(
        [
            py,
            "scripts/live/build_starting_pitchers_rotowire.py",
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

    _run(
        [
            py,
            "scripts/spine/build_model_spine_game.py",
            "--season",
            str(args.season),
            "--date",
            args.date,
            "--config",
            args.config,
        ],
        repo_root,
    )

    out_path = processed_live_dir / f"model_spine_game_{args.season}_{args.date}.parquet"
    print("")
    print(f"daily_out={out_path}")


if __name__ == "__main__":
    main()
