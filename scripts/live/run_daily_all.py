#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def get_today_date() -> str:
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Joe Plumber daily stack.")
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--season", type=str, default="2026")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument(
        "--seasons-history",
        nargs="+",
        type=int,
        default=[2025, 2026],
        help="Seasons used to rebuild cross-season history/rollings.",
    )
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
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
    py = sys.executable

    date_str = args.date or get_today_date()
    season = str(args.season)

    print(f"=== JOE PLUMBER DAILY STACK :: {date_str} ===")
    print(f"repo_root={repo_root}")
    print(f"season={season}")
    print(f"history_seasons={args.seasons_history}")

    if not args.skip_ingest:
        _run(
            [
                py,
                "scripts/live/run_live_pregame_pipeline.py",
                "--season",
                season,
                "--date",
                date_str,
                "--config",
                args.config,
                "--force",
            ],
            repo_root,
        )

    if not args.skip_history:
        _run(
            [
                py,
                "scripts/features/build_historical_bridge.py",
                "--seasons",
                *[str(s) for s in args.seasons_history],
                "--config",
                args.config,
            ],
            repo_root,
        )

        _run(
            [
                py,
                "scripts/features/build_cross_season_rollings.py",
                "--config",
                args.config,
            ],
            repo_root,
        )

        _run(
            [
                py,
                "scripts/features/build_statcast_game_tables.py",
                "--seasons",
                *[str(s) for s in args.seasons_history],
                "--config",
                args.config,
            ],
            repo_root,
        )

        _run(
            [
                py,
                "scripts/features/build_cross_season_statcast_rollings.py",
                "--config",
                args.config,
            ],
            repo_root,
        )

    _run(
        [
            py,
            "scripts/features/build_daily_feature_views.py",
            "--season",
            season,
            "--date",
            date_str,
            "--config",
            args.config,
        ],
        repo_root,
    )

    _run(
        [
            py,
            "scripts/live/run_daily_nrfi.py",
            "--season",
            season,
            "--date",
            date_str,
            "--config",
            args.config,
        ],
        repo_root,
    )

    _run(
        [
            py,
            "scripts/live/run_daily_moneyline.py",
            "--season",
            season,
            "--date",
            date_str,
            "--config",
            args.config,
        ],
        repo_root,
    )

    _run(
        [
            py,
            "scripts/live/run_daily_hr.py",
            "--season",
            season,
            "--date",
            date_str,
            "--config",
            args.config,
        ],
        repo_root,
    )

    _run(
        [
            py,
            "scripts/live/run_daily_rbi.py",
            "--season",
            season,
            "--date",
            date_str,
            "--config",
            args.config,
        ],
        repo_root,
    )

    print("")
    print("✅ Joe Plumber daily stack complete")
    print(f"date={date_str}")
    print(f"season={season}")


if __name__ == "__main__":
    main()