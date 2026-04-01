#!/usr/bin/env python3
"""
Run full Layer 1 ingest for a season.

Purpose
-------
Orchestrates all ingest steps:
- schedule
- game metadata
- weather
- lineups (optional)
- plate appearances
- parks (optional)

This is the ONLY script you run to build raw truth for a season.

STRICT RULES:
- No modeling logic
- No feature engineering
- No targets
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n>>> RUNNING: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full ingest pipeline for a season.")

    parser.add_argument("--season", type=int, required=True)

    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)

    parser.add_argument("--with-lineups", action="store_true")
    parser.add_argument("--with-parks", action="store_true")

    parser.add_argument("--pa-input", type=str, required=True)
    parser.add_argument("--lineups-input", type=str, default=None)
    parser.add_argument("--parks-input", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    python = "python"

    print("\n==============================")
    print("JOE PLUMBER INGEST PIPELINE")
    print(f"SEASON: {args.season}")
    print("==============================\n")

    # ---------------------------
    # 1. Schedule
    # ---------------------------
    run([
        python,
        "scripts/ingest/ingest_schedule_games.py",
        "--season", str(args.season),
        "--start-date", str(args.start_date or ""),
        "--end-date", str(args.end_date or ""),
    ])

    # ---------------------------
    # 2. Game Metadata
    # ---------------------------
    run([
        python,
        "scripts/ingest/ingest_game_metadata.py",
        "--season", str(args.season),
        "--start-date", str(args.start_date or ""),
        "--end-date", str(args.end_date or ""),
    ])

    # ---------------------------
    # 3. Weather (NOAA METAR)
    # ---------------------------
    run([
        python,
        "scripts/ingest/ingest_weather_games.py",
        "--season", str(args.season),
        "--start-date", str(args.start_date or ""),
        "--end-date", str(args.end_date or ""),
    ])

    # ---------------------------
    # 4. Plate Appearances
    # ---------------------------
    run([
        python,
        "scripts/ingest/ingest_plate_appearances.py",
        "--season", str(args.season),
        "--input", args.pa_input,
        "--source", "statcast",
    ])

    # ---------------------------
    # 5. Lineups (Optional)
    # ---------------------------
    if args.with_lineups and args.lineups_input:
        run([
            python,
            "scripts/ingest/ingest_lineups.py",
            "--season", str(args.season),
            "--date", str(args.start_date),
            "--projected-lineups-input", args.lineups_input,
            "--projected-source", "rotowire",
        ])

    # ---------------------------
    # 6. Parks (Optional)
    # ---------------------------
    if args.with_parks and args.parks_input:
        run([
            python,
            "scripts/ingest/ingest_parks.py",
            "--input", args.parks_input,
            "--source", "mlb_statsapi",
            "--season", str(args.season),
        ])

    print("\n✅ INGEST COMPLETE — RAW TRUTH BUILT\n")


if __name__ == "__main__":
    main()
