from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test schedule ingest probable starters coverage.")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        "scripts/ingest/ingest_schedule_games.py",
        "--date",
        args.date,
        "--season",
        str(args.season),
        "--game-types",
        "S,R",
        "--config",
        str(args.config),
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        print(f"warning: ingest command failed (non-fatal for smoke): {exc}")

    out_path = REPO_ROOT / "data" / "raw" / "live" / f"games_schedule_{args.season}_{args.date}.parquet"
    if not out_path.exists():
        print(f"date-scoped schedule file not found: {out_path}")
        return

    df = pd.read_parquet(out_path)
    n = len(df)
    home = int(df.get("home_probable_pitcher_id", pd.Series(dtype="Int64")).notna().sum()) if n else 0
    away = int(df.get("away_probable_pitcher_id", pd.Series(dtype="Int64")).notna().sum()) if n else 0
    print(f"rows={n} home_probables={home} ({(home/n*100.0 if n else 0.0):.2f}%) away_probables={away} ({(away/n*100.0 if n else 0.0):.2f}%)")
    print("smoke complete (non-fatal if probables are unavailable).")


if __name__ == "__main__":
    main()
