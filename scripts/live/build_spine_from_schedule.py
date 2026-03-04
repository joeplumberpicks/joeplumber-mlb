from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build minimal live spine from ingested schedule games.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_spine_from_schedule.log")
    log_header("scripts/live/build_spine_from_schedule.py", repo_root, config_path, dirs)

    date_ts = pd.to_datetime(args.date, format="%Y-%m-%d", errors="raise")
    in_path = dirs["raw_dir"] / "live" / f"games_schedule_{args.season}.parquet"
    out_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"

    if out_path.exists() and not args.force:
        raise FileExistsError(f"Output already exists (use --force): {out_path}")
    if not in_path.exists():
        raise FileNotFoundError(f"Schedule parquet not found: {in_path}")

    sched = read_parquet(in_path)
    if "game_date" not in sched.columns:
        raise ValueError(f"Expected game_date in schedule parquet: {in_path}")

    sched["game_date"] = pd.to_datetime(sched["game_date"], errors="coerce")
    day = sched[sched["game_date"].dt.date == date_ts.date()].copy()

    out = pd.DataFrame(
        {
            "game_pk": pd.to_numeric(day.get("game_pk"), errors="coerce").astype("Int64"),
            "game_date": pd.to_datetime(day.get("game_date"), errors="coerce"),
            "season": pd.to_numeric(day.get("season"), errors="coerce").fillna(args.season).astype("Int64"),
            "home_team": day.get("home_team"),
            "away_team": day.get("away_team"),
            "venue_id": pd.to_numeric(day.get("venue_id"), errors="coerce").astype("Int64"),
            "home_sp_id": pd.to_numeric(day.get("home_probable_pitcher_id"), errors="coerce").astype("Int64"),
            "away_sp_id": pd.to_numeric(day.get("away_probable_pitcher_id"), errors="coerce").astype("Int64"),
            "home_sp_name": day.get("home_probable_pitcher_name"),
            "away_sp_name": day.get("away_probable_pitcher_name"),
        }
    )

    out = out.dropna(subset=["game_pk"]).copy()
    out["game_pk"] = out["game_pk"].astype("int64")
    out["season"] = out["season"].astype("int64")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    write_parquet(out, out_path)
    home_present = int(out["home_sp_id"].notna().sum()) if len(out) else 0
    away_present = int(out["away_sp_id"].notna().sum()) if len(out) else 0
    home_pct = (home_present / len(out) * 100.0) if len(out) else 0.0
    away_pct = (away_present / len(out) * 100.0) if len(out) else 0.0
    logging.info(
        "live schedule spine rows=%s out=%s home_sp_id_present=%s (%.2f%%) away_sp_id_present=%s (%.2f%%)",
        len(out),
        out_path,
        home_present,
        home_pct,
        away_present,
        away_pct,
    )
    print(f"Row count [model_spine_game_{args.season}_{args.date}]: {len(out):,}")
    print(f"Writing to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
