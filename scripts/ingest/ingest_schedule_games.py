from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import write_parquet
from src.utils.logging import configure_logging, log_header

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest MLB schedule games to raw/live parquet.")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--start", default=None, help="YYYY-MM-DD")
    p.add_argument("--end", default=None, help="YYYY-MM-DD")
    p.add_argument("--game-types", default="S,R", help="Comma-separated game types (default: S,R)")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _fetch_schedule(season: int, start_date: str, end_date: str) -> dict:
    params = {
        "sportId": 1,
        "season": season,
        "startDate": start_date,
        "endDate": end_date,
        "hydrate": "team,venue",
    }
    resp = requests.get(SCHEDULE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _to_rows(payload: dict, season: int, game_types: set[str]) -> list[dict]:
    rows: list[dict] = []
    for day in payload.get("dates", []):
        for g in day.get("games", []):
            gt = str(g.get("gameType", ""))
            if gt not in game_types:
                continue
            teams = g.get("teams", {})
            away = teams.get("away", {}).get("team", {})
            home = teams.get("home", {}).get("team", {})
            venue = g.get("venue", {})
            status = g.get("status", {}).get("detailedState") or g.get("status", {}).get("abstractGameState")
            home_prob = teams.get("home", {}).get("probablePitcher")
            away_prob = teams.get("away", {}).get("probablePitcher")

            rows.append(
                {
                    "game_date": pd.to_datetime(g.get("officialDate"), errors="coerce"),
                    "game_pk": pd.to_numeric(g.get("gamePk"), errors="coerce"),
                    "season": int(season),
                    "game_type": gt,
                    "away_team": away.get("name"),
                    "home_team": home.get("name"),
                    "away_team_id": pd.to_numeric(away.get("id"), errors="coerce"),
                    "home_team_id": pd.to_numeric(home.get("id"), errors="coerce"),
                    "venue_id": pd.to_numeric(venue.get("id"), errors="coerce"),
                    "venue_name": venue.get("name"),
                    "status": status,
                    "doubleheader": g.get("doubleHeader"),
                    "game_num": pd.to_numeric(g.get("gameNumber"), errors="coerce"),
                    "start_time_utc": pd.to_datetime(g.get("gameDate"), errors="coerce", utc=True).tz_convert(None)
                    if g.get("gameDate")
                    else pd.NaT,
                    "home_probable_pitcher_id": home_prob.get("id") if home_prob else None,
                    "away_probable_pitcher_id": away_prob.get("id") if away_prob else None,
                    "home_probable_pitcher_name": home_prob.get("fullName") if home_prob else None,
                    "away_probable_pitcher_name": away_prob.get("fullName") if away_prob else None,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "ingest_schedule_games.log")
    log_header("scripts/ingest/ingest_schedule_games.py", repo_root, config_path, dirs)

    start_date = args.start or args.date
    end_date = args.end or args.date
    game_types = {x.strip().upper() for x in args.game_types.split(",") if x.strip()}
    out_path = args.out or (dirs["raw_dir"] / "live" / f"games_schedule_{args.season}.parquet")

    payload = _fetch_schedule(args.season, start_date, end_date)
    rows = _to_rows(payload, args.season, game_types)
    df = pd.DataFrame(rows)

    # stable dtypes
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "game_date",
                "game_pk",
                "season",
                "game_type",
                "away_team",
                "home_team",
                "away_team_id",
                "home_team_id",
                "venue_id",
                "venue_name",
                "status",
                "doubleheader",
                "game_num",
                "start_time_utc",
                "home_probable_pitcher_id",
                "away_probable_pitcher_id",
                "home_probable_pitcher_name",
                "away_probable_pitcher_name",
            ]
        )
    else:
        int_cols = ["game_pk", "season", "away_team_id", "home_team_id", "venue_id", "game_num"]
        for c in int_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["home_probable_pitcher_id"] = pd.to_numeric(df.get("home_probable_pitcher_id"), errors="coerce").astype("Int64")
    df["away_probable_pitcher_id"] = pd.to_numeric(df.get("away_probable_pitcher_id"), errors="coerce").astype("Int64")

    write_parquet(df, out_path)
    home_present = int(df["home_probable_pitcher_id"].notna().sum()) if len(df) else 0
    away_present = int(df["away_probable_pitcher_id"].notna().sum()) if len(df) else 0
    home_pct = (home_present / len(df) * 100.0) if len(df) else 0.0
    away_pct = (away_present / len(df) * 100.0) if len(df) else 0.0
    logging.info(
        "schedule ingest rows=%s out=%s home_probables=%s (%.2f%%) away_probables=%s (%.2f%%)",
        len(df),
        out_path,
        home_present,
        home_pct,
        away_present,
        away_pct,
    )
    print(f"Row count [games_schedule_{args.season}]: {len(df):,}")
    print(f"Writing to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
