from __future__ import annotations

"""Ingest MLB parks from StatsAPI into Drive-rooted raw/reference storage."""

import argparse
import json
import sys
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.checks import print_rowcount
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import write_parquet
from src.utils.logging import configure_logging, log_header

STATSAPI_VENUES_URL = "https://statsapi.mlb.com/api/v1/venues?sportId=1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest MLB parks metadata from StatsAPI.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()


def _fetch_venues() -> list[dict]:
    with urlopen(STATSAPI_VENUES_URL, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload.get("venues", [])


def _venues_to_df(venues: list[dict], season: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for venue in venues:
        location = venue.get("location", {}) or {}
        timezone = venue.get("timeZone", {}) or {}
        default_coords = (location.get("defaultCoordinates") or {})
        rows.append(
            {
                "park_id": venue.get("id"),
                "venue_id": venue.get("id"),
                "park_name": venue.get("name"),
                "lat": default_coords.get("latitude"),
                "lon": default_coords.get("longitude"),
                "roofType": venue.get("roofType"),
                "tz": timezone.get("id"),
                "season": season,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["venue_id"]).reset_index(drop=True)
    return df


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "run_ingest_parks.log")
    log_header("scripts/ingest/run_ingest_parks.py", repo_root, config_path, dirs)
    print(f"Args: season={args.season}, force={args.force}")

    raw_out_path = dirs["raw_dir"] / "by_season" / f"parks_{args.season}.parquet"
    ref_out_path = dirs["reference_dir"] / "parks_master.parquet"

    if raw_out_path.exists() and not args.force:
        print(f"Using existing parks file (force=False): {raw_out_path.resolve()}")
        return

    venues = _fetch_venues()
    parks_df = _venues_to_df(venues, args.season)
    print_rowcount(f"parks_{args.season}", parks_df)

    if len(parks_df) == 0:
        raise RuntimeError("Parks ingest returned 0 rows; failing loudly.")
    if len(parks_df) < 30:
        raise RuntimeError(f"Parks ingest returned fewer than 30 rows ({len(parks_df)}); failing loudly.")

    print(f"Writing to: {raw_out_path.resolve()}")
    write_parquet(parks_df, raw_out_path)

    print(f"Writing to: {ref_out_path.resolve()}")
    write_parquet(parks_df.drop(columns=["season"], errors="ignore"), ref_out_path)


if __name__ == "__main__":
    main()
