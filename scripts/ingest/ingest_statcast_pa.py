from __future__ import annotations

"""Statcast plate appearance ingest utilities."""

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount
from src.utils.io import read_parquet, write_parquet

REQUIRED_COLUMNS = [
    "game_pk",
    "game_date",
    "inning",
    "inning_topbot",
    "batter",
    "pitcher",
    "events",
    "event_type",
    "description",
    "home_team",
    "away_team",
]
DEDUP_KEYS = ["game_pk", "at_bat_number", "pitch_number"]


def _parse_day(value: str | None, default_day: date) -> date:
    if value is None:
        return default_day
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_chunks(start_day: date, end_day: date, chunk_days: int) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = start_day
    while current <= end_day:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_day)
        ranges.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)
    return ranges


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "events" not in out.columns:
        out["events"] = pd.NA
    if "event_type" not in out.columns:
        out["event_type"] = out["events"]
    out["event_type"] = out["event_type"].fillna(out["events"])

    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    return out


def _dedupe_pa(df: pd.DataFrame) -> pd.DataFrame:
    keys_present = [k for k in DEDUP_KEYS if k in df.columns]
    if len(keys_present) == len(DEDUP_KEYS):
        return df.drop_duplicates(subset=keys_present).reset_index(drop=True)
    return df.drop_duplicates().reset_index(drop=True)


def ingest_statcast_pa(
    dirs: dict[str, Path],
    season: int,
    start: str | None,
    end: str | None,
    chunk_days: int,
    force: bool = False,
) -> Path:
    """Pull Statcast data in chunks and write raw/by_season/pa_{season}.parquet."""
    from pybaseball import statcast

    raw_by_season = dirs["raw_dir"] / "by_season"
    raw_by_season.mkdir(parents=True, exist_ok=True)
    output_path = raw_by_season / f"pa_{season}.parquet"

    default_start = date(season, 3, 1)
    default_end = date(season, 11, 30)
    start_day = _parse_day(start, default_start)
    end_day = _parse_day(end, default_end)
    if end_day < start_day:
        raise ValueError(f"end date {end_day} is before start date {start_day}")

    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(start_day, end_day, chunk_days):
        print(f"Fetching Statcast chunk: {chunk_start} -> {chunk_end}")
        chunk_df = statcast(start_dt=chunk_start.isoformat(), end_dt=chunk_end.isoformat())
        if chunk_df is None or chunk_df.empty:
            continue
        frames.append(chunk_df)

    fetched_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=REQUIRED_COLUMNS)
    fetched_df = _normalize_columns(fetched_df)

    if output_path.exists() and not force:
        existing_df = _normalize_columns(read_parquet(output_path))
        combined = pd.concat([existing_df, fetched_df], ignore_index=True)
    else:
        combined = fetched_df

    combined = _dedupe_pa(combined)
    combined = combined[REQUIRED_COLUMNS + [c for c in combined.columns if c not in REQUIRED_COLUMNS]]

    print_rowcount(f"pa_{season}", combined)
    print(f"Writing to: {output_path.resolve()}")
    write_parquet(combined, output_path)

    if not output_path.exists():
        raise FileNotFoundError(f"Failed to create PA parquet: {output_path.resolve()}")
    return output_path
