from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
from pybaseball import statcast


@dataclass(frozen=True)
class StatcastValidationSummary:
    rows: int
    unique_batters: int
    unique_pitchers: int
    null_rate_pitch_type: float
    null_rate_plate_x: float
    null_rate_plate_z: float


def resolve_date_window(season: int, start: str | None, end: str | None) -> tuple[str, str]:
    """Resolve the pull window for a season in YYYY-MM-DD format."""
    default_start = date(season, 1, 1)
    default_end = date(season, 12, 31)

    start_date = datetime.strptime(start, "%Y-%m-%d").date() if start else default_start
    end_date = datetime.strptime(end, "%Y-%m-%d").date() if end else default_end

    if start_date > end_date:
        raise ValueError(f"start date {start_date} is after end date {end_date}")

    if start_date.year > season or end_date.year < season:
        raise ValueError(
            f"date window {start_date} to {end_date} does not overlap season {season}"
        )

    return start_date.isoformat(), end_date.isoformat()


def _date_chunks(start_date: date, end_date: date, chunk_days: int) -> list[tuple[date, date]]:
    chunks: list[tuple[date, date]] = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_date)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _fetch_chunk_with_retries(
    start_date: date,
    end_date: date,
    max_retries: int,
    backoff_seconds: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Pulling Statcast chunk %s to %s (attempt %d/%d)",
                start_date,
                end_date,
                attempt,
                max_retries,
            )
            chunk = statcast(start_dt=start_date.isoformat(), end_dt=end_date.isoformat())
            if chunk is None:
                return pd.DataFrame()
            return chunk
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            logger.warning(
                "Statcast pull failed for %s to %s on attempt %d/%d: %s",
                start_date,
                end_date,
                attempt,
                max_retries,
                exc,
            )
            if attempt < max_retries:
                sleep_for = backoff_seconds * (2 ** (attempt - 1))
                logger.info("Sleeping %.1fs before retry", sleep_for)
                time.sleep(sleep_for)

    assert last_error is not None
    raise RuntimeError(
        f"Failed to pull Statcast data for {start_date} to {end_date} after {max_retries} attempts"
    ) from last_error


def pull_statcast_pitches(
    start: str,
    end: str,
    *,
    chunk_days: int = 7,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Pull pitch-by-pitch Statcast data for a date range."""
    log = logger or logging.getLogger(__name__)

    start_date = datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date()
    if start_date > end_date:
        raise ValueError("start must be <= end")

    chunks = _date_chunks(start_date, end_date, chunk_days=chunk_days)
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in chunks:
        chunk = _fetch_chunk_with_retries(
            chunk_start,
            chunk_end,
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            logger=log,
        )
        if chunk.empty:
            log.info("No Statcast data returned for chunk %s to %s", chunk_start, chunk_end)
            continue

        frames.append(chunk)
        log.info(
            "Fetched %d rows for chunk %s to %s",
            len(chunk),
            chunk_start,
            chunk_end,
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def normalize_statcast_pitches(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Normalize key fields and deterministically dedupe Statcast pitch-level rows."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "game_date",
                "game_pk",
                "at_bat_number",
                "pitch_number",
                "pitcher_id",
                "batter_id",
                "pitch_type",
                "plate_x",
                "plate_z",
            ]
        )

    normalized = df.copy()
    rename_map = {
        "pitcher": "pitcher_id",
        "batter": "batter_id",
    }
    normalized = normalized.rename(columns=rename_map)

    for col in ["pitcher_id", "batter_id", "game_pk", "at_bat_number", "pitch_number"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype("Int64")

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce").dt.date

    for col in ["plate_x", "plate_z"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if "pitch_type" in normalized.columns:
        normalized["pitch_type"] = normalized["pitch_type"].astype("string")

    normalized["season"] = season

    dedupe_keys = ["game_pk", "at_bat_number", "pitch_number"]
    available_dedupe_keys = [c for c in dedupe_keys if c in normalized.columns]

    sort_cols = [c for c in ["game_date", *available_dedupe_keys, "pitcher_id", "batter_id"] if c in normalized.columns]
    if sort_cols:
        normalized = normalized.sort_values(sort_cols, kind="mergesort", na_position="last")

    if available_dedupe_keys:
        normalized = normalized.drop_duplicates(subset=available_dedupe_keys, keep="first")
    else:
        normalized = normalized.drop_duplicates(keep="first")

    normalized = normalized.reset_index(drop=True)
    return normalized


def validate_statcast_pitches(df: pd.DataFrame) -> StatcastValidationSummary:
    """Build the requested validation summary for Statcast pitch-level data."""
    rows = int(len(df))

    unique_batters = int(df["batter_id"].nunique(dropna=True)) if "batter_id" in df.columns else 0
    unique_pitchers = int(df["pitcher_id"].nunique(dropna=True)) if "pitcher_id" in df.columns else 0

    def null_rate(column: str) -> float:
        if column not in df.columns or rows == 0:
            return 0.0
        return float(df[column].isna().mean())

    return StatcastValidationSummary(
        rows=rows,
        unique_batters=unique_batters,
        unique_pitchers=unique_pitchers,
        null_rate_pitch_type=null_rate("pitch_type"),
        null_rate_plate_x=null_rate("plate_x"),
        null_rate_plate_z=null_rate("plate_z"),
    )
