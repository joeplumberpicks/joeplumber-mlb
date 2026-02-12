from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd

try:
    from pybaseball import statcast as _pybaseball_statcast
except ImportError:  # pragma: no cover - environment dependent
    _pybaseball_statcast = None


PRIMARY_DEDUPE_KEY = ["game_pk", "at_bat_number", "pitch_number"]
PRIMARY_NULL_RATE_THRESHOLD = 0.05
FALLBACK_DEDUPE_CANDIDATES = [
    "game_date",
    "batter_id",
    "pitcher_id",
    "inning",
    "inning_topbot",
    "balls",
    "strikes",
    "outs_when_up",
    "pitch_type",
    "plate_x",
    "plate_z",
    "release_speed",
    "description",
    "events",
]


@dataclass(frozen=True)
class StatcastValidationSummary:
    rows: int
    unique_batters: int
    unique_pitchers: int
    null_rate_pitch_type: float
    null_rate_plate_x: float
    null_rate_plate_z: float
    dedupe_key: str


def _require_pybaseball() -> Callable[..., pd.DataFrame]:
    if _pybaseball_statcast is None:
        raise RuntimeError(
            "Missing dependency 'pybaseball'. Install project requirements with: "
            "pip install -r requirements.txt"
        )
    return _pybaseball_statcast


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


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_chunks(start: str, end: str, chunk_days: int) -> list[tuple[str, str]]:
    """Return inclusive date chunks between start/end as YYYY-MM-DD tuples."""
    start_date = _parse_date(start)
    end_date = _parse_date(end)

    if chunk_days < 1:
        raise ValueError("chunk_days must be >= 1")
    if start_date > end_date:
        raise ValueError("start must be <= end")

    chunks: list[tuple[str, str]] = []
    cursor = start_date
    while cursor <= end_date:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_date)
        chunks.append((cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _retry_sleep_seconds(base: float, attempt: int, cap_seconds: float = 30.0) -> float:
    expo = min(base * (2 ** (attempt - 1)), cap_seconds)
    return min(expo + random.uniform(0.0, 1.0), cap_seconds)


def fetch_statcast_chunk(
    start: str,
    end: str,
    *,
    retries: int = 4,
    backoff_seconds: float = 2.0,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Fetch one Statcast chunk with retries/backoff and per-attempt logging."""
    log = logger or logging.getLogger(__name__)
    statcast = _require_pybaseball()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        start_time = time.perf_counter()
        try:
            log.info("chunk=%s..%s attempt=%d/%d starting", start, end, attempt, retries)
            frame = statcast(start_dt=start, end_dt=end)
            frame = pd.DataFrame() if frame is None else frame
            duration = time.perf_counter() - start_time
            log.info(
                "chunk=%s..%s attempt=%d/%d rows=%d duration_s=%.2f",
                start,
                end,
                attempt,
                retries,
                len(frame),
                duration,
            )
            return frame
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            last_error = exc
            duration = time.perf_counter() - start_time
            log.warning(
                "chunk=%s..%s attempt=%d/%d failed duration_s=%.2f error=%s",
                start,
                end,
                attempt,
                retries,
                duration,
                exc,
            )
            if attempt < retries:
                sleep_seconds = _retry_sleep_seconds(backoff_seconds, attempt)
                log.info("chunk=%s..%s sleeping_s=%.2f before retry", start, end, sleep_seconds)
                time.sleep(sleep_seconds)

    assert last_error is not None
    raise RuntimeError(
        f"Failed to pull Statcast data for {start} to {end} after {retries} attempts"
    ) from last_error


def pull_statcast_pitches(
    start: str,
    end: str,
    *,
    chunk_days: int = 7,
    max_retries: int = 4,
    backoff_seconds: float = 2.0,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Pull pitch-by-pitch Statcast data for a date range."""
    log = logger or logging.getLogger(__name__)
    frames: list[pd.DataFrame] = []

    for chunk_start, chunk_end in date_chunks(start, end, chunk_days):
        frame = fetch_statcast_chunk(
            chunk_start,
            chunk_end,
            retries=max_retries,
            backoff_seconds=backoff_seconds,
            logger=log,
        )
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _choose_dedupe_strategy(df: pd.DataFrame, logger: logging.Logger | None = None) -> tuple[str, list[str]]:
    log = logger or logging.getLogger(__name__)

    missing = [col for col in PRIMARY_DEDUPE_KEY if col not in df.columns]
    if missing:
        log.warning("Primary dedupe key unavailable; missing columns=%s", missing)
        fallback_cols = [col for col in FALLBACK_DEDUPE_CANDIDATES if col in df.columns]
        return "fallback", fallback_cols

    null_rates = {col: float(df[col].isna().mean()) for col in PRIMARY_DEDUPE_KEY}
    log.info("Primary dedupe key null rates: %s", null_rates)
    if any(rate > PRIMARY_NULL_RATE_THRESHOLD for rate in null_rates.values()):
        log.warning(
            "Primary dedupe key null rates too high; threshold=%.2f rates=%s",
            PRIMARY_NULL_RATE_THRESHOLD,
            null_rates,
        )
        fallback_cols = [col for col in FALLBACK_DEDUPE_CANDIDATES if col in df.columns]
        return "fallback", fallback_cols

    return "primary", PRIMARY_DEDUPE_KEY


def _build_fallback_hash(df: pd.DataFrame, candidate_cols: list[str]) -> pd.Series:
    if not candidate_cols:
        payload = df.index.to_series().astype("string")
        return payload.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())

    payload = df[candidate_cols].copy()
    for col in payload.columns:
        if pd.api.types.is_float_dtype(payload[col]):
            payload[col] = payload[col].round(4)
        payload[col] = payload[col].astype("string").fillna("<NA>")
    merged = payload.agg("|".join, axis=1)
    return merged.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest())


def normalize_statcast_pitches(
    df: pd.DataFrame,
    season: int,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Normalize key fields and deterministically dedupe Statcast pitch-level rows."""
    log = logger or logging.getLogger(__name__)
    if df.empty:
        out = pd.DataFrame(
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
        out.attrs["dedupe_key"] = "primary"
        return out

    normalized = df.copy().rename(columns={"pitcher": "pitcher_id", "batter": "batter_id"})

    for col in ["pitcher_id", "batter_id", "game_pk", "at_bat_number", "pitch_number", "inning", "balls", "strikes", "outs_when_up"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").astype("Int64")

    if "game_date" in normalized.columns:
        normalized["game_date"] = pd.to_datetime(normalized["game_date"], errors="coerce").dt.date

    for col in ["plate_x", "plate_z", "release_speed"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    for col in ["pitch_type", "inning_topbot", "description", "events"]:
        if col in normalized.columns:
            normalized[col] = normalized[col].astype("string")

    normalized["season"] = season

    strategy, key_cols = _choose_dedupe_strategy(normalized, logger=log)
    sort_cols = [
        col
        for col in ["game_date", "game_pk", "at_bat_number", "pitch_number", "pitcher_id", "batter_id"]
        if col in normalized.columns
    ]
    if sort_cols:
        normalized = normalized.sort_values(sort_cols, kind="mergesort", na_position="last")

    pre_rows = len(normalized)
    if strategy == "primary":
        deduped = normalized.drop_duplicates(subset=PRIMARY_DEDUPE_KEY, keep="first")
    else:
        key_hash = _build_fallback_hash(normalized, key_cols)
        deduped = normalized.assign(_fallback_dedupe_key=key_hash).drop_duplicates(
            subset=["_fallback_dedupe_key"], keep="first"
        )
        deduped = deduped.drop(columns=["_fallback_dedupe_key"])

    deduped = deduped.reset_index(drop=True)
    post_rows = len(deduped)
    drop_rate = 0.0 if pre_rows == 0 else (pre_rows - post_rows) / pre_rows
    if drop_rate > 0.02:
        log.warning(
            "high duplicate rate detected during dedupe: dropped=%d total=%d rate=%.4f",
            pre_rows - post_rows,
            pre_rows,
            drop_rate,
        )

    deduped.attrs["dedupe_key"] = strategy
    deduped.attrs["dedupe_drop_rate"] = drop_rate
    return deduped


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
        dedupe_key=str(df.attrs.get("dedupe_key", "unknown")),
    )
