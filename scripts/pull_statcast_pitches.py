#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import tempfile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.statcast_ingestion import (
    date_chunks,
    fetch_statcast_chunk,
    normalize_statcast_pitches,
    resolve_date_window,
    validate_statcast_pitches,
)
from src.utils.io import load_config, read_parquet, write_parquet

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull pitch-by-pitch Statcast data and save parquet output."
    )
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2024")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--chunk-days", type=int, default=7, help="Date chunk size for pulls")
    parser.add_argument("--retries", type=int, default=4, help="Retries per chunk")
    parser.add_argument("--backoff-seconds", type=float, default=2.0, help="Exponential retry backoff base")
    parser.add_argument("--resume", action="store_true", help="Skip chunks that already exist")
    parser.add_argument("--force", action="store_true", help="Re-pull and overwrite existing chunks")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _resolve_raw_statcast_dir(config: dict, logger: logging.Logger) -> Path:
    try:
        from src.utils.drive import resolve_data_dirs

        data_dirs = resolve_data_dirs(config)
        raw_dir = data_dirs["raw"]
    except ImportError:
        logger.info("Drive utilities not available; continuing without Drive sync.")
        raw_dir = REPO_ROOT / config["paths"]["raw"]
    except Exception as exc:
        logger.warning("Falling back to local raw path due to drive resolution error: %s", exc)
        raw_dir = REPO_ROOT / config["paths"]["raw"]

    return raw_dir / "statcast"


def _chunk_file(chunks_dir: Path, season: int, chunk_start: str, chunk_end: str) -> Path:
    return chunks_dir / f"pitches_{season}_{chunk_start}_{chunk_end}.parquet"


def _write_failure_report(failures_dir: Path, season: int, failed_chunks: list[dict]) -> Path:
    failures_dir.mkdir(parents=True, exist_ok=True)
    report_path = failures_dir / f"pitches_{season}_failures.json"
    payload = {"season": season, "failed_chunks": failed_chunks}
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def _atomic_write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="tmp_statcast_", suffix=".parquet", delete=False, dir=output_path.parent) as handle:
        temp_path = Path(handle.name)
    try:
        write_parquet(df, temp_path)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def main() -> None:
    setup_logging()
    logger = logging.getLogger("pull_statcast_pitches")
    args = parse_args()

    if args.resume and args.force:
        raise ValueError("--resume and --force cannot be used together")

    start, end = resolve_date_window(args.season, args.start, args.end)
    logger.info("Starting Statcast ingestion for season=%s window=%s..%s", args.season, start, end)

    config = load_config()
    statcast_dir = _resolve_raw_statcast_dir(config, logger)
    chunks_dir = statcast_dir / "chunks"
    failures_dir = statcast_dir / "failures"
    output_path = statcast_dir / f"pitches_{args.season}.parquet"

    chunk_ranges = date_chunks(start, end, args.chunk_days)
    failed_chunks: list[dict] = []
    chunk_paths: list[Path] = []

    chunks_dir.mkdir(parents=True, exist_ok=True)
    for chunk_start, chunk_end in chunk_ranges:
        chunk_path = _chunk_file(chunks_dir, args.season, chunk_start, chunk_end)

        if args.resume and chunk_path.exists():
            logger.info("Skipping existing chunk due to --resume: %s", chunk_path)
            chunk_paths.append(chunk_path)
            continue

        if args.force and chunk_path.exists():
            chunk_path.unlink()

        try:
            chunk_df = fetch_statcast_chunk(
                chunk_start,
                chunk_end,
                retries=args.retries,
                backoff_seconds=args.backoff_seconds,
                logger=logger,
            )
            write_parquet(chunk_df, chunk_path)
            chunk_paths.append(chunk_path)
        except Exception as exc:
            failed_chunks.append(
                {"start": chunk_start, "end": chunk_end, "error": str(exc)}
            )
            logger.error("Chunk failed permanently: %s..%s error=%s", chunk_start, chunk_end, exc)

    if failed_chunks:
        report_path = _write_failure_report(failures_dir, args.season, failed_chunks)
        logger.error("Wrote failure report to %s", report_path)
        raise SystemExit(1)

    if not chunk_paths:
        normalized = normalize_statcast_pitches(pd.DataFrame(), season=args.season, logger=logger)
    else:
        frames = [read_parquet(path) for path in sorted(chunk_paths)]
        raw_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        normalized = normalize_statcast_pitches(raw_df, season=args.season, logger=logger)

    summary = validate_statcast_pitches(normalized)
    logger.info("Validation summary")
    logger.info("rows=%d", summary.rows)
    logger.info("unique_batters=%d", summary.unique_batters)
    logger.info("unique_pitchers=%d", summary.unique_pitchers)
    logger.info("null_rate_pitch_type=%.4f", summary.null_rate_pitch_type)
    logger.info("null_rate_plate_x=%.4f", summary.null_rate_plate_x)
    logger.info("null_rate_plate_z=%.4f", summary.null_rate_plate_z)
    logger.info("dedupe_key=%s", summary.dedupe_key)

    _atomic_write_parquet(normalized, output_path)
    logger.info("Wrote Statcast pitch data to %s", output_path)


if __name__ == "__main__":
    main()
