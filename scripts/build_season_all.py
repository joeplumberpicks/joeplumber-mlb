#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import load_config, read_parquet, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full season processed + Statcast layers.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--skip-statcast", action="store_true")
    parser.add_argument("--max-games", type=int, default=None)
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _run_script(script_name: str, args: list[str], logger: logging.Logger) -> None:
    script_path = REPO_ROOT / "scripts" / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")

    cmd = [sys.executable, str(script_path), *args]
    logger.info("RUN %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed ({result.returncode}): {script_name}")


def _processed_dir() -> Path:
    cfg = load_config()
    rel = cfg.get("paths", {}).get("processed", "data/processed")
    return REPO_ROOT / rel


def _raw_dir() -> Path:
    cfg = load_config()
    rel = cfg.get("paths", {}).get("raw", "data/raw")
    return REPO_ROOT / rel


def _ensure_core_tables(
    season: int,
    start: str | None,
    end: str | None,
    max_games: int | None,
    logger: logging.Logger,
) -> tuple[Path, Path]:
    processed = _processed_dir()
    games_path = processed / "games.parquet"
    events_path = processed / "events_pa.parquet"

    logger.info("Step 1/5: ensure core processed tables")
    logger.info("Expected core paths: games=%s events=%s", games_path, events_path)

    if not games_path.exists():
        if not (REPO_ROOT / "scripts/build_spine.py").exists():
            raise FileNotFoundError(
                f"Missing {games_path}. build_spine.py not found. Please run your games ingestion first."
            )
        args = ["--season", str(season)]
        if start:
            args += ["--start", start]
        if end:
            args += ["--end", end]
        _run_script("build_spine.py", args, logger)

    if not events_path.exists():
        if not (REPO_ROOT / "scripts/build_events.py").exists():
            raise FileNotFoundError(
                f"Missing {events_path}. build_events.py not found. Please run your events ingestion first."
            )
        args = ["--season", str(season), "--force"]
        if start:
            args += ["--start", start]
        if end:
            args += ["--end", end]
        if max_games is not None:
            args += ["--max_games", str(max_games)]
        _run_script("build_events.py", args, logger)

    if not games_path.exists() or not events_path.exists():
        raise FileNotFoundError(
            "Core tables still missing after orchestration. Missing: "
            f"games_exists={games_path.exists()} events_exists={events_path.exists()}"
        )

    logger.info("Core tables ready")
    return games_path, events_path


def _validate_spine(spine_path: Path, max_games: int | None, logger: logging.Logger) -> int:
    if not spine_path.exists():
        raise FileNotFoundError(f"Canonical spine not found after build: {spine_path}")
    df = read_parquet(spine_path)
    rows = len(df)
    logger.info("model_spine_game rows=%d path=%s", rows, spine_path)
    if max_games is None and rows < 200:
        logger.warning("model_spine_game row count is low for full season: rows=%d (<200)", rows)
    return rows


def _validate_statcast_pitches(path: Path, max_games: int | None, logger: logging.Logger) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Statcast pitches not found after pull: {path}")
    df = read_parquet(path)
    rows = len(df)
    logger.info("statcast pitches rows=%d path=%s", rows, path)
    if max_games is None and rows <= 100_000:
        logger.warning("Statcast pitch volume is unexpectedly low: rows=%d (<=100000)", rows)
    return rows


def _validate_game_context(path: Path, spine_rows: int, logger: logging.Logger) -> int:
    if not path.exists():
        raise FileNotFoundError(f"statcast_game_context output not found: {path}")
    df = read_parquet(path)
    rows = len(df)
    expected = spine_rows * 2
    logger.info("statcast_game_context rows=%d expected_approx=%d path=%s", rows, expected, path)
    if expected > 0 and rows < 0.9 * expected:
        logger.warning("statcast_game_context coverage low: rows=%d expected_approx=%d", rows, expected)
    return rows


def _build_matchups_from_events(season: int, logger: logging.Logger) -> Path:
    processed = _processed_dir()
    events_path = processed / "events_pa.parquet"
    matchups_path = processed / f"matchups_{season}.parquet"
    events = read_parquet(events_path)

    required = {"game_date", "batter_id", "pitcher_id"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Cannot create matchups file from events_pa; missing columns: {sorted(missing)}")

    out = events[["game_date", "batter_id", "pitcher_id"]].copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["batter_id"] = pd.to_numeric(out["batter_id"], errors="coerce").astype("Int64")
    out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")
    out["batter_stand"] = "UNK"
    out["pitcher_throws"] = "UNK"
    out = out.dropna(subset=["game_date", "batter_id", "pitcher_id"]).drop_duplicates(
        ["game_date", "batter_id", "pitcher_id"], keep="first"
    )
    out = out.sort_values(["game_date", "pitcher_id", "batter_id"], kind="mergesort").reset_index(drop=True)

    write_parquet(out, matchups_path)
    logger.info("Created minimal matchup file rows=%d path=%s", len(out), matchups_path)
    return matchups_path


def main() -> None:
    setup_logging()
    log = logging.getLogger("build_season_all")
    args = parse_args()

    log.info("Starting full-season build for season=%s", args.season)

    _ensure_core_tables(args.season, args.start, args.end, args.max_games, log)

    log.info("Step 2/5: build canonical spine")
    spine_args = ["--season", str(args.season)]
    if args.start:
        spine_args += ["--start", args.start]
    if args.end:
        spine_args += ["--end", args.end]
    _run_script("build_model_spine_game.py", spine_args, log)
    spine_path = _processed_dir() / "model_spine_game.parquet"
    spine_rows = _validate_spine(spine_path, args.max_games, log)

    if not args.skip_statcast:
        log.info("Step 3/5: pull Statcast pitches")
        pull_args = ["--season", str(args.season), "--resume"]
        if args.start:
            pull_args += ["--start", args.start]
        if args.end:
            pull_args += ["--end", args.end]
        _run_script("pull_statcast_pitches.py", pull_args, log)
        pitches_path = _raw_dir() / "statcast" / f"pitches_{args.season}.parquet"
        _validate_statcast_pitches(pitches_path, args.max_games, log)

        log.info("Step 4/5: build Statcast game context")
        context_args = ["--season", str(args.season)]
        if args.max_games is not None:
            context_args += ["--max-games", str(args.max_games), "--allow-partial"]
        _run_script("build_statcast_game_context.py", context_args, log)
        context_path = _processed_dir() / "statcast" / f"statcast_game_context_{args.season}.parquet"
        _validate_game_context(context_path, spine_rows, log)

        log.info("Step 5/5: optional Statcast feature layers")
        optional_scripts = [
            "build_pitcher_mix.py",
            "build_hitter_pitchtype.py",
        ]
        for script_name in optional_scripts:
            if (REPO_ROOT / "scripts" / script_name).exists():
                _run_script(script_name, ["--season", str(args.season)], log)

        if (REPO_ROOT / "scripts" / "build_ppmi_matchup.py").exists():
            matchups_path = _build_matchups_from_events(args.season, log)
            _run_script(
                "build_ppmi_matchup.py",
                ["--season", str(args.season), "--matchups", str(matchups_path)],
                log,
            )
    else:
        log.info("Statcast steps skipped (--skip-statcast).")

    log.info("Season build completed successfully.")


if __name__ == "__main__":
    main()
