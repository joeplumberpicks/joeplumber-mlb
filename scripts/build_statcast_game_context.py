#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.statcast_game_context import build_and_write_statcast_game_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build shared Statcast game context features for starter matchups."
    )
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2024")
    parser.add_argument(
        "--pitches",
        type=str,
        default=None,
        help="Optional pitches parquet path (default: data/raw/statcast/pitches_{season}.parquet)",
    )
    parser.add_argument(
        "--spine",
        type=str,
        default=None,
        help="Optional spine parquet path (default discovery under data/processed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output parquet path (default: data/processed/statcast/statcast_game_context_{season}.parquet)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow writing output even when starter coverage falls below hard-fail threshold.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional quick-test limiter: use first N games from spine.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _discover_spine_path() -> Path:
    attempted = [
        Path("data/processed/model_spine_game.parquet"),
        Path("data/processed/model_spine.parquet"),
        Path("data/processed/spine_game.parquet"),
        Path("data/processed/spine.parquet"),
    ]
    for path in attempted:
        if path.exists():
            return path

    attempted_text = "\n".join(f"- {p}" for p in attempted)
    raise FileNotFoundError(f"No default spine parquet found. Attempted:\n{attempted_text}")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("build_statcast_game_context")

    pitches_path = Path(args.pitches) if args.pitches else Path(
        f"data/raw/statcast/pitches_{args.season}.parquet"
    )
    spine_path = Path(args.spine) if args.spine else _discover_spine_path()
    output_path = Path(args.output) if args.output else Path(
        f"data/processed/statcast/statcast_game_context_{args.season}.parquet"
    )

    logger.info("resolved_pitches_input=%s", pitches_path)
    logger.info("resolved_spine_input=%s", spine_path)

    build_and_write_statcast_game_context(
        season=args.season,
        pitches_path=pitches_path,
        spine_path=spine_path,
        output_path=output_path,
        allow_partial=args.allow_partial,
        max_games=args.max_games,
        logger=logger,
    )


if __name__ == "__main__":
    main()
