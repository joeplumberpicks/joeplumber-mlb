#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.ppmi_matchup import build_and_write_ppmi_matchup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Statcast PPMI matchup feature table.")
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2024")
    parser.add_argument(
        "--pitcher-mix",
        type=str,
        default=None,
        help="Optional pitcher mix parquet path (default: data/processed/statcast/pitcher_mix_{season}.parquet)",
    )
    parser.add_argument(
        "--hitter-pitchtype",
        type=str,
        default=None,
        help="Optional hitter pitch-type parquet path (default: data/processed/statcast/hitter_pitchtype_{season}.parquet)",
    )
    parser.add_argument(
        "--matchups",
        type=str,
        default=None,
        help=(
            "Optional matchup parquet path. Default behavior: try data/processed/model_spine_game.parquet "
            "if it has batter/pitcher rows, else data/processed/matchups_{season}.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output parquet path (default: data/processed/statcast/ppmi_matchup_{season}.parquet)",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    args = parse_args()
    logger = logging.getLogger("build_ppmi_matchup")

    processed_dir = Path("data/processed")
    pitcher_mix = Path(args.pitcher_mix) if args.pitcher_mix else Path(
        f"data/processed/statcast/pitcher_mix_{args.season}.parquet"
    )
    hitter_pitchtype = Path(args.hitter_pitchtype) if args.hitter_pitchtype else Path(
        f"data/processed/statcast/hitter_pitchtype_{args.season}.parquet"
    )
    matchups = Path(args.matchups) if args.matchups else None
    output = Path(args.output) if args.output else Path(
        f"data/processed/statcast/ppmi_matchup_{args.season}.parquet"
    )

    build_and_write_ppmi_matchup(
        season=args.season,
        pitcher_mix_path=pitcher_mix,
        hitter_pitchtype_path=hitter_pitchtype,
        processed_dir=processed_dir,
        output_path=output,
        matchups_path=matchups,
        logger=logger,
    )


if __name__ == "__main__":
    main()
