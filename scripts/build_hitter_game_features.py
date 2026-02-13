#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.hitter_game_features import build_and_write_hitter_game_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hitter-game feature table.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--events", type=str, default="data/processed/events_pa.parquet")
    p.add_argument("--games", type=str, default="data/processed/games.parquet")
    p.add_argument("--spine", type=str, default="data/processed/model_spine_game.parquet")
    p.add_argument("--game-features", type=str, default=None, help="default data/processed/model_features_game_{season}.parquet")
    p.add_argument("--ppmi", type=str, default=None, help="default data/processed/statcast/ppmi_matchup_{season}.parquet")
    p.add_argument("--hitter-pitchtype", type=str, default=None, help="default data/processed/statcast/hitter_pitchtype_{season}.parquet")
    p.add_argument("--pitcher-mix", type=str, default=None, help="default data/processed/statcast/pitcher_mix_{season}.parquet")
    p.add_argument("--batter-rolling", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="default data/processed/model_features_hitter_game_{season}.parquet")
    p.add_argument("--allow-partial", action="store_true")
    return p.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def main() -> None:
    setup_logging()
    a = parse_args()
    logger = logging.getLogger("build_hitter_game_features")

    build_and_write_hitter_game_features(
        season=a.season,
        events_path=Path(a.events),
        games_path=Path(a.games),
        spine_path=Path(a.spine),
        game_features_path=Path(a.game_features) if a.game_features else Path(f"data/processed/model_features_game_{a.season}.parquet"),
        ppmi_path=Path(a.ppmi) if a.ppmi else Path(f"data/processed/statcast/ppmi_matchup_{a.season}.parquet"),
        hitter_pitchtype_path=Path(a.hitter_pitchtype) if a.hitter_pitchtype else Path(f"data/processed/statcast/hitter_pitchtype_{a.season}.parquet"),
        pitcher_mix_path=Path(a.pitcher_mix) if a.pitcher_mix else Path(f"data/processed/statcast/pitcher_mix_{a.season}.parquet"),
        batter_rolling_path=Path(a.batter_rolling) if a.batter_rolling else None,
        output_path=Path(a.output) if a.output else Path(f"data/processed/model_features_hitter_game_{a.season}.parquet"),
        start=a.start,
        end=a.end,
        allow_partial=a.allow_partial,
        logger=logger,
    )


if __name__ == "__main__":
    main()
