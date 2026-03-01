from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.rolling import build_rolling_features
from src.utils.io import write_parquet


def run_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="rolling_smoke_") as tmp:
        processed_dir = Path(tmp) / "processed"
        by_season = processed_dir / "by_season"
        by_season.mkdir(parents=True, exist_ok=True)

        games = pd.DataFrame(
            {
                "game_pk": [1, 2],
                "game_date": ["2024-04-01", "2024-04-02"],
                "home_team": ["A", "B"],
                "away_team": ["C", "D"],
                "park_id": [10, 11],
            }
        )
        batter_game = pd.DataFrame(
            {
                "game_pk": [1, 1, 2, 2],
                "batter_id": [101, 102, 101, 103],
                "batter_team": ["C", "C", "B", "D"],
                "game_date": ["2024-04-01", "2024-04-01", "2024-04-02", "2024-04-02"],
                "sample_stat": [1.0, 2.0, 3.0, 4.0],
            }
        )
        pitcher_game = pd.DataFrame(
            {
                "game_pk": [1, 2],
                "pitcher_id": [201, 202],
                "game_date": ["2024-04-01", "2024-04-02"],
                "sample_stat": [5.0, 6.0],
            }
        )

        write_parquet(games, by_season / "games_2024.parquet")
        write_parquet(batter_game, by_season / "batter_game_2024.parquet")
        write_parquet(pitcher_game, by_season / "pitcher_game_2024.parquet")

        assert len(batter_game) > len(games), "Expected batter_game rows > games rows"
        nonnull_rate = pd.to_datetime(batter_game["game_date"], errors="coerce").notna().mean()
        assert nonnull_rate > 0.95, "Expected game_date non-null rate > 0.95"

        outputs = build_rolling_features({"processed_dir": processed_dir}, windows=[3], shift_n=1)
        batter_roll = pd.read_parquet(outputs["batter_game_rolling"])
        pitcher_roll = pd.read_parquet(outputs["pitcher_game_rolling"])

        assert not batter_roll.empty, "Expected non-empty batter rolling output"
        assert not pitcher_roll.empty, "Expected non-empty pitcher rolling output"
        assert "batter" in batter_roll.columns, "Expected canonical batter column in rolling output"
        assert "batter_team" in batter_roll.columns, "Expected batter_team to be preserved in rolling output"
        assert "pitcher" in pitcher_roll.columns, "Expected canonical pitcher column in rolling output"

    print("Smoke test passed: rolling pipeline grain/date/id expectations satisfied.")


if __name__ == "__main__":
    run_smoke()
