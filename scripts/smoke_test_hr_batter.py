from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.marts.build_hr_batter_features import build_hr_batter_features
from src.utils.io import write_parquet


def run_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="hr_batter_smoke_") as tmp:
        root = Path(tmp)
        dirs = {
            "processed_dir": root / "processed",
            "marts_dir": root / "marts",
        }
        (dirs["processed_dir"] / "by_season").mkdir(parents=True, exist_ok=True)
        dirs["marts_dir"].mkdir(parents=True, exist_ok=True)

        batter_game = pd.DataFrame(
            {
                "game_pk": [1, 1, 2],
                "batter_id": [101, 102, 103],
                "batter_team": ["NYY", "NYY", "BOS"],
                "bat_hr": [0, 1, 0],
                "game_date": ["2024-04-01", "2024-04-01", "2024-04-02"],
                "season": [2024, 2024, 2024],
            }
        )
        spine = pd.DataFrame(
            {
                "game_pk": [1, 2],
                "game_date": ["2024-04-01", "2024-04-02"],
                "home_team": ["NYY", "BOS"],
                "away_team": ["TOR", "NYY"],
                "home_sp_id": [9001, 9002],
                "away_sp_id": [9101, 9102],
                "park_id": [31, 32],
                "season": [2024, 2024],
            }
        )
        batter_roll = pd.DataFrame(
            {
                "game_pk": [1, 1, 2],
                "batter_id": [101, 102, 103],
                "bat_recent": [0.1, 0.2, 0.3],
            }
        )
        pitcher_roll = pd.DataFrame(
            {
                "game_pk": [1, 2],
                "pitcher_id": [9101, 9102],
                "pit_recent": [0.4, 0.5],
            }
        )

        write_parquet(batter_game, dirs["processed_dir"] / "by_season" / "batter_game_2024.parquet")
        write_parquet(spine, dirs["processed_dir"] / "model_spine_game.parquet")
        write_parquet(batter_roll, dirs["processed_dir"] / "batter_game_rolling.parquet")
        write_parquet(pitcher_roll, dirs["processed_dir"] / "pitcher_game_rolling.parquet")

        out_path = build_hr_batter_features(dirs, season=2024)
        out = pd.read_parquet(out_path)

        assert "target_hr" in out.columns, "target_hr missing"
        assert len(out) == len(batter_game), "Expected batter-game grain in hr_batter_features"
        assert "opp_sp_id" in out.columns, "opp_sp_id missing"

    print("Smoke test passed: hr batter mart builds with expected grain and target.")


if __name__ == "__main__":
    run_smoke()
