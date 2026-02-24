from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.rolling import _rolling_from_source


def run_smoke() -> None:
    batter_df = pd.DataFrame(
        {
            "batter_id": [101, 101, 101],
            "game_date": ["2024-04-01", "2024-04-02", "2024-04-03"],
            "game_pk": [1, 2, 3],
            "sample_stat": [1.0, 2.0, 3.0],
        }
    )
    batter_roll = _rolling_from_source(batter_df, "batter", windows=[3], shift_n=1)
    assert "batter" in batter_roll.columns, "Expected canonical batter column in rolling output"

    pitcher_df = pd.DataFrame(
        {
            "pitcher_id": [201, 201, 201],
            "game_date": ["2024-04-01", "2024-04-02", "2024-04-03"],
            "game_pk": [1, 2, 3],
            "sample_stat": [4.0, 5.0, 6.0],
        }
    )
    pitcher_roll = _rolling_from_source(pitcher_df, "pitcher", windows=[3], shift_n=1)
    assert "pitcher" in pitcher_roll.columns, "Expected canonical pitcher column in rolling output"

    print("Smoke test passed: rolling id detection works for batter_id and pitcher_id.")


if __name__ == "__main__":
    run_smoke()
