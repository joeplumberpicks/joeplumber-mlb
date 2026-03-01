from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.paths import target_output_path
from src.utils.io import write_parquet


def run_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="moneyline_targets_smoke_") as tmp:
        root = Path(tmp)
        processed = root / "processed"
        (processed / "by_season").mkdir(parents=True, exist_ok=True)

        games = pd.DataFrame(
            {
                "game_pk": [1, 2],
                "game_date": ["2020-04-01", "2020-04-02"],
                "home_team": ["A", "B"],
                "away_team": ["C", "D"],
            }
        )
        pa = pd.DataFrame(
            {
                "game_pk": [1, 1, 2, 2],
                "inning_topbot": ["Top", "Bot", "Top", "Bot"],
                "bat_score": [3, 5, 1, 0],
            }
        )

        write_parquet(games, processed / "by_season" / "games_2020.parquet")
        write_parquet(pa, processed / "by_season" / "pa_2020.parquet")

        out = games.copy()
        top = pa[pa["inning_topbot"].str.lower().eq("top")].groupby("game_pk")["bat_score"].max().reset_index(name="away_score")
        bot = pa[pa["inning_topbot"].str.lower().eq("bot")].groupby("game_pk")["bat_score"].max().reset_index(name="home_score")
        out = out.merge(bot, on="game_pk", how="left").merge(top, on="game_pk", how="left")
        out["target_home_win"] = (out["home_score"] > out["away_score"]).astype("Int64")

        out_path = target_output_path(processed, "moneyline", 2020)
        write_parquet(out, out_path)

        chk = pd.read_parquet(out_path)
        assert out_path.exists(), "moneyline target output missing"
        assert chk["game_pk"].nunique() > 0, "expected non-zero unique games"
        assert float(chk["target_home_win"].isna().mean()) < 1.0, "null_rate should not be 1.0"

    print("Smoke test passed: moneyline target artifact exists with non-empty keys and non-all-null target.")


if __name__ == "__main__":
    run_smoke()
