from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount, require_files
from src.utils.io import read_parquet, write_parquet

MART_SCHEMAS = {
    "hr_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "target_hr"],
    "nrfi_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "target_nrfi"],
    "moneyline_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "target_home_win"],
    "hitter_props_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "target_hitter_prop"],
    "pitcher_props_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "target_pitcher_prop"],
}


def _base_mart(spine: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    base_cols = [c for c in ["game_pk", "game_date", "season", "home_team", "away_team"] if c in spine.columns]
    df = spine[base_cols].copy() if not spine.empty else pd.DataFrame(columns=base_cols)
    for col in schema:
        if col not in df.columns:
            df[col] = pd.NA
    return df[schema]


def build_marts(dirs: dict[str, Path]) -> dict[str, Path]:
    spine_path = dirs["processed_dir"] / "model_spine_game.parquet"
    require_files([spine_path], "mart_build_model_spine")
    spine = read_parquet(spine_path)

    optional_inputs = [
        dirs["processed_dir"] / "batter_game_rolling.parquet",
        dirs["processed_dir"] / "pitcher_game_rolling.parquet",
        dirs["processed_dir"] / "by_season" / "parks_2024.parquet",
        dirs["processed_dir"] / "by_season" / "weather_game_2024.parquet",
    ]
    for path in optional_inputs:
        if not path.exists():
            print(f"WARNING: optional input missing, continuing with scaffold: {path.resolve()}")

    outputs: dict[str, Path] = {}
    for filename, schema in MART_SCHEMAS.items():
        mart_df = _base_mart(spine, schema)
        print_rowcount(filename.replace('.parquet', ''), mart_df)
        out_path = dirs["marts_dir"] / filename
        print(f"Writing to: {out_path.resolve()}")
        write_parquet(mart_df, out_path)
        outputs[filename] = out_path
    return outputs
