from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount
from src.utils.io import read_parquet, write_parquet


BATTER_SCHEMA = ["game_pk", "batter_id", "game_date", "season", "batters_faced", "hits", "hr"]
PITCHER_SCHEMA = ["game_pk", "pitcher_id", "game_date", "season", "outs", "er", "k"]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _rolling_from_source(source_df: pd.DataFrame, id_col: str, windows: list[int], shift_n: int) -> pd.DataFrame:
    if source_df.empty:
        return source_df

    df = source_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values([id_col, "game_date"]) 

    numeric_cols = [c for c in df.columns if c not in {"game_pk", id_col, "game_date", "season"}]
    for col in numeric_cols:
        for win in windows:
            roll_col = f"{col}_roll{win}"
            df[roll_col] = (
                df.groupby(id_col)[col]
                .transform(lambda s: s.shift(shift_n).rolling(win, min_periods=1).mean())
            )
    return df


def build_rolling_features(dirs: dict[str, Path], windows: list[int], shift_n: int) -> dict[str, Path]:
    processed_dir = dirs["processed_dir"]

    batter_source_path = processed_dir / "batter_game_logs.parquet"
    pitcher_source_path = processed_dir / "pitcher_game_logs.parquet"

    if batter_source_path.exists():
        batter_source = read_parquet(batter_source_path)
    else:
        batter_source = _empty(BATTER_SCHEMA)

    if pitcher_source_path.exists():
        pitcher_source = read_parquet(pitcher_source_path)
    else:
        pitcher_source = _empty(PITCHER_SCHEMA)

    batter_roll = _rolling_from_source(batter_source, "batter_id", windows, shift_n)
    pitcher_roll = _rolling_from_source(pitcher_source, "pitcher_id", windows, shift_n)

    batter_out = processed_dir / "batter_game_rolling.parquet"
    pitcher_out = processed_dir / "pitcher_game_rolling.parquet"
    print_rowcount("batter_game_rolling", batter_roll)
    print_rowcount("pitcher_game_rolling", pitcher_roll)
    print(f"Writing to: {batter_out.resolve()}")
    write_parquet(batter_roll, batter_out)
    print(f"Writing to: {pitcher_out.resolve()}")
    write_parquet(pitcher_roll, pitcher_out)

    return {"batter_game_rolling": batter_out, "pitcher_game_rolling": pitcher_out}
