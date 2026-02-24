from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount
from src.utils.io import read_parquet, write_parquet


def _rolling_from_source(source_df: pd.DataFrame, id_col: str, windows: list[int], shift_n: int) -> pd.DataFrame:
    if source_df.empty:
        return source_df

    df = source_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values([id_col, "game_date", "game_pk" if "game_pk" in df.columns else id_col])

    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in {id_col, "season"}]
    for col in numeric_cols:
        for win in windows:
            df[f"{col}_roll{win}"] = (
                df.groupby(id_col)[col].transform(lambda s: s.shift(shift_n).rolling(win, min_periods=1).mean())
            )

    return df


def _load_season_logs(processed_dir: Path, prefix: str) -> pd.DataFrame:
    by_season = processed_dir / "by_season"
    files = sorted(by_season.glob(f"{prefix}_*.parquet"))
    frames: list[pd.DataFrame] = []
    for path in files:
        df = read_parquet(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_rolling_features(dirs: dict[str, Path], windows: list[int], shift_n: int) -> dict[str, Path]:
    processed_dir = dirs["processed_dir"]

    batter_source = _load_season_logs(processed_dir, "batter_game")
    pitcher_source = _load_season_logs(processed_dir, "pitcher_game")

    batter_roll = _rolling_from_source(batter_source, "batter", windows, shift_n) if not batter_source.empty else pd.DataFrame()
    pitcher_roll = _rolling_from_source(pitcher_source, "pitcher", windows, shift_n) if not pitcher_source.empty else pd.DataFrame()

    batter_out = processed_dir / "batter_game_rolling.parquet"
    pitcher_out = processed_dir / "pitcher_game_rolling.parquet"
    print_rowcount("batter_game_rolling", batter_roll)
    print_rowcount("pitcher_game_rolling", pitcher_roll)
    print(f"Writing to: {batter_out.resolve()}")
    write_parquet(batter_roll, batter_out)
    print(f"Writing to: {pitcher_out.resolve()}")
    write_parquet(pitcher_roll, pitcher_out)

    if not batter_roll.empty:
        print(f"Sample batter rolling cols: {batter_roll.columns[:12].tolist()}")
    if not pitcher_roll.empty:
        print(f"Sample pitcher rolling cols: {pitcher_roll.columns[:12].tolist()}")

    return {"batter_game_rolling": batter_out, "pitcher_game_rolling": pitcher_out}
