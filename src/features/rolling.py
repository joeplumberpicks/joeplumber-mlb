from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount
from src.utils.io import read_parquet, write_parquet

BATTER_ID_CANDIDATES = ["batter", "batter_id", "mlbam_batter_id", "player_id"]
PITCHER_ID_CANDIDATES = ["pitcher", "pitcher_id", "mlbam_pitcher_id", "player_id"]


def _detect_id_column(df: pd.DataFrame, candidates: list[str], entity: str) -> str:
    for col in candidates:
        if col in df.columns:
            logging.info("Selected %s id column: %s", entity, col)
            return col
    raise ValueError(
        f"Could not find {entity} id column. Candidates: {candidates}. Available columns: {list(df.columns)}"
    )


def _normalize_entity_df(source_df: pd.DataFrame, entity: str) -> tuple[pd.DataFrame, str]:
    if source_df.empty:
        canonical = "batter" if entity == "batter" else "pitcher"
        return source_df.copy(), canonical

    if "game_date" not in source_df.columns:
        raise ValueError(f"Missing required game_date column for {entity} rolling. Available: {list(source_df.columns)}")

    candidates = BATTER_ID_CANDIDATES if entity == "batter" else PITCHER_ID_CANDIDATES
    detected_id = _detect_id_column(source_df, candidates, entity)
    canonical_id = "batter" if entity == "batter" else "pitcher"

    df = source_df.copy()
    if detected_id != canonical_id:
        df[canonical_id] = df[detected_id]

    before_rows = len(df)
    logging.info("%s rows before game_date coercion: %s", entity, before_rows)
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).copy()
    logging.info("%s rows after game_date coercion/dropna: %s", entity, len(df))

    sort_cols = [canonical_id, "game_date", "game_pk" if "game_pk" in df.columns else canonical_id]
    df = df.sort_values(sort_cols)
    return df, canonical_id


def _rolling_from_source(source_df: pd.DataFrame, entity: str, windows: list[int], shift_n: int) -> pd.DataFrame:
    if source_df.empty:
        return source_df.copy()

    df, canonical_id = _normalize_entity_df(source_df, entity)
    if df.empty:
        return df

    logging.info("Applying %s rolling windows: %s with shift=%s", entity, windows, shift_n)
    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns if c not in {canonical_id, "season"}]
    for col in numeric_cols:
        for win in windows:
            df[f"{col}_roll{win}"] = (
                df.groupby(canonical_id)[col].transform(lambda s: s.shift(shift_n).rolling(win, min_periods=1).mean())
            )

    return df


def _load_season_logs(processed_dir: Path, prefix: str) -> pd.DataFrame:
    by_season = processed_dir / "by_season"
    files = sorted(by_season.glob(f"{prefix}_*.parquet"))
    frames: list[pd.DataFrame] = []
    for path in files:
        frames.append(read_parquet(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_rolling_features(dirs: dict[str, Path], windows: list[int], shift_n: int) -> dict[str, Path]:
    processed_dir = dirs["processed_dir"]

    batter_source = _load_season_logs(processed_dir, "batter_game")
    pitcher_source = _load_season_logs(processed_dir, "pitcher_game")

    batter_roll = _rolling_from_source(batter_source, "batter", windows, shift_n)
    pitcher_roll = _rolling_from_source(pitcher_source, "pitcher", windows, shift_n)

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
