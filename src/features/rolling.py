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

    # Elite HR Engine feature blocks (safe/no-leakage with shift).
    if entity == "batter":
        launch_speed = pd.to_numeric(df.get("launch_speed"), errors="coerce") if "launch_speed" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Float64")
        launch_angle = pd.to_numeric(df.get("launch_angle"), errors="coerce") if "launch_angle" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Float64")

        df["barrel_proxy"] = ((launch_speed >= 95) & launch_angle.between(20, 35)).astype("Int64")
        df["hard_hit"] = (launch_speed >= 95).astype("Int64")
        df["sweet_spot"] = launch_angle.between(10, 35).astype("Int64")

        df["hr_power_index"] = launch_speed * 0.4 + launch_angle * 0.3 + pd.to_numeric(df["barrel_proxy"], errors="coerce") * 0.3

        for w in [3, 7, 15, 30]:
            df[f"barrel_rate_roll{w}"] = (
                df.groupby(canonical_id)["barrel_proxy"].transform(lambda s: s.shift(shift_n).rolling(w, min_periods=1).mean())
            )
            df[f"hard_hit_rate_roll{w}"] = (
                df.groupby(canonical_id)["hard_hit"].transform(lambda s: s.shift(shift_n).rolling(w, min_periods=1).mean())
            )
            df[f"sweet_spot_rate_roll{w}"] = (
                df.groupby(canonical_id)["sweet_spot"].transform(lambda s: s.shift(shift_n).rolling(w, min_periods=1).mean())
            )

        for w in [7, 15, 30]:
            df[f"hr_power_index_roll{w}_mean"] = (
                df.groupby(canonical_id)["hr_power_index"].transform(lambda s: s.shift(shift_n).rolling(w, min_periods=1).mean())
            )
            df[f"hr_power_index_roll{w}_max"] = (
                df.groupby(canonical_id)["hr_power_index"].transform(lambda s: s.shift(shift_n).rolling(w, min_periods=1).max())
            )

    if entity == "pitcher":
        launch_speed = pd.to_numeric(df.get("launch_speed"), errors="coerce") if "launch_speed" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Float64")
        launch_angle = pd.to_numeric(df.get("launch_angle"), errors="coerce") if "launch_angle" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Float64")

        df["barrel_proxy"] = ((launch_speed >= 95) & launch_angle.between(20, 35)).astype("Int64")
        df["hard_hit"] = (launch_speed >= 95).astype("Int64")

        if "is_hr_allowed" in df.columns:
            df["pitcher_hr_allowed"] = pd.to_numeric(df["is_hr_allowed"], errors="coerce")
        elif "hr_allowed" in df.columns:
            df["pitcher_hr_allowed"] = pd.to_numeric(df["hr_allowed"], errors="coerce")
        elif "hr" in df.columns:
            df["pitcher_hr_allowed"] = pd.to_numeric(df["hr"], errors="coerce")
        else:
            df["pitcher_hr_allowed"] = pd.NA

        if "contact_rate" in df.columns:
            contact_base = pd.to_numeric(df["contact_rate"], errors="coerce")
        elif "is_in_play" in df.columns:
            contact_base = pd.to_numeric(df["is_in_play"], errors="coerce")
        elif "k" in df.columns:
            contact_base = 1.0 - pd.to_numeric(df["k"], errors="coerce")
        else:
            contact_base = pd.Series(pd.NA, index=df.index, dtype="Float64")
        df["pit_contact_proxy"] = contact_base

        for w in [7, 15, 30]:
            df[f"pit_hr_rate_roll{w}"] = (
                df.groupby(canonical_id)["pitcher_hr_allowed"].transform(lambda s: s.shift(shift_n).rolling(w, min_periods=1).mean())
            )
            df[f"pit_barrel_rate_roll{w}"] = (
                df.groupby(canonical_id)["barrel_proxy"].transform(lambda s: s.shift(shift_n).rolling(w, min_periods=1).mean())
            )
            df[f"pit_hard_hit_rate_roll{w}"] = (
                df.groupby(canonical_id)["hard_hit"].transform(lambda s: s.shift(shift_n).rolling(w, min_periods=1).mean())
            )
            df[f"pit_contact_rate_roll{w}"] = (
                df.groupby(canonical_id)["pit_contact_proxy"].transform(lambda s: s.shift(shift_n).rolling(w, min_periods=1).mean())
            )

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
