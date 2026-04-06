from __future__ import annotations

import pandas as pd


def _ensure_binary(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _ensure_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    return df


def _ensure_required_base_columns(pa: pd.DataFrame) -> pd.DataFrame:
    df = pa.copy()

    for col in ["game_pk", "batter_id", "pitcher_id"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    if "game_date" not in df.columns:
        raise KeyError("Missing required column: game_date")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    for col in [
        "is_hit",
        "is_hr",
        "is_rbi",
        "is_bb",
        "is_so",
        "is_barrel",
        "is_hard_hit",
        "is_1b",
        "is_2b",
        "is_3b",
    ]:
        df = _ensure_binary(df, col)

    for col in ["launch_speed", "launch_angle"]:
        df = _ensure_numeric(df, col, default=0.0)

    if "total_bases" not in df.columns:
        df["total_bases"] = (
            df["is_1b"]
            + 2 * df["is_2b"]
            + 3 * df["is_3b"]
            + 4 * df["is_hr"]
        )
    else:
        df["total_bases"] = pd.to_numeric(df["total_bases"], errors="coerce").fillna(0.0)

    if "batter_name" not in df.columns:
        df["batter_name"] = pd.NA
    if "pitcher_name" not in df.columns:
        df["pitcher_name"] = pd.NA

    return df


def build_batter_game(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per batter-game from plate appearance/statcast data.
    """
    df = _ensure_required_base_columns(pa)

    grouped = (
        df.groupby(["game_pk", "game_date", "batter_id"], dropna=False)
        .agg(
            batter_name=("batter_name", "first"),
            pa=("batter_id", "count"),
            hits=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            rbi=("is_rbi", "sum"),
            tb=("total_bases", "sum"),
            ev_mean=("launch_speed", "mean"),
            ev_max=("launch_speed", "max"),
            la_mean=("launch_angle", "mean"),
            barrels=("is_barrel", "sum"),
            hardhit=("is_hard_hit", "sum"),
            bb=("is_bb", "sum"),
            so=("is_so", "sum"),
        )
        .reset_index()
    )

    grouped["barrel_rate"] = grouped["barrels"] / grouped["pa"].where(grouped["pa"].ne(0))
    grouped["hardhit_rate"] = grouped["hardhit"] / grouped["pa"].where(grouped["pa"].ne(0))
    grouped["k_rate"] = grouped["so"] / grouped["pa"].where(grouped["pa"].ne(0))
    grouped["bb_rate"] = grouped["bb"] / grouped["pa"].where(grouped["pa"].ne(0))

    grouped = grouped.sort_values(["batter_id", "game_date", "game_pk"], kind="stable").reset_index(drop=True)
    return grouped


def build_pitcher_game(pa: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per pitcher-game from plate appearance/statcast data.
    """
    df = _ensure_required_base_columns(pa)

    grouped = (
        df.groupby(["game_pk", "game_date", "pitcher_id"], dropna=False)
        .agg(
            pitcher_name=("pitcher_name", "first"),
            bf=("pitcher_id", "count"),
            hits_allowed=("is_hit", "sum"),
            hr_allowed=("is_hr", "sum"),
            bb=("is_bb", "sum"),
            so=("is_so", "sum"),
            ev_mean=("launch_speed", "mean"),
            ev_max=("launch_speed", "max"),
            barrels=("is_barrel", "sum"),
            hardhit=("is_hard_hit", "sum"),
        )
        .reset_index()
    )

    grouped["barrel_rate_allowed"] = grouped["barrels"] / grouped["bf"].where(grouped["bf"].ne(0))
    grouped["hardhit_rate_allowed"] = grouped["hardhit"] / grouped["bf"].where(grouped["bf"].ne(0))
    grouped["k_rate"] = grouped["so"] / grouped["bf"].where(grouped["bf"].ne(0))
    grouped["bb_rate"] = grouped["bb"] / grouped["bf"].where(grouped["bf"].ne(0))

    grouped = grouped.sort_values(["pitcher_id", "game_date", "game_pk"], kind="stable").reset_index(drop=True)
    return grouped