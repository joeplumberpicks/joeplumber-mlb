from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def safe_numeric(series: pd.Series, fill_value: float | int | None = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


def safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return num / den.where(den.ne(0))


def logistic_score(x: pd.Series | np.ndarray | float, center: float = 0.0, scale: float = 1.0) -> pd.Series:
    s = pd.Series(x, copy=False, dtype="float64")
    scale = 1.0 if scale == 0 else float(scale)
    z = (s - center) / scale
    return 1.0 / (1.0 + np.exp(-z))


def zscore_by_slate(df: pd.DataFrame, value_cols: Iterable[str], group_col: str = "game_date") -> pd.DataFrame:
    out = df.copy()
    for col in value_cols:
        if col not in out.columns:
            continue
        g = out.groupby(group_col, dropna=False)[col]
        mean = g.transform("mean")
        std = g.transform("std").replace(0, np.nan)
        out[f"{col}_z"] = ((pd.to_numeric(out[col], errors="coerce") - mean) / std).clip(-2.5, 2.5)
    return out


def add_group_rolling_sums(
    df: pd.DataFrame,
    *,
    group_col: str,
    sort_cols: list[str],
    value_cols: list[str],
    windows: tuple[int, ...] = (3, 7, 15, 30),
    min_periods: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([group_col, *sort_cols], kind="stable").reset_index(drop=True)

    for col in value_cols:
        base = pd.to_numeric(out[col], errors="coerce")
        shifted = base.groupby(out[group_col], dropna=False).shift(1)
        for w in windows:
            out[f"{col}_roll{w}"] = (
                shifted.groupby(out[group_col], dropna=False)
                .rolling(w, min_periods=min_periods)
                .sum()
                .reset_index(level=0, drop=True)
            )

    return out


def add_group_rolling_means(
    df: pd.DataFrame,
    *,
    group_col: str,
    sort_cols: list[str],
    value_cols: list[str],
    windows: tuple[int, ...] = (3, 7, 15, 30),
    min_periods: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([group_col, *sort_cols], kind="stable").reset_index(drop=True)

    for col in value_cols:
        base = pd.to_numeric(out[col], errors="coerce")
        shifted = base.groupby(out[group_col], dropna=False).shift(1)
        for w in windows:
            out[f"{col}_roll{w}"] = (
                shifted.groupby(out[group_col], dropna=False)
                .rolling(w, min_periods=min_periods)
                .mean()
                .reset_index(level=0, drop=True)
            )

    return out


def add_group_rolling_rates(
    df: pd.DataFrame,
    *,
    numerators: dict[str, str],
    denominators: dict[str, str],
    windows: tuple[int, ...] = (3, 7, 15, 30),
) -> pd.DataFrame:
    out = df.copy()
    for feature_name, num_base in numerators.items():
        den_base = denominators[feature_name]
        for w in windows:
            num_col = f"{num_base}_roll{w}"
            den_col = f"{den_base}_roll{w}"
            if num_col in out.columns and den_col in out.columns:
                out[f"{feature_name}_roll{w}"] = safe_rate(out[num_col], out[den_col])
    return out


def latest_row_per_key(df: pd.DataFrame, key_col: str, sort_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.sort_values([key_col, *sort_cols], kind="stable").drop_duplicates(subset=[key_col], keep="last")
    return out.reset_index(drop=True)
