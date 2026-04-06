from __future__ import annotations

import pandas as pd


def apply_rolling(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    cols: list[str],
    windows: tuple[int, ...] = (3, 7, 15, 30),
    shift: int = 1,
) -> pd.DataFrame:
    """
    Apply grouped rolling means with leakage protection via shift().

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.
    group_col : str
        Entity id column, e.g. batter_id or pitcher_id.
    date_col : str
        Date column used for chronological ordering.
    cols : list[str]
        Numeric columns to roll.
    windows : tuple[int, ...]
        Rolling window sizes.
    shift : int
        Leakage-prevention shift. Use 1 for prior-games-only features.
    """
    out = df.copy()

    if group_col not in out.columns:
        raise KeyError(f"Missing group_col: {group_col}")
    if date_col not in out.columns:
        raise KeyError(f"Missing date_col: {date_col}")

    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    sort_cols = [c for c in [group_col, date_col, "game_pk"] if c in out.columns]
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    valid_cols: list[str] = []
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            valid_cols.append(c)

    for w in windows:
        for c in valid_cols:
            out[f"{c}_roll{w}"] = (
                out.groupby(group_col, dropna=False)[c]
                .transform(lambda x: x.shift(shift).rolling(w, min_periods=1).mean())
            )

    return out