from __future__ import annotations

import math

import numpy as np
import pandas as pd

SAFE_ENGINEERED_COLS = {
    "lineup_slot", "lineup_slot_numeric", "lineup_bucket_top", "lineup_bucket_mid", "lineup_bucket_bottom",
    "expected_batting_order_pa", "lineup_confidence", "bat_ab_per_game_roll15", "bat_pa_per_game_roll15", "expected_ab_proxy",
    "park_factor_hits_hist_shrunk", "park_factor_runs_hist_shrunk", "park_factor_xbh_hist_shrunk", "park_factor_babip_hist", "park_factor_avg_launch_speed_hist", "park_factor_avg_launch_angle_hist",
    "park_factor_hits_2026_roll", "park_factor_hr_2026_roll", "park_factor_xbh_2026_roll", "park_factor_runs_2026_roll", "park_factor_babip_2026_roll", "park_factor_avg_launch_speed_2026_roll", "park_factor_avg_launch_angle_2026_roll",
    "park_factor_dynamic_weight", "park_factor_hist_weight", "dynamic_sample_confidence",
    "park_factor_hits_blend", "park_factor_runs_blend", "park_factor_xbh_blend", "park_factor_babip_blend", "park_factor_avg_launch_speed_blend", "park_factor_avg_launch_angle_blend",
    "temperature", "wind_speed", "weather_wind_out", "weather_wind_in",
}
ROLL_SUFFIXES = ("_roll3", "_roll7", "_roll15", "_roll30")
LINEUP_PA_MAP = {1: 4.65, 2: 4.55, 3: 4.45, 4: 4.35, 5: 4.25, 6: 4.10, 7: 3.95, 8: 3.80, 9: 3.70}
LIVE_ONLY_FEATURE_NAMES = {
    "lineup_slot", "lineup_slot_numeric", "expected_batting_order_pa", "expected_ab_proxy",
    "lineup_confidence", "lineup_status_confirmed", "lineup_status_projected", "lineup_status_fallback",
    "temperature", "wind_speed", "weather_wind_out", "weather_wind_in",
}


def season_series(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df.get("season"), errors="coerce")
    if s.notna().any():
        return s
    return pd.to_datetime(df.get("game_date"), errors="coerce").dt.year


def filter_season_range(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    s = season_series(df)
    return df[(s >= start) & (s <= end)].copy()


def select_safe_numeric_features(
    df: pd.DataFrame,
    excluded: set[str],
    extra_safe_cols: set[str] | None = None,
    exclude_live_only: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    safe_cols = set(SAFE_ENGINEERED_COLS)
    if extra_safe_cols:
        safe_cols.update(extra_safe_cols)

    numeric_features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded]
    unsafe_features = [c for c in numeric_features if not (c in safe_cols or c.endswith(ROLL_SUFFIXES))]
    if exclude_live_only:
        unsafe_features = unsafe_features + [c for c in numeric_features if c in LIVE_ONLY_FEATURE_NAMES and c not in unsafe_features]
    candidate = [c for c in numeric_features if c not in unsafe_features]

    keep: list[str] = []
    dropped_all_null: list[str] = []
    for c in candidate:
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            keep.append(c)
        else:
            dropped_all_null.append(c)

    return keep, unsafe_features, dropped_all_null


def classify_feature_sets(columns: list[str]) -> tuple[list[str], list[str]]:
    live_only = [c for c in columns if c in LIVE_ONLY_FEATURE_NAMES]
    historical = [c for c in columns if c not in LIVE_ONLY_FEATURE_NAMES]
    return historical, live_only


def coerce_lineup_slot_numeric(df: pd.DataFrame) -> pd.Series:
    for c in ["lineup_slot_numeric", "lineup_slot", "bat_order", "batting_order", "lineup_position", "order"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def lineup_expected_pa_from_slot(slot: pd.Series) -> pd.Series:
    return pd.to_numeric(slot, errors="coerce").map(LINEUP_PA_MAP)


def poisson_prob_at_least(mu: float | pd.Series | np.ndarray, threshold: int) -> float | pd.Series | np.ndarray:
    if threshold <= 0:
        if isinstance(mu, pd.Series):
            return pd.Series(1.0, index=mu.index)
        if isinstance(mu, np.ndarray):
            return np.ones_like(mu, dtype=float)
        return 1.0

    def _single(v: float) -> float:
        if not np.isfinite(v) or v <= 0:
            return 0.0
        cdf = 0.0
        for k in range(threshold):
            cdf += math.exp(-v) * (v**k) / math.factorial(k)
        return float(np.clip(1.0 - cdf, 0.0, 1.0))

    if isinstance(mu, pd.Series):
        return mu.astype(float).map(_single)
    if isinstance(mu, np.ndarray):
        return np.array([_single(float(x)) for x in mu], dtype=float)
    return _single(float(mu))
