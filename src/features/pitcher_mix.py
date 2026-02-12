from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.statcast_helpers import is_ball_in_play, is_swing, is_whiff

REQUIRED_BASE_COLUMNS = ["pitcher_id", "pitch_type", "game_date"]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.astype("float64").replace(0.0, np.nan)
    return numerator.astype("float64") / den


def _normalize_columns(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    out = df.copy()

    if "pitcher" in out.columns and "pitcher_id" not in out.columns:
        out = out.rename(columns={"pitcher": "pitcher_id"})

    if "pitcher_id" in out.columns:
        out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if "pitch_type" in out.columns:
        out["pitch_type"] = out["pitch_type"].astype("string").str.strip().str.upper()

    if "batter_stand" in out.columns:
        out["batter_stand"] = out["batter_stand"].astype("string").str.strip().str.upper().fillna("UNK")
        out.loc[out["batter_stand"] == "", "batter_stand"] = "UNK"
    else:
        out["batter_stand"] = "UNK"

    return out


def _zone_indicator(df: pd.DataFrame) -> pd.Series:
    if "zone" in df.columns:
        zone_num = pd.to_numeric(df["zone"], errors="coerce")
        return zone_num.between(1, 9)

    if "plate_x" in df.columns and "plate_z" in df.columns:
        plate_x = pd.to_numeric(df["plate_x"], errors="coerce")
        plate_z = pd.to_numeric(df["plate_z"], errors="coerce")
        return plate_x.abs().le(0.83) & plate_z.between(1.5, 3.5)

    return pd.Series(np.nan, index=df.index, dtype="float64")


def load_pitches(input_path: Path) -> pd.DataFrame:
    """Load Statcast pitch parquet with a clear error if missing."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing Statcast pitch parquet: {input_path}. "
            "Run scripts/pull_statcast_pitches.py first."
        )
    return pd.read_parquet(input_path, engine="pyarrow")


def build_pitcher_mix_features(
    pitches: pd.DataFrame,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build pitcher pitch-mix and quality metrics by batter handedness and pitch type."""
    log = logger or logging.getLogger(__name__)
    df = _normalize_columns(pitches, logger=log)

    missing_base = [c for c in ["pitcher_id", "game_date"] if c not in df.columns]
    if missing_base:
        raise ValueError(f"Statcast pitches missing required columns: {missing_base}")

    total_rows = len(df)
    pitch_type_null_rate = float(df["pitch_type"].isna().mean()) if "pitch_type" in df.columns else 1.0
    plate_x_null_rate = float(df["plate_x"].isna().mean()) if "plate_x" in df.columns else 1.0
    plate_z_null_rate = float(df["plate_z"].isna().mean()) if "plate_z" in df.columns else 1.0

    log.info("rows_loaded=%d", total_rows)
    log.info("unique_pitchers=%d", int(df["pitcher_id"].nunique(dropna=True)))
    log.info("null_rate_pitch_type=%.4f", pitch_type_null_rate)
    log.info("null_rate_plate_x=%.4f", plate_x_null_rate)
    log.info("null_rate_plate_z=%.4f", plate_z_null_rate)

    if "pitch_type" not in df.columns:
        raise ValueError("Statcast pitches missing 'pitch_type' column required for pitch mix output")

    top_pitch_types = (
        df["pitch_type"].value_counts(dropna=True).head(10).rename_axis("pitch_type").reset_index(name="count")
    )
    log.info("top_pitch_types=%s", top_pitch_types.to_dict(orient="records"))

    df = df.dropna(subset=["pitcher_id", "pitch_type"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "pitcher_id",
                "batter_stand",
                "pitch_type",
                "pitches",
                "pitch_usage_pct",
                "avg_velo",
                "zone_pct",
                "swing_rate",
                "whiff_rate",
                "inplay_rate",
                "hr_per_bip",
                "xwoba_on_contact",
                "hard_hit_rate",
            ]
        )

    group_cols = ["pitcher_id", "batter_stand", "pitch_type"]
    if "pitcher_hand" in df.columns:
        df["pitcher_hand"] = df["pitcher_hand"].astype("string").str.strip().str.upper()
        group_cols = ["pitcher_id", "pitcher_hand", "batter_stand", "pitch_type"]

    zone_indicator = _zone_indicator(df)
    has_zone = bool(zone_indicator.notna().any())

    swing = is_swing(df["description"] if "description" in df.columns else None, df["pitch_type"])
    whiff = is_whiff(df["description"] if "description" in df.columns else None, df["pitch_type"])
    bip = is_ball_in_play(
        df["description"] if "description" in df.columns else None,
        df["events"] if "events" in df.columns else None,
        df["type"] if "type" in df.columns else None,
        df["bb_type"] if "bb_type" in df.columns else None,
        fallback_index=df.index,
    )

    hr = (
        df["events"].astype("string").str.lower().eq("home_run").fillna(False)
        if "events" in df.columns
        else pd.Series(False, index=df.index)
    )

    prepared = df.assign(
        _swing=swing.astype("int64"),
        _whiff=whiff.astype("int64"),
        _bip=bip.astype("int64"),
        _hr=hr.astype("int64"),
    )

    if has_zone:
        prepared["_zone"] = zone_indicator.astype("float64")
    else:
        prepared["_zone"] = np.nan

    if "release_speed" in prepared.columns:
        prepared["release_speed"] = pd.to_numeric(prepared["release_speed"], errors="coerce")
    else:
        prepared["release_speed"] = np.nan

    if "estimated_woba_using_speedangle" in prepared.columns:
        prepared["estimated_woba_using_speedangle"] = pd.to_numeric(
            prepared["estimated_woba_using_speedangle"], errors="coerce"
        )

    if "launch_speed" in prepared.columns:
        prepared["launch_speed"] = pd.to_numeric(prepared["launch_speed"], errors="coerce")

    agg = prepared.groupby(group_cols, dropna=False, observed=False).agg(
        pitches=("pitch_type", "size"),
        avg_velo=("release_speed", "mean"),
        zone_pct=("_zone", "mean"),
        swings=("_swing", "sum"),
        whiffs=("_whiff", "sum"),
        balls_in_play=("_bip", "sum"),
        hrs=("_hr", "sum"),
    )

    if "estimated_woba_using_speedangle" in prepared.columns:
        xwoba = (
            prepared.loc[prepared["_bip"] == 1]
            .groupby(group_cols, dropna=False, observed=False)["estimated_woba_using_speedangle"]
            .mean()
            .rename("xwoba_on_contact")
        )
        agg = agg.join(xwoba, how="left")
    else:
        agg["xwoba_on_contact"] = np.nan

    if "launch_speed" in prepared.columns:
        hard_hit = (
            prepared.loc[prepared["_bip"] == 1]
            .assign(_hard_hit=lambda x: x["launch_speed"].ge(95).astype("int64"))
            .groupby(group_cols, dropna=False, observed=False)["_hard_hit"]
            .mean()
            .rename("hard_hit_rate")
        )
        agg = agg.join(hard_hit, how="left")
    else:
        agg["hard_hit_rate"] = np.nan

    total_group_cols = ["pitcher_id", "batter_stand"]
    if "pitcher_hand" in group_cols:
        total_group_cols = ["pitcher_id", "pitcher_hand", "batter_stand"]

    totals = (
        prepared.groupby(total_group_cols, dropna=False, observed=False)
        .size()
        .rename("total_pitches")
        .reset_index()
    )

    out = agg.reset_index().merge(totals, on=total_group_cols, how="left")
    out["pitch_usage_pct"] = _safe_divide(out["pitches"], out["total_pitches"])
    out["swing_rate"] = _safe_divide(out["swings"], out["pitches"])
    out["whiff_rate"] = _safe_divide(out["whiffs"], out["swings"])
    out["inplay_rate"] = _safe_divide(out["balls_in_play"], out["swings"])
    out["hr_per_bip"] = _safe_divide(out["hrs"], out["balls_in_play"])

    keep_cols = [
        "pitcher_id",
        "batter_stand",
        "pitch_type",
        "pitches",
        "pitch_usage_pct",
        "avg_velo",
        "zone_pct",
        "swing_rate",
        "whiff_rate",
        "inplay_rate",
        "hr_per_bip",
        "xwoba_on_contact",
        "hard_hit_rate",
    ]
    if "pitcher_hand" in out.columns:
        keep_cols.insert(1, "pitcher_hand")

    out = out[keep_cols].sort_values(
        by=["pitcher_id", "batter_stand", "pitch_usage_pct", "pitch_type"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )

    log.info("groups_created=%d", len(out))
    return out.reset_index(drop=True)


def build_and_write_pitcher_mix(
    season: int,
    input_path: Path,
    output_path: Path,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Load pitches, compute pitcher mix features, and write parquet output."""
    log = logger or logging.getLogger(__name__)
    pitches = load_pitches(input_path)
    features = build_pitcher_mix_features(pitches, logger=log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_pitcher_mix season=%s rows=%d path=%s", season, len(features), output_path)
    return features
