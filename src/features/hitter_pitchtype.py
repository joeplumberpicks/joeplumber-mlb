from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.statcast_helpers import is_ball_in_play, is_swing, is_whiff


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.astype("float64").replace(0.0, np.nan)
    return numerator.astype("float64") / den


def load_pitches(input_path: Path) -> pd.DataFrame:
    """Load Statcast pitch parquet with a clear error if missing."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing Statcast pitch parquet: {input_path}. "
            "Run scripts/pull_statcast_pitches.py first."
        )
    return pd.read_parquet(input_path, engine="pyarrow")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "batter" in out.columns and "batter_id" not in out.columns:
        out = out.rename(columns={"batter": "batter_id"})
    if "pitcher" in out.columns and "pitcher_id" not in out.columns:
        out = out.rename(columns={"pitcher": "pitcher_id"})

    if "batter_id" in out.columns:
        out["batter_id"] = pd.to_numeric(out["batter_id"], errors="coerce").astype("Int64")
    if "pitcher_id" in out.columns:
        out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if "pitch_type" in out.columns:
        out["pitch_type"] = out["pitch_type"].astype("string").str.strip().str.upper()

    if "pitcher_throws" in out.columns:
        out["pitcher_throws"] = out["pitcher_throws"].astype("string").str.strip().str.upper()
        out["pitcher_throws"] = out["pitcher_throws"].replace("", pd.NA).fillna("UNK")
    else:
        out["pitcher_throws"] = "UNK"

    return out


def build_hitter_pitchtype_features(
    pitches: pd.DataFrame,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build hitter pitch-type swing/quality features by pitcher handedness."""
    log = logger or logging.getLogger(__name__)
    df = _normalize_columns(pitches)

    missing_required = [c for c in ["batter_id", "pitch_type"] if c not in df.columns]
    if missing_required:
        raise ValueError(f"Statcast pitches missing required columns: {missing_required}")

    rows_loaded = len(df)
    null_rate_pitch_type = float(df["pitch_type"].isna().mean()) if rows_loaded else 0.0
    null_rate_plate_x = float(df["plate_x"].isna().mean()) if "plate_x" in df.columns and rows_loaded else 1.0
    null_rate_plate_z = float(df["plate_z"].isna().mean()) if "plate_z" in df.columns and rows_loaded else 1.0

    has_description = "description" in df.columns
    has_events = "events" in df.columns
    has_launch_speed = "launch_speed" in df.columns
    has_xwoba = "estimated_woba_using_speedangle" in df.columns

    log.info("rows_loaded=%d", rows_loaded)
    log.info("unique_batters=%d", int(df["batter_id"].nunique(dropna=True)))
    log.info("null_rate_pitch_type=%.4f", null_rate_pitch_type)
    log.info("null_rate_plate_x=%.4f", null_rate_plate_x)
    log.info("null_rate_plate_z=%.4f", null_rate_plate_z)
    log.info(
        "optional_columns description=%s events=%s launch_speed=%s estimated_woba_using_speedangle=%s",
        has_description,
        has_events,
        has_launch_speed,
        has_xwoba,
    )

    df = df.dropna(subset=["batter_id", "pitch_type"]).copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "batter_id",
                "pitcher_throws",
                "pitch_type",
                "pitches",
                "pitch_usage_pct",
                "swings",
                "swing_rate",
                "whiffs",
                "whiff_rate",
                "balls_in_play",
                "inplay_rate",
                "hr_per_bip",
                "avg_ev",
                "hard_hit_rate",
                "xwoba_on_contact",
            ]
        )

    swing = is_swing(df["description"] if has_description else None, df["pitch_type"]) if has_description else pd.Series(np.nan, index=df.index)
    whiff = is_whiff(df["description"] if has_description else None, df["pitch_type"]) if has_description else pd.Series(np.nan, index=df.index)
    inplay = (
        is_ball_in_play(
            df["description"] if has_description else None,
            df["events"] if has_events else None,
            df["type"] if "type" in df.columns else None,
            df["bb_type"] if "bb_type" in df.columns else None,
            fallback_index=df.index,
        )
        if has_description
        else pd.Series(np.nan, index=df.index)
    )
    if has_description:
        missing_desc = df["description"].isna()
        swing = swing.astype("float64").mask(missing_desc)
        whiff = whiff.astype("float64").mask(missing_desc)
        inplay = inplay.astype("float64").mask(missing_desc)

    hr = (
        df["events"].astype("string").str.lower().eq("home_run").fillna(False)
        if has_events
        else pd.Series(np.nan, index=df.index)
    )

    prepared = df.assign(
        _swing=swing,
        _whiff=whiff,
        _bip=inplay,
        _hr=hr,
        _has_description=(~df["description"].isna()).astype("int64") if has_description else 0,
    )

    if has_launch_speed:
        prepared["launch_speed"] = pd.to_numeric(prepared["launch_speed"], errors="coerce")
    if has_xwoba:
        prepared["estimated_woba_using_speedangle"] = pd.to_numeric(
            prepared["estimated_woba_using_speedangle"], errors="coerce"
        )

    group_cols = ["batter_id", "pitcher_throws", "pitch_type"]

    agg = prepared.groupby(group_cols, dropna=False, observed=False).agg(
        pitches=("pitch_type", "size"),
        swings=("_swing", "sum"),
        whiffs=("_whiff", "sum"),
        balls_in_play=("_bip", "sum"),
        hrs=("_hr", "sum"),
        description_rows=("_has_description", "sum"),
    )

    totals = (
        prepared.groupby(["batter_id", "pitcher_throws"], dropna=False, observed=False)
        .size()
        .rename("total_pitches")
        .reset_index()
    )

    out = agg.reset_index().merge(totals, on=["batter_id", "pitcher_throws"], how="left")
    out["pitch_usage_pct"] = _safe_divide(out["pitches"], out["total_pitches"])

    if has_description:
        no_desc_group = out["description_rows"].eq(0)
        out.loc[no_desc_group, ["swings", "whiffs", "balls_in_play"]] = np.nan
        out["swing_rate"] = _safe_divide(out["swings"], out["pitches"])
        out["whiff_rate"] = _safe_divide(out["whiffs"], out["swings"])
        out["inplay_rate"] = _safe_divide(out["balls_in_play"], out["swings"])
    else:
        out["swings"] = np.nan
        out["whiffs"] = np.nan
        out["balls_in_play"] = np.nan
        out["swing_rate"] = np.nan
        out["whiff_rate"] = np.nan
        out["inplay_rate"] = np.nan

    if has_events and has_description:
        out["hr_per_bip"] = _safe_divide(out["hrs"], out["balls_in_play"])
    else:
        out["hr_per_bip"] = np.nan

    if has_launch_speed and has_description:
        contact = prepared.loc[prepared["_bip"] == True].copy()  # noqa: E712
        ev = (
            contact.groupby(group_cols, dropna=False, observed=False)["launch_speed"]
            .mean()
            .rename("avg_ev")
        )
        hard_hit = (
            contact.assign(_hard_hit=lambda x: x["launch_speed"].ge(95).astype("float64"))
            .groupby(group_cols, dropna=False, observed=False)["_hard_hit"]
            .mean()
            .rename("hard_hit_rate")
        )
        out = out.merge(ev.reset_index(), on=group_cols, how="left")
        out = out.merge(hard_hit.reset_index(), on=group_cols, how="left")
    else:
        out["avg_ev"] = np.nan
        out["hard_hit_rate"] = np.nan

    if has_xwoba and has_description:
        xwoba = (
            prepared.loc[prepared["_bip"] == True]  # noqa: E712
            .groupby(group_cols, dropna=False, observed=False)["estimated_woba_using_speedangle"]
            .mean()
            .rename("xwoba_on_contact")
            .reset_index()
        )
        out = out.merge(xwoba, on=group_cols, how="left")
    else:
        out["xwoba_on_contact"] = np.nan

    top_usage = (
        out.groupby("pitch_type", dropna=False, observed=False)["pitches"].sum().sort_values(ascending=False)
    )
    usage_pct = (top_usage / top_usage.sum()).head(10).reset_index(name="usage_pct")
    log.info("top_pitch_types_by_usage_pct=%s", usage_pct.to_dict(orient="records"))

    keep_cols = [
        "batter_id",
        "pitcher_throws",
        "pitch_type",
        "pitches",
        "pitch_usage_pct",
        "swings",
        "swing_rate",
        "whiffs",
        "whiff_rate",
        "balls_in_play",
        "inplay_rate",
        "hr_per_bip",
        "avg_ev",
        "hard_hit_rate",
        "xwoba_on_contact",
    ]

    out = out.drop(columns=["description_rows"], errors="ignore")

    out = out[keep_cols].sort_values(
        by=["batter_id", "pitcher_throws", "pitch_usage_pct", "pitch_type"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )

    log.info("groups_written=%d", len(out))
    return out.reset_index(drop=True)


def build_and_write_hitter_pitchtype(
    season: int,
    input_path: Path,
    output_path: Path,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Load pitches, compute hitter pitch-type features, and write parquet output."""
    log = logger or logging.getLogger(__name__)
    pitches = load_pitches(input_path)
    features = build_hitter_pitchtype_features(pitches, logger=log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False, engine="pyarrow")
    log.info(
        "wrote_hitter_pitchtype season=%s rows=%d path=%s",
        season,
        len(features),
        output_path,
    )
    return features
