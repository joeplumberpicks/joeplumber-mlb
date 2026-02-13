from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.statcast_helpers import is_ball_in_play, is_swing, is_whiff


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.astype("float64").replace(0.0, np.nan)
    return numerator.astype("float64") / den


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _resolve_spine_mapping(df: pd.DataFrame) -> dict[str, str]:
    cols = df.columns
    mapping = {
        "game_date": _first_existing(cols, ["game_date", "date", "game_dt"]),
        "game_pk": _first_existing(cols, ["game_pk", "game_id", "mlb_game_id"]),
        "home_team": _first_existing(cols, ["home_team", "home_team_name", "home_name"]),
        "away_team": _first_existing(cols, ["away_team", "away_team_name", "away_name"]),
        "home_sp_id": _first_existing(cols, ["home_sp_id", "home_starter_id", "home_starting_pitcher_id"]),
        "away_sp_id": _first_existing(cols, ["away_sp_id", "away_starter_id", "away_starting_pitcher_id"]),
    }
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(
            "Unable to map required spine columns. Missing mappings for: "
            f"{missing}. Available columns: {list(cols)}"
        )
    return mapping  # type: ignore[return-value]


def _zone_indicator(df: pd.DataFrame) -> pd.Series:
    if "zone" in df.columns:
        zone_num = pd.to_numeric(df["zone"], errors="coerce")
        return zone_num.between(1, 9)
    if "plate_x" in df.columns and "plate_z" in df.columns:
        px = pd.to_numeric(df["plate_x"], errors="coerce")
        pz = pd.to_numeric(df["plate_z"], errors="coerce")
        return px.abs().le(0.83) & pz.between(1.5, 3.5)
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _aggregate_pitcher_game_metrics(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["game_pk", "pitcher_id"])

    out = df.copy()
    has_description = "description" in out.columns
    has_events = "events" in out.columns
    has_launch_speed = "launch_speed" in out.columns
    has_xwoba = "estimated_woba_using_speedangle" in out.columns

    out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")

    swing = is_swing(out["description"], out["pitch_type"]) if has_description else pd.Series(np.nan, index=out.index)
    whiff = is_whiff(out["description"], out["pitch_type"]) if has_description else pd.Series(np.nan, index=out.index)
    bip = (
        is_ball_in_play(
            out["description"] if has_description else None,
            out["events"] if has_events else None,
            out["type"] if "type" in out.columns else None,
            out["bb_type"] if "bb_type" in out.columns else None,
            fallback_index=out.index,
        )
        if has_description
        else pd.Series(np.nan, index=out.index)
    )

    if has_description:
        missing_desc = out["description"].isna()
        swing = swing.astype("float64").mask(missing_desc)
        whiff = whiff.astype("float64").mask(missing_desc)
        bip = bip.astype("float64").mask(missing_desc)

    hr = (
        out["events"].astype("string").str.lower().eq("home_run").fillna(False)
        if has_events
        else pd.Series(np.nan, index=out.index)
    )

    out["_zone"] = _zone_indicator(out).astype("float64")
    out["_swing"] = swing
    out["_whiff"] = whiff
    out["_bip"] = bip
    out["_hr"] = hr

    out["release_speed"] = pd.to_numeric(out.get("release_speed"), errors="coerce")
    if has_launch_speed:
        out["launch_speed"] = pd.to_numeric(out["launch_speed"], errors="coerce")
    if has_xwoba:
        out["estimated_woba_using_speedangle"] = pd.to_numeric(out["estimated_woba_using_speedangle"], errors="coerce")

    agg = out.groupby(["game_pk", "pitcher_id"], dropna=False, observed=False).agg(
        pitches_thrown=("pitcher_id", "size"),
        zone_pct=("_zone", "mean"),
        swings=("_swing", "sum"),
        whiffs=("_whiff", "sum"),
        balls_in_play=("_bip", "sum"),
        hrs=("_hr", "sum"),
        avg_velo=("release_speed", "mean"),
    ).reset_index()

    agg["swing_rate"] = _safe_divide(agg["swings"], agg["pitches_thrown"])
    agg["whiff_rate"] = _safe_divide(agg["whiffs"], agg["swings"])
    agg["k_like_rate"] = agg["swing_rate"] * agg["whiff_rate"]
    agg["hr_allowed_rate"] = _safe_divide(agg["hrs"], agg["balls_in_play"])

    if has_launch_speed:
        hh = (
            out.loc[out["_bip"] == True]  # noqa: E712
            .assign(_hard_hit=lambda x: x["launch_speed"].ge(95).astype("float64"))
            .groupby(["game_pk", "pitcher_id"], dropna=False, observed=False)["_hard_hit"]
            .mean()
            .rename("hard_hit_allowed_rate")
            .reset_index()
        )
        agg = agg.merge(hh, on=["game_pk", "pitcher_id"], how="left")
    else:
        agg["hard_hit_allowed_rate"] = np.nan

    if has_xwoba:
        xw = (
            out.loc[out["_bip"] == True]  # noqa: E712
            .groupby(["game_pk", "pitcher_id"], dropna=False, observed=False)["estimated_woba_using_speedangle"]
            .mean()
            .rename("xwoba_on_contact_allowed")
            .reset_index()
        )
        agg = agg.merge(xw, on=["game_pk", "pitcher_id"], how="left")
    else:
        agg["xwoba_on_contact_allowed"] = np.nan

    keep = [
        "game_pk", "pitcher_id", "pitches_thrown", "zone_pct", "swing_rate", "whiff_rate",
        "k_like_rate", "avg_velo", "hard_hit_allowed_rate", "xwoba_on_contact_allowed", "hr_allowed_rate",
    ]
    agg = agg[keep]
    if prefix:
        rename = {c: f"{prefix}{c}" for c in keep if c not in {"game_pk", "pitcher_id"}}
        agg = agg.rename(columns=rename)
    return agg


def _validate_output_coverage(
    expected_starters: int,
    output_rows: int,
    count_missing_starter_ids: int,
    count_no_matching_pitches: int,
    spine_path: Path,
    pitches_path: Path,
    allow_partial: bool,
    logger: logging.Logger,
) -> None:
    if expected_starters <= 0:
        return

    if output_rows < int(np.ceil(0.90 * expected_starters)):
        logger.warning(
            "Low starter coverage: expected_starters=%d output_rows=%d count_missing_starter_ids=%d "
            "count_no_matching_pitches=%d spine_path=%s pitches_path=%s",
            expected_starters,
            output_rows,
            count_missing_starter_ids,
            count_no_matching_pitches,
            spine_path,
            pitches_path,
        )

    if output_rows < int(np.ceil(0.50 * expected_starters)) and not allow_partial:
        raise SystemExit(2)


def build_statcast_game_context(
    season: int,
    pitches: pd.DataFrame,
    spine: pd.DataFrame,
    *,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build one-row-per-starter-per-game context features from Statcast pitch data."""
    log = logger or logging.getLogger(__name__)

    spm = _resolve_spine_mapping(spine)
    sp = spine.copy()
    sp[spm["game_date"]] = pd.to_datetime(sp[spm["game_date"]], errors="coerce")

    if "season" in sp.columns:
        sp = sp.loc[pd.to_numeric(sp["season"], errors="coerce") == season].copy()
    else:
        sp = sp.loc[sp[spm["game_date"]].dt.year == season].copy()

    base_cols = [spm[k] for k in ["game_date", "game_pk", "home_team", "away_team", "home_sp_id", "away_sp_id"]]
    sp = sp[base_cols].drop_duplicates(subset=[spm["game_pk"]], keep="first").copy()

    home_sp = pd.to_numeric(sp[spm["home_sp_id"]], errors="coerce").astype("Int64")
    away_sp = pd.to_numeric(sp[spm["away_sp_id"]], errors="coerce").astype("Int64")
    count_missing_starter_ids = int(home_sp.isna().sum() + away_sp.isna().sum())

    home = pd.DataFrame(
        {
            "game_date": sp[spm["game_date"]],
            "game_pk": pd.to_numeric(sp[spm["game_pk"]], errors="coerce").astype("Int64"),
            "side": "home",
            "pitcher_id": home_sp,
            "pitcher_team": sp[spm["home_team"]].astype("string"),
            "opponent_team": sp[spm["away_team"]].astype("string"),
        }
    )
    away = pd.DataFrame(
        {
            "game_date": sp[spm["game_date"]],
            "game_pk": pd.to_numeric(sp[spm["game_pk"]], errors="coerce").astype("Int64"),
            "side": "away",
            "pitcher_id": away_sp,
            "pitcher_team": sp[spm["away_team"]].astype("string"),
            "opponent_team": sp[spm["home_team"]].astype("string"),
        }
    )

    starters = pd.concat([home, away], ignore_index=True)
    starters = starters.dropna(subset=["game_pk", "pitcher_id"]).reset_index(drop=True)

    p = pitches.copy()
    if "pitcher" in p.columns and "pitcher_id" not in p.columns:
        p = p.rename(columns={"pitcher": "pitcher_id"})
    if "game_pk" not in p.columns or "pitcher_id" not in p.columns:
        raise ValueError("pitches parquet must contain game_pk and pitcher_id (or pitcher)")

    p["game_pk"] = pd.to_numeric(p["game_pk"], errors="coerce").astype("Int64")
    p["pitcher_id"] = pd.to_numeric(p["pitcher_id"], errors="coerce").astype("Int64")

    full = _aggregate_pitcher_game_metrics(p, prefix="")

    if "inning" in p.columns:
        p_inn1 = p.loc[pd.to_numeric(p["inning"], errors="coerce") == 1].copy()
        inn1 = _aggregate_pitcher_game_metrics(p_inn1, prefix="inn1_").rename(
            columns={"inn1_pitches_thrown": "inn1_pitches"}
        )
    else:
        inn1 = pd.DataFrame(columns=["game_pk", "pitcher_id"])

    out = starters.merge(full, on=["game_pk", "pitcher_id"], how="left")
    out = out.merge(inn1, on=["game_pk", "pitcher_id"], how="left")

    for col in [
        "inn1_pitches", "inn1_zone_pct", "inn1_swing_rate", "inn1_whiff_rate",
        "inn1_hard_hit_allowed_rate", "inn1_xwoba_on_contact_allowed", "inn1_hr_allowed_rate",
    ]:
        if col not in out.columns:
            out[col] = np.nan

    keep_cols = [
        "game_date", "game_pk", "side", "pitcher_id", "pitcher_team", "opponent_team",
        "pitches_thrown", "zone_pct", "swing_rate", "whiff_rate", "k_like_rate", "avg_velo",
        "hard_hit_allowed_rate", "xwoba_on_contact_allowed", "hr_allowed_rate", "inn1_pitches",
        "inn1_zone_pct", "inn1_swing_rate", "inn1_whiff_rate", "inn1_hard_hit_allowed_rate",
        "inn1_xwoba_on_contact_allowed", "inn1_hr_allowed_rate",
    ]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out[keep_cols].sort_values(["game_date", "game_pk", "side"], kind="mergesort").reset_index(drop=True)

    # Attach coverage stats for caller-level guardrails.
    out.attrs["games_in_spine"] = int(sp[spm["game_pk"]].nunique(dropna=True))
    out.attrs["expected_starters"] = int((out.attrs["games_in_spine"] * 2) - count_missing_starter_ids)
    out.attrs["count_missing_starter_ids"] = count_missing_starter_ids
    out.attrs["count_no_matching_pitches"] = int(out["pitches_thrown"].isna().sum())

    log.info("rows_loaded_pitches=%d", len(p))
    log.info("games_in_spine=%d", out.attrs["games_in_spine"])
    log.info("starter_rows_created=%d", len(starters))
    log.info(
        "starters_without_pitch_matches=%d (%.2f%%)",
        out.attrs["count_no_matching_pitches"],
        0.0 if len(out) == 0 else (out.attrs["count_no_matching_pitches"] / len(out)) * 100.0,
    )

    return out


def build_and_write_statcast_game_context(
    season: int,
    pitches_path: Path,
    spine_path: Path,
    output_path: Path,
    *,
    allow_partial: bool = False,
    max_games: int | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Load inputs, build Statcast game context, and write parquet output."""
    log = logger or logging.getLogger(__name__)

    if not pitches_path.exists():
        raise FileNotFoundError(f"Missing pitches parquet: {pitches_path}")
    if not spine_path.exists():
        raise FileNotFoundError(f"Missing spine parquet: {spine_path}")

    log.info("resolved_pitches_path=%s", pitches_path)
    log.info("resolved_spine_path=%s", spine_path)

    pitches = pd.read_parquet(pitches_path, engine="pyarrow")
    spine = pd.read_parquet(spine_path, engine="pyarrow")

    if max_games is not None:
        spine_mapping = _resolve_spine_mapping(spine)
        spine = spine.sort_values(spine_mapping["game_date"], kind="mergesort").head(max_games).copy()

    spm = _resolve_spine_mapping(spine)
    game_dates = pd.to_datetime(spine[spm["game_date"]], errors="coerce")
    log.info(
        "spine_games_loaded=%d date_range=%s..%s",
        int(pd.to_numeric(spine[spm["game_pk"]], errors="coerce").nunique(dropna=True)),
        game_dates.min(),
        game_dates.max(),
    )

    out = build_statcast_game_context(season=season, pitches=pitches, spine=spine, logger=log)

    _validate_output_coverage(
        expected_starters=int(out.attrs.get("expected_starters", 0)),
        output_rows=len(out),
        count_missing_starter_ids=int(out.attrs.get("count_missing_starter_ids", 0)),
        count_no_matching_pitches=int(out.attrs.get("count_no_matching_pitches", 0)),
        spine_path=spine_path,
        pitches_path=pitches_path,
        allow_partial=allow_partial,
        logger=log,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_statcast_game_context season=%s rows=%d path=%s", season, len(out), output_path)
    return out
