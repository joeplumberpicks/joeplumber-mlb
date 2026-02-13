from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.statcast_helpers import is_ball_in_play, is_swing, is_whiff


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _normalize_hand(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    return s.where(s.isin(["R", "L"]), "UNK")


def _zone_indicator(df: pd.DataFrame) -> pd.Series:
    if "zone" in df.columns:
        zone_num = pd.to_numeric(df["zone"], errors="coerce")
        return zone_num.between(1, 9)
    if "plate_x" in df.columns and "plate_z" in df.columns:
        px = pd.to_numeric(df["plate_x"], errors="coerce")
        pz = pd.to_numeric(df["plate_z"], errors="coerce")
        return px.abs().le(0.83) & pz.between(1.5, 3.5)
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    den = b.astype(float).replace(0.0, np.nan)
    return a.astype(float) / den


def _resolve_batting_team(pitches: pd.DataFrame, games: pd.DataFrame | None = None) -> pd.Series:
    cols = pitches.columns
    batting_col = _first_existing(cols, ["batting_team", "bat_team", "offense_team", "team_batting"])
    if batting_col is not None:
        return pitches[batting_col].astype("string")

    if games is None:
        return pd.Series(pd.NA, index=pitches.index, dtype="string")

    gcols = games.columns
    game_pk_col = _first_existing(gcols, ["game_pk", "game_id", "mlb_game_id"])
    home_col = _first_existing(gcols, ["home_team", "home_team_name", "home_name"])
    away_col = _first_existing(gcols, ["away_team", "away_team_name", "away_name"])
    if game_pk_col is None or home_col is None or away_col is None:
        return pd.Series(pd.NA, index=pitches.index, dtype="string")

    inning_half_col = _first_existing(cols, ["inning_topbot", "inning_half", "topbot", "half_inning"])
    game_pk_pitch_col = _first_existing(cols, ["game_pk", "game_id", "mlb_game_id"])
    if inning_half_col is None or game_pk_pitch_col is None:
        return pd.Series(pd.NA, index=pitches.index, dtype="string")

    g = games[[game_pk_col, home_col, away_col]].copy().drop_duplicates(subset=[game_pk_col], keep="first")
    g.columns = ["game_pk", "home_team", "away_team"]

    p = pitches[[game_pk_pitch_col, inning_half_col]].copy()
    p.columns = ["game_pk", "inning_topbot"]
    p["game_pk"] = pd.to_numeric(p["game_pk"], errors="coerce").astype("Int64")

    g["game_pk"] = pd.to_numeric(g["game_pk"], errors="coerce").astype("Int64")
    merged = p.merge(g, on="game_pk", how="left")

    top = merged["inning_topbot"].astype("string").str.strip().str.lower().str.startswith("top")
    out = pd.Series(pd.NA, index=merged.index, dtype="string")
    out = out.where(~top, merged["away_team"].astype("string"))
    out = out.where(top, merged["home_team"].astype("string"))
    return out


def _rolling_features(daily: pd.DataFrame, days: str, suffix: str, zone_available: bool) -> pd.DataFrame:
    out = daily.sort_values(["team", "vs_pitcher_hand", "game_date"], kind="mergesort").copy()
    keys = ["team", "vs_pitcher_hand"]
    base_cols = [
        "pitches",
        "swings",
        "whiffs",
        "pitches_out_zone",
        "swings_out_zone",
        "pitches_in_zone",
        "swings_in_zone",
    ]

    rolled_parts: list[pd.DataFrame] = []
    for _, g in out.groupby(keys, observed=False, dropna=False):
        g = g.sort_values("game_date", kind="mergesort").copy()
        gi = g.set_index("game_date")
        sums = gi[base_cols].rolling(days, closed="left").sum().fillna(0.0)
        sums = sums.reset_index()
        sums["team"] = g["team"].values
        sums["vs_pitcher_hand"] = g["vs_pitcher_hand"].values
        rolled_parts.append(sums)

    rolled = pd.concat(rolled_parts, ignore_index=True) if rolled_parts else out[["game_date", "team", "vs_pitcher_hand"]].copy()

    rolled[f"swing_rate_{suffix}"] = _safe_divide(rolled["swings"], rolled["pitches"])
    rolled[f"whiff_rate_{suffix}"] = _safe_divide(rolled["whiffs"], rolled["swings"])
    rolled[f"contact_rate_{suffix}"] = 1.0 - rolled[f"whiff_rate_{suffix}"]

    if zone_available:
        rolled[f"chase_rate_{suffix}"] = _safe_divide(rolled["swings_out_zone"], rolled["pitches_out_zone"])
        rolled[f"z_swing_rate_{suffix}"] = _safe_divide(rolled["swings_in_zone"], rolled["pitches_in_zone"])
    else:
        rolled[f"chase_rate_{suffix}"] = np.nan
        rolled[f"z_swing_rate_{suffix}"] = np.nan

    return rolled[[
        "game_date",
        "team",
        "vs_pitcher_hand",
        f"swing_rate_{suffix}",
        f"whiff_rate_{suffix}",
        f"contact_rate_{suffix}",
        f"chase_rate_{suffix}",
        f"z_swing_rate_{suffix}",
    ]]


def build_offense_discipline(
    season: int,
    pitches: pd.DataFrame,
    *,
    games: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)

    p = pitches.copy()
    date_col = _first_existing(p.columns, ["game_date", "date", "game_dt"])
    if date_col is None:
        raise ValueError("pitches parquet missing game_date/date column")
    p["game_date"] = pd.to_datetime(p[date_col], errors="coerce").dt.normalize()

    if "description" not in p.columns:
        raise ValueError("pitches parquet missing description column")

    hand_col = _first_existing(p.columns, ["pitcher_throws", "pitcher_hand"])
    if hand_col is not None:
        p["vs_pitcher_hand"] = _normalize_hand(p[hand_col])
    else:
        p["vs_pitcher_hand"] = "UNK"

    p["team"] = _resolve_batting_team(p, games=games).astype("string")
    p["description"] = p["description"].astype("string").str.strip().str.lower()

    p = p.loc[p["game_date"].dt.year == int(season)].copy()
    if start:
        p = p.loc[p["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        p = p.loc[p["game_date"] <= pd.to_datetime(end).normalize()].copy()

    p["swing"] = is_swing(p["description"])
    p["whiff"] = is_whiff(p["description"])
    p["contact"] = p["swing"] & (~p["whiff"])
    p["in_play"] = is_ball_in_play(
        p["description"],
        p["events"] if "events" in p.columns else None,
        p["type"] if "type" in p.columns else None,
        p["bb_type"] if "bb_type" in p.columns else None,
        p.index,
    )

    zone = _zone_indicator(p)
    zone_available = bool(zone.notna().any())
    p["in_zone"] = zone if zone_available else np.nan

    p = p.dropna(subset=["game_date", "team"]).copy()
    p["team"] = p["team"].astype("string")

    p["pitches"] = 1.0
    p["swings"] = p["swing"].astype(float)
    p["whiffs"] = p["whiff"].astype(float)

    if zone_available:
        p["pitches_out_zone"] = (~p["in_zone"]).astype(float)
        p["swings_out_zone"] = (p["swing"] & (~p["in_zone"])).astype(float)
        p["pitches_in_zone"] = p["in_zone"].astype(float)
        p["swings_in_zone"] = (p["swing"] & p["in_zone"]).astype(float)
    else:
        p["pitches_out_zone"] = 0.0
        p["swings_out_zone"] = 0.0
        p["pitches_in_zone"] = 0.0
        p["swings_in_zone"] = 0.0

    daily = (
        p.groupby(["game_date", "team", "vs_pitcher_hand"], dropna=False, observed=False)
        .agg(
            pitches=("pitches", "sum"),
            swings=("swings", "sum"),
            whiffs=("whiffs", "sum"),
            pitches_out_zone=("pitches_out_zone", "sum"),
            swings_out_zone=("swings_out_zone", "sum"),
            pitches_in_zone=("pitches_in_zone", "sum"),
            swings_in_zone=("swings_in_zone", "sum"),
        )
        .reset_index()
    )

    r14 = _rolling_features(daily, "14D", "14d", zone_available)
    r30 = _rolling_features(daily, "30D", "30d", zone_available)

    out = r14.merge(r30, on=["game_date", "team", "vs_pitcher_hand"], how="outer")
    out = out.sort_values(["game_date", "team", "vs_pitcher_hand"], kind="mergesort").reset_index(drop=True)

    total_pitches = len(pitches)
    batting_resolved_pct = float(p["team"].notna().mean()) if len(p) else 0.0
    hand_resolved_pct = float((p["vs_pitcher_hand"] != "UNK").mean()) if len(p) else 0.0

    log.info("offense_total_pitches_loaded=%d", total_pitches)
    log.info("offense_batting_team_resolved_pct=%.4f", batting_resolved_pct)
    log.info("offense_pitcher_hand_resolved_pct=%.4f", hand_resolved_pct)
    log.info("offense_zone_metrics_available=%s", zone_available)
    log.info("offense_rows_written=%d", len(out))

    return out


def build_and_write_offense_discipline(
    season: int,
    pitches_path: Path,
    output_path: Path,
    *,
    games_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)
    if not pitches_path.exists():
        raise FileNotFoundError(f"Missing pitches parquet: {pitches_path}")

    pitches = pd.read_parquet(pitches_path, engine="pyarrow")
    games = pd.read_parquet(games_path, engine="pyarrow") if games_path and games_path.exists() else None

    out = build_offense_discipline(
        season=season,
        pitches=pitches,
        games=games,
        start=start,
        end=end,
        logger=log,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_offense_discipline season=%s rows=%d path=%s", season, len(out), output_path)
    return out
