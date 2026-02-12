from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").round(0).astype("Int64")


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _derive_opponent(df: pd.DataFrame, games: pd.DataFrame | None) -> pd.Series:
    opp_col = _first_existing(df.columns, ["opponent_team", "opp_team", "opponent"])
    if opp_col:
        return df[opp_col].astype("string")

    if games is None:
        return pd.Series(pd.NA, index=df.index, dtype="string")

    gpk_col = _first_existing(games.columns, ["game_pk", "game_id", "mlb_game_id"])
    home_col = _first_existing(games.columns, ["home_team", "home_team_name", "home_name"])
    away_col = _first_existing(games.columns, ["away_team", "away_team_name", "away_name"])
    team_col = _first_existing(df.columns, ["team", "batting_team", "bat_team", "team_id"])
    d_gpk = _first_existing(df.columns, ["game_pk", "game_id", "mlb_game_id"])
    if not all([gpk_col, home_col, away_col, team_col, d_gpk]):
        return pd.Series(pd.NA, index=df.index, dtype="string")

    g = games[[gpk_col, home_col, away_col]].drop_duplicates(subset=[gpk_col], keep="first").copy()
    g.columns = ["game_pk", "home_team", "away_team"]
    g["game_pk"] = _to_int(g["game_pk"])

    p = df[[d_gpk, team_col]].copy()
    p.columns = ["game_pk", "team"]
    p["game_pk"] = _to_int(p["game_pk"])
    p["team"] = p["team"].astype("string")

    m = p.merge(g, on="game_pk", how="left")
    opp = pd.Series(pd.NA, index=m.index, dtype="string")
    is_home = m["team"] == m["home_team"].astype("string")
    is_away = m["team"] == m["away_team"].astype("string")
    opp = opp.where(~is_home, m["away_team"].astype("string"))
    opp = opp.where(~is_away, m["home_team"].astype("string"))
    return opp


def _build_from_events(events: pd.DataFrame, games: pd.DataFrame | None) -> pd.DataFrame:
    cols = events.columns
    date_col = _first_existing(cols, ["game_date", "date", "game_dt"])
    gpk_col = _first_existing(cols, ["game_pk", "game_id", "mlb_game_id"])
    batter_col = _first_existing(cols, ["batter_id", "batter", "hitter_id", "player_id"])
    team_col = _first_existing(cols, ["batting_team", "bat_team", "team", "team_id"])
    events_col = _first_existing(cols, ["events", "event", "hit_type"])
    if not all([date_col, gpk_col, batter_col, team_col]):
        raise ValueError("events_pa missing required identifiers for hitter targets build")

    df = events.copy()
    df["game_date"] = _to_date(df[date_col])
    df["game_pk"] = _to_int(df[gpk_col])
    df["batter_id"] = _to_int(df[batter_col])
    df["team"] = df[team_col].astype("string")

    ev = df[events_col].astype("string").str.strip().str.lower() if events_col else pd.Series(pd.NA, index=df.index, dtype="string")

    # targets
    hr = ev.eq("home_run").astype(float)
    hits = ev.isin(["single", "double", "triple", "home_run"]).astype(float)
    tb = pd.Series(0.0, index=df.index)
    tb = tb.where(~ev.eq("single"), 1.0)
    tb = tb.where(~ev.eq("double"), 2.0)
    tb = tb.where(~ev.eq("triple"), 3.0)
    tb = tb.where(~ev.eq("home_run"), 4.0)

    rbi_col = _first_existing(cols, ["rbi", "runs_batted_in"])
    bb_col = _first_existing(cols, ["bb", "walks", "bases_on_balls"])
    runs_col = _first_existing(cols, ["runs", "r"])

    df["hr"] = hr
    df["hits"] = hits
    df["tb"] = tb
    df["rbi"] = _to_float(df[rbi_col]) if rbi_col else np.nan
    if bb_col:
        df["bb"] = _to_float(df[bb_col])
    else:
        df["bb"] = ev.isin(["walk", "intent_walk"]).astype(float)
    df["runs"] = _to_float(df[runs_col]) if runs_col else np.nan

    out = (
        df.groupby(["game_date", "game_pk", "batter_id", "team"], dropna=False, observed=False)
        .agg(hr=("hr", "max"), hits=("hits", "sum"), tb=("tb", "sum"), rbi=("rbi", "sum"), bb=("bb", "sum"), runs=("runs", "sum"))
        .reset_index()
    )
    out["opponent_team"] = _derive_opponent(out, games)
    return out


def _build_from_batter_game(bg: pd.DataFrame, games: pd.DataFrame | None) -> pd.DataFrame:
    cols = bg.columns
    date_col = _first_existing(cols, ["game_date", "date", "game_dt"])
    gpk_col = _first_existing(cols, ["game_pk", "game_id", "mlb_game_id"])
    batter_col = _first_existing(cols, ["batter_id", "batter", "hitter_id", "player_id"])
    team_col = _first_existing(cols, ["team", "batting_team", "team_id"])
    if not all([date_col, gpk_col, batter_col, team_col]):
        raise ValueError("batter_game missing required identifiers for fallback build")

    out = bg.copy()
    out["game_date"] = _to_date(out[date_col])
    out["game_pk"] = _to_int(out[gpk_col])
    out["batter_id"] = _to_int(out[batter_col])
    out["team"] = out[team_col].astype("string")

    map_cols = {
        "hr": ["hr", "home_runs"],
        "hits": ["hits", "h"],
        "tb": ["tb", "total_bases"],
        "rbi": ["rbi", "runs_batted_in"],
        "bb": ["bb", "walks"],
        "runs": ["runs", "r"],
    }
    for k, cand in map_cols.items():
        c = _first_existing(cols, cand)
        out[k] = _to_float(out[c]) if c else np.nan

    out["opponent_team"] = _derive_opponent(out, games)
    return out[["game_date", "game_pk", "batter_id", "team", "opponent_team", "hr", "hits", "tb", "rbi", "bb", "runs"]]


def build_hitter_game_targets(
    season: int,
    *,
    events_pa: pd.DataFrame | None,
    games: pd.DataFrame | None = None,
    batter_game: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)

    out: pd.DataFrame
    used_fallback = False
    if events_pa is not None:
        try:
            out = _build_from_events(events_pa, games)
        except Exception:
            if batter_game is None:
                raise
            used_fallback = True
            out = _build_from_batter_game(batter_game, games)
    elif batter_game is not None:
        used_fallback = True
        out = _build_from_batter_game(batter_game, games)
    else:
        raise FileNotFoundError("Need events_pa.parquet or batter_game.parquet to build hitter targets")

    out = out.loc[out["game_date"].dt.year == int(season)].copy()
    if start:
        out = out.loc[out["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        out = out.loc[out["game_date"] <= pd.to_datetime(end).normalize()].copy()

    out = out.drop_duplicates(subset=["game_pk", "batter_id"], keep="first")
    out = out.sort_values(["game_date", "game_pk", "batter_id"], kind="mergesort").reset_index(drop=True)

    for c in ["hr", "hits", "tb", "rbi", "bb", "runs"]:
        log.info("hitter_target_null_rate_%s=%.4f", c, float(out[c].isna().mean()) if len(out) else 0.0)
    if used_fallback:
        log.warning("hitter_targets_used_fallback_batter_game=True")
    log.info("hitter_targets_rows_written=%d", len(out))
    return out


def build_and_write_hitter_game_targets(
    season: int,
    output_path: Path,
    *,
    events_path: Path | None = None,
    games_path: Path | None = None,
    batter_game_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)
    events_df = pd.read_parquet(events_path, engine="pyarrow") if events_path and events_path.exists() else None
    games_df = pd.read_parquet(games_path, engine="pyarrow") if games_path and games_path.exists() else None

    if batter_game_path is None:
        p = Path("data/processed/batter_game.parquet")
        batter_game_path = p if p.exists() else None
    batter_df = pd.read_parquet(batter_game_path, engine="pyarrow") if batter_game_path and batter_game_path.exists() else None

    out = build_hitter_game_targets(
        season=season,
        events_pa=events_df,
        games=games_df,
        batter_game=batter_df,
        start=start,
        end=end,
        logger=log,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_hitter_game_targets season=%s rows=%d path=%s", season, len(out), output_path)
    return out
