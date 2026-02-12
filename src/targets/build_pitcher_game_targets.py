from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(0).astype("Int64")


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _discover_pitcher_game_path(override: Path | None = None) -> Path:
    if override is not None:
        return override
    attempted = [
        Path("data/processed/pitcher_game.parquet"),
        Path("data/processed/team_pitching_game.parquet"),
        Path("data/processed/pitching_game.parquet"),
        Path("data/processed/bullpen_game_logs.parquet"),
    ]
    for p in attempted:
        if p.exists():
            return p
    attempted_text = "\n".join(f"- {p}" for p in attempted)
    raise FileNotFoundError(f"No pitcher game log parquet found. Attempted:\n{attempted_text}")


def _derive_opponent_team(df: pd.DataFrame, games: pd.DataFrame | None) -> pd.Series:
    opp_col = _first_existing(df.columns, ["opponent_team", "opp_team", "opp", "opponent"])
    if opp_col is not None:
        return df[opp_col].astype("string")

    if games is None:
        return pd.Series(pd.NA, index=df.index, dtype="string")

    gpk_col = _first_existing(games.columns, ["game_pk", "game_id", "mlb_game_id"])
    home_col = _first_existing(games.columns, ["home_team", "home_team_name", "home_name"])
    away_col = _first_existing(games.columns, ["away_team", "away_team_name", "away_name"])
    if not gpk_col or not home_col or not away_col:
        return pd.Series(pd.NA, index=df.index, dtype="string")

    team_col = _first_existing(df.columns, ["team", "team_id", "pitching_team", "fielding_team", "club"])
    gpk_pitch_col = _first_existing(df.columns, ["game_pk", "game_id", "mlb_game_id"])
    if not team_col or not gpk_pitch_col:
        return pd.Series(pd.NA, index=df.index, dtype="string")

    g = games[[gpk_col, home_col, away_col]].copy().drop_duplicates(subset=[gpk_col], keep="first")
    g.columns = ["game_pk", "home_team", "away_team"]
    g["game_pk"] = _to_int(g["game_pk"])

    p = df[[gpk_pitch_col, team_col]].copy()
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


def build_pitcher_game_targets(
    season: int,
    pitcher_game: pd.DataFrame,
    *,
    games: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)

    cols = pitcher_game.columns
    date_col = _first_existing(cols, ["game_date", "date", "game_dt"])
    gpk_col = _first_existing(cols, ["game_pk", "game_id", "mlb_game_id"])
    pid_col = _first_existing(cols, ["pitcher_id", "pitcher", "player_id", "mlb_id"])
    team_col = _first_existing(cols, ["team", "team_id", "pitching_team", "fielding_team", "club"])
    if not date_col or not gpk_col or not pid_col or not team_col:
        raise ValueError(
            "pitcher game table missing required identifiers (game_date, game_pk, pitcher_id, team). "
            f"Available columns: {list(cols)}"
        )

    so_col = _first_existing(cols, ["strikeouts", "so", "k"])
    bb_col = _first_existing(cols, ["walks", "bb", "bases_on_balls"])
    er_col = _first_existing(cols, ["earned_runs", "er"])
    outs_col = _first_existing(cols, ["outs", "outs_recorded", "ip_outs", "outs_pitched"])
    ip_col = _first_existing(cols, ["innings_pitched", "ip"])

    pg = pitcher_game.copy()
    pg["game_date"] = _to_date(pg[date_col])
    pg["game_pk"] = _to_int(pg[gpk_col])
    pg["pitcher_id"] = _to_int(pg[pid_col])
    pg["team"] = pg[team_col].astype("string")

    pg = pg.loc[pg["game_date"].dt.year == int(season)].copy()
    if start:
        pg = pg.loc[pg["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        pg = pg.loc[pg["game_date"] <= pd.to_datetime(end).normalize()].copy()

    pg["opponent_team"] = _derive_opponent_team(pg, games)
    pg["strikeouts"] = _to_float(pg[so_col]) if so_col else np.nan
    pg["walks"] = _to_float(pg[bb_col]) if bb_col else np.nan
    pg["earned_runs"] = _to_float(pg[er_col]) if er_col else np.nan

    if outs_col:
        pg["outs"] = _to_float(pg[outs_col])
    elif ip_col:
        pg["outs"] = _to_float(pg[ip_col]) * 3.0
    else:
        pg["outs"] = np.nan

    if ip_col:
        pg["innings_pitched"] = _to_float(pg[ip_col])
    else:
        pg["innings_pitched"] = _safe_ip_from_outs(pg["outs"])

    out_cols = [
        "game_date",
        "game_pk",
        "pitcher_id",
        "team",
        "opponent_team",
        "strikeouts",
        "walks",
        "outs",
        "innings_pitched",
        "earned_runs",
    ]
    out = pg[out_cols].drop_duplicates(subset=["game_pk", "pitcher_id"], keep="first")
    out = out.sort_values(["game_date", "game_pk", "pitcher_id"], kind="mergesort").reset_index(drop=True)

    for c in ["strikeouts", "walks", "outs", "innings_pitched", "earned_runs"]:
        log.info("pitcher_target_null_rate_%s=%.4f", c, float(out[c].isna().mean()) if len(out) else 0.0)
    log.info("pitcher_targets_rows_written=%d", len(out))

    missing_targets = [name for name, col in [("strikeouts", so_col), ("walks", bb_col), ("earned_runs", er_col)] if col is None]
    if not outs_col and not ip_col:
        missing_targets.append("outs/innings_pitched")
    if missing_targets:
        log.warning("pitcher_targets_unavailable_inputs=%s", missing_targets)

    return out


def _safe_ip_from_outs(outs: pd.Series) -> pd.Series:
    return pd.to_numeric(outs, errors="coerce") / 3.0


def build_and_write_pitcher_game_targets(
    season: int,
    output_path: Path,
    *,
    pitcher_game_path: Path | None = None,
    games_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)
    resolved_pg = _discover_pitcher_game_path(pitcher_game_path)
    if not resolved_pg.exists():
        raise FileNotFoundError(f"Missing pitcher game parquet: {resolved_pg}")

    pg = pd.read_parquet(resolved_pg, engine="pyarrow")
    games = pd.read_parquet(games_path, engine="pyarrow") if games_path and games_path.exists() else None

    out = build_pitcher_game_targets(
        season=season,
        pitcher_game=pg,
        games=games,
        start=start,
        end=end,
        logger=log,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_pitcher_game_targets season=%s rows=%d path=%s", season, len(out), output_path)
    return out
