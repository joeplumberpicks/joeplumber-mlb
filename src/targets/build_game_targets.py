from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def _to_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(0).astype("Int64")


def _load_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} parquet: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def _derive_final_runs(games: pd.DataFrame, game_runs: pd.DataFrame | None) -> pd.DataFrame:
    out = games.copy()

    home_run_col = _first_existing(out.columns, ["home_runs", "home_runs_final", "home_score", "home_final_runs"])
    away_run_col = _first_existing(out.columns, ["away_runs", "away_runs_final", "away_score", "away_final_runs"])

    if home_run_col and away_run_col:
        out["home_runs"] = _to_int_series(out[home_run_col])
        out["away_runs"] = _to_int_series(out[away_run_col])
        return out

    if game_runs is None:
        raise ValueError(
            "Unable to derive final runs from games.parquet and no game_runs source provided. "
            "Expected games columns like home_runs/away_runs or game_runs columns like home_runs_final/away_runs_final."
        )

    gr = game_runs.copy()
    gr_game_pk = _first_existing(gr.columns, ["game_pk", "game_id"])
    gr_home = _first_existing(gr.columns, ["home_runs_final", "home_runs", "home_score"])
    gr_away = _first_existing(gr.columns, ["away_runs_final", "away_runs", "away_score"])

    if not gr_game_pk or not gr_home or not gr_away:
        raise ValueError(
            "game_runs.parquet does not have columns needed to derive final runs. "
            f"Found columns={list(gr.columns)}; expected game_pk + home/away final runs fields."
        )

    gr = gr.rename(columns={gr_game_pk: "game_pk_tmp", gr_home: "home_runs_tmp", gr_away: "away_runs_tmp"})
    gr["game_pk_tmp"] = _to_int_series(gr["game_pk_tmp"])
    gr = gr[["game_pk_tmp", "home_runs_tmp", "away_runs_tmp"]].drop_duplicates("game_pk_tmp", keep="first")

    out["game_pk_tmp"] = _to_int_series(out["game_pk"])
    out = out.merge(gr, on="game_pk_tmp", how="left")
    out["home_runs"] = _to_int_series(out["home_runs_tmp"])
    out["away_runs"] = _to_int_series(out["away_runs_tmp"])
    out = out.drop(columns=["game_pk_tmp", "home_runs_tmp", "away_runs_tmp"])
    return out


def _derive_home_win(df: pd.DataFrame) -> pd.Series:
    winner_home_col = _first_existing(df.columns, ["home_win", "is_home_win"])
    if winner_home_col is not None:
        return _to_int_series(df[winner_home_col]).fillna(0)

    winner_col = _first_existing(df.columns, ["winner_home_away", "winner_side", "winner"])
    if winner_col is not None:
        vals = df[winner_col].astype("string").str.strip().str.lower()
        mapped = vals.map({"home": 1, "away": 0, "h": 1, "a": 0})
        if mapped.notna().any():
            return _to_int_series(mapped).fillna(0)

    return _to_int_series((df["home_runs"] > df["away_runs"]).astype("Int64")).fillna(0)


def _derive_inning1_from_game_runs(game_runs: pd.DataFrame) -> pd.DataFrame | None:
    gr = game_runs.copy()
    game_pk_col = _first_existing(gr.columns, ["game_pk", "game_id"])
    home1_col = _first_existing(gr.columns, ["home_runs_1st", "home_inn1_runs", "home_first_inning_runs"])
    away1_col = _first_existing(gr.columns, ["away_runs_1st", "away_inn1_runs", "away_first_inning_runs"])
    if not game_pk_col or not home1_col or not away1_col:
        return None

    out = gr[[game_pk_col, home1_col, away1_col]].copy()
    out = out.rename(columns={game_pk_col: "game_pk", home1_col: "home_inn1_runs", away1_col: "away_inn1_runs"})
    out["game_pk"] = _to_int_series(out["game_pk"])
    out["home_inn1_runs"] = _to_int_series(out["home_inn1_runs"])
    out["away_inn1_runs"] = _to_int_series(out["away_inn1_runs"])
    out = out.drop_duplicates("game_pk", keep="first")
    return out


def _derive_inning1_from_events(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame | None:
    ev = events.copy()
    game_pk_col = _first_existing(ev.columns, ["game_pk", "game_id"])
    inning_col = _first_existing(ev.columns, ["inning", "inning_num", "inning_number"])
    runs_col = _first_existing(ev.columns, ["runs_on_play", "runs_scored_play", "runs"])
    if not game_pk_col or not inning_col or not runs_col:
        return None

    # batting team resolution
    bat_team_col = _first_existing(ev.columns, ["batting_team_id", "bat_team_id", "offense_team_id"])
    half_col = _first_existing(ev.columns, ["inning_half", "half_inning", "top_bottom", "inning_topbot"])

    if not bat_team_col and not half_col:
        return None

    ev["game_pk"] = _to_int_series(ev[game_pk_col])
    ev["inning_tmp"] = pd.to_numeric(ev[inning_col], errors="coerce")
    ev["runs_tmp"] = pd.to_numeric(ev[runs_col], errors="coerce").fillna(0)
    ev = ev.loc[ev["inning_tmp"] == 1].copy()

    if ev.empty:
        return pd.DataFrame(columns=["game_pk", "home_inn1_runs", "away_inn1_runs"])

    g = games.copy()
    g["game_pk"] = _to_int_series(g["game_pk"])
    g["home_team_id"] = _to_int_series(g["home_team_id"]) if "home_team_id" in g.columns else pd.Series(pd.NA, index=g.index, dtype="Int64")
    g["away_team_id"] = _to_int_series(g["away_team_id"]) if "away_team_id" in g.columns else pd.Series(pd.NA, index=g.index, dtype="Int64")

    if bat_team_col:
        ev["batting_team_id"] = _to_int_series(ev[bat_team_col])
        base = ev.groupby(["game_pk", "batting_team_id"], dropna=False, observed=False)["runs_tmp"].sum().reset_index()
        home = g[["game_pk", "home_team_id"]].merge(
            base,
            left_on=["game_pk", "home_team_id"],
            right_on=["game_pk", "batting_team_id"],
            how="left",
        )
        away = g[["game_pk", "away_team_id"]].merge(
            base,
            left_on=["game_pk", "away_team_id"],
            right_on=["game_pk", "batting_team_id"],
            how="left",
        )
        out = g[["game_pk"]].drop_duplicates().copy()
        out["home_inn1_runs"] = _to_int_series(home["runs_tmp"].fillna(0))
        out["away_inn1_runs"] = _to_int_series(away["runs_tmp"].fillna(0))
        return out

    # fallback using inning-half text if batting team id absent
    half = ev[half_col].astype("string").str.strip().str.lower()
    ev["is_top"] = half.isin(["top", "t", "away"])
    ev["is_bottom"] = half.isin(["bottom", "bot", "b", "home"])

    agg = ev.groupby("game_pk", dropna=False, observed=False).agg(
        away_inn1_runs=("runs_tmp", lambda x: x[ev.loc[x.index, "is_top"]].sum()),
        home_inn1_runs=("runs_tmp", lambda x: x[ev.loc[x.index, "is_bottom"]].sum()),
    ).reset_index()
    agg["home_inn1_runs"] = _to_int_series(agg["home_inn1_runs"])
    agg["away_inn1_runs"] = _to_int_series(agg["away_inn1_runs"])
    return agg[["game_pk", "home_inn1_runs", "away_inn1_runs"]]


def build_game_targets(
    season: int,
    games: pd.DataFrame,
    *,
    game_runs: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build standardized game-level targets for multi-market modeling."""
    log = logger or logging.getLogger(__name__)

    required_games = {"game_pk", "game_date", "season", "home_team_name", "away_team_name"}
    missing = required_games - set(games.columns)
    if missing:
        raise ValueError(f"games.parquet missing required columns: {sorted(missing)}")

    g = games.copy()
    g = g.loc[pd.to_numeric(g["season"], errors="coerce") == int(season)].copy()
    g["game_date"] = _to_date(g["game_date"])
    g["game_pk"] = _to_int_series(g["game_pk"])
    if start:
        g = g.loc[g["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        g = g.loc[g["game_date"] <= pd.to_datetime(end).normalize()].copy()

    g = g.rename(columns={"home_team_name": "home_team", "away_team_name": "away_team"})
    g = _derive_final_runs(g, game_runs=game_runs)
    g["home_win"] = _derive_home_win(g)

    inn1_df: pd.DataFrame | None = None
    if game_runs is not None:
        inn1_df = _derive_inning1_from_game_runs(game_runs)

    if inn1_df is None and events is not None:
        inn1_df = _derive_inning1_from_events(events, g)

    if inn1_df is None:
        raise ValueError(
            "Unable to derive inning-1 runs. Provide game_runs.parquet with inning-1 split columns "
            "(home_runs_1st/away_runs_1st) or events_pa.parquet with inning and runs columns plus batting-team or inning-half info."
        )

    inn1_df["game_pk"] = _to_int_series(inn1_df["game_pk"])
    out = g.merge(inn1_df[["game_pk", "home_inn1_runs", "away_inn1_runs"]], on="game_pk", how="left")

    out["home_runs"] = _to_int_series(out["home_runs"])
    out["away_runs"] = _to_int_series(out["away_runs"])
    out["total_runs"] = _to_int_series(out["home_runs"] + out["away_runs"])
    out["home_inn1_runs"] = _to_int_series(out["home_inn1_runs"])
    out["away_inn1_runs"] = _to_int_series(out["away_inn1_runs"])

    inn1_total = out["home_inn1_runs"].fillna(0) + out["away_inn1_runs"].fillna(0)
    out["yrfi"] = _to_int_series((inn1_total > 0).astype("Int64"))
    out["nrfi"] = _to_int_series((inn1_total == 0).astype("Int64"))

    keep_cols = [
        "game_date",
        "game_pk",
        "home_team",
        "away_team",
        "home_runs",
        "away_runs",
        "total_runs",
        "home_win",
        "yrfi",
        "nrfi",
        "home_inn1_runs",
        "away_inn1_runs",
    ]
    out = out[keep_cols].sort_values(["game_date", "game_pk"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    null_home_runs = float(out["home_runs"].isna().mean()) if len(out) else 0.0
    null_away_runs = float(out["away_runs"].isna().mean()) if len(out) else 0.0
    null_home_win = float(out["home_win"].isna().mean()) if len(out) else 0.0
    inn1_success = float((out["home_inn1_runs"].notna() & out["away_inn1_runs"].notna()).mean() * 100.0) if len(out) else 0.0

    log.info("games_loaded=%d", len(g))
    log.info("games_written=%d", len(out))
    log.info("null_rate_home_runs=%.4f", null_home_runs)
    log.info("null_rate_away_runs=%.4f", null_away_runs)
    log.info("null_rate_home_win=%.4f", null_home_win)
    log.info("inning1_derivation_success_pct=%.2f", inn1_success)

    return out


def build_and_write_game_targets(
    season: int,
    games_path: Path,
    output_path: Path,
    *,
    game_runs_path: Path | None = None,
    events_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Load required source tables, build game targets, and write parquet output."""
    log = logger or logging.getLogger(__name__)

    games = _load_required(games_path, "games")

    game_runs = None
    if game_runs_path is not None and game_runs_path.exists():
        game_runs = pd.read_parquet(game_runs_path, engine="pyarrow")

    events = None
    if events_path is not None and events_path.exists():
        events = pd.read_parquet(events_path, engine="pyarrow")

    out = build_game_targets(
        season=season,
        games=games,
        game_runs=game_runs,
        events=events,
        start=start,
        end=end,
        logger=log,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_game_targets season=%s rows=%d path=%s", season, len(out), output_path)
    return out
