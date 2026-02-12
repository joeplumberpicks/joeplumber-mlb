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


def _coerce_bool_flag(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    lowered = series.astype("string").str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y", "starter", "sp"})


def _resolve_pitching_mapping(df: pd.DataFrame) -> dict[str, str | None]:
    cols = df.columns
    mapping: dict[str, str | None] = {
        "game_date": _first_existing(cols, ["game_date", "date", "game_dt"]),
        "team": _first_existing(cols, ["team", "team_id", "pitching_team", "fielding_team", "club"]),
        "pitcher_id": _first_existing(cols, ["pitcher_id", "pitcher", "player_id", "mlb_id"]),
        "is_starter": _first_existing(cols, ["is_starter", "starter", "started", "gs", "start_flag"]),
        "role": _first_existing(cols, ["role", "pitcher_role"]),
        "innings_pitched": _first_existing(cols, ["innings_pitched", "ip"]),
        "outs": _first_existing(cols, ["outs", "outs_recorded", "ip_outs", "outs_pitched"]),
        "strikeouts": _first_existing(cols, ["strikeouts", "k", "so"]),
        "walks": _first_existing(cols, ["walks", "bb", "bases_on_balls"]),
        "home_runs": _first_existing(cols, ["home_runs", "hr", "home_runs_allowed", "hr_allowed"]),
        "batters_faced": _first_existing(cols, ["batters_faced", "bf"]),
    }

    required = ["game_date", "team", "pitcher_id", "strikeouts", "walks"]
    missing = [c for c in required if mapping[c] is None]
    if missing:
        raise ValueError(
            f"Pitching game logs missing required columns: {missing}. Available columns: {list(cols)}"
        )

    if mapping["is_starter"] is None and mapping["role"] is None:
        raise ValueError(
            "Could not identify starter flag/role in pitching game logs. Expected one of "
            "[is_starter, starter, started, gs, start_flag, role]."
        )

    if mapping["innings_pitched"] is None and mapping["outs"] is None:
        raise ValueError(
            "Pitching game logs missing innings measure. Expected one of "
            "[innings_pitched, ip, outs, outs_recorded, ip_outs, outs_pitched]."
        )

    return mapping


def _resolve_spine_mapping(df: pd.DataFrame) -> dict[str, str]:
    cols = df.columns
    mapping = {
        "game_date": _first_existing(cols, ["game_date", "date", "game_dt"]),
        "home_team": _first_existing(cols, ["home_team", "home_team_name", "home_name"]),
        "away_team": _first_existing(cols, ["away_team", "away_team_name", "away_name"]),
    }
    missing = [k for k, v in mapping.items() if v is None]
    if missing:
        raise ValueError(f"spine missing required team/date columns: {missing}. Available columns: {list(cols)}")
    return mapping  # type: ignore[return-value]


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    den = b.astype(float).replace(0.0, np.nan)
    return a.astype(float) / den


def _aggregate_bullpen_team_game(
    pitching: pd.DataFrame,
    mapping: dict[str, str | None],
    *,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, str]:
    p = pitching.copy()

    p["game_date"] = pd.to_datetime(p[mapping["game_date"]], errors="coerce").dt.normalize()
    p["team"] = p[mapping["team"]].astype("string")
    p["pitcher_id"] = pd.to_numeric(p[mapping["pitcher_id"]], errors="coerce").astype("Int64")

    if mapping["innings_pitched"] is not None:
        p["ip"] = pd.to_numeric(p[mapping["innings_pitched"]], errors="coerce")
    else:
        p["ip"] = np.nan

    if mapping["outs"] is not None:
        p["outs"] = pd.to_numeric(p[mapping["outs"]], errors="coerce")
        p["ip"] = p["ip"].fillna(p["outs"] / 3.0)
    else:
        p["outs"] = p["ip"] * 3.0

    p["k"] = pd.to_numeric(p[mapping["strikeouts"]], errors="coerce").fillna(0.0)
    p["bb"] = pd.to_numeric(p[mapping["walks"]], errors="coerce").fillna(0.0)

    if mapping["home_runs"] is not None:
        p["hr"] = pd.to_numeric(p[mapping["home_runs"]], errors="coerce")
    else:
        p["hr"] = np.nan

    if mapping["batters_faced"] is not None:
        p["bf"] = pd.to_numeric(p[mapping["batters_faced"]], errors="coerce")
    else:
        p["bf"] = np.nan

    if mapping["is_starter"] is not None:
        is_starter = _coerce_bool_flag(p[mapping["is_starter"]])
    else:
        role = p[mapping["role"]].astype("string").str.strip().str.lower()
        is_starter = role.str.contains("starter") | role.eq("sp")

    rel = p.loc[~is_starter].copy()

    agg = (
        rel.groupby(["game_date", "team"], dropna=False, observed=False)
        .agg(
            bp_ip=("ip", "sum"),
            bp_outs=("outs", "sum"),
            bp_k=("k", "sum"),
            bp_bb=("bb", "sum"),
            bp_hr=("hr", "sum"),
            bp_bf=("bf", "sum"),
        )
        .reset_index()
    )

    if mapping["home_runs"] is None:
        agg["bp_hr"] = np.nan
    if mapping["batters_faced"] is None:
        agg["bp_bf"] = np.nan

    if mapping["batters_faced"] is not None:
        denom = agg["bp_bf"]
        denom_label = "bp_bf"
        agg["bp_pa_proxy"] = np.nan
    else:
        denom = agg["bp_outs"]
        denom_label = "bp_outs"
        agg["bp_pa_proxy"] = agg["bp_outs"]

    agg["bp_k_rate"] = _safe_divide(agg["bp_k"], denom)
    agg["bp_bb_rate"] = _safe_divide(agg["bp_bb"], denom)
    agg["bp_hr9"] = _safe_divide(agg["bp_hr"], agg["bp_ip"]) * 9.0

    logger.info("bullpen_rate_denominator=%s", denom_label)
    return agg, denom_label


def _compute_workload(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        agg["bp_ip_last3"] = []
        agg["bp_ip_last7"] = []
        return agg

    out = agg.sort_values(["team", "game_date"], kind="mergesort").copy()

    def _window_sum(g: pd.DataFrame, window: str) -> pd.Series:
        vals = (
            g.set_index("game_date")["bp_ip"]
            .rolling(window, closed="left")
            .sum()
            .fillna(0.0)
            .to_numpy()
        )
        return pd.Series(vals, index=g.index, dtype="float64")

    out["bp_ip_last3"] = out.groupby("team", group_keys=False, observed=False).apply(
        lambda g: _window_sum(g, "3D")
    )
    out["bp_ip_last7"] = out.groupby("team", group_keys=False, observed=False).apply(
        lambda g: _window_sum(g, "7D")
    )
    return out


def build_bullpen_context(
    season: int,
    spine: pd.DataFrame,
    pitching_game: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)

    sm = _resolve_spine_mapping(spine)
    sp = spine.copy()
    sp[sm["game_date"]] = pd.to_datetime(sp[sm["game_date"]], errors="coerce").dt.normalize()
    if "season" in sp.columns:
        sp = sp.loc[pd.to_numeric(sp["season"], errors="coerce") == int(season)].copy()
    else:
        sp = sp.loc[sp[sm["game_date"]].dt.year == int(season)].copy()

    if start:
        sp = sp.loc[sp[sm["game_date"]] >= pd.to_datetime(start).normalize()].copy()
    if end:
        sp = sp.loc[sp[sm["game_date"]] <= pd.to_datetime(end).normalize()].copy()

    schedule_teams = pd.concat(
        [
            pd.DataFrame({"game_date": sp[sm["game_date"]], "team": sp[sm["home_team"]].astype("string")}),
            pd.DataFrame({"game_date": sp[sm["game_date"]], "team": sp[sm["away_team"]].astype("string")}),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["game_date", "team"], keep="first")

    pm = _resolve_pitching_mapping(pitching_game)
    agg, _ = _aggregate_bullpen_team_game(pitching_game, pm, logger=log)

    agg = agg.loc[agg["game_date"].dt.year == int(season)].copy()
    if start:
        agg = agg.loc[agg["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        agg = agg.loc[agg["game_date"] <= pd.to_datetime(end).normalize()].copy()

    agg = _compute_workload(agg)

    out = schedule_teams.merge(agg, on=["game_date", "team"], how="left")
    out = out.sort_values(["game_date", "team"], kind="mergesort").reset_index(drop=True)

    teams_processed = int(out["team"].nunique())
    team_games_written = int(len(out))
    missing_denom_pct = float(out["bp_k_rate"].isna().mean()) if len(out) else 0.0

    log.info("bullpen_teams_processed=%d", teams_processed)
    log.info("bullpen_team_games_written=%d", team_games_written)
    log.info("bullpen_missing_core_denominator_pct=%.4f", missing_denom_pct)

    return out


def _discover_pitching_game_path(override_path: Path | None = None) -> Path:
    if override_path is not None:
        return override_path

    attempted = [
        Path("data/processed/pitcher_game.parquet"),
        Path("data/processed/team_pitching_game.parquet"),
        Path("data/processed/pitching_game.parquet"),
        Path("data/processed/bullpen_game_logs.parquet"),
    ]
    for path in attempted:
        if path.exists():
            return path

    attempted_text = "\n".join(f"- {p}" for p in attempted)
    raise FileNotFoundError(f"No pitching game log parquet found. Attempted:\n{attempted_text}")


def build_and_write_bullpen_context(
    season: int,
    spine_path: Path,
    output_path: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    pitcher_game_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)
    if not spine_path.exists():
        raise FileNotFoundError(f"Missing canonical spine parquet: {spine_path}")

    resolved_pitcher_game = _discover_pitching_game_path(pitcher_game_path)
    if not resolved_pitcher_game.exists():
        raise FileNotFoundError(f"Missing pitching game parquet: {resolved_pitcher_game}")

    log.info("resolved_spine_input=%s", spine_path)
    log.info("resolved_pitcher_game_input=%s", resolved_pitcher_game)

    spine = pd.read_parquet(spine_path, engine="pyarrow")
    pitching_game = pd.read_parquet(resolved_pitcher_game, engine="pyarrow")

    out = build_bullpen_context(
        season=season,
        spine=spine,
        pitching_game=pitching_game,
        start=start,
        end=end,
        logger=log,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_bullpen_context season=%s rows=%d path=%s", season, len(out), output_path)
    return out
