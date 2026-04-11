from __future__ import annotations

import math
import numpy as np
import pandas as pd


PARK_FACTORS = {
    "ARI": {"park_hr_factor": 1.03, "park_run_factor": 1.01},
    "ATL": {"park_hr_factor": 1.02, "park_run_factor": 1.01},
    "BAL": {"park_hr_factor": 1.05, "park_run_factor": 1.01},
    "BOS": {"park_hr_factor": 1.04, "park_run_factor": 1.02},
    "CHC": {"park_hr_factor": 1.00, "park_run_factor": 1.00},
    "CHW": {"park_hr_factor": 1.03, "park_run_factor": 1.01},
    "CIN": {"park_hr_factor": 1.18, "park_run_factor": 1.07},
    "CLE": {"park_hr_factor": 0.96, "park_run_factor": 0.98},
    "COL": {"park_hr_factor": 1.30, "park_run_factor": 1.18},
    "DET": {"park_hr_factor": 0.90, "park_run_factor": 0.95},
    "HOU": {"park_hr_factor": 0.97, "park_run_factor": 0.99},
    "KC":  {"park_hr_factor": 0.92, "park_run_factor": 0.97},
    "LAA": {"park_hr_factor": 1.04, "park_run_factor": 1.01},
    "LAD": {"park_hr_factor": 1.02, "park_run_factor": 1.01},
    "MIA": {"park_hr_factor": 0.92, "park_run_factor": 0.95},
    "MIL": {"park_hr_factor": 1.06, "park_run_factor": 1.03},
    "MIN": {"park_hr_factor": 1.02, "park_run_factor": 1.01},
    "NYM": {"park_hr_factor": 0.93, "park_run_factor": 0.96},
    "NYY": {"park_hr_factor": 1.14, "park_run_factor": 1.03},
    "OAK": {"park_hr_factor": 0.98, "park_run_factor": 0.98},
    "PHI": {"park_hr_factor": 1.08, "park_run_factor": 1.04},
    "PIT": {"park_hr_factor": 0.96, "park_run_factor": 0.97},
    "SD":  {"park_hr_factor": 0.90, "park_run_factor": 0.94},
    "SEA": {"park_hr_factor": 0.95, "park_run_factor": 0.96},
    "SF":  {"park_hr_factor": 0.90, "park_run_factor": 0.94},
    "STL": {"park_hr_factor": 0.94, "park_run_factor": 0.97},
    "TB":  {"park_hr_factor": 0.92, "park_run_factor": 0.95},
    "TEX": {"park_hr_factor": 1.08, "park_run_factor": 1.05},
    "TOR": {"park_hr_factor": 1.00, "park_run_factor": 1.00},
    "WSH": {"park_hr_factor": 1.00, "park_run_factor": 1.00},
}


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def _safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out


def _to_string_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _to_num(series: pd.Series, default: float | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if default is not None:
        s = s.fillna(default)
    return s


def _sigmoid(x: pd.Series | np.ndarray) -> pd.Series:
    x = pd.Series(x, dtype="float64")
    return 1.0 / (1.0 + np.exp(-x.clip(-20, 20)))


def _normalize_team(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.upper().str.strip()
    return s.replace({
        "ATH": "OAK",
        "AZ": "ARI",
        "WSN": "WSH",
        "WAS": "WSH",
        "CWS": "CHW",
    })


def _normalize_is_home(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "y": True,
            "n": False,
            "yes": True,
            "no": False,
            "home": True,
            "away": False,
            "@": False,
            "vs": True,
        })
        .astype("boolean")
    )


def _first_numeric(df: pd.DataFrame, candidates: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    found = False
    for c in candidates:
        if c in df.columns:
            found = True
            out = out.combine_first(pd.to_numeric(df[c], errors="coerce"))
    if not found:
        out = pd.Series(default, index=df.index, dtype="float64")
    if not np.isnan(default):
        out = out.fillna(default)
    return out


def _latest_rows_by_id(
    df: pd.DataFrame,
    id_candidates: list[str],
    slate_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
    out = _safe_copy(df)
    id_col = _pick_col(out, id_candidates, required=True)
    date_col = _pick_col(out, ["game_date"], required=True)

    out[id_col] = _to_string_id(out[id_col])
    out = out[out[date_col].notna()].copy()
    out = out[out[date_col] < slate_date].copy()

    sort_cols = [id_col, date_col]
    if "game_pk" in out.columns:
        sort_cols.append("game_pk")

    out = out.sort_values(sort_cols, kind="mergesort")
    out = out.drop_duplicates(subset=[id_col], keep="last").reset_index(drop=True)
    return out, id_col


def _expected_pa_from_slot(slot: pd.Series) -> pd.Series:
    s = pd.to_numeric(slot, errors="coerce")
    pa = pd.Series(4.15, index=s.index, dtype="float64")
    pa = np.where(s == 1, 4.85, pa)
    pa = np.where(s == 2, 4.70, pa)
    pa = np.where(s == 3, 4.60, pa)
    pa = np.where(s == 4, 4.50, pa)
    pa = np.where(s == 5, 4.35, pa)
    pa = np.where(s == 6, 4.20, pa)
    pa = np.where(s == 7, 4.05, pa)
    pa = np.where(s == 8, 3.95, pa)
    pa = np.where(s == 9, 3.85, pa)
    return pd.Series(pa, index=s.index, dtype="float64")


def _build_player_hr_expectation(
    lineups: pd.DataFrame,
    spine: pd.DataFrame,
    batter_roll: pd.DataFrame,
    pitcher_roll: pd.DataFrame,
    slate_date: pd.Timestamp,
) -> pd.DataFrame:
    lu = _safe_copy(lineups)
    sp = _safe_copy(spine)
    br = _safe_copy(batter_roll)
    pr = _safe_copy(pitcher_roll)

    lu["team"] = _normalize_team(lu["team"])
    if "opponent" in lu.columns:
        lu["opponent"] = _normalize_team(lu["opponent"])
    lu["is_home"] = _normalize_is_home(lu["is_home"])

    sp["home_team"] = _normalize_team(sp["home_team"])
    sp["away_team"] = _normalize_team(sp["away_team"])

    batter_id_col = _pick_col(lu, ["batter_id", "player_id"], required=True)
    slot_col = _pick_col(lu, ["batting_order", "lineup_slot", "order_spot", "slot"], required=False)

    lu[batter_id_col] = _to_string_id(lu[batter_id_col])

    sp_small = sp[[
        "game_pk",
        "game_date",
        "home_team",
        "away_team",
        "home_starter_pitcher_id",
        "away_starter_pitcher_id",
        "temperature_f",
        "wind_mph",
        "weather_wind_out",
        "weather_wind_in",
        "weather_crosswind",
    ]].copy()

    sp_small["home_starter_pitcher_id"] = _to_string_id(sp_small["home_starter_pitcher_id"])
    sp_small["away_starter_pitcher_id"] = _to_string_id(sp_small["away_starter_pitcher_id"])

    df = lu.merge(sp_small, on=["game_pk", "game_date"], how="left")

    df["opp_pitcher_id"] = np.where(
        df["is_home"].eq(True),
        df["away_starter_pitcher_id"],
        df["home_starter_pitcher_id"],
    )
    df["opp_pitcher_id"] = _to_string_id(df["opp_pitcher_id"])

    # latest batter row
    br_latest, br_id = _latest_rows_by_id(br, ["batter_id", "player_id"], slate_date)
    bat_cols = [c for c in br_latest.columns if c.startswith("bat_")]
    br_small = br_latest[[br_id, *bat_cols]].copy()

    df = df.merge(
        br_small,
        left_on=batter_id_col,
        right_on=br_id,
        how="left",
    )
    if br_id in df.columns and br_id != batter_id_col:
        df = df.drop(columns=[br_id])

    # latest pitcher row
    pr_latest, pr_id = _latest_rows_by_id(pr, ["pitcher_id"], slate_date)
    pit_cols = [c for c in pr_latest.columns if c.startswith("pit_")]
    pr_small = pr_latest[[pr_id, *pit_cols]].copy()

    df = df.merge(
        pr_small,
        left_on="opp_pitcher_id",
        right_on=pr_id,
        how="left",
    )
    if pr_id in df.columns:
        df = df.drop(columns=[pr_id])

    # park factors by venue home team
    df["park_team"] = df["home_team"]
    df["park_hr_factor"] = 1.00
    df["park_run_factor"] = 1.00

    for team, vals in PARK_FACTORS.items():
        mask = df["park_team"].astype("string").eq(team)
        df.loc[mask, "park_hr_factor"] = vals["park_hr_factor"]
        df.loc[mask, "park_run_factor"] = vals["park_run_factor"]

    # hitter profile
    df["barrel_rate"] = _first_numeric(df, ["bat_barrel_rate_roll30", "bat_barrel_rate_roll15", "bat_barrel_rate_roll7", "bat_barrel_rate_roll3"], default=np.nan)
    df["iso"] = _first_numeric(df, ["bat_iso_roll30", "bat_iso_roll15", "bat_iso_roll7", "bat_iso_roll3"], default=np.nan)
    df["hr_per_pa"] = _first_numeric(df, ["bat_hr_per_pa_roll30", "bat_hr_per_pa_roll15", "bat_hr_per_pa_roll7", "bat_hr_per_pa_roll3"], default=np.nan)
    df["hard_hit_rate"] = _first_numeric(df, ["bat_hard_hit_rate_roll30", "bat_hard_hit_rate_roll15", "bat_hard_hit_rate_roll7", "bat_hard_hit_rate_roll3"], default=np.nan)
    df["flyball"] = _first_numeric(df, ["bat_fb_rate_roll30", "bat_fb_rate_roll15", "bat_fb_rate_roll7", "bat_fb_rate_roll3"], default=np.nan)
    df["pulled_air"] = _first_numeric(df, ["bat_pulled_air_rate_roll30", "bat_pulled_air_rate_roll15", "bat_pulled_air_rate_roll7", "bat_pulled_air_rate_roll3"], default=np.nan)
    df["ev"] = _first_numeric(df, ["bat_avg_ev_roll30", "bat_avg_ev_roll15", "bat_avg_ev_roll7", "bat_avg_ev_roll3"], default=np.nan)
    df["la"] = _first_numeric(df, ["bat_avg_la_roll30", "bat_avg_la_roll15", "bat_avg_la_roll7", "bat_avg_la_roll3"], default=np.nan)

    # pitcher overlay
    df["pit_hr9"] = _first_numeric(df, ["pit_hr9_roll30", "pit_hr9_roll15", "pit_hr9_roll7", "pit_hr9_roll3"], default=np.nan)
    df["pit_barrel_rate"] = _first_numeric(df, ["pit_barrel_rate_roll30", "pit_barrel_rate_roll15", "pit_barrel_rate_roll7", "pit_barrel_rate_roll3"], default=np.nan)
    df["pit_hard_hit_rate"] = _first_numeric(df, ["pit_hard_hit_rate_roll30", "pit_hard_hit_rate_roll15", "pit_hard_hit_rate_roll7", "pit_hard_hit_rate_roll3"], default=np.nan)
    df["pit_bb_rate"] = _first_numeric(df, ["pit_bb_rate_roll30", "pit_bb_rate_roll15", "pit_bb_rate_roll7", "pit_bb_rate_roll3"], default=np.nan)
    df["pit_k_rate"] = _first_numeric(df, ["pit_k_rate_roll30", "pit_k_rate_roll15", "pit_k_rate_roll7", "pit_k_rate_roll3"], default=np.nan)

    # environment
    df["temperature_f"] = _first_numeric(df, ["temperature_f"], default=72.0)
    df["wind_mph"] = _first_numeric(df, ["wind_mph"], default=0.0)
    df["weather_wind_out"] = _first_numeric(df, ["weather_wind_out"], default=0.0)
    df["weather_wind_in"] = _first_numeric(df, ["weather_wind_in"], default=0.0)
    df["weather_crosswind"] = _first_numeric(df, ["weather_crosswind"], default=0.0)

    # fill sparse early-season values with soft medians/defaults
    fill_defaults = {
        "barrel_rate": 0.055,
        "iso": 0.155,
        "hr_per_pa": 0.028,
        "hard_hit_rate": 0.38,
        "flyball": 0.35,
        "pulled_air": 0.11,
        "ev": 89.0,
        "la": 13.0,
        "pit_hr9": 1.10,
        "pit_barrel_rate": 0.08,
        "pit_hard_hit_rate": 0.40,
        "pit_bb_rate": 0.08,
        "pit_k_rate": 0.22,
    }
    for c, default in fill_defaults.items():
        s = pd.to_numeric(df[c], errors="coerce")
        med = s.dropna().median()
        df[c] = s.fillna(float(med) if pd.notna(med) else default)

    # expected PA weight
    if slot_col is not None:
        df["lineup_spot"] = pd.to_numeric(df[slot_col], errors="coerce")
    else:
        df["lineup_spot"] = 6.0
    df["expected_pa"] = _expected_pa_from_slot(df["lineup_spot"])

    # -----------------------------
    # CONTACT-ONLY / NEUTRAL HR RISK
    # -----------------------------
    neutral_score = (
        df["barrel_rate"] * 6.2
        + df["iso"] * 2.4
        + df["hr_per_pa"] * 9.0
        + df["hard_hit_rate"] * 1.4
        + df["flyball"] * 1.1
        + df["pulled_air"] * 0.9
        + ((df["ev"] - 88.0).clip(-5, 10) * 0.06)
        + ((df["la"] - 12.0).clip(-10, 18) * 0.03)
    )

    matchup_score = (
        df["pit_hr9"] * 0.55
        + df["pit_barrel_rate"] * 3.5
        + df["pit_hard_hit_rate"] * 1.4
        + df["pit_bb_rate"] * 0.35
        - df["pit_k_rate"] * 0.90
    )

    slot_adj = (
        (df["lineup_spot"] <= 2).astype(float) * 0.08
        + (df["lineup_spot"] == 3).astype(float) * 0.14
        + (df["lineup_spot"] == 4).astype(float) * 0.16
        + (df["lineup_spot"] == 5).astype(float) * 0.08
        - (df["lineup_spot"] >= 7).astype(float) * 0.10
        - (df["lineup_spot"] >= 8).astype(float) * 0.08
    )

    df["neutral_hr_score"] = neutral_score + matchup_score + slot_adj
    df["p_hr_neutral_pa"] = _sigmoid(-5.10 + df["neutral_hr_score"] * 0.33)

    # -----------------------------
    # CPW / ENVIRONMENTAL DELTA
    # -----------------------------
    temp_boost = ((df["temperature_f"] - 70.0) / 15.0).clip(-2.0, 2.0)
    wind_out_effect = df["wind_mph"] * df["weather_wind_out"]
    wind_in_effect = df["wind_mph"] * df["weather_wind_in"]
    crosswind_effect = df["wind_mph"] * df["weather_crosswind"]

    # Ballpark-Pal-style idea:
    # CPW - C_ONLY = environment contribution
    env_delta = (
        temp_boost * (0.020 + df["barrel_rate"] * 0.18)
        + wind_out_effect * (0.006 + df["flyball"] * 0.030 + df["pulled_air"] * 0.018)
        - wind_in_effect * (0.008 + df["flyball"] * 0.032)
        - crosswind_effect * (0.002 + df["flyball"] * 0.010)
        + (df["park_hr_factor"] - 1.0) * (
            0.35
            + df["flyball"] * 0.60
            + df["pulled_air"] * 0.55
            + df["barrel_rate"] * 0.75
        )
        + (df["park_run_factor"] - 1.0) * 0.10
    )

    # modest contact-quality park effect
    contact_quality_delta = (
        df["hard_hit_rate"] * (df["park_hr_factor"] - 1.0) * 0.18
        + ((df["ev"] - 90.0).clip(-8, 10) * 0.01)
        + ((df["la"] - 14.0).clip(-10, 18) * 0.006)
    )

    df["env_hr_delta_player"] = env_delta + contact_quality_delta
    df["p_hr_today_pa"] = (df["p_hr_neutral_pa"] * (1.0 + df["env_hr_delta_player"])).clip(0.0005, 0.35)

    # convert per-PA to per-game expected HR contribution
    df["expected_hr_neutral_player"] = df["p_hr_neutral_pa"] * df["expected_pa"]
    df["expected_hr_today_player"] = df["p_hr_today_pa"] * df["expected_pa"]

    return df


def build_no_hr_game_features(
    spine: pd.DataFrame,
    lineups: pd.DataFrame,
    batter_roll: pd.DataFrame,
    pitcher_roll: pd.DataFrame,
) -> pd.DataFrame:
    sp = _safe_copy(spine)

    game_pk_col = _pick_col(sp, ["game_pk"], required=True)
    game_date_col = _pick_col(sp, ["game_date"], required=True)
    home_team_col = _pick_col(sp, ["home_team"], required=True)
    away_team_col = _pick_col(sp, ["away_team"], required=True)

    slate_date = pd.to_datetime(sp[game_date_col], errors="coerce").dropna().max()
    if pd.isna(slate_date):
        raise ValueError("Unable to determine slate_date from spine.game_date")

    players = _build_player_hr_expectation(
        lineups=lineups,
        spine=spine,
        batter_roll=batter_roll,
        pitcher_roll=pitcher_roll,
        slate_date=slate_date,
    )

    team_agg = (
        players.groupby([game_pk_col, game_date_col, "team"], dropna=False)
        .agg(
            lineup_count=("player_name", "count") if "player_name" in players.columns else ("team", "count"),
            expected_pa=("expected_pa", "sum"),
            expected_hr_neutral=("expected_hr_neutral_player", "sum"),
            expected_hr_today=("expected_hr_today_player", "sum"),
            env_hr_delta=("env_hr_delta_player", "sum"),
            avg_barrel_rate=("barrel_rate", "mean"),
            avg_iso=("iso", "mean"),
            avg_hr_per_pa=("hr_per_pa", "mean"),
            avg_hard_hit_rate=("hard_hit_rate", "mean"),
            avg_flyball=("flyball", "mean"),
            avg_pulled_air=("pulled_air", "mean"),
        )
        .reset_index()
    )

    home_team_agg = team_agg.copy().rename(
        columns={
            "team": home_team_col,
            "lineup_count": "home_lineup_count",
            "expected_pa": "home_expected_pa",
            "expected_hr_neutral": "home_expected_hr_neutral",
            "expected_hr_today": "home_expected_hr_today",
            "env_hr_delta": "home_env_hr_delta",
            "avg_barrel_rate": "home_avg_barrel_rate",
            "avg_iso": "home_avg_iso",
            "avg_hr_per_pa": "home_avg_hr_per_pa",
            "avg_hard_hit_rate": "home_avg_hard_hit_rate",
            "avg_flyball": "home_avg_flyball",
            "avg_pulled_air": "home_avg_pulled_air",
        }
    )

    away_team_agg = team_agg.copy().rename(
        columns={
            "team": away_team_col,
            "lineup_count": "away_lineup_count",
            "expected_pa": "away_expected_pa",
            "expected_hr_neutral": "away_expected_hr_neutral",
            "expected_hr_today": "away_expected_hr_today",
            "env_hr_delta": "away_env_hr_delta",
            "avg_barrel_rate": "away_avg_barrel_rate",
            "avg_iso": "away_avg_iso",
            "avg_hr_per_pa": "away_avg_hr_per_pa",
            "avg_hard_hit_rate": "away_avg_hard_hit_rate",
            "avg_flyball": "away_avg_flyball",
            "avg_pulled_air": "away_avg_pulled_air",
        }
    )

    out = sp.merge(
        home_team_agg,
        on=[game_pk_col, game_date_col, home_team_col],
        how="left",
    ).merge(
        away_team_agg,
        on=[game_pk_col, game_date_col, away_team_col],
        how="left",
    )

    # game totals
    out["expected_hr_neutral"] = (
        _to_num(out["home_expected_hr_neutral"], 0.0)
        + _to_num(out["away_expected_hr_neutral"], 0.0)
    )
    out["expected_hr_today"] = (
        _to_num(out["home_expected_hr_today"], 0.0)
        + _to_num(out["away_expected_hr_today"], 0.0)
    )
    out["env_hr_delta"] = out["expected_hr_today"] - out["expected_hr_neutral"]

    # lineup completeness penalties
    out["home_lineup_count"] = _to_num(out["home_lineup_count"], 0.0)
    out["away_lineup_count"] = _to_num(out["away_lineup_count"], 0.0)

    out["lineup_penalty"] = (
        (out["home_lineup_count"] < 7).astype(float) * 0.08
        + (out["away_lineup_count"] < 7).astype(float) * 0.08
    )

    # modest uncertainty add if starter ids missing
    starter_penalty = pd.Series(0.0, index=out.index, dtype="float64")
    for c in ["home_starter_pitcher_id", "away_starter_pitcher_id", "home_sp_id", "away_sp_id"]:
        if c in out.columns:
            starter_penalty = starter_penalty + out[c].isna().astype(float) * 0.05

    out["expected_hr_today_adj"] = (out["expected_hr_today"] + out["lineup_penalty"] + starter_penalty).clip(lower=0.02)

    # Poisson no-HR probability
    out["p_no_hr_game_est"] = np.exp(-out["expected_hr_today_adj"].clip(0.02, 6.0))
    out["p_yes_hr_game_est"] = 1.0 - out["p_no_hr_game_est"]

    # keep a raw score for downstream runners if needed
    out["no_hr_game_score_raw"] = -out["expected_hr_today_adj"]

    out = out.drop_duplicates(subset=[game_pk_col]).reset_index(drop=True)
    return out