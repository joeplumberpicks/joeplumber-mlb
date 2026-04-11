from __future__ import annotations

import numpy as np
import pandas as pd


PARK_FACTORS = {
    "ARI": {"park_hr_factor": 1.03, "park_run_factor": 1.01, "park_1b_factor": 1.00, "park_2b3b_factor": 1.03},
    "ATL": {"park_hr_factor": 1.02, "park_run_factor": 1.01, "park_1b_factor": 0.99, "park_2b3b_factor": 1.00},
    "BAL": {"park_hr_factor": 1.05, "park_run_factor": 1.01, "park_1b_factor": 0.99, "park_2b3b_factor": 1.01},
    "BOS": {"park_hr_factor": 1.04, "park_run_factor": 1.02, "park_1b_factor": 1.01, "park_2b3b_factor": 1.08},
    "CHC": {"park_hr_factor": 1.00, "park_run_factor": 1.00, "park_1b_factor": 1.00, "park_2b3b_factor": 1.00},
    "CHW": {"park_hr_factor": 1.03, "park_run_factor": 1.01, "park_1b_factor": 1.00, "park_2b3b_factor": 1.00},
    "CIN": {"park_hr_factor": 1.18, "park_run_factor": 1.07, "park_1b_factor": 0.98, "park_2b3b_factor": 0.97},
    "CLE": {"park_hr_factor": 0.96, "park_run_factor": 0.98, "park_1b_factor": 1.00, "park_2b3b_factor": 1.00},
    "COL": {"park_hr_factor": 1.30, "park_run_factor": 1.18, "park_1b_factor": 1.08, "park_2b3b_factor": 1.15},
    "DET": {"park_hr_factor": 0.90, "park_run_factor": 0.95, "park_1b_factor": 1.00, "park_2b3b_factor": 1.05},
    "HOU": {"park_hr_factor": 0.97, "park_run_factor": 0.99, "park_1b_factor": 0.99, "park_2b3b_factor": 1.00},
    "KC":  {"park_hr_factor": 0.92, "park_run_factor": 0.97, "park_1b_factor": 1.03, "park_2b3b_factor": 1.07},
    "LAA": {"park_hr_factor": 1.04, "park_run_factor": 1.01, "park_1b_factor": 0.99, "park_2b3b_factor": 1.00},
    "LAD": {"park_hr_factor": 1.02, "park_run_factor": 1.01, "park_1b_factor": 1.00, "park_2b3b_factor": 1.00},
    "MIA": {"park_hr_factor": 0.92, "park_run_factor": 0.95, "park_1b_factor": 1.00, "park_2b3b_factor": 1.00},
    "MIL": {"park_hr_factor": 1.06, "park_run_factor": 1.03, "park_1b_factor": 0.99, "park_2b3b_factor": 0.99},
    "MIN": {"park_hr_factor": 1.02, "park_run_factor": 1.01, "park_1b_factor": 0.99, "park_2b3b_factor": 1.01},
    "NYM": {"park_hr_factor": 0.93, "park_run_factor": 0.96, "park_1b_factor": 1.00, "park_2b3b_factor": 0.98},
    "NYY": {"park_hr_factor": 1.14, "park_run_factor": 1.03, "park_1b_factor": 0.98, "park_2b3b_factor": 0.96},
    "OAK": {"park_hr_factor": 0.98, "park_run_factor": 0.98, "park_1b_factor": 1.00, "park_2b3b_factor": 1.00},
    "PHI": {"park_hr_factor": 1.08, "park_run_factor": 1.04, "park_1b_factor": 0.99, "park_2b3b_factor": 1.00},
    "PIT": {"park_hr_factor": 0.96, "park_run_factor": 0.97, "park_1b_factor": 1.00, "park_2b3b_factor": 1.02},
    "SD":  {"park_hr_factor": 0.90, "park_run_factor": 0.94, "park_1b_factor": 1.01, "park_2b3b_factor": 1.01},
    "SEA": {"park_hr_factor": 0.95, "park_run_factor": 0.96, "park_1b_factor": 0.99, "park_2b3b_factor": 1.00},
    "SF":  {"park_hr_factor": 0.90, "park_run_factor": 0.94, "park_1b_factor": 1.00, "park_2b3b_factor": 1.04},
    "STL": {"park_hr_factor": 0.94, "park_run_factor": 0.97, "park_1b_factor": 1.00, "park_2b3b_factor": 1.00},
    "TB":  {"park_hr_factor": 0.92, "park_run_factor": 0.95, "park_1b_factor": 0.99, "park_2b3b_factor": 1.00},
    "TEX": {"park_hr_factor": 1.08, "park_run_factor": 1.05, "park_1b_factor": 1.00, "park_2b3b_factor": 1.02},
    "TOR": {"park_hr_factor": 1.00, "park_run_factor": 1.00, "park_1b_factor": 1.00, "park_2b3b_factor": 1.00},
    "WSH": {"park_hr_factor": 1.00, "park_run_factor": 1.00, "park_1b_factor": 1.00, "park_2b3b_factor": 1.00},
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


def _normalize_team(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.upper().str.strip()
    return s.replace({"ATH": "OAK", "AZ": "ARI", "WSN": "WSH", "WAS": "WSH", "CWS": "CHW"})


def _normalize_is_home(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
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
            }
        )
        .astype("boolean")
    )


def _first_numeric(df: pd.DataFrame, candidates: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    found = False
    for c in candidates:
        if c in df.columns:
            found = True
            s = pd.to_numeric(df[c], errors="coerce")
            out = out.combine_first(s)
    if not found:
        out = pd.Series(default, index=df.index, dtype="float64")
    return out.fillna(default) if not np.isnan(default) else out


def _attach_context(lineups: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    lu = _safe_copy(lineups)
    sp = _safe_copy(spine)

    game_pk_col = _pick_col(lu, ["game_pk"], required=True)
    lu_is_home_col = _pick_col(lu, ["is_home"], required=True)
    team_col = _pick_col(lu, ["team"], required=True)

    lu[lu_is_home_col] = _normalize_is_home(lu[lu_is_home_col])
    lu[team_col] = _normalize_team(lu[team_col])

    sp_game_pk = _pick_col(sp, ["game_pk"], required=True)
    home_sp_col = _pick_col(sp, ["home_starter_pitcher_id", "home_sp_id", "home_pitcher_id"], required=True)
    away_sp_col = _pick_col(sp, ["away_starter_pitcher_id", "away_sp_id", "away_pitcher_id"], required=True)
    home_team_col = _pick_col(sp, ["home_team"], required=True)
    away_team_col = _pick_col(sp, ["away_team"], required=True)

    keep = [
        sp_game_pk,
        home_sp_col,
        away_sp_col,
        home_team_col,
        away_team_col,
        *[c for c in ["temperature_f", "wind_mph", "weather_wind_out", "weather_wind_in", "weather_crosswind"] if c in sp.columns],
    ]
    spj = sp[keep].copy().rename(columns={sp_game_pk: game_pk_col})
    spj[home_team_col] = _normalize_team(spj[home_team_col])
    spj[away_team_col] = _normalize_team(spj[away_team_col])

    lu = lu.merge(spj, on=game_pk_col, how="left")

    lu["opp_pitcher_id"] = pd.NA
    lu.loc[lu[lu_is_home_col].eq(True), "opp_pitcher_id"] = lu.loc[lu[lu_is_home_col].eq(True), away_sp_col]
    lu.loc[lu[lu_is_home_col].eq(False), "opp_pitcher_id"] = lu.loc[lu[lu_is_home_col].eq(False), home_sp_col]

    lu["opponent"] = pd.NA
    lu.loc[lu[lu_is_home_col].eq(True), "opponent"] = lu.loc[lu[lu_is_home_col].eq(True), away_team_col]
    lu.loc[lu[lu_is_home_col].eq(False), "opponent"] = lu.loc[lu[lu_is_home_col].eq(False), home_team_col]

    lu["park_team"] = lu[home_team_col]
    return lu


def _apply_park_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["park_hr_factor"] = 1.00
    out["park_run_factor"] = 1.00
    out["park_1b_factor"] = 1.00
    out["park_2b3b_factor"] = 1.00
    for team, vals in PARK_FACTORS.items():
        mask = out["park_team"].astype("string").eq(team)
        for k, v in vals.items():
            out.loc[mask, k] = v
    return out


def build_tb2_features(
    spine: pd.DataFrame,
    lineups: pd.DataFrame,
    batter_roll: pd.DataFrame,
    pitcher_roll: pd.DataFrame,
) -> pd.DataFrame:
    sp = _safe_copy(spine)
    lu = _attach_context(lineups, sp)
    br = _safe_copy(batter_roll)
    pr = _safe_copy(pitcher_roll)

    lu_batter_id = _pick_col(lu, ["batter_id", "player_id"], required=True)
    lu_game_date = _pick_col(lu, ["game_date"], required=True)

    br_batter_id = _pick_col(br, ["batter_id", "player_id"], required=True)
    br_game_date = _pick_col(br, ["game_date"], required=False)

    left_on = [lu_batter_id]
    right_on = [br_batter_id]
    if br_game_date is not None:
        left_on.append(lu_game_date)
        right_on.append(br_game_date)

    br_keep = [c for c in br.columns if c in set(right_on) or c.startswith("bat_")]
    if br_keep:
        df = lu.merge(br[br_keep].copy(), left_on=left_on, right_on=right_on, how="left")
    else:
        df = lu.copy()

    pr_pitcher_id = _pick_col(pr, ["pitcher_id"], required=True)
    pr_game_date = _pick_col(pr, ["game_date"], required=False)

    left_on = ["opp_pitcher_id"]
    right_on = [pr_pitcher_id]
    if pr_game_date is not None:
        left_on.append(lu_game_date)
        right_on.append(pr_game_date)

    pr_keep = [c for c in pr.columns if c in set(right_on) or c.startswith("pit_")]
    pr_use = pr[pr_keep].copy()

    rename_map = {}
    for c in pr_use.columns:
        if c == pr_pitcher_id:
            rename_map[c] = "opp_pitcher_id"
        elif c != pr_game_date:
            rename_map[c] = f"opp_{c}"
    pr_use = pr_use.rename(columns=rename_map)

    df = df.merge(
        pr_use,
        left_on=left_on,
        right_on=[rename_map.get(c, c) for c in right_on],
        how="left",
    )

    df = _apply_park_factors(df)

    df["tb_per_pa"] = _first_numeric(df, ["bat_tb_per_pa_roll30", "bat_tb_per_pa_roll15", "bat_tb_per_pa_roll7", "bat_tb_per_pa_roll3"])
    df["iso"] = _first_numeric(df, ["bat_iso_roll30", "bat_iso_roll15", "bat_iso_roll7", "bat_iso_roll3"])
    df["hard_hit_rate"] = _first_numeric(df, ["bat_hard_hit_rate_roll30", "bat_hard_hit_rate_roll15", "bat_hard_hit_rate_roll7", "bat_hard_hit_rate_roll3"])
    df["barrel_rate"] = _first_numeric(df, ["bat_barrel_rate_roll30", "bat_barrel_rate_roll15", "bat_barrel_rate_roll7", "bat_barrel_rate_roll3"])
    df["hr_per_pa"] = _first_numeric(df, ["bat_hr_per_pa_roll30", "bat_hr_per_pa_roll15", "bat_hr_per_pa_roll7", "bat_hr_per_pa_roll3"])
    df["ev"] = _first_numeric(df, ["bat_avg_ev_roll30", "bat_avg_ev_roll15", "bat_avg_ev_roll7", "bat_avg_ev_roll3"])
    df["la"] = _first_numeric(df, ["bat_avg_la_roll30", "bat_avg_la_roll15", "bat_avg_la_roll7", "bat_avg_la_roll3"])
    df["lineup_spot"] = _first_numeric(df, ["batting_order", "lineup_slot", "order_spot", "slot"], default=6.0)
    df["lineup_weight"] = _first_numeric(df, ["lineup_weight"], default=1.0)

    df["opp_pitcher_hard_hit_rate"] = _first_numeric(df, ["opp_pit_hard_hit_rate_roll30", "opp_pit_hard_hit_rate_roll15", "opp_pit_hard_hit_rate_roll7", "opp_pit_hard_hit_rate_roll3"])
    df["opp_pitcher_barrel_rate"] = _first_numeric(df, ["opp_pit_barrel_rate_roll30", "opp_pit_barrel_rate_roll15", "opp_pit_barrel_rate_roll7", "opp_pit_barrel_rate_roll3"])
    df["opp_pitcher_hr9"] = _first_numeric(df, ["opp_pit_hr9_roll30", "opp_pit_hr9_roll15", "opp_pit_hr9_roll7", "opp_pit_hr9_roll3"])

    df["temperature_f"] = _first_numeric(df, ["temperature_f"], default=72.0)
    df["wind_mph"] = _first_numeric(df, ["wind_mph"], default=0.0)
    df["weather_wind_out"] = _first_numeric(df, ["weather_wind_out"], default=0.0)
    df["weather_wind_in"] = _first_numeric(df, ["weather_wind_in"], default=0.0)
    df["weather_crosswind"] = _first_numeric(df, ["weather_crosswind"], default=0.0)

    df["env_temp_boost"] = ((df["temperature_f"] - 70.0) / 15.0).clip(-2.0, 2.0)
    df["env_wind_out_effect"] = df["wind_mph"] * df["weather_wind_out"]
    df["env_wind_in_effect"] = df["wind_mph"] * df["weather_wind_in"]
    df["env_crosswind_effect"] = df["wind_mph"] * df["weather_crosswind"]

    df["env_tb2_boost"] = (
        df["env_temp_boost"] * 0.05
        + df["env_wind_out_effect"] * 0.010
        - df["env_wind_in_effect"] * 0.010
        - df["env_crosswind_effect"] * 0.004
        + (df["park_1b_factor"] - 1.0) * 0.20
        + (df["park_2b3b_factor"] - 1.0) * 0.50
    )

    df["tb2_weather_delta"] = (
        df["env_tb2_boost"]
        + df["tb_per_pa"].fillna(0.0) * (df["park_2b3b_factor"] - 1.0) * 0.45
        + df["hard_hit_rate"].fillna(0.0) * (df["park_2b3b_factor"] - 1.0) * 0.25
        + df["iso"].fillna(0.0) * (df["park_hr_factor"] - 1.0) * 0.20
    )

    return df