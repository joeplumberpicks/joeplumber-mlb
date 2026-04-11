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
            s = pd.to_numeric(df[c], errors="coerce")
            out = out.combine_first(s)
    if not found:
        out = pd.Series(default, index=df.index, dtype="float64")
    if not np.isnan(default):
        out = out.fillna(default)
    return out


def _latest_batter_rollings(lineups: pd.DataFrame, batter_roll: pd.DataFrame) -> pd.DataFrame:
    lu = lineups.copy()
    br = batter_roll.copy()

    lu["game_date"] = pd.to_datetime(lu["game_date"], errors="coerce")
    br["game_date"] = pd.to_datetime(br["game_date"], errors="coerce")

    lu_batter_id = _pick_col(lu, ["batter_id", "player_id"], required=True)
    br_batter_id = _pick_col(br, ["batter_id", "player_id"], required=True)

    br_cols = [c for c in br.columns if c.startswith("bat_")]
    br_small = br[[br_batter_id, "game_date", *br_cols]].copy()

    lu = lu.sort_values([lu_batter_id, "game_date"]).reset_index(drop=True)
    br_small = br_small.sort_values([br_batter_id, "game_date"]).reset_index(drop=True)

    merged = pd.merge_asof(
        lu,
        br_small,
        left_on="game_date",
        right_on="game_date",
        left_by=lu_batter_id,
        right_by=br_batter_id,
        direction="backward",
        allow_exact_matches=False,
    )

    if br_batter_id in merged.columns and br_batter_id != lu_batter_id:
        merged = merged.drop(columns=[br_batter_id])

    return merged


def _latest_pitcher_rollings(df: pd.DataFrame, pitcher_roll: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pr = pitcher_roll.copy()

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    pr["game_date"] = pd.to_datetime(pr["game_date"], errors="coerce")

    pr_pitcher_id = _pick_col(pr, ["pitcher_id"], required=True)
    pr_cols = [c for c in pr.columns if c.startswith("pit_")]
    pr_small = pr[[pr_pitcher_id, "game_date", *pr_cols]].copy()

    out = out.sort_values(["opp_pitcher_id", "game_date"]).reset_index(drop=True)
    pr_small = pr_small.sort_values([pr_pitcher_id, "game_date"]).reset_index(drop=True)

    merged = pd.merge_asof(
        out,
        pr_small,
        left_on="game_date",
        right_on="game_date",
        left_by="opp_pitcher_id",
        right_by=pr_pitcher_id,
        direction="backward",
        allow_exact_matches=False,
    )

    rename_map = {c: f"opp_{c}" for c in pr_cols}
    merged = merged.rename(columns=rename_map)

    if pr_pitcher_id in merged.columns:
        merged = merged.drop(columns=[pr_pitcher_id])

    return merged


def build_rbi_features(
    spine: pd.DataFrame,
    lineups: pd.DataFrame,
    batter_roll: pd.DataFrame,
    pitcher_roll: pd.DataFrame,
) -> pd.DataFrame:
    sp = spine.copy()
    lu = lineups.copy()

    sp["game_date"] = pd.to_datetime(sp["game_date"], errors="coerce")
    lu["game_date"] = pd.to_datetime(lu["game_date"], errors="coerce")

    lu["team"] = _normalize_team(lu["team"])
    if "opponent" in lu.columns:
        lu["opponent"] = _normalize_team(lu["opponent"])
    lu["is_home"] = _normalize_is_home(lu["is_home"])

    sp["home_team"] = _normalize_team(sp["home_team"])
    sp["away_team"] = _normalize_team(sp["away_team"])

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

    df = lu.merge(
        sp_small,
        on=["game_pk", "game_date"],
        how="left",
    )

    df["opp_pitcher_id"] = np.where(
        df["is_home"].eq(True),
        df["away_starter_pitcher_id"],
        df["home_starter_pitcher_id"],
    )

    df["park_team"] = df["home_team"]

    df = _latest_batter_rollings(df, batter_roll)
    df = _latest_pitcher_rollings(df, pitcher_roll)

    df["park_hr_factor"] = 1.00
    df["park_run_factor"] = 1.00
    df["park_1b_factor"] = 1.00
    df["park_2b3b_factor"] = 1.00

    for team, vals in PARK_FACTORS.items():
        mask = df["park_team"].astype("string").eq(team)
        for k, v in vals.items():
            df.loc[mask, k] = v

    df["tb_per_pa"] = _first_numeric(df, ["bat_tb_per_pa_roll30", "bat_tb_per_pa_roll15", "bat_tb_per_pa_roll7", "bat_tb_per_pa_roll3"])
    df["hr_per_pa"] = _first_numeric(df, ["bat_hr_per_pa_roll30", "bat_hr_per_pa_roll15", "bat_hr_per_pa_roll7", "bat_hr_per_pa_roll3"])
    df["bb_rate"] = _first_numeric(df, ["bat_bb_rate_roll30", "bat_bb_rate_roll15", "bat_bb_rate_roll7", "bat_bb_rate_roll3"])
    df["hard_hit_rate"] = _first_numeric(df, ["bat_hard_hit_rate_roll30", "bat_hard_hit_rate_roll15", "bat_hard_hit_rate_roll7", "bat_hard_hit_rate_roll3"])
    df["barrel_rate"] = _first_numeric(df, ["bat_barrel_rate_roll30", "bat_barrel_rate_roll15", "bat_barrel_rate_roll7", "bat_barrel_rate_roll3"])
    df["iso"] = _first_numeric(df, ["bat_iso_roll30", "bat_iso_roll15", "bat_iso_roll7", "bat_iso_roll3"])
    df["lineup_spot"] = _first_numeric(df, ["batting_order", "lineup_slot", "order_spot", "slot"], default=6.0)
    df["lineup_weight"] = _first_numeric(df, ["lineup_weight"], default=1.0)

    df["opp_pitcher_bb_rate"] = _first_numeric(df, ["opp_pit_bb_rate_roll30", "opp_pit_bb_rate_roll15", "opp_pit_bb_rate_roll7", "opp_pit_bb_rate_roll3"])
    df["opp_pitcher_hr9"] = _first_numeric(df, ["opp_pit_hr9_roll30", "opp_pit_hr9_roll15", "opp_pit_hr9_roll7", "opp_pit_hr9_roll3"])
    df["opp_pitcher_hard_hit_rate"] = _first_numeric(df, ["opp_pit_hard_hit_rate_roll30", "opp_pit_hard_hit_rate_roll15", "opp_pit_hard_hit_rate_roll7", "opp_pit_hard_hit_rate_roll3"])
    df["opp_pitcher_barrel_rate"] = _first_numeric(df, ["opp_pit_barrel_rate_roll30", "opp_pit_barrel_rate_roll15", "opp_pit_barrel_rate_roll7", "opp_pit_barrel_rate_roll3"])

    df["temperature_f"] = _first_numeric(df, ["temperature_f"], default=72.0)
    df["wind_mph"] = _first_numeric(df, ["wind_mph"], default=0.0)
    df["weather_wind_out"] = _first_numeric(df, ["weather_wind_out"], default=0.0)
    df["weather_wind_in"] = _first_numeric(df, ["weather_wind_in"], default=0.0)
    df["weather_crosswind"] = _first_numeric(df, ["weather_crosswind"], default=0.0)

    df["env_temp_boost"] = ((df["temperature_f"] - 70.0) / 15.0).clip(-2.0, 2.0)
    df["env_wind_out_effect"] = df["wind_mph"] * df["weather_wind_out"]
    df["env_wind_in_effect"] = df["wind_mph"] * df["weather_wind_in"]
    df["env_crosswind_effect"] = df["wind_mph"] * df["weather_crosswind"]

    df["env_run_boost"] = (
        df["env_temp_boost"] * 0.05
        + df["env_wind_out_effect"] * 0.008
        - df["env_wind_in_effect"] * 0.010
        - df["env_crosswind_effect"] * 0.003
        + (df["park_run_factor"] - 1.0) * 0.55
    )

    df["rbi_weather_delta"] = (
        df["env_run_boost"]
        + df["tb_per_pa"].fillna(0.0) * (df["park_run_factor"] - 1.0) * 0.40
        + df["hr_per_pa"].fillna(0.0) * (df["park_hr_factor"] - 1.0) * 0.30
        + df["bb_rate"].fillna(0.0) * df["opp_pitcher_bb_rate"].fillna(0.0) * 0.20
    )

    return df