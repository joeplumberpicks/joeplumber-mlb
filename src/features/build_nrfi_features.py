from __future__ import annotations

import pandas as pd

from src.features.rolling import latest_row_per_key, logistic_score, safe_numeric


def _prep_latest_pitcher_roll(pitcher_roll: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    df = pitcher_roll.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    cutoff = pd.to_datetime(as_of_date)
    df = df.loc[df["game_date"] < cutoff].copy()
    return latest_row_per_key(df, "pitcher_id", ["game_date", "game_pk"])


def _prep_latest_batter_roll(batter_roll: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    df = batter_roll.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    cutoff = pd.to_datetime(as_of_date)
    df = df.loc[df["game_date"] < cutoff].copy()
    return latest_row_per_key(df, "batter_id", ["game_date", "game_pk"])


def _lineup_topn_features(lineups_today: pd.DataFrame, batter_roll_latest: pd.DataFrame, team_col: str = "team", n: int = 3) -> pd.DataFrame:
    if lineups_today.empty:
        return pd.DataFrame(columns=[team_col])

    lu = lineups_today.copy()
    lu["batting_order"] = safe_numeric(lu.get("batting_order"))
    lu = lu.sort_values([team_col, "batting_order"], kind="stable")
    lu = lu.loc[lu["batting_order"].le(n) | lu["batting_order"].isna()].copy()

    br = batter_roll_latest.copy()
    keep = [c for c in br.columns if c in {
        "batter_id",
        "hit_rate_roll7", "hit_rate_roll15", "hit_rate_roll30",
        "hr_rate_roll7", "hr_rate_roll15", "hr_rate_roll30",
        "rbi_pa_rate_roll7", "rbi_pa_rate_roll15", "rbi_pa_rate_roll30",
        "tb_pa_rate_roll7", "tb_pa_rate_roll15", "tb_pa_rate_roll30",
        "barrel_rate_roll7", "barrel_rate_roll15", "barrel_rate_roll30",
        "hard_hit_rate_roll7", "hard_hit_rate_roll15", "hard_hit_rate_roll30",
        "contact_rate_roll7", "contact_rate_roll15", "contact_rate_roll30",
        "whiff_rate_roll7", "whiff_rate_roll15", "whiff_rate_roll30",
        "bb_rate_roll7", "bb_rate_roll15", "bb_rate_roll30",
        "so_rate_roll7", "so_rate_roll15", "so_rate_roll30",
    }]
    br = br[keep]

    merged = lu.merge(br, left_on="player_id", right_on="batter_id", how="left")
    agg_map = {c: "mean" for c in merged.columns if c.endswith(("roll7", "roll15", "roll30"))}
    out = merged.groupby(team_col, dropna=False).agg(agg_map).reset_index()
    return out


def build_nrfi_features(
    spine_today: pd.DataFrame,
    lineups_today: pd.DataFrame,
    batter_roll_latest: pd.DataFrame,
    pitcher_roll_latest: pd.DataFrame,
) -> pd.DataFrame:
    df = spine_today.copy()

    away_p = pitcher_roll_latest.add_prefix("away_sp_")
    home_p = pitcher_roll_latest.add_prefix("home_sp_")

    df = df.merge(away_p, left_on="away_starter_pitcher_id", right_on="away_sp_pitcher_id", how="left")
    df = df.merge(home_p, left_on="home_starter_pitcher_id", right_on="home_sp_pitcher_id", how="left")

    away_lu = lineups_today.loc[lineups_today["team"].eq(lineups_today["team"])].copy()
    away_top3 = _lineup_topn_features(lineups_today.rename(columns={"team": "away_team"}), batter_roll_latest, team_col="away_team", n=3)
    home_top3 = _lineup_topn_features(lineups_today.rename(columns={"team": "home_team"}), batter_roll_latest, team_col="home_team", n=3)

    away_top3 = away_top3.add_prefix("away_top3_").rename(columns={"away_top3_away_team": "away_team"})
    home_top3 = home_top3.add_prefix("home_top3_").rename(columns={"home_top3_home_team": "home_team":})

    # manual rename fix for parser-safe paste
    if "home_top3_home_team" in home_top3.columns:
        home_top3 = home_top3.rename(columns={"home_top3_home_team": "home_team"})
    if "away_top3_away_team" in away_top3.columns:
        away_top3 = away_top3.rename(columns={"away_top3_away_team": "away_team"})

    df = df.merge(away_top3, on="away_team", how="left")
    df = df.merge(home_top3, on="home_team", how="left")

    df["top3_ob_quality"] = (
        safe_numeric(df.get("away_top3_bb_rate_roll15"), 0.0)
        + safe_numeric(df.get("away_top3_hit_rate_roll15"), 0.0)
        + safe_numeric(df.get("home_top3_bb_rate_roll15"), 0.0)
        + safe_numeric(df.get("home_top3_hit_rate_roll15"), 0.0)
    ) / 2.0

    df["sp_command_quality"] = (
        safe_numeric(df.get("away_sp_k_rate_roll15"), 0.0)
        - safe_numeric(df.get("away_sp_bb_rate_roll15"), 0.0)
        + safe_numeric(df.get("home_sp_k_rate_roll15"), 0.0)
        - safe_numeric(df.get("home_sp_bb_rate_roll15"), 0.0)
    ) / 2.0

    df["sp_contact_suppression"] = (
        -safe_numeric(df.get("away_sp_hard_hit_rate_allowed_roll15"), 0.0)
        -safe_numeric(df.get("away_sp_barrel_rate_allowed_roll15"), 0.0)
        -safe_numeric(df.get("home_sp_hard_hit_rate_allowed_roll15"), 0.0)
        -safe_numeric(df.get("home_sp_barrel_rate_allowed_roll15"), 0.0)
    ) / 2.0

    df["park_weather_run_boost"] = (
        safe_numeric(df.get("park_factor_runs"), 1.0)
        + 0.015 * safe_numeric(df.get("temperature_f"), 72.0)
        + 0.04 * safe_numeric(df.get("weather_wind_out"), 0.0)
        - 0.03 * safe_numeric(df.get("weather_wind_in"), 0.0)
    )

    raw = (
        4.00 * df["top3_ob_quality"].fillna(0)
        - 3.00 * df["sp_command_quality"].fillna(0)
        - 2.00 * df["sp_contact_suppression"].fillna(0)
        + 0.80 * df["park_weather_run_boost"].fillna(0)
    )

    df["nrfi_score_raw"] = -raw
    df["p_nrfi"] = logistic_score(df["nrfi_score_raw"], center=-0.25, scale=0.65).clip(0.05, 0.95)
    df["p_yrfi"] = (1.0 - df["p_nrfi"]).clip(0.05, 0.95)

    return df.sort_values(["game_date", "scheduled_start_time_et", "game_pk"], kind="stable").reset_index(drop=True)
