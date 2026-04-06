from __future__ import annotations

import pandas as pd

from src.features.rolling import logistic_score, safe_numeric


def _team_lineup_summary(lineups_today: pd.DataFrame, batter_roll_latest: pd.DataFrame, team_key: str = "team", top_n: int = 9) -> pd.DataFrame:
    lu = lineups_today.copy()
    lu["batting_order"] = safe_numeric(lu.get("batting_order"))
    lu = lu.sort_values([team_key, "batting_order"], kind="stable")
    lu = lu.loc[lu["batting_order"].le(top_n) | lu["batting_order"].isna()].copy()

    keep = [c for c in batter_roll_latest.columns if c in {
        "batter_id",
        "hit_rate_roll15", "hit_rate_roll30",
        "hr_rate_roll15", "hr_rate_roll30",
        "rbi_pa_rate_roll15", "rbi_pa_rate_roll30",
        "tb_pa_rate_roll15", "tb_pa_rate_roll30",
        "barrel_rate_roll15", "barrel_rate_roll30",
        "hard_hit_rate_roll15", "hard_hit_rate_roll30",
        "bb_rate_roll15", "bb_rate_roll30",
        "so_rate_roll15", "so_rate_roll30",
        "contact_rate_roll15", "contact_rate_roll30",
    }]
    br = batter_roll_latest[keep].copy()

    merged = lu.merge(br, left_on="player_id", right_on="batter_id", how="left")
    agg_map = {c: "mean" for c in merged.columns if c.endswith(("roll15", "roll30"))}
    out = merged.groupby(team_key, dropna=False).agg(agg_map).reset_index()
    return out


def build_moneyline_features(
    spine_today: pd.DataFrame,
    lineups_today: pd.DataFrame,
    batter_roll_latest: pd.DataFrame,
    pitcher_roll_latest: pd.DataFrame,
) -> pd.DataFrame:
    df = spine_today.copy()

    away_sp = pitcher_roll_latest.add_prefix("away_sp_")
    home_sp = pitcher_roll_latest.add_prefix("home_sp_")
    df = df.merge(away_sp, left_on="away_starter_pitcher_id", right_on="away_sp_pitcher_id", how="left")
    df = df.merge(home_sp, left_on="home_starter_pitcher_id", right_on="home_sp_pitcher_id", how="left")

    away_off = _team_lineup_summary(lineups_today.rename(columns={"team": "away_team"}), batter_roll_latest, team_key="away_team")
    home_off = _team_lineup_summary(lineups_today.rename(columns={"team": "home_team"}), batter_roll_latest, team_key="home_team")
    away_off = away_off.add_prefix("away_off_").rename(columns={"away_off_away_team": "away_team"})
    home_off = home_off.add_prefix("home_off_").rename(columns={"home_off_home_team": "home_team"})

    if "away_off_away_team" in away_off.columns:
        away_off = away_off.rename(columns={"away_off_away_team": "away_team"})
    if "home_off_home_team" in home_off.columns:
        home_off = home_off.rename(columns={"home_off_home_team": "home_team"})

    df = df.merge(away_off, on="away_team", how="left")
    df = df.merge(home_off, on="home_team", how="left")

    df["away_offense_strength"] = (
        1.3 * safe_numeric(df.get("away_off_tb_pa_rate_roll30"), 0)
        + 1.1 * safe_numeric(df.get("away_off_hit_rate_roll30"), 0)
        + 1.3 * safe_numeric(df.get("away_off_barrel_rate_roll30"), 0)
        + 0.7 * safe_numeric(df.get("away_off_bb_rate_roll30"), 0)
        - 0.5 * safe_numeric(df.get("away_off_so_rate_roll30"), 0)
    )

    df["home_offense_strength"] = (
        1.3 * safe_numeric(df.get("home_off_tb_pa_rate_roll30"), 0)
        + 1.1 * safe_numeric(df.get("home_off_hit_rate_roll30"), 0)
        + 1.3 * safe_numeric(df.get("home_off_barrel_rate_roll30"), 0)
        + 0.7 * safe_numeric(df.get("home_off_bb_rate_roll30"), 0)
        - 0.5 * safe_numeric(df.get("home_off_so_rate_roll30"), 0)
    )

    df["away_sp_strength"] = (
        1.4 * safe_numeric(df.get("away_sp_k_rate_roll30"), 0)
        - 1.0 * safe_numeric(df.get("away_sp_bb_rate_roll30"), 0)
        - 1.6 * safe_numeric(df.get("away_sp_hr_rate_roll30"), 0)
        - 1.3 * safe_numeric(df.get("away_sp_barrel_rate_allowed_roll30"), 0)
        - 0.8 * safe_numeric(df.get("away_sp_hard_hit_rate_allowed_roll30"), 0)
    )

    df["home_sp_strength"] = (
        1.4 * safe_numeric(df.get("home_sp_k_rate_roll30"), 0)
        - 1.0 * safe_numeric(df.get("home_sp_bb_rate_roll30"), 0)
        - 1.6 * safe_numeric(df.get("home_sp_hr_rate_roll30"), 0)
        - 1.3 * safe_numeric(df.get("home_sp_barrel_rate_allowed_roll30"), 0)
        - 0.8 * safe_numeric(df.get("home_sp_hard_hit_rate_allowed_roll30"), 0)
    )

    df["away_run_env"] = (
        safe_numeric(df.get("park_factor_runs"), 1.0)
        + 0.02 * safe_numeric(df.get("temperature_f"), 72)
        + 0.06 * safe_numeric(df.get("weather_wind_out"), 0)
        - 0.04 * safe_numeric(df.get("weather_wind_in"), 0)
    )
    df["home_run_env"] = df["away_run_env"]

    df["home_edge_raw"] = (
        1.7 * (df["home_offense_strength"] - df["away_offense_strength"])
        + 2.0 * (df["home_sp_strength"] - df["away_sp_strength"])
        + 0.18  # mild home-field
    )

    df["p_home_win"] = logistic_score(df["home_edge_raw"], center=0.0, scale=0.9).clip(0.05, 0.95)
    df["p_away_win"] = (1.0 - df["p_home_win"]).clip(0.05, 0.95)

    df["away_implied_runs"] = (
        3.85
        + 1.8 * df["away_offense_strength"].fillna(0)
        - 1.2 * df["home_sp_strength"].fillna(0)
        + 0.25 * df["away_run_env"].fillna(0)
    ).clip(2.0, 8.5)

    df["home_implied_runs"] = (
        4.00
        + 1.8 * df["home_offense_strength"].fillna(0)
        - 1.2 * df["away_sp_strength"].fillna(0)
        + 0.25 * df["home_run_env"].fillna(0)
    ).clip(2.0, 8.5)

    df["projected_total"] = (df["away_implied_runs"] + df["home_implied_runs"]).clip(5.5, 13.5)
    df["projected_margin_home"] = df["home_implied_runs"] - df["away_implied_runs"]

    return df.sort_values(["game_date", "scheduled_start_time_et", "game_pk"], kind="stable").reset_index(drop=True)
