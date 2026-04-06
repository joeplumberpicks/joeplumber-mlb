from __future__ import annotations

import pandas as pd

from src.features.rolling import logistic_score, safe_numeric, zscore_by_slate


def build_hr_features(
    spine_today: pd.DataFrame,
    lineups_today: pd.DataFrame,
    batter_roll_latest: pd.DataFrame,
    pitcher_roll_latest: pd.DataFrame,
) -> pd.DataFrame:
    lu = lineups_today.copy()
    lu["batting_order"] = safe_numeric(lu.get("batting_order"))

    spine_cols = [
        "game_pk", "game_date", "season", "away_team", "home_team", "venue_id",
        "temperature_f", "weather_wind_out", "weather_wind_in",
        "park_factor_hr", "park_factor_runs",
        "away_starter_pitcher_id", "home_starter_pitcher_id",
        "away_starter_pitcher_name", "home_starter_pitcher_name",
    ]
    spine_cols = [c for c in spine_cols if c in spine_today.columns]
    games = spine_today[spine_cols].copy()

    lu = lu.merge(games, on=["game_pk", "game_date", "season"], how="left")

    lu["opp_pitcher_id"] = lu.apply(
        lambda r: r["home_starter_pitcher_id"] if str(r.get("team")) == str(r.get("away_team")) else r["away_starter_pitcher_id"],
        axis=1,
    )
    lu["opp_pitcher_name"] = lu.apply(
        lambda r: r["home_starter_pitcher_name"] if str(r.get("team")) == str(r.get("away_team")) else r["away_starter_pitcher_name"],
        axis=1,
    )

    hit_keep = [c for c in batter_roll_latest.columns if c in {
        "batter_id", "batter_name", "pa_roll7", "pa_roll15", "pa_roll30",
        "ab_roll7", "ab_roll15", "ab_roll30",
        "hr_rate_roll7", "hr_rate_roll15", "hr_rate_roll30",
        "tb_pa_rate_roll7", "tb_pa_rate_roll15", "tb_pa_rate_roll30",
        "barrel_rate_roll7", "barrel_rate_roll15", "barrel_rate_roll30",
        "hard_hit_rate_roll7", "hard_hit_rate_roll15", "hard_hit_rate_roll30",
        "fb_rate_roll7", "fb_rate_roll15", "fb_rate_roll30",
        "avg_ev_roll7", "avg_ev_roll15", "avg_ev_roll30",
        "avg_la_roll7", "avg_la_roll15", "avg_la_roll30",
        "hit_rate_roll7", "hit_rate_roll15", "hit_rate_roll30",
    }]
    pit_keep = [c for c in pitcher_roll_latest.columns if c in {
        "pitcher_id", "pitcher_name",
        "hr_rate_roll7", "hr_rate_roll15", "hr_rate_roll30",
        "barrel_rate_allowed_roll7", "barrel_rate_allowed_roll15", "barrel_rate_allowed_roll30",
        "hard_hit_rate_allowed_roll7", "hard_hit_rate_allowed_roll15", "hard_hit_rate_allowed_roll30",
        "fb_rate_allowed_roll7", "fb_rate_allowed_roll15", "fb_rate_allowed_roll30",
        "avg_ev_allowed_roll7", "avg_ev_allowed_roll15", "avg_ev_allowed_roll30",
        "avg_la_allowed_roll7", "avg_la_allowed_roll15", "avg_la_allowed_roll30",
        "bb_rate_roll15", "k_rate_roll15",
    }]

    df = lu.merge(batter_roll_latest[hit_keep], left_on="player_id", right_on="batter_id", how="left")
    df = df.merge(pitcher_roll_latest[pit_keep].add_prefix("opp_"), left_on="opp_pitcher_id", right_on="opp_pitcher_id", how="left")

    df["expected_pa"] = (
        3.5
        + 0.08 * (10 - safe_numeric(df.get("batting_order"), 5.0))
        + 0.12 * safe_numeric(df.get("pa_roll15"), 4.0)
    ).clip(3.2, 5.3)

    df["hhr_form"] = (
        1.7 * safe_numeric(df.get("barrel_rate_roll15"), 0)
        + 1.2 * safe_numeric(df.get("hard_hit_rate_roll15"), 0)
        + 0.8 * safe_numeric(df.get("fb_rate_roll15"), 0)
        + 0.010 * safe_numeric(df.get("avg_ev_roll15"), 88)
        + 0.015 * safe_numeric(df.get("avg_la_roll15"), 12)
    )

    df["pei"] = (
        1.7 * safe_numeric(df.get("opp_hr_rate_roll15"), 0)
        + 1.5 * safe_numeric(df.get("opp_barrel_rate_allowed_roll15"), 0)
        + 1.1 * safe_numeric(df.get("opp_hard_hit_rate_allowed_roll15"), 0)
        + 0.7 * safe_numeric(df.get("opp_fb_rate_allowed_roll15"), 0)
    )

    df["pwi"] = (
        safe_numeric(df.get("park_factor_hr"), 1.0)
        + 0.02 * safe_numeric(df.get("temperature_f"), 72)
        + 0.10 * safe_numeric(df.get("weather_wind_out"), 0)
        - 0.07 * safe_numeric(df.get("weather_wind_in"), 0)
    )

    df["hr_raw_score"] = (
        1.8 * df["hhr_form"].fillna(0)
        + 1.5 * df["pei"].fillna(0)
        + 0.9 * df["pwi"].fillna(0)
        + 0.18 * df["expected_pa"].fillna(4.2)
    )

    df = zscore_by_slate(df, ["hr_raw_score"], group_col="game_date")
    z = safe_numeric(df.get("hr_raw_score_z"), 0.0)
    df["p_hr"] = logistic_score(z, center=0.55, scale=0.95).clip(0.03, 0.28)

    return df.sort_values(["game_date", "p_hr", "hr_raw_score"], ascending=[True, False, False], kind="stable").reset_index(drop=True)
