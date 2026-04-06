from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.rolling import safe_numeric


def _coalesce(df: pd.DataFrame, candidates: list[str], default: Any = np.nan) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _prep_raw_statcast(pa_df: pd.DataFrame) -> pd.DataFrame:
    df = pa_df.copy()

    df["game_date"] = pd.to_datetime(_coalesce(df, ["game_date"]), errors="coerce")
    df["season"] = safe_numeric(_coalesce(df, ["season"])).astype("Int64")
    df["game_pk"] = safe_numeric(_coalesce(df, ["game_pk"])).astype("Int64")
    df["batter_id"] = safe_numeric(_coalesce(df, ["batter_id", "batter"])).astype("Int64")
    df["pitcher_id"] = safe_numeric(_coalesce(df, ["pitcher_id", "pitcher"])).astype("Int64")

    df["batter_name"] = _coalesce(df, ["batter_name", "player_name", "batter"]).astype("string")
    df["pitcher_name"] = _coalesce(df, ["pitcher_name", "pitcher"]).astype("string")
    df["batting_team"] = _coalesce(df, ["batting_team"]).astype("string")
    df["fielding_team"] = _coalesce(df, ["fielding_team"]).astype("string")

    for col in [
        "is_pa", "is_ab", "is_hit", "is_1b", "is_2b", "is_3b", "is_hr",
        "is_bb", "is_hbp", "is_so", "is_rbi", "is_sac_fly", "is_reached_on_error"
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype("boolean")
        else:
            df[col] = False

    df["rbi"] = safe_numeric(_coalesce(df, ["rbi"]), fill_value=0.0)
    df["runs_scored_on_pa"] = safe_numeric(_coalesce(df, ["runs_scored_on_pa"]), fill_value=0.0)
    df["outs_before_pa"] = safe_numeric(_coalesce(df, ["outs_before_pa"]))
    df["outs_after_pa"] = safe_numeric(_coalesce(df, ["outs_after_pa"]))

    # Rich Statcast / pitch columns if present
    rich_numeric = {
        "launch_speed": ["launch_speed", "hit_speed", "exit_velocity"],
        "launch_angle": ["launch_angle"],
        "hit_distance_sc": ["hit_distance_sc", "hit_distance"],
        "estimated_ba_using_speedangle": ["estimated_ba_using_speedangle", "xba"],
        "estimated_woba_using_speedangle": ["estimated_woba_using_speedangle", "xwoba"],
        "woba_value": ["woba_value"],
        "delta_home_win_exp": ["delta_home_win_exp"],
        "delta_run_exp": ["delta_run_exp"],
        "zone": ["zone"],
        "plate_x": ["plate_x"],
        "plate_z": ["plate_z"],
        "sz_top": ["sz_top"],
        "sz_bot": ["sz_bot"],
        "release_speed": ["release_speed", "start_speed"],
        "release_spin_rate": ["release_spin_rate", "spin_rate_deprecated", "spin_rate"],
    }
    for out_col, cands in rich_numeric.items():
        df[out_col] = safe_numeric(_coalesce(df, cands))

    rich_text = {
        "bb_type": ["bb_type"],
        "description": ["description", "des"],
        "events": ["events", "event_type"],
        "pitch_type": ["pitch_type"],
        "stand": ["stand"],
        "p_throws": ["p_throws"],
        "inning_topbot": ["inning_topbot"],
    }
    for out_col, cands in rich_text.items():
        df[out_col] = _coalesce(df, cands).astype("string").str.lower()

    # Derived indicators
    df["barrel"] = (
        df["launch_speed"].ge(98)
        & df["launch_angle"].between(26, 30, inclusive="both")
    ).fillna(False).astype(int)

    df["hard_hit"] = df["launch_speed"].ge(95).fillna(False).astype(int)
    df["sweet_spot"] = df["launch_angle"].between(8, 32, inclusive="both").fillna(False).astype(int)

    df["gb"] = df["bb_type"].eq("ground_ball").fillna(False).astype(int)
    df["fb"] = df["bb_type"].eq("fly_ball").fillna(False).astype(int)
    df["ld"] = df["bb_type"].eq("line_drive").fillna(False).astype(int)
    df["pu"] = df["bb_type"].isin(["popup", "pop_up"]).fillna(False).astype(int)

    df["tb"] = (
        df["is_1b"].astype(int)
        + 2 * df["is_2b"].astype(int)
        + 3 * df["is_3b"].astype(int)
        + 4 * df["is_hr"].astype(int)
    )

    desc = df["description"].fillna("")
    events = df["events"].fillna("")
    zone = df["zone"]

    strike_desc = {
        "called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"
    }
    swing_desc = {
        "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "hit_into_play_no_out",
        "hit_into_play_score", "foul_bunt", "missed_bunt"
    }
    contact_desc = {
        "foul", "foul_tip", "hit_into_play", "hit_into_play_no_out", "hit_into_play_score", "foul_bunt"
    }

    df["pitch_seen"] = 1
    df["zone_pitch"] = zone.between(1, 9, inclusive="both").fillna(False).astype(int)
    df["swing"] = desc.isin(swing_desc).astype(int)
    df["whiff"] = desc.isin({"swinging_strike", "swinging_strike_blocked", "missed_bunt"}).astype(int)
    df["contact"] = desc.isin(contact_desc).astype(int)
    df["called_strike"] = desc.eq("called_strike").astype(int)
    df["csw"] = (df["called_strike"].eq(1) | df["whiff"].eq(1)).astype(int)
    df["chase_swing"] = ((df["zone_pitch"] == 0) & (df["swing"] == 1)).astype(int)
    df["zone_contact"] = ((df["zone_pitch"] == 1) & (df["contact"] == 1)).astype(int)
    df["zone_swing"] = ((df["zone_pitch"] == 1) & (df["swing"] == 1)).astype(int)

    df["first_pitch"] = safe_numeric(_coalesce(df, ["pitch_number_start", "pitch_number_end"]), fill_value=1).eq(1).astype(int)
    df["first_pitch_swing"] = ((df["first_pitch"] == 1) & (df["swing"] == 1)).astype(int)
    df["first_pitch_strike"] = ((df["first_pitch"] == 1) & (df["description"].isin(strike_desc))).astype(int)

    return df


def build_batter_game_statcast(pa_df: pd.DataFrame) -> pd.DataFrame:
    df = _prep_raw_statcast(pa_df)

    grouped = (
        df.groupby(["season", "game_date", "game_pk", "batting_team", "fielding_team", "batter_id", "batter_name"], dropna=False)
        .agg(
            pa=("is_pa", "sum"),
            ab=("is_ab", "sum"),
            hits=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            rbi=("rbi", "sum"),
            tb=("tb", "sum"),
            bb=("is_bb", "sum"),
            so=("is_so", "sum"),
            hbp=("is_hbp", "sum"),
            runs_scored=("runs_scored_on_pa", "sum"),
            barrels=("barrel", "sum"),
            hard_hits=("hard_hit", "sum"),
            sweet_spots=("sweet_spot", "sum"),
            gb=("gb", "sum"),
            fb=("fb", "sum"),
            ld=("ld", "sum"),
            pu=("pu", "sum"),
            pitch_seen=("pitch_seen", "sum"),
            swings=("swing", "sum"),
            whiffs=("whiff", "sum"),
            contacts=("contact", "sum"),
            zone_pitches=("zone_pitch", "sum"),
            chase_swings=("chase_swing", "sum"),
            zone_contacts=("zone_contact", "sum"),
            zone_swings=("zone_swing", "sum"),
            first_pitch_seen=("first_pitch", "sum"),
            first_pitch_swings=("first_pitch_swing", "sum"),
            first_pitch_strikes_seen=("first_pitch_strike", "sum"),
            avg_ev=("launch_speed", "mean"),
            max_ev=("launch_speed", "max"),
            avg_la=("launch_angle", "mean"),
            avg_distance=("hit_distance_sc", "mean"),
            xba_mean=("estimated_ba_using_speedangle", "mean"),
            xwoba_mean=("estimated_woba_using_speedangle", "mean"),
            woba_value_mean=("woba_value", "mean"),
            avg_release_speed=("release_speed", "mean"),
            avg_spin_seen=("release_spin_rate", "mean"),
        )
        .reset_index()
    )

    grouped["barrel_rate"] = grouped["barrels"] / grouped["ab"].where(grouped["ab"].ne(0))
    grouped["hard_hit_rate"] = grouped["hard_hits"] / grouped["ab"].where(grouped["ab"].ne(0))
    grouped["sweet_spot_rate"] = grouped["sweet_spots"] / grouped["ab"].where(grouped["ab"].ne(0))
    grouped["gb_rate"] = grouped["gb"] / grouped["ab"].where(grouped["ab"].ne(0))
    grouped["fb_rate"] = grouped["fb"] / grouped["ab"].where(grouped["ab"].ne(0))
    grouped["ld_rate"] = grouped["ld"] / grouped["ab"].where(grouped["ab"].ne(0))
    grouped["pu_rate"] = grouped["pu"] / grouped["ab"].where(grouped["ab"].ne(0))
    grouped["whiff_rate"] = grouped["whiffs"] / grouped["swings"].where(grouped["swings"].ne(0))
    grouped["contact_rate"] = grouped["contacts"] / grouped["swings"].where(grouped["swings"].ne(0))
    grouped["chase_rate"] = grouped["chase_swings"] / (grouped["pitch_seen"] - grouped["zone_pitches"]).where((grouped["pitch_seen"] - grouped["zone_pitches"]).ne(0))
    grouped["zone_contact_rate"] = grouped["zone_contacts"] / grouped["zone_swings"].where(grouped["zone_swings"].ne(0))
    grouped["first_pitch_swing_rate"] = grouped["first_pitch_swings"] / grouped["first_pitch_seen"].where(grouped["first_pitch_seen"].ne(0))
    grouped["first_pitch_strike_seen_rate"] = grouped["first_pitch_strikes_seen"] / grouped["first_pitch_seen"].where(grouped["first_pitch_seen"].ne(0))
    grouped["hit_rate"] = grouped["hits"] / grouped["ab"].where(grouped["ab"].ne(0))
    grouped["hr_rate"] = grouped["hr"] / grouped["ab"].where(grouped["ab"].ne(0))
    grouped["rbi_pa_rate"] = grouped["rbi"] / grouped["pa"].where(grouped["pa"].ne(0))
    grouped["tb_pa_rate"] = grouped["tb"] / grouped["pa"].where(grouped["pa"].ne(0))

    return grouped.sort_values(["batter_id", "game_date", "game_pk"], kind="stable").reset_index(drop=True)


def build_pitcher_game_statcast(pa_df: pd.DataFrame) -> pd.DataFrame:
    df = _prep_raw_statcast(pa_df)

    grouped = (
        df.groupby(["season", "game_date", "game_pk", "fielding_team", "batting_team", "pitcher_id", "pitcher_name"], dropna=False)
        .agg(
            batters_faced=("is_pa", "sum"),
            ab_allowed=("is_ab", "sum"),
            hits_allowed=("is_hit", "sum"),
            hr_allowed=("is_hr", "sum"),
            bb_allowed=("is_bb", "sum"),
            hbp_allowed=("is_hbp", "sum"),
            so=("is_so", "sum"),
            rbi_allowed=("rbi", "sum"),
            runs_allowed=("runs_scored_on_pa", "sum"),
            barrels_allowed=("barrel", "sum"),
            hard_hits_allowed=("hard_hit", "sum"),
            sweet_spots_allowed=("sweet_spot", "sum"),
            gb_allowed=("gb", "sum"),
            fb_allowed=("fb", "sum"),
            ld_allowed=("ld", "sum"),
            pu_allowed=("pu", "sum"),
            pitch_seen=("pitch_seen", "sum"),
            swings_against=("swing", "sum"),
            whiffs_generated=("whiff", "sum"),
            contacts_allowed=("contact", "sum"),
            zone_pitches=("zone_pitch", "sum"),
            chase_swings_induced=("chase_swing", "sum"),
            zone_contacts_allowed=("zone_contact", "sum"),
            zone_swings_against=("zone_swing", "sum"),
            first_pitch_seen=("first_pitch", "sum"),
            first_pitch_strikes=("first_pitch_strike", "sum"),
            called_strikes=("called_strike", "sum"),
            csw_events=("csw", "sum"),
            avg_ev_allowed=("launch_speed", "mean"),
            max_ev_allowed=("launch_speed", "max"),
            avg_la_allowed=("launch_angle", "mean"),
            avg_distance_allowed=("hit_distance_sc", "mean"),
            xba_allowed=("estimated_ba_using_speedangle", "mean"),
            xwoba_allowed=("estimated_woba_using_speedangle", "mean"),
            avg_release_speed=("release_speed", "mean"),
            avg_spin_rate=("release_spin_rate", "mean"),
        )
        .reset_index()
    )

    grouped["barrel_rate_allowed"] = grouped["barrels_allowed"] / grouped["ab_allowed"].where(grouped["ab_allowed"].ne(0))
    grouped["hard_hit_rate_allowed"] = grouped["hard_hits_allowed"] / grouped["ab_allowed"].where(grouped["ab_allowed"].ne(0))
    grouped["sweet_spot_rate_allowed"] = grouped["sweet_spots_allowed"] / grouped["ab_allowed"].where(grouped["ab_allowed"].ne(0))
    grouped["gb_rate_allowed"] = grouped["gb_allowed"] / grouped["ab_allowed"].where(grouped["ab_allowed"].ne(0))
    grouped["fb_rate_allowed"] = grouped["fb_allowed"] / grouped["ab_allowed"].where(grouped["ab_allowed"].ne(0))
    grouped["ld_rate_allowed"] = grouped["ld_allowed"] / grouped["ab_allowed"].where(grouped["ab_allowed"].ne(0))
    grouped["whiff_rate"] = grouped["whiffs_generated"] / grouped["swings_against"].where(grouped["swings_against"].ne(0))
    grouped["contact_rate_allowed"] = grouped["contacts_allowed"] / grouped["swings_against"].where(grouped["swings_against"].ne(0))
    grouped["chase_rate_induced"] = grouped["chase_swings_induced"] / (grouped["pitch_seen"] - grouped["zone_pitches"]).where((grouped["pitch_seen"] - grouped["zone_pitches"]).ne(0))
    grouped["zone_contact_rate_allowed"] = grouped["zone_contacts_allowed"] / grouped["zone_swings_against"].where(grouped["zone_swings_against"].ne(0))
    grouped["first_pitch_strike_rate"] = grouped["first_pitch_strikes"] / grouped["first_pitch_seen"].where(grouped["first_pitch_seen"].ne(0))
    grouped["csw_rate"] = grouped["csw_events"] / grouped["pitch_seen"].where(grouped["pitch_seen"].ne(0))
    grouped["k_rate"] = grouped["so"] / grouped["batters_faced"].where(grouped["batters_faced"].ne(0))
    grouped["bb_rate"] = grouped["bb_allowed"] / grouped["batters_faced"].where(grouped["batters_faced"].ne(0))
    grouped["hr_rate"] = grouped["hr_allowed"] / grouped["batters_faced"].where(grouped["batters_faced"].ne(0))
    grouped["hit_rate_allowed"] = grouped["hits_allowed"] / grouped["batters_faced"].where(grouped["batters_faced"].ne(0))
    grouped["runs_rate_allowed"] = grouped["runs_allowed"] / grouped["batters_faced"].where(grouped["batters_faced"].ne(0))

    return grouped.sort_values(["pitcher_id", "game_date", "game_pk"], kind="stable").reset_index(drop=True)
