from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.pa_outcome_model import PaOutcomeArtifact, predict_pa_outcome_proba


PA_CLASSES = ["out", "walk_hbp", "single", "double", "triple", "home_run"]
DROP_DEAD_FEATURES = {
    "pitch_number_start",
    "pitch_number_end",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
}
HALF_INNING_PA_CAP = 25


@dataclass
class GameState:
    inning: int = 1
    half: str = "TOP"
    outs: int = 0
    on_1b: int = 0
    on_2b: int = 0
    on_3b: int = 0
    score_away: int = 0
    score_home: int = 0
    batter_idx_away: int = 0
    batter_idx_home: int = 0
    bf_starter_away: int = 0
    bf_starter_home: int = 0
    using_bullpen_away: int = 0
    using_bullpen_home: int = 0


def encode_base_state(on_1b: int, on_2b: int, on_3b: int) -> str:
    runners = []
    if on_1b:
        runners.append("1")
    if on_2b:
        runners.append("2")
    if on_3b:
        runners.append("3")
    return "".join(runners)


def inning_bucket(inning: int) -> str:
    if inning <= 3:
        return "early"
    if inning <= 6:
        return "mid"
    return "late"


def _get_num(row: dict[str, Any], key: str) -> float:
    val = row.get(key, np.nan)
    return float(val) if pd.notna(val) else np.nan


def _clip_int(value: float, low: int, high: int) -> int:
    return int(max(low, min(high, round(value))))


def estimate_starter_bf_cap(starter_row: dict[str, Any]) -> int:
    """
    Dynamic starter hook with workload-first, quality-fallback logic.

    Priority:
    1) Use rolling batters faced directly when available.
    2) Otherwise estimate leash from quality/workload proxies.
    """
    bf30 = _get_num(starter_row, "pit_batters_faced_roll30")
    bf15 = _get_num(starter_row, "pit_batters_faced_roll15")
    bf7 = _get_num(starter_row, "pit_batters_faced_roll7")

    if pd.notna(bf30) and bf30 > 0:
        return _clip_int(bf30, 18, 30)
    if pd.notna(bf15) and bf15 > 0:
        return _clip_int(bf15 + 2, 18, 30)
    if pd.notna(bf7) and bf7 > 0:
        return _clip_int(bf7 + 6, 18, 30)

    k_rate = _get_num(starter_row, "pit_k_rate_roll30")
    bb_rate = _get_num(starter_row, "pit_bb_rate_roll30")
    hit_rate = _get_num(starter_row, "pit_hit_rate_roll30")
    hr_rate = _get_num(starter_row, "pit_hr_rate_roll30")
    outs_pg = _get_num(starter_row, "pit_outs_per_game_roll30")
    ip30 = _get_num(starter_row, "pit_ip_roll30")

    score = 0.0

    if pd.notna(k_rate):
        score += 20.0 * k_rate
    if pd.notna(bb_rate):
        score -= 12.0 * bb_rate
    if pd.notna(hit_rate):
        score -= 10.0 * hit_rate
    if pd.notna(hr_rate):
        score -= 15.0 * hr_rate
    if pd.notna(outs_pg):
        score += 3.0 * outs_pg
    if pd.notna(ip30):
        score += 0.2 * ip30

    if score >= 7.5:
        return 27
    if score >= 5.5:
        return 25
    if score >= 3.5:
        return 23
    if score >= 1.5:
        return 21
    return 19


def apply_pa_outcome(
    state: GameState,
    outcome: str,
) -> tuple[GameState, int]:
    runs_scored = 0

    on_1b = state.on_1b
    on_2b = state.on_2b
    on_3b = state.on_3b
    outs = state.outs

    if outcome == "out":
        outs += 1

    elif outcome == "walk_hbp":
        if on_1b and on_2b and on_3b:
            runs_scored += 1
        new_on_3b = 1 if (on_3b or on_2b) else 0
        new_on_2b = 1 if (on_2b or on_1b) else 0
        new_on_1b = 1
        on_1b, on_2b, on_3b = new_on_1b, new_on_2b, new_on_3b

    elif outcome == "single":
        runs_scored += on_3b
        runs_scored += on_2b
        new_on_3b = on_1b
        new_on_2b = 0
        new_on_1b = 1
        on_1b, on_2b, on_3b = new_on_1b, new_on_2b, new_on_3b

    elif outcome == "double":
        runs_scored += on_3b + on_2b
        new_on_3b = on_1b
        new_on_2b = 1
        new_on_1b = 0
        on_1b, on_2b, on_3b = new_on_1b, new_on_2b, new_on_3b

    elif outcome == "triple":
        runs_scored += on_1b + on_2b + on_3b
        on_1b, on_2b, on_3b = 0, 0, 1

    elif outcome == "home_run":
        runs_scored += 1 + on_1b + on_2b + on_3b
        on_1b, on_2b, on_3b = 0, 0, 0

    else:
        raise ValueError(f"Unknown outcome: {outcome}")

    state.outs = outs
    state.on_1b = on_1b
    state.on_2b = on_2b
    state.on_3b = on_3b

    if state.half == "TOP":
        state.score_away += runs_scored
    else:
        state.score_home += runs_scored

    return state, runs_scored


def advance_half_inning(state: GameState) -> GameState:
    state.outs = 0
    state.on_1b = 0
    state.on_2b = 0
    state.on_3b = 0

    if state.half == "TOP":
        state.half = "BOT"
    else:
        state.half = "TOP"
        state.inning += 1

    return state


def choose_outcome_from_probs(
    probs: dict[str, float],
    rng: np.random.Generator,
) -> str:
    p = np.array([max(0.0, float(probs.get(k, 0.0))) for k in PA_CLASSES], dtype=float)
    total = p.sum()
    if total <= 0:
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        p = p / total
    return str(rng.choice(PA_CLASSES, p=p))


def build_pre_pa_row(
    batter_row: dict[str, Any],
    pitcher_row: dict[str, Any],
    state: GameState,
) -> dict[str, Any]:
    row: dict[str, Any] = {}

    row["inning"] = state.inning
    row["outs_before_pa"] = state.outs
    row["inning_topbot"] = state.half
    row["base_state_before"] = encode_base_state(state.on_1b, state.on_2b, state.on_3b)
    row["on_1b"] = state.on_1b
    row["on_2b"] = state.on_2b
    row["on_3b"] = state.on_3b
    row["base_runner_count"] = state.on_1b + state.on_2b + state.on_3b
    row["risp_flag"] = int(state.on_2b == 1 or state.on_3b == 1)
    row["bases_empty_flag"] = int((state.on_1b + state.on_2b + state.on_3b) == 0)
    row["is_top_inning"] = int(state.half == "TOP")
    row["is_bot_inning"] = int(state.half == "BOT")
    row["two_out_flag"] = int(state.outs == 2)
    row["inning_bucket"] = inning_bucket(state.inning)

    row["batter_id"] = batter_row.get("batter_id")
    row["pitcher_id"] = pitcher_row.get("pitcher_id")
    row["lineup_slot"] = batter_row.get("lineup_slot")

    row.update(batter_row)
    row.update(pitcher_row)

    pairs_diff = [
        ("matchup_hit_rate_diff", "bat_hit_rate_roll30", "pit_hit_rate_roll30"),
        ("matchup_hr_rate_diff", "bat_hr_rate_roll30", "pit_hr_rate_roll30"),
        ("matchup_bb_rate_diff", "bat_bb_rate_roll30", "pit_bb_rate_roll30"),
        ("matchup_k_pressure_diff", "bat_so_rate_roll30", "pit_k_rate_roll30"),
        ("matchup_power_diff", "bat_tb_per_pa_roll30", "pit_tb_allowed_per_bf_roll30"),
    ]
    for new_col, a, b in pairs_diff:
        av = _get_num(row, a)
        bv = _get_num(row, b)
        row[new_col] = av - bv if pd.notna(av) and pd.notna(bv) else np.nan

    pairs_x = [
        ("matchup_hr_pressure_x", "bat_hr_rate_roll30", "pit_hr_rate_roll30"),
        ("matchup_hit_pressure_x", "bat_hit_rate_roll30", "pit_hit_rate_roll30"),
        ("matchup_walk_pressure_x", "bat_bb_rate_roll30", "pit_bb_rate_roll30"),
        ("matchup_k_pressure_x", "bat_so_rate_roll30", "pit_k_rate_roll30"),
    ]
    for new_col, a, b in pairs_x:
        av = _get_num(row, a)
        bv = _get_num(row, b)
        row[new_col] = av * bv if pd.notna(av) and pd.notna(bv) else np.nan

    return row


def _cache_key(
    batter_row: dict[str, Any],
    pitcher_row: dict[str, Any],
    state: GameState,
    pitcher_mode: str,
) -> tuple:
    return (
        int(batter_row.get("batter_id")),
        int(pitcher_row.get("pitcher_id")),
        pitcher_mode,
        int(batter_row.get("lineup_slot", -1)) if pd.notna(batter_row.get("lineup_slot", np.nan)) else -1,
        state.inning,
        state.half,
        state.outs,
        state.on_1b,
        state.on_2b,
        state.on_3b,
    )


def get_pa_probabilities_fast(
    artifact: PaOutcomeArtifact,
    batter_row: dict[str, Any],
    pitcher_row: dict[str, Any],
    state: GameState,
    cache: dict[tuple, dict[str, float]],
    feature_columns: list[str],
    pitcher_mode: str,
) -> dict[str, float]:
    key = _cache_key(batter_row, pitcher_row, state, pitcher_mode)
    if key in cache:
        return cache[key]

    sim_row = build_pre_pa_row(
        batter_row=batter_row,
        pitcher_row=pitcher_row,
        state=state,
    )

    usable_feature_columns = [c for c in feature_columns if c not in DROP_DEAD_FEATURES]
    sim_df = pd.DataFrame([{c: sim_row.get(c, np.nan) for c in usable_feature_columns}])

    proba = predict_pa_outcome_proba(artifact, sim_df).iloc[0].to_dict()
    probs = {
        "out": float(proba.get("p_out", 0.0)),
        "walk_hbp": float(proba.get("p_walk_hbp", 0.0)),
        "single": float(proba.get("p_single", 0.0)),
        "double": float(proba.get("p_double", 0.0)),
        "triple": float(proba.get("p_triple", 0.0)),
        "home_run": float(proba.get("p_home_run", 0.0)),
    }

    cache[key] = probs
    return probs


def simulate_single_game_fast(
    artifact: PaOutcomeArtifact,
    lineup_away: pd.DataFrame,
    lineup_home: pd.DataFrame,
    starter_away: pd.Series,
    starter_home: pd.Series,
    bullpen_away: pd.Series | None,
    bullpen_home: pd.Series | None,
    starter_bf_cap_away: int | None,
    starter_bf_cap_home: int | None,
    rng: np.random.Generator,
    max_innings: int = 9,
    extra_innings_cap: int = 12,
    return_pa_log: bool = False,
) -> dict[str, Any]:
    state = GameState()
    pa_log: list[dict[str, Any]] = []

    lineup_away = lineup_away.reset_index(drop=True).copy()
    lineup_home = lineup_home.reset_index(drop=True).copy()

    away_rows = [lineup_away.iloc[i].to_dict() for i in range(len(lineup_away))]
    home_rows = [lineup_home.iloc[i].to_dict() for i in range(len(lineup_home))]
    starter_away_dict = starter_away.to_dict()
    starter_home_dict = starter_home.to_dict()
    bullpen_away_dict = bullpen_away.to_dict() if bullpen_away is not None else None
    bullpen_home_dict = bullpen_home.to_dict() if bullpen_home is not None else None

    dynamic_bf_cap_away = starter_bf_cap_away if starter_bf_cap_away is not None else estimate_starter_bf_cap(starter_away_dict)
    dynamic_bf_cap_home = starter_bf_cap_home if starter_bf_cap_home is not None else estimate_starter_bf_cap(starter_home_dict)

    cache: dict[tuple, dict[str, float]] = {}
    feature_columns = list(artifact.feature_columns)

    current_half_pa_count = 0

    while True:
        if state.inning > max_innings:
            if state.score_away != state.score_home:
                break
            if state.inning > extra_innings_cap:
                break

        if current_half_pa_count >= HALF_INNING_PA_CAP:
            state.outs = 3

        if state.half == "TOP":
            batter_idx = state.batter_idx_away % len(away_rows)
            batter_row = away_rows[batter_idx]

            if bullpen_home_dict is not None and state.bf_starter_home >= dynamic_bf_cap_home:
                pitcher_row = bullpen_home_dict
                pitcher_mode = "bullpen_home"
                state.using_bullpen_home = 1
            else:
                pitcher_row = starter_home_dict
                pitcher_mode = "starter_home"

        else:
            batter_idx = state.batter_idx_home % len(home_rows)
            batter_row = home_rows[batter_idx]

            if bullpen_away_dict is not None and state.bf_starter_away >= dynamic_bf_cap_away:
                pitcher_row = bullpen_away_dict
                pitcher_mode = "bullpen_away"
                state.using_bullpen_away = 1
            else:
                pitcher_row = starter_away_dict
                pitcher_mode = "starter_away"

        probs = get_pa_probabilities_fast(
            artifact=artifact,
            batter_row=batter_row,
            pitcher_row=pitcher_row,
            state=state,
            cache=cache,
            feature_columns=feature_columns,
            pitcher_mode=pitcher_mode,
        )

        outcome = choose_outcome_from_probs(probs, rng=rng)

        pa_record = {
            "inning": state.inning,
            "half": state.half,
            "outs_before_pa": state.outs,
            "base_state_before": encode_base_state(state.on_1b, state.on_2b, state.on_3b),
            "batter_id": batter_row.get("batter_id"),
            "pitcher_id": pitcher_row.get("pitcher_id"),
            "pitcher_mode": pitcher_mode,
            "outcome": outcome,
            "half_inning_pa_count": current_half_pa_count,
            **{f"p_{k}": float(v) for k, v in probs.items()},
        }

        state, runs_scored = apply_pa_outcome(state, outcome)
        pa_record["runs_scored_on_pa"] = runs_scored
        pa_record["score_away_after"] = state.score_away
        pa_record["score_home_after"] = state.score_home

        if state.half == "TOP":
            state.batter_idx_away += 1
            if pitcher_mode == "starter_home":
                state.bf_starter_home += 1
        else:
            state.batter_idx_home += 1
            if pitcher_mode == "starter_away":
                state.bf_starter_away += 1

        current_half_pa_count += 1
        pa_log.append(pa_record)

        if state.outs >= 3:
            state = advance_half_inning(state)
            current_half_pa_count = 0

        if state.inning >= max_innings and state.half == "TOP" and state.score_home > state.score_away:
            break

    out = {
        "score_away": int(state.score_away),
        "score_home": int(state.score_home),
        "total_runs": int(state.score_away + state.score_home),
        "home_win": int(state.score_home > state.score_away),
        "away_win": int(state.score_away > state.score_home),
        "is_tie_cap": int(state.score_home == state.score_away),
        "cache_size": int(len(cache)),
        "used_bullpen_away": int(state.using_bullpen_away),
        "used_bullpen_home": int(state.using_bullpen_home),
        "bf_starter_away": int(state.bf_starter_away),
        "bf_starter_home": int(state.bf_starter_home),
        "starter_bf_cap_away": int(dynamic_bf_cap_away),
        "starter_bf_cap_home": int(dynamic_bf_cap_home),
    }

    if return_pa_log:
        out["pa_log"] = pd.DataFrame(pa_log)

    return out


def summarize_sim_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(results)

    return {
        "n_sims": int(len(df)),
        "home_win_pct": float(df["home_win"].mean()),
        "away_win_pct": float(df["away_win"].mean()),
        "tie_cap_pct": float(df["is_tie_cap"].mean()),
        "mean_home_runs_scored": float(df["score_home"].mean()),
        "mean_away_runs_scored": float(df["score_away"].mean()),
        "mean_total_runs": float(df["total_runs"].mean()),
        "median_total_runs": float(df["total_runs"].median()),
        "p_total_le_7_5": float((df["total_runs"] <= 7).mean()),
        "p_total_ge_8_5": float((df["total_runs"] >= 9).mean()),
        "mean_cache_size": float(df["cache_size"].mean()) if "cache_size" in df.columns else np.nan,
        "bullpen_used_pct_away": float(df["used_bullpen_away"].mean()) if "used_bullpen_away" in df.columns else np.nan,
        "bullpen_used_pct_home": float(df["used_bullpen_home"].mean()) if "used_bullpen_home" in df.columns else np.nan,
        "mean_bf_starter_away": float(df["bf_starter_away"].mean()) if "bf_starter_away" in df.columns else np.nan,
        "mean_bf_starter_home": float(df["bf_starter_home"].mean()) if "bf_starter_home" in df.columns else np.nan,
        "mean_starter_bf_cap_away": float(df["starter_bf_cap_away"].mean()) if "starter_bf_cap_away" in df.columns else np.nan,
        "mean_starter_bf_cap_home": float(df["starter_bf_cap_home"].mean()) if "starter_bf_cap_home" in df.columns else np.nan,
    }
