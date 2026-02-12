from __future__ import annotations

import pandas as pd

SWING_DESCRIPTIONS = {
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "swinging_strike",
    "swinging_strike_blocked",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "foul_pitchout",
    "swinging_pitchout",
}

WHIFF_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
    "swinging_pitchout",
}

INPLAY_DESCRIPTIONS = {
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

INPLAY_EVENTS = {
    "single",
    "double",
    "triple",
    "home_run",
    "field_out",
    "grounded_into_double_play",
    "force_out",
    "double_play",
    "field_error",
    "fielders_choice",
    "fielders_choice_out",
    "sac_fly",
    "sac_bunt",
}


def _lower_string(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def is_swing(description: pd.Series | None, pitch_type_series: pd.Series | None = None) -> pd.Series:
    """Return boolean mask for swings based on Statcast pitch descriptions."""
    if description is None:
        if pitch_type_series is None:
            return pd.Series(False, index=pd.RangeIndex(0))
        return pd.Series(False, index=pitch_type_series.index)

    desc = _lower_string(description)
    swings = desc.isin(SWING_DESCRIPTIONS)
    return swings.fillna(False)


def is_whiff(description: pd.Series | None, pitch_type_series: pd.Series | None = None) -> pd.Series:
    """Return boolean mask for swinging misses."""
    if description is None:
        if pitch_type_series is None:
            return pd.Series(False, index=pd.RangeIndex(0))
        return pd.Series(False, index=pitch_type_series.index)

    desc = _lower_string(description)
    whiffs = desc.isin(WHIFF_DESCRIPTIONS)
    return whiffs.fillna(False)


def is_ball_in_play(
    description: pd.Series | None,
    events: pd.Series | None,
    type_col: pd.Series | None,
    bb_type: pd.Series | None,
    fallback_index: pd.Index,
) -> pd.Series:
    """Return boolean mask for balls put in play."""
    out = pd.Series(False, index=fallback_index)

    if description is not None:
        out = out | _lower_string(description).isin(INPLAY_DESCRIPTIONS).fillna(False)

    if events is not None:
        out = out | _lower_string(events).isin(INPLAY_EVENTS).fillna(False)

    if type_col is not None:
        out = out | (_lower_string(type_col) == "x").fillna(False)

    if bb_type is not None:
        out = out | bb_type.notna()

    return out
