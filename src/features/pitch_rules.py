from __future__ import annotations

"""Pitch event classification rules."""

import math
from typing import Any

SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_score",
    "hit_into_play_no_out",
    "foul_bunt",
    "missed_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
CONTACT_DESCRIPTIONS = {"foul", "foul_tip", "hit_into_play", "hit_into_play_score", "hit_into_play_no_out"}

FASTBALL_TYPES = {"FF", "FT", "SI", "FC", "FA"}
BREAKING_TYPES = {"SL", "CU", "KC", "SV", "CS", "ST"}
OFFSPEED_TYPES = {"CH", "FS", "FO", "SC", "KN", "EP"}


def pitch_group(pitch_type: Any) -> str:
    # Statcast occasionally emits float NaN for pitch_type, so guard before string normalization.
    if pitch_type is None or (isinstance(pitch_type, float) and math.isnan(pitch_type)):
        pt_raw = ""
    else:
        pt_raw = str(pitch_type)

    pt = pt_raw.strip().upper()
    if pt in FASTBALL_TYPES:
        return "fastball"
    if pt in BREAKING_TYPES:
        return "breaking"
    if pt in OFFSPEED_TYPES:
        return "offspeed"
    return "other"
