from __future__ import annotations

from src.utils.team_normalize import canonical_team_abbr


_TEAM_MAP = {
    "KCR": "KC",
    "KAN": "KC",
    "WSH": "WSN",
    "NYY": "NYY",
    "NYM": "NYM",
    "LAA": "LAA",
    "LAD": "LAD",
    "CHW": "CWS",
    "CHA": "CWS",
    "SFG": "SF",
    "SDP": "SD",
    "TBR": "TB",
}


def normalize_team_abbr(x: str | None) -> str:
    if x is None:
        return "UNK"
    raw = str(x).strip().upper()
    if not raw:
        return "UNK"
    if raw in _TEAM_MAP:
        return _TEAM_MAP[raw]
    can = canonical_team_abbr(raw)
    if can != "UNK":
        return _TEAM_MAP.get(can, can)
    if len(raw) in (2, 3) and raw.isalpha():
        return _TEAM_MAP.get(raw, raw)
    return "UNK"
