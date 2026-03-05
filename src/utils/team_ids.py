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

_TEAM_ID_TO_ABBR = {
    108: "LAA",
    109: "AZ",
    110: "BAL",
    111: "BOS",
    112: "CHC",
    113: "CIN",
    114: "CLE",
    115: "COL",
    116: "DET",
    117: "HOU",
    118: "KC",
    119: "LAD",
    120: "WSN",
    121: "NYM",
    133: "OAK",
    134: "PIT",
    135: "SD",
    136: "SEA",
    137: "SF",
    138: "STL",
    139: "TB",
    140: "TEX",
    141: "TOR",
    142: "MIN",
    143: "PHI",
    144: "ATL",
    145: "CWS",
    146: "MIA",
    147: "NYY",
    158: "MIL",
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


def team_id_to_abbr(team_id: int | str | None) -> str:
    try:
        tid = int(team_id) if team_id is not None else None
    except (TypeError, ValueError):
        tid = None
    if tid is None:
        return "UNK"
    return _TEAM_ID_TO_ABBR.get(tid, "UNK")
