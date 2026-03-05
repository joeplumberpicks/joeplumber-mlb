from __future__ import annotations

import re

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

NAME_TO_ABBR = {
    "arizona diamondbacks": "AZ",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CWS",
    "cincinnati reds": "CIN",
    "colorado rockies": "COL",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "seattle mariners": "SEA",
    "st louis cardinals": "STL",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSN",
}


def _clean_team_token(s: str) -> str:
    s = str(s).lower().replace(".", " ").replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_team_abbr(x: str | None) -> str:
    if x is None:
        return "UNK"
    if isinstance(x, float) and x != x:
        return "UNK"

    cleaned = _clean_team_token(str(x))
    if not cleaned:
        return "UNK"

    raw = cleaned.upper()
    if raw in _TEAM_MAP:
        return _TEAM_MAP[raw]
    can = canonical_team_abbr(raw)
    if can != "UNK":
        return _TEAM_MAP.get(can, can)
    if len(raw) in (2, 3) and raw.isalpha():
        return _TEAM_MAP.get(raw, raw)
    if cleaned in NAME_TO_ABBR:
        return NAME_TO_ABBR[cleaned]
    return "UNK"


def team_id_to_abbr(team_id: int | str | None) -> str:
    try:
        tid = int(team_id) if team_id is not None else None
    except (TypeError, ValueError):
        tid = None
    if tid is None:
        return "UNK"
    return _TEAM_ID_TO_ABBR.get(tid, "UNK")
