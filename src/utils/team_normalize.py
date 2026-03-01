from __future__ import annotations

"""Team normalization helpers for historical naming changes."""


def canonical_team_abbr(name_or_abbr: str | None, season: int | None = None) -> str:
    if not name_or_abbr:
        return "UNK"
    raw = str(name_or_abbr).strip().upper()

    lookup = {
        "CLEVELAND INDIANS": "CLE",
        "CLEVELAND GUARDIANS": "CLE",
        "GUARDIANS": "CLE",
        "INDIANS": "CLE",
        "CLE": "CLE",
        "OAKLAND ATHLETICS": "OAK",
        "ATHLETICS": "OAK",
        "A'S": "OAK",
        "A’S": "OAK",
        "ATH": "OAK",
        "OAK": "OAK",
        "TAMPA BAY RAYS": "TB",
        "TAMPA BAY DEVIL RAYS": "TB",
        "TB": "TB",
    }
    if raw in lookup:
        return lookup[raw]

    # Best effort fallback for already-standard abbreviations.
    if len(raw) in (2, 3, 4) and raw.isalpha():
        return raw[:3]

    return "UNK"
