from __future__ import annotations

"""Stable park identity mapping utilities."""

from pathlib import Path

import pandas as pd

from src.utils.io import read_csv
from src.utils.team_normalize import canonical_team_abbr


def _normalize_name(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip().lower()


def build_canonical_park_key(team: str, park_name: object, lat: object, lon: object, venue_id: object, park_id: object) -> str:
    team_norm = canonical_team_abbr(team)
    name_norm = _normalize_name(park_name)
    if name_norm and pd.notna(lat) and pd.notna(lon):
        return f"{team_norm}|{name_norm}|{lat}|{lon}"
    if pd.notna(venue_id):
        return f"venue:{venue_id}"
    if pd.notna(park_id):
        return f"park:{park_id}"
    return f"{team_norm}|unknown_park"


def load_park_overrides(overrides_path: Path) -> pd.DataFrame:
    if not overrides_path.exists():
        return pd.DataFrame(
            columns=[
                "season_start",
                "season_end",
                "team",
                "park_name_contains",
                "venue_id",
                "park_id_override",
                "notes",
            ]
        )
    return read_csv(overrides_path)


def resolve_park_for_game(game_row: pd.Series, parks_master_df: pd.DataFrame) -> dict[str, object]:
    venue_candidates = ["venue_id", "park_id"]
    team = canonical_team_abbr(game_row.get("home_team"), int(game_row.get("season", 0) or 0))

    for col in venue_candidates:
        val = game_row.get(col)
        if pd.notna(val) and col in parks_master_df.columns:
            match = parks_master_df[parks_master_df[col] == val]
            if not match.empty:
                m = match.iloc[0]
                park_id = m.get("park_id", val)
                venue_id = m.get("venue_id", park_id)
                park_name = m.get("park_name")
                return {
                    "park_id": park_id,
                    "venue_id": venue_id,
                    "park_name": park_name,
                    "canonical_park_key": build_canonical_park_key(
                        team,
                        park_name,
                        m.get("lat"),
                        m.get("lon"),
                        venue_id,
                        park_id,
                    ),
                }

    park_id = game_row.get("park_id")
    venue_id = game_row.get("venue_id", park_id)
    park_name = game_row.get("park_name")
    return {
        "park_id": park_id,
        "venue_id": venue_id,
        "park_name": park_name,
        "canonical_park_key": build_canonical_park_key(team, park_name, None, None, venue_id, park_id),
    }
