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


def _normalize_team(value: object, season: int | None = None) -> str:
    if value is None:
        return "UNK"
    try:
        if pd.isna(value):
            return "UNK"
    except Exception:
        pass
    return canonical_team_abbr(value, season)

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
    season_raw = pd.to_numeric(game_row.get("season"), errors="coerce")
    season = int(season_raw) if pd.notna(season_raw) else None
    team = _normalize_team(game_row.get("home_team"), season)

    def _resolved_row_to_result(row: pd.Series, fallback_val: object | None = None) -> dict[str, object]:
        park_id = row.get("park_id", fallback_val)
        venue_id = row.get("venue_id", park_id)
        park_name = row.get("park_name")
        return {
            "park_id": park_id,
            "venue_id": venue_id,
            "park_name": park_name,
            "canonical_park_key": build_canonical_park_key(
                team,
                park_name,
                row.get("lat"),
                row.get("lon"),
                venue_id,
                park_id,
            ),
        }

    for col in venue_candidates:
        val = game_row.get(col)
        if pd.notna(val) and col in parks_master_df.columns:
            match = parks_master_df[parks_master_df[col] == val]
            if not match.empty:
                if "season" in match.columns and season is not None:
                    season_match = match[pd.to_numeric(match["season"], errors="coerce") == season]
                    if not season_match.empty:
                        match = season_match
                sort_cols = [c for c in ["team", "venue_id", "park_id"] if c in match.columns]
                if sort_cols:
                    match = match.sort_values(by=sort_cols, kind="stable")
                return _resolved_row_to_result(match.iloc[0], val)

    park_name = game_row.get("park_name")
    if _normalize_name(park_name) == "" and not parks_master_df.empty:
        parks_team = parks_master_df.copy()
        if "team" in parks_team.columns:
            parks_team["__team_norm"] = parks_team["team"].map(lambda x: _normalize_team(x, season))
            team_match = parks_team[parks_team["__team_norm"] == team]
        else:
            team_match = parks_team.iloc[0:0]

        if not team_match.empty:
            venue_val = game_row.get("venue_id")
            if pd.notna(venue_val) and "venue_id" in team_match.columns:
                venue_match = team_match[team_match["venue_id"] == venue_val]
                if not venue_match.empty:
                    team_match = venue_match
            if "season" in team_match.columns and season is not None:
                season_match = team_match[pd.to_numeric(team_match["season"], errors="coerce") == season]
                if not season_match.empty:
                    team_match = season_match
            sort_cols = [c for c in ["team", "venue_id", "park_id"] if c in team_match.columns]
            if sort_cols:
                team_match = team_match.sort_values(by=sort_cols, kind="stable")
            return _resolved_row_to_result(team_match.iloc[0])

    park_id = game_row.get("park_id")
    venue_id = game_row.get("venue_id", park_id)
    return {
        "park_id": park_id,
        "venue_id": venue_id,
        "park_name": park_name,
        "canonical_park_key": build_canonical_park_key(team, park_name, None, None, venue_id, park_id),
    }
