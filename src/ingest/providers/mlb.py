"""
MLB Data Provider

Purpose
-------
Handles all raw data pulls from MLB Stats API.

This is a LOW-LEVEL provider:
- No normalization
- No schema enforcement
- No feature logic

Returns raw JSON or lightly flattened DataFrames.

Used by:
- schedule.py
- games.py
- plate_appearances.py
"""

from __future__ import annotations

import requests
from typing import Any, Dict, List

import pandas as pd


BASE_URL = "https://statsapi.mlb.com/api/v1"


# ---------------------------------------------------------------------
# Core Request Helper
# ---------------------------------------------------------------------

def _get(endpoint: str, params: dict | None = None) -> dict:
    """
    Internal helper for MLB API GET requests.
    """
    url = f"{BASE_URL}{endpoint}"
    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------

def fetch_schedule(
    date: str,
    *,
    sport_id: int = 1,
    game_types: str = "R",
) -> dict:
    """
    Fetch MLB schedule for a given date.

    Returns raw JSON.
    """
    return _get(
        "/schedule",
        params={
            "sportId": sport_id,
            "date": date,
            "gameTypes": game_types,
        },
    )


def fetch_schedule_dataframe(date: str) -> pd.DataFrame:
    """
    Fetch and lightly flatten schedule into DataFrame.
    """
    data = fetch_schedule(date)

    rows: List[Dict[str, Any]] = []

    for d in data.get("dates", []):
        for game in d.get("games", []):
            rows.append(
                {
                    "game_pk": game.get("gamePk"),
                    "game_date": game.get("officialDate"),
                    "season": game.get("season"),
                    "game_type": game.get("gameType"),
                    "status": game.get("status", {}).get("abstractGameState"),
                    "status_detailed": game.get("status", {}).get("detailedState"),
                    "coded_game_state": game.get("status", {}).get("codedGameState"),
                    "away_team": game.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation"),
                    "home_team": game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation"),
                    "away_team_id": game.get("teams", {}).get("away", {}).get("team", {}).get("id"),
                    "home_team_id": game.get("teams", {}).get("home", {}).get("team", {}).get("id"),
                    "venue_id": game.get("venue", {}).get("id"),
                    "venue_name": game.get("venue", {}).get("name"),
                    "scheduled_start_time_utc": game.get("gameDate"),
                    "doubleheader_flag": game.get("doubleHeader"),
                    "game_number": game.get("gameNumber"),
                    "day_night": game.get("dayNight"),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Game Feed (Live / Final Game Data)
# ---------------------------------------------------------------------

def fetch_game_feed(game_pk: int) -> dict:
    """
    Fetch full live game feed (boxscore + plays).
    """
    return _get(f"/game/{game_pk}/feed/live")


# ---------------------------------------------------------------------
# Boxscore
# ---------------------------------------------------------------------

def fetch_boxscore(game_pk: int) -> dict:
    """
    Fetch boxscore for a game.
    """
    return _get(f"/game/{game_pk}/boxscore")


# ---------------------------------------------------------------------
# Play-by-Play / Plate Appearances
# ---------------------------------------------------------------------

def fetch_play_by_play(game_pk: int) -> dict:
    """
    Fetch play-by-play data for a game.
    """
    return _get(f"/game/{game_pk}/playByPlay")


def extract_plate_appearances_from_feed(feed: dict) -> pd.DataFrame:
    """
    Extract plate appearances from MLB live feed JSON.

    NOTE:
    This is still RAW extraction (no feature logic).
    """
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])

    rows = []

    for idx, play in enumerate(plays):
        matchup = play.get("matchup", {})
        result = play.get("result", {})
        about = play.get("about", {})

        rows.append(
            {
                "pa_index": idx,
                "game_pk": feed.get("gamePk"),
                "game_date": feed.get("gameData", {}).get("datetime", {}).get("officialDate"),
                "inning": about.get("inning"),
                "inning_topbot": about.get("halfInning"),
                "batter_id": matchup.get("batter", {}).get("id"),
                "batter_name": matchup.get("batter", {}).get("fullName"),
                "pitcher_id": matchup.get("pitcher", {}).get("id"),
                "pitcher_name": matchup.get("pitcher", {}).get("fullName"),
                "event_type": result.get("eventType"),
                "event_text": result.get("description"),
                "rbi": result.get("rbi"),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Venues / Parks
# ---------------------------------------------------------------------

def fetch_venues() -> dict:
    """
    Fetch all MLB venues.
    """
    return _get("/venues", params={"sportId": 1})


def fetch_venues_dataframe() -> pd.DataFrame:
    """
    Flatten venue data.
    """
    data = fetch_venues()

    rows = []

    for v in data.get("venues", []):
        location = v.get("location", {})

        rows.append(
            {
                "venue_id": v.get("id"),
                "venue_name": v.get("name"),
                "city": location.get("city"),
                "state": location.get("stateAbbrev"),
                "country": location.get("country"),
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
                "time_zone": v.get("timeZone", {}).get("id"),
                "roof_type": v.get("roofType"),
                "turf_type": v.get("turfType"),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------

def test_connection() -> bool:
    """
    Simple health check to confirm MLB API is reachable.
    """
    try:
        _get("/sports")
        return True
    except Exception:
        return False


__all__ = [
    "fetch_schedule",
    "fetch_schedule_dataframe",
    "fetch_game_feed",
    "fetch_boxscore",
    "fetch_play_by_play",
    "extract_plate_appearances_from_feed",
    "fetch_venues",
    "fetch_venues_dataframe",
    "test_connection",
]
