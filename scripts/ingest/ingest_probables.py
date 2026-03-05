from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def fetch_probables_for_date(date: str) -> pd.DataFrame:
    """Fetch probable starters keyed by game_pk for a single date.

    Returns empty frame with expected columns if data is unavailable.
    """
    cols = [
        "game_pk",
        "home_probable_pitcher_id",
        "away_probable_pitcher_id",
        "home_probable_pitcher_name",
        "away_probable_pitcher_name",
    ]

    try:
        params = {
            "sportId": 1,
            "date": date,
            "hydrate": "probablePitcher,team",
        }
        resp = requests.get(SCHEDULE_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=cols)

    rows: list[dict] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            home_prob = teams.get("home", {}).get("probablePitcher")
            away_prob = teams.get("away", {}).get("probablePitcher")
            rows.append(
                {
                    "game_pk": pd.to_numeric(game.get("gamePk"), errors="coerce"),
                    "home_probable_pitcher_id": home_prob.get("id") if home_prob else None,
                    "away_probable_pitcher_id": away_prob.get("id") if away_prob else None,
                    "home_probable_pitcher_name": home_prob.get("fullName") if home_prob else None,
                    "away_probable_pitcher_name": away_prob.get("fullName") if away_prob else None,
                }
            )

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["home_probable_pitcher_id"] = pd.to_numeric(df["home_probable_pitcher_id"], errors="coerce").astype("Int64")
    df["away_probable_pitcher_id"] = pd.to_numeric(df["away_probable_pitcher_id"], errors="coerce").astype("Int64")
    return df[cols]
