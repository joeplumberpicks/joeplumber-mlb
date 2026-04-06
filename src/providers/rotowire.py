from __future__ import annotations

import re
from io import StringIO
from typing import Any

import pandas as pd
import requests


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


TEAM_ALIASES = {
    "ARI": "ARI",
    "AZ": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CHW": "CHW",
    "CWS": "CHW",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KC",
    "KCR": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "OAK": "OAK",
    "ATH": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SD",
    "SDP": "SD",
    "SEA": "SEA",
    "SF": "SF",
    "SFG": "SF",
    "STL": "STL",
    "TB": "TB",
    "TBR": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WSH",
    "WAS": "WSH",
}


def _request_text(url: str, request_timeout: int = 30) -> str:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=request_timeout)
    response.raise_for_status()
    return response.text


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join([str(x).strip() for x in tup if str(x).strip() and str(x) != "nan"]).strip()
            for tup in out.columns
        ]
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def _norm_colname(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _normalize_team(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().upper()
    return TEAM_ALIASES.get(s, s)


def _normalize_is_home(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().lower()
    mapping = {
        "home": True,
        "away": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "vs": True,
        "@": False,
    }
    return mapping.get(s)


def _safe_get(row: pd.Series, candidates: list[str]) -> Any:
    for c in candidates:
        if c in row.index:
            return row[c]
    return None


def read_html_tables(
    url: str,
    request_timeout: int = 30,
    match: str | re.Pattern[str] | None = None,
    verbose: bool = False,
) -> list[pd.DataFrame]:
    html = _request_text(url, request_timeout=request_timeout)

    kwargs: dict[str, Any] = {}
    if match is not None:
        kwargs["match"] = match

    tables = pd.read_html(StringIO(html), **kwargs)
    tables = [_flatten_columns(t) for t in tables]

    if verbose:
        print(f"Rotowire tables found: {len(tables)}")

    return tables


def extract_lineups(table: pd.DataFrame, lineup_status: str = "confirmed") -> pd.DataFrame:
    df = _flatten_columns(table).copy()
    original_cols = list(df.columns)
    norm_map = {c: _norm_colname(c) for c in df.columns}
    df = df.rename(columns=norm_map)

    player_col = None
    for c in ["player", "name", "batter", "hitter"]:
        if c in df.columns:
            player_col = c
            break

    team_col = None
    for c in ["team", "tm"]:
        if c in df.columns:
            team_col = c
            break

    opp_col = None
    for c in ["opp", "opponent"]:
        if c in df.columns:
            opp_col = c
            break

    slot_col = None
    for c in ["order", "batting_order", "lineup_slot", "slot"]:
        if c in df.columns:
            slot_col = c
            break

    pid_col = None
    for c in ["player_id", "mlbam_id", "mlb_id", "id"]:
        if c in df.columns:
            pid_col = c
            break

    is_home_col = None
    for c in ["is_home", "home_away", "location", "venue"]:
        if c in df.columns:
            is_home_col = c
            break

    game_date_col = None
    for c in ["game_date", "date"]:
        if c in df.columns:
            game_date_col = c
            break

    if player_col is None or team_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["player_name"] = df[player_col].astype("string").str.strip()
    out["team"] = df[team_col].map(_normalize_team).astype("string")

    if opp_col is not None:
        out["opponent"] = df[opp_col].map(_normalize_team).astype("string")
    else:
        out["opponent"] = pd.Series(pd.NA, index=df.index, dtype="string")

    if slot_col is not None:
        out["lineup_slot"] = pd.to_numeric(df[slot_col], errors="coerce").astype("Int64")
    else:
        out["lineup_slot"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    if pid_col is not None:
        out["player_id"] = pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
    else:
        out["player_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    if is_home_col is not None:
        out["is_home"] = df[is_home_col].map(_normalize_is_home).astype("boolean")
    else:
        out["is_home"] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    if game_date_col is not None:
        out["game_date"] = pd.to_datetime(df[game_date_col], errors="coerce").dt.date.astype("string")
    else:
        out["game_date"] = pd.Series(pd.NA, index=df.index, dtype="string")

    out["lineup_status"] = lineup_status
    out["source"] = "rotowire"
    out["source_columns"] = ", ".join(original_cols)

    out = out.dropna(subset=["player_name", "team"], how="any")
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def extract_starting_pitchers(table: pd.DataFrame, starter_status: str = "probable") -> pd.DataFrame:
    df = _flatten_columns(table).copy()
    original_cols = list(df.columns)
    norm_map = {c: _norm_colname(c) for c in df.columns}
    df = df.rename(columns=norm_map)

    pitcher_col = None
    for c in ["pitcher", "starter", "player", "name"]:
        if c in df.columns:
            pitcher_col = c
            break

    team_col = None
    for c in ["team", "tm"]:
        if c in df.columns:
            team_col = c
            break

    opp_col = None
    for c in ["opp", "opponent"]:
        if c in df.columns:
            opp_col = c
            break

    pid_col = None
    for c in ["pitcher_id", "player_id", "mlbam_id", "mlb_id", "id"]:
        if c in df.columns:
            pid_col = c
            break

    is_home_col = None
    for c in ["is_home", "home_away", "location", "venue"]:
        if c in df.columns:
            is_home_col = c
            break

    game_date_col = None
    for c in ["game_date", "date"]:
        if c in df.columns:
            game_date_col = c
            break

    if pitcher_col is None or team_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["pitcher_name"] = df[pitcher_col].astype("string").str.strip()
    out["team"] = df[team_col].map(_normalize_team).astype("string")

    if opp_col is not None:
        out["opponent"] = df[opp_col].map(_normalize_team).astype("string")
    else:
        out["opponent"] = pd.Series(pd.NA, index=df.index, dtype="string")

    if pid_col is not None:
        out["pitcher_id"] = pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
    else:
        out["pitcher_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    if is_home_col is not None:
        out["is_home"] = df[is_home_col].map(_normalize_is_home).astype("boolean")
    else:
        out["is_home"] = pd.Series(pd.NA, index=df.index, dtype="boolean")

    if game_date_col is not None:
        out["game_date"] = pd.to_datetime(df[game_date_col], errors="coerce").dt.date.astype("string")
    else:
        out["game_date"] = pd.Series(pd.NA, index=df.index, dtype="string")

    out["starter_status"] = starter_status
    out["source"] = "rotowire"
    out["source_columns"] = ", ".join(original_cols)

    out = out.dropna(subset=["pitcher_name", "team"], how="any")
    out = out.drop_duplicates().reset_index(drop=True)
    return out