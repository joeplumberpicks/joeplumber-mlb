from __future__ import annotations

import re
from datetime import datetime
from io import StringIO
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


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

PLAYER_HREF_RE = re.compile(r"/baseball/player/[^/]+-(\d+)")
STATUS_RE = re.compile(r"(Confirmed|Expected)\s+Lineup", re.I)
DATE_RE = re.compile(r"Starting MLB lineups for ([A-Za-z]+ \d{1,2}, \d{4})", re.I)


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


def _extract_rotowire_id(href: str | None) -> int | None:
    if not href:
        return None
    m = PLAYER_HREF_RE.search(href)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_page_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%B %d, %Y")
        return dt.date().isoformat()
    except Exception:
        return None


def _team_tokens(text: str) -> list[str]:
    tokens = re.findall(r"\b[A-Z]{2,3}\b", text.upper())
    out: list[str] = []
    for tok in tokens:
        t = _normalize_team(tok)
        if t in TEAM_ALIASES.values() and t not in out:
            out.append(t)
    return out


def _smallest_lineup_cards(soup: BeautifulSoup) -> list[Tag]:
    """
    Find the smallest ancestor containers that appear to represent one team-side lineup card.
    """
    candidates: list[Tag] = []
    seen: set[int] = set()

    for status_node in soup.find_all(string=STATUS_RE):
        cur = status_node.parent
        best: Tag | None = None

        # walk upward to find a compact container with 10-ish player links
        for _ in range(8):
            if cur is None or not isinstance(cur, Tag):
                break

            player_links = cur.find_all("a", href=PLAYER_HREF_RE)
            text = " ".join(cur.stripped_strings)
            n_links = len(player_links)

            if STATUS_RE.search(text) and 8 <= n_links <= 12:
                best = cur

            cur = cur.parent

        if best is not None and id(best) not in seen:
            seen.add(id(best))
            candidates.append(best)

    # preserve document order
    return candidates


def _parse_card(card: Tag, page_date: str | None) -> dict[str, Any] | None:
    text = " ".join(card.stripped_strings)
    if not STATUS_RE.search(text):
        return None

    status_match = STATUS_RE.search(text)
    if not status_match:
        return None
    lineup_status = status_match.group(1).lower()

    team_candidates = _team_tokens(text)
    team = team_candidates[0] if team_candidates else None

    player_links = []
    for a in card.find_all("a", href=PLAYER_HREF_RE):
        name = a.get_text(" ", strip=True)
        href = a.get("href")
        rid = _extract_rotowire_id(href)
        if name:
            player_links.append(
                {
                    "name": name,
                    "href": href,
                    "rotowire_id": rid,
                }
            )

    # Typical card order is: starter first, then 9 lineup batters
    if len(player_links) < 2:
        return None

    starter = player_links[0]
    lineup_players = player_links[1:10]

    if not lineup_players:
        return None

    return {
        "team": team,
        "game_date": page_date,
        "lineup_status": lineup_status,
        "starter_status": "probable",
        "starter": starter,
        "lineup_players": lineup_players,
    }


def _pair_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Pair consecutive cards as away/home when possible.
    This matches the visible page layout where one game displays two team cards in order.  [oai_citation:1‡RotoWire](https://www.rotowire.com/baseball/daily-lineups.php)
    """
    out: list[dict[str, Any]] = []

    i = 0
    while i < len(cards):
        left = cards[i].copy()
        right = cards[i + 1].copy() if i + 1 < len(cards) else None

        if right is not None:
            left["opponent"] = right.get("team")
            left["is_home"] = False

            right["opponent"] = left.get("team")
            right["is_home"] = True

            out.append(left)
            out.append(right)
            i += 2
        else:
            left["opponent"] = None
            left["is_home"] = None
            out.append(left)
            i += 1

    return out


def _cards_to_synthetic_tables(cards: list[dict[str, Any]]) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []

    for card in cards:
        team = card.get("team")
        opp = card.get("opponent")
        game_date = card.get("game_date")
        is_home = card.get("is_home")
        lineup_status = card.get("lineup_status", "confirmed")
        starter_status = card.get("starter_status", "probable")

        starter = card["starter"]
        starter_df = pd.DataFrame(
            [
                {
                    "record_type": "starter",
                    "team": team,
                    "opp": opp,
                    "is_home": is_home,
                    "game_date": game_date,
                    "pitcher": starter["name"],
                    "rotowire_id": starter["rotowire_id"],
                    "pitcher_id": pd.NA,   # resolver layer should map this later
                    "starter_status": starter_status,
                }
            ]
        )
        tables.append(starter_df)

        lineup_rows = []
        for idx, p in enumerate(card["lineup_players"], start=1):
            lineup_rows.append(
                {
                    "record_type": "lineup",
                    "team": team,
                    "opp": opp,
                    "is_home": is_home,
                    "game_date": game_date,
                    "player": p["name"],
                    "rotowire_id": p["rotowire_id"],
                    "player_id": pd.NA,    # resolver layer should map this later
                    "order": idx,
                    "lineup_status": lineup_status,
                }
            )

        lineup_df = pd.DataFrame(lineup_rows)
        tables.append(lineup_df)

    return tables


def _parse_rotowire_cards(html: str, verbose: bool = False) -> list[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text("\n", strip=True)
    page_date = _parse_page_date(page_text)

    card_tags = _smallest_lineup_cards(soup)
    parsed_cards: list[dict[str, Any]] = []

    for card in card_tags:
        parsed = _parse_card(card, page_date)
        if parsed is not None:
            parsed_cards.append(parsed)

    if not parsed_cards:
        raise ValueError("No lineup cards found in Rotowire page HTML")

    parsed_cards = _pair_cards(parsed_cards)
    tables = _cards_to_synthetic_tables(parsed_cards)

    if verbose:
        lineup_tables = sum(1 for t in tables if "record_type" in t.columns and t["record_type"].eq("lineup").all())
        starter_tables = sum(1 for t in tables if "record_type" in t.columns and t["record_type"].eq("starter").all())
        print(f"Rotowire synthetic lineup tables: {lineup_tables}")
        print(f"Rotowire synthetic starter tables: {starter_tables}")

    return tables


def read_html_tables(
    url: str,
    request_timeout: int = 30,
    match: str | re.Pattern[str] | None = None,
    verbose: bool = False,
) -> list[pd.DataFrame]:
    """
    Try HTML tables first. If none exist, fall back to parsing Rotowire lineup cards.
    """
    html = _request_text(url, request_timeout=request_timeout)

    kwargs: dict[str, Any] = {}
    if match is not None:
        kwargs["match"] = match

    try:
        tables = pd.read_html(StringIO(html), **kwargs)
        tables = [_flatten_columns(t) for t in tables]
        if verbose:
            print(f"Rotowire HTML tables found: {len(tables)}")
        return tables
    except ValueError:
        if verbose:
            print("Rotowire page returned no HTML tables; falling back to card parser")
        return _parse_rotowire_cards(html, verbose=verbose)


def extract_lineups(table: pd.DataFrame, lineup_status: str = "confirmed") -> pd.DataFrame:
    df = _flatten_columns(table).copy()
    original_cols = list(df.columns)
    norm_map = {c: _norm_colname(c) for c in df.columns}
    df = df.rename(columns=norm_map)

    # synthetic card-parser output
    if "record_type" in df.columns:
        df = df.loc[df["record_type"].astype("string").eq("lineup")].copy()
        if df.empty:
            return pd.DataFrame()

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

    rid_col = None
    for c in ["rotowire_id"]:
        if c in df.columns:
            rid_col = c
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

    if rid_col is not None:
        out["rotowire_id"] = pd.to_numeric(df[rid_col], errors="coerce").astype("Int64")
    else:
        out["rotowire_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

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

    # synthetic card-parser output
    if "record_type" in df.columns:
        df = df.loc[df["record_type"].astype("string").eq("starter")].copy()
        if df.empty:
            return pd.DataFrame()

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

    rid_col = None
    for c in ["rotowire_id"]:
        if c in df.columns:
            rid_col = c
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

    if rid_col is not None:
        out["rotowire_id"] = pd.to_numeric(df[rid_col], errors="coerce").astype("Int64")
    else:
        out["rotowire_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

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