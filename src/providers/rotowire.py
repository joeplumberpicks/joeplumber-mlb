from __future__ import annotations

import re
from datetime import datetime
from io import StringIO
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


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
DATE_RE = re.compile(r"Starting MLB lineups for ([A-Za-z]+ \d{1,2}, \d{4})", re.I)
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*(AM|PM)\s*ET$", re.I)
STATUS_RE = re.compile(r"^(Confirmed|Expected)\s+Lineup$", re.I)
POSITION_RE = re.compile(r"^(C|1B|2B|3B|SS|LF|CF|RF|DH)$", re.I)
HAND_RE = re.compile(r"^[RLS]$", re.I)
ERA_RE = re.compile(r"ERA", re.I)


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


def _parse_page_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    try:
        dt = datetime.strptime(m.group(1), "%B %d, %Y")
        return dt.date().isoformat()
    except Exception:
        return None


def _name_to_rotowire_ids(soup: BeautifulSoup) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in soup.find_all("a", href=PLAYER_HREF_RE):
        name = a.get_text(" ", strip=True)
        href = a.get("href", "")
        m = PLAYER_HREF_RE.search(href)
        if not name or not m:
            continue
        try:
            out[name] = int(m.group(1))
        except Exception:
            continue
    return out


def _clean_tokens(soup: BeautifulSoup) -> list[str]:
    tokens = []
    for s in soup.stripped_strings:
        tok = str(s).strip()
        if tok:
            tokens.append(tok)
    return tokens


def _find_time_indices(tokens: list[str]) -> list[int]:
    return [i for i, tok in enumerate(tokens) if TIME_RE.match(tok)]


def _find_first_two_teams(tokens: list[str]) -> tuple[str | None, str | None]:
    found: list[str] = []
    for tok in tokens:
        t = _normalize_team(tok)
        if t in TEAM_ALIASES.values():
            found.append(t)
            if len(found) == 2:
                return found[0], found[1]
    return None, None


def _find_pitcher_before_status(tokens: list[str], status_idx: int) -> tuple[str | None, str | None]:
    # look backwards for "... Pitcher Name, R/L, ERA line, Confirmed Lineup"
    for i in range(max(0, status_idx - 6), status_idx):
        pass

    for i in range(status_idx - 1, max(-1, status_idx - 8), -1):
        if i - 2 >= 0 and HAND_RE.match(tokens[i - 1]) and ERA_RE.search(tokens[i]):
            name = tokens[i - 2]
            throws = tokens[i - 1].upper()
            return name, throws
    return None, None


def _parse_lineup_after_status(
    tokens: list[str],
    start_idx: int,
    team: str | None,
    opponent: str | None,
    is_home: bool | None,
    game_date: str | None,
    lineup_status: str,
    name_to_rw_id: dict[str, int],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    i = start_idx + 1
    order = 1

    while i < len(tokens) and order <= 9:
        tok = tokens[i]

        if tok == "Home Run Odds" or tok == "Starting Pitcher Intel" or STATUS_RE.match(tok):
            break

        if POSITION_RE.match(tok):
            position = tok.upper()
            name = tokens[i + 1] if i + 1 < len(tokens) else None
            hand_bat = tokens[i + 2].upper() if i + 2 < len(tokens) and HAND_RE.match(tokens[i + 2]) else None

            if name and name not in {"Home Run Odds", "Starting Pitcher Intel"}:
                rows.append(
                    {
                        "record_type": "lineup",
                        "team": team,
                        "opp": opponent,
                        "is_home": is_home,
                        "game_date": game_date,
                        "player": name,
                        "rotowire_id": name_to_rw_id.get(name),
                        "player_id": pd.NA,
                        "order": order,
                        "position": position,
                        "handedness_bat": hand_bat,
                        "lineup_status": lineup_status,
                    }
                )
                order += 1

            i += 3
            continue

        i += 1

    return rows, i


def _parse_game_segment(tokens: list[str], page_date: str | None, name_to_rw_id: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    away_team, home_team = _find_first_two_teams(tokens)
    if away_team is None or home_team is None:
        return rows

    status_indices = [i for i, tok in enumerate(tokens) if STATUS_RE.match(tok)]
    if len(status_indices) < 2:
        return rows

    # away side
    away_status_idx = status_indices[0]
    away_status = STATUS_RE.match(tokens[away_status_idx]).group(1).lower()  # type: ignore[union-attr]
    away_pitcher_name, away_throws = _find_pitcher_before_status(tokens, away_status_idx)
    if away_pitcher_name:
        rows.append(
            {
                "record_type": "starter",
                "team": away_team,
                "opp": home_team,
                "is_home": False,
                "game_date": page_date,
                "pitcher": away_pitcher_name,
                "rotowire_id": name_to_rw_id.get(away_pitcher_name),
                "pitcher_id": pd.NA,
                "throws": away_throws,
                "starter_status": "probable",
            }
        )

    away_lineup_rows, scan_idx = _parse_lineup_after_status(
        tokens=tokens,
        start_idx=away_status_idx,
        team=away_team,
        opponent=home_team,
        is_home=False,
        game_date=page_date,
        lineup_status=away_status,
        name_to_rw_id=name_to_rw_id,
    )
    rows.extend(away_lineup_rows)

    # home side
    home_status_idx = None
    for idx in status_indices[1:]:
        if idx > scan_idx:
            home_status_idx = idx
            break
    if home_status_idx is None:
        home_status_idx = status_indices[1]

    home_status = STATUS_RE.match(tokens[home_status_idx]).group(1).lower()  # type: ignore[union-attr]
    home_pitcher_name, home_throws = _find_pitcher_before_status(tokens, home_status_idx)
    if home_pitcher_name:
        rows.append(
            {
                "record_type": "starter",
                "team": home_team,
                "opp": away_team,
                "is_home": True,
                "game_date": page_date,
                "pitcher": home_pitcher_name,
                "rotowire_id": name_to_rw_id.get(home_pitcher_name),
                "pitcher_id": pd.NA,
                "throws": home_throws,
                "starter_status": "probable",
            }
        )

    home_lineup_rows, _ = _parse_lineup_after_status(
        tokens=tokens,
        start_idx=home_status_idx,
        team=home_team,
        opponent=away_team,
        is_home=True,
        game_date=page_date,
        lineup_status=home_status,
        name_to_rw_id=name_to_rw_id,
    )
    rows.extend(home_lineup_rows)

    return rows


def _parse_rotowire_cards(html: str, verbose: bool = False) -> list[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text("\n", strip=True)
    page_date = _parse_page_date(page_text)
    name_to_rw_id = _name_to_rotowire_ids(soup)
    tokens = _clean_tokens(soup)

    time_indices = _find_time_indices(tokens)
    if not time_indices:
        raise ValueError("No game time markers found in Rotowire page")

    all_rows: list[dict[str, Any]] = []

    for idx, start in enumerate(time_indices):
        end = time_indices[idx + 1] if idx + 1 < len(time_indices) else len(tokens)
        segment = tokens[start:end]
        all_rows.extend(_parse_game_segment(segment, page_date, name_to_rw_id))

    if not all_rows:
        raise ValueError("No lineup or starter rows parsed from Rotowire page")

    df = pd.DataFrame(all_rows)

    if verbose:
        n_lineup = int(df["record_type"].eq("lineup").sum()) if "record_type" in df.columns else 0
        n_starter = int(df["record_type"].eq("starter").sum()) if "record_type" in df.columns else 0
        print(f"Rotowire parsed lineup rows: {n_lineup}")
        print(f"Rotowire parsed starter rows: {n_starter}")

    return [df]


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

    try:
        tables = pd.read_html(StringIO(html), **kwargs)
        tables = [_flatten_columns(t) for t in tables]
        if verbose:
            print(f"Rotowire HTML tables found: {len(tables)}")
        return tables
    except ValueError:
        if verbose:
            print("Rotowire page returned no HTML tables; falling back to token parser")
        return _parse_rotowire_cards(html, verbose=verbose)


def extract_lineups(table: pd.DataFrame, lineup_status: str = "confirmed") -> pd.DataFrame:
    df = _flatten_columns(table).copy()
    original_cols = list(df.columns)
    norm_map = {c: _norm_colname(c) for c in df.columns}
    df = df.rename(columns=norm_map)

    if "record_type" in df.columns:
        df = df.loc[df["record_type"].astype("string").eq("lineup")].copy()
        if df.empty:
            return pd.DataFrame()

    player_col = next((c for c in ["player", "name", "batter", "hitter"] if c in df.columns), None)
    team_col = next((c for c in ["team", "tm"] if c in df.columns), None)
    opp_col = next((c for c in ["opp", "opponent"] if c in df.columns), None)
    slot_col = next((c for c in ["order", "batting_order", "lineup_slot", "slot"] if c in df.columns), None)
    pid_col = next((c for c in ["player_id", "mlbam_id", "mlb_id", "id"] if c in df.columns), None)
    rid_col = "rotowire_id" if "rotowire_id" in df.columns else None
    is_home_col = next((c for c in ["is_home", "home_away", "location", "venue"] if c in df.columns), None)
    game_date_col = next((c for c in ["game_date", "date"] if c in df.columns), None)
    hb_col = "handedness_bat" if "handedness_bat" in df.columns else None
    pos_col = "position" if "position" in df.columns else None

    if player_col is None or team_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["player_name"] = df[player_col].astype("string").str.strip()
    out["team"] = df[team_col].map(_normalize_team).astype("string")
    out["opponent"] = (
        df[opp_col].map(_normalize_team).astype("string")
        if opp_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["lineup_slot"] = (
        pd.to_numeric(df[slot_col], errors="coerce").astype("Int64")
        if slot_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    out["player_id"] = (
        pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
        if pid_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    out["rotowire_id"] = (
        pd.to_numeric(df[rid_col], errors="coerce").astype("Int64")
        if rid_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    out["is_home"] = (
        df[is_home_col].astype("boolean")
        if is_home_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="boolean")
    )
    out["game_date"] = (
        pd.to_datetime(df[game_date_col], errors="coerce").dt.date.astype("string")
        if game_date_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["handedness_bat"] = (
        df[hb_col].astype("string")
        if hb_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["position"] = (
        df[pos_col].astype("string")
        if pos_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
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

    if "record_type" in df.columns:
        df = df.loc[df["record_type"].astype("string").eq("starter")].copy()
        if df.empty:
            return pd.DataFrame()

    pitcher_col = next((c for c in ["pitcher", "starter", "player", "name"] if c in df.columns), None)
    team_col = next((c for c in ["team", "tm"] if c in df.columns), None)
    opp_col = next((c for c in ["opp", "opponent"] if c in df.columns), None)
    pid_col = next((c for c in ["pitcher_id", "player_id", "mlbam_id", "mlb_id", "id"] if c in df.columns), None)
    rid_col = "rotowire_id" if "rotowire_id" in df.columns else None
    is_home_col = next((c for c in ["is_home", "home_away", "location", "venue"] if c in df.columns), None)
    game_date_col = next((c for c in ["game_date", "date"] if c in df.columns), None)
    throws_col = "throws" if "throws" in df.columns else None

    if pitcher_col is None or team_col is None:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["pitcher_name"] = df[pitcher_col].astype("string").str.strip()
    out["team"] = df[team_col].map(_normalize_team).astype("string")
    out["opponent"] = (
        df[opp_col].map(_normalize_team).astype("string")
        if opp_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["pitcher_id"] = (
        pd.to_numeric(df[pid_col], errors="coerce").astype("Int64")
        if pid_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    out["rotowire_id"] = (
        pd.to_numeric(df[rid_col], errors="coerce").astype("Int64")
        if rid_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="Int64")
    )
    out["is_home"] = (
        df[is_home_col].astype("boolean")
        if is_home_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="boolean")
    )
    out["game_date"] = (
        pd.to_datetime(df[game_date_col], errors="coerce").dt.date.astype("string")
        if game_date_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["throws"] = (
        df[throws_col].astype("string")
        if throws_col is not None
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    out["starter_status"] = starter_status
    out["source"] = "rotowire"
    out["source_columns"] = ", ".join(original_cols)

    out = out.dropna(subset=["pitcher_name", "team"], how="any")
    out = out.drop_duplicates().reset_index(drop=True)
    return out