from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

SOURCE_NAME = "rotowire"
SOURCE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"
SOURCE_URL_FALLBACK = "https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow"

POS_SET = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "OF"}
HAND_SET = {"L", "R", "S"}

TEAM_NAME_TO_ABBR = {
    "angels": "LAA",
    "astros": "HOU",
    "athletics": "ATH",
    "ath": "ATH",
    "blue jays": "TOR",
    "braves": "ATL",
    "brewers": "MIL",
    "cardinals": "STL",
    "cubs": "CHC",
    "diamondbacks": "ARI",
    "dodgers": "LAD",
    "giants": "SF",
    "guardians": "CLE",
    "mariners": "SEA",
    "marlins": "MIA",
    "mets": "NYM",
    "nationals": "WSH",
    "orioles": "BAL",
    "padres": "SD",
    "phillies": "PHI",
    "pirates": "PIT",
    "rangers": "TEX",
    "rays": "TB",
    "reds": "CIN",
    "red sox": "BOS",
    "rockies": "COL",
    "royals": "KC",
    "tigers": "DET",
    "twins": "MIN",
    "white sox": "CWS",
    "yankees": "NYY",
}

ABBR_TO_CANONICAL = {
    "ARI": "ARIZONA DIAMONDBACKS",
    "ATL": "ATLANTA BRAVES",
    "BAL": "BALTIMORE ORIOLES",
    "BOS": "BOSTON RED SOX",
    "CHC": "CHICAGO CUBS",
    "CWS": "CHICAGO WHITE SOX",
    "CIN": "CINCINNATI REDS",
    "CLE": "CLEVELAND GUARDIANS",
    "COL": "COLORADO ROCKIES",
    "DET": "DETROIT TIGERS",
    "HOU": "HOUSTON ASTROS",
    "KC": "KANSAS CITY ROYALS",
    "LAA": "LOS ANGELES ANGELS",
    "LAD": "LOS ANGELES DODGERS",
    "MIA": "MIAMI MARLINS",
    "MIL": "MILWAUKEE BREWERS",
    "MIN": "MINNESOTA TWINS",
    "NYM": "NEW YORK METS",
    "NYY": "NEW YORK YANKEES",
    "ATH": "ATHLETICS",
    "OAK": "ATHLETICS",
    "PHI": "PHILADELPHIA PHILLIES",
    "PIT": "PITTSBURGH PIRATES",
    "SD": "SAN DIEGO PADRES",
    "SF": "SAN FRANCISCO GIANTS",
    "SFG": "SAN FRANCISCO GIANTS",
    "SEA": "SEATTLE MARINERS",
    "STL": "ST. LOUIS CARDINALS",
    "TB": "TAMPA BAY RAYS",
    "TBR": "TAMPA BAY RAYS",
    "TEX": "TEXAS RANGERS",
    "TOR": "TORONTO BLUE JAYS",
    "WSH": "WASHINGTON NATIONALS",
}

NAME_TO_CANONICAL = {
    "ARIZONA DIAMONDBACKS": "ARIZONA DIAMONDBACKS",
    "ATLANTA BRAVES": "ATLANTA BRAVES",
    "BALTIMORE ORIOLES": "BALTIMORE ORIOLES",
    "BOSTON RED SOX": "BOSTON RED SOX",
    "CHICAGO CUBS": "CHICAGO CUBS",
    "CHICAGO WHITE SOX": "CHICAGO WHITE SOX",
    "CINCINNATI REDS": "CINCINNATI REDS",
    "CLEVELAND GUARDIANS": "CLEVELAND GUARDIANS",
    "COLORADO ROCKIES": "COLORADO ROCKIES",
    "DETROIT TIGERS": "DETROIT TIGERS",
    "HOUSTON ASTROS": "HOUSTON ASTROS",
    "KANSAS CITY ROYALS": "KANSAS CITY ROYALS",
    "LOS ANGELES ANGELS": "LOS ANGELES ANGELS",
    "LOS ANGELES DODGERS": "LOS ANGELES DODGERS",
    "MIAMI MARLINS": "MIAMI MARLINS",
    "MILWAUKEE BREWERS": "MILWAUKEE BREWERS",
    "MINNESOTA TWINS": "MINNESOTA TWINS",
    "NEW YORK METS": "NEW YORK METS",
    "NEW YORK YANKEES": "NEW YORK YANKEES",
    "ATHLETICS": "ATHLETICS",
    "PHILADELPHIA PHILLIES": "PHILADELPHIA PHILLIES",
    "PITTSBURGH PIRATES": "PITTSBURGH PIRATES",
    "SAN DIEGO PADRES": "SAN DIEGO PADRES",
    "SAN FRANCISCO GIANTS": "SAN FRANCISCO GIANTS",
    "SEATTLE MARINERS": "SEATTLE MARINERS",
    "ST. LOUIS CARDINALS": "ST. LOUIS CARDINALS",
    "ST LOUIS CARDINALS": "ST. LOUIS CARDINALS",
    "TAMPA BAY RAYS": "TAMPA BAY RAYS",
    "TEXAS RANGERS": "TEXAS RANGERS",
    "TORONTO BLUE JAYS": "TORONTO BLUE JAYS",
    "WASHINGTON NATIONALS": "WASHINGTON NATIONALS",
}

STATUS_LABELS = {"Expected Lineup", "Confirmed Lineup", "Unknown Lineup"}
JUNK_PATTERNS = [
    r"^Expected Lineup$",
    r"^Confirmed Lineup$",
    r"^Unknown Lineup$",
    r"^Starting Pitcher Intel$",
    r"^Pitcher Intel$",
    r"^Watch Now$",
    r"^Tickets$",
    r"^Alerts$",
    r"^LINE$",
    r"^O/U$",
    r"^Umpire:",
    r"lineup has not been posted yet",
    r"^\d{1,2}:\d{2}\s*(AM|PM)\s*ET$",
    r"^\d+$",
]
JUNK_RE = re.compile("|".join(f"(?:{p})" for p in JUNK_PATTERNS), flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape projected lineups for live props.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    cset = set(cols)
    for c in candidates:
        if c in cset:
            return c
    return None


def _norm_name(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _normalize_text(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _clean_player_name(s: str) -> str | None:
    s = _normalize_text(s)
    if not s:
        return None

    s = re.sub(r"^NONE\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(?:Expected Lineup|Confirmed Lineup|Unknown Lineup)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(?:SP|RP|P|C|1B|2B|3B|SS|LF|CF|RF|OF|DH)\s+", "", s)
    s = re.sub(r"\s+\b(?:L|R|S)\b$", "", s)
    s = re.sub(r"^\d{1,2}\.?\s*", "", s)
    s = _normalize_text(s).strip(" -–|")

    if not s:
        return None
    if JUNK_RE.search(s):
        return None
    if len(s.split()) < 2:
        return None
    if _to_canonical_team(s) is not None:
        return None
    return s


def _extract_position_and_name(raw_text: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse strings like:
    - 'RF B. Nimmo L'
    - '1 RF Brandon Nimmo L'
    - 'Brandon Nimmo RF L'
    - 'NONE RF B. Nimmo'
    """
    t = _normalize_text(raw_text)
    if not t:
        return None, None, None

    t = re.sub(r"^NONE\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\d{1,2}\.?\s+", "", t)
    t = _normalize_text(t)

    if not t or JUNK_RE.search(t):
        return None, None, None
    if "pitcher intel" in t.lower():
        return None, None, None
    if "era" in t.lower() and len(t.split()) <= 6:
        return None, None, None

    tokens = t.split()
    bats = None

    if tokens and tokens[-1].upper() in HAND_SET:
        bats = tokens[-1].upper()
        tokens = tokens[:-1]

    if not tokens:
        return None, None, None

    pos = None

    if tokens and tokens[0].upper() in POS_SET:
        pos = tokens[0].upper()
        tokens = tokens[1:]
    elif len(tokens) >= 2 and tokens[-1].upper() in POS_SET:
        pos = tokens[-1].upper()
        tokens = tokens[:-1]

    name = " ".join(tokens).strip()
    name = _clean_player_name(name)
    if not name:
        return None, None, None

    return pos, name, bats


def _resolve_player_ids(out: pd.DataFrame, batter_path: Path, slate_date: pd.Timestamp) -> pd.DataFrame:
    if not batter_path.exists() or out.empty:
        return out

    batter = pd.read_parquet(batter_path).copy()
    team_col = _pick(list(batter.columns), ["batter_team", "team", "team_abbrev", "team_name", "batting_team"])
    bid_col = _pick(list(batter.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(batter.columns), ["player_name", "batter_name", "name"])

    if team_col is None or bid_col is None or name_col is None:
        logging.warning("projected_lineups id resolver skipped missing columns batter_cols=%s", sorted(batter.columns))
        return out

    batter["game_date"] = pd.to_datetime(batter.get("game_date"), errors="coerce")
    batter = batter[batter["game_date"] < slate_date].copy()
    if batter.empty:
        return out

    batter["canonical_team"] = batter[team_col].map(_to_canonical_team)
    batter["batter_id"] = pd.to_numeric(batter[bid_col], errors="coerce").astype("Int64")
    batter["name_norm"] = _norm_name(batter[name_col])

    lookup = (
        batter.sort_values(["game_date"])
        .dropna(subset=["batter_id", "canonical_team"])
        .drop_duplicates(subset=["canonical_team", "name_norm"], keep="last")[["canonical_team", "name_norm", "batter_id"]]
    )

    out["name_norm"] = _norm_name(out["player_name"])
    fill_mask = out["batter_id"].isna()
    if fill_mask.any():
        resolved = out.loc[fill_mask, ["canonical_team", "name_norm"]].merge(
            lookup,
            on=["canonical_team", "name_norm"],
            how="left",
        )["batter_id"]
        out.loc[fill_mask, "batter_id"] = pd.to_numeric(resolved, errors="coerce").astype("Int64").values

    return out.drop(columns=["name_norm"], errors="ignore")


def _team_aliases(spine: pd.DataFrame) -> dict[str, str]:
    aliases: dict[str, str] = {}

    for col in ["home_team", "away_team"]:
        if col not in spine.columns:
            continue
        vals = spine[col].dropna().astype(str).str.upper().unique().tolist()
        for v in vals:
            aliases[v] = v

    for k, v in TEAM_NAME_TO_ABBR.items():
        aliases[k.upper()] = v
        aliases[k.lower()] = v

    for abbr in ABBR_TO_CANONICAL:
        aliases[abbr] = abbr

    aliases["A'S"] = "ATH"
    aliases["ATHLETICS"] = "ATH"
    aliases["ATH"] = "ATH"
    aliases["OAK"] = "ATH"
    aliases["OAKLAND"] = "ATH"

    return aliases


def _extract_team_abbr(team_token: str, aliases: dict[str, str]) -> str | None:
    raw = _normalize_text(team_token)
    if not raw:
        return None

    upper = raw.upper()
    lower = raw.lower()

    if upper in aliases:
        val = aliases[upper]
        return val if len(val) <= 4 else None

    if lower in aliases:
        val = aliases[lower]
        return val if len(val) <= 4 else None

    for k, v in TEAM_NAME_TO_ABBR.items():
        if k in lower:
            return v

    return None


def _to_canonical_team(team_token: str) -> str | None:
    tok = _normalize_text(team_token).upper()
    if not tok:
        return None

    if tok in ABBR_TO_CANONICAL:
        return ABBR_TO_CANONICAL[tok]
    if tok in NAME_TO_CANONICAL:
        return NAME_TO_CANONICAL[tok]

    tok_clean = re.sub(r"[^A-Z ]+", " ", tok)
    tok_clean = re.sub(r"\s+", " ", tok_clean).strip()

    if tok_clean in ABBR_TO_CANONICAL:
        return ABBR_TO_CANONICAL[tok_clean]
    if tok_clean in NAME_TO_CANONICAL:
        return NAME_TO_CANONICAL[tok_clean]

    for k, abbr in TEAM_NAME_TO_ABBR.items():
        if k.upper() in tok_clean:
            return ABBR_TO_CANONICAL.get(abbr)

    return None


def _fetch_rotowire_html(url: str) -> tuple[str, dict[str, object]]:
    resp = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    resp.raise_for_status()
    html = resp.text
    content_len = len(html)

    checks = {
        "Expected Lineup": "Expected Lineup" in html,
        "Starting Pitcher Intel": "Starting Pitcher Intel" in html,
        "NYY": "NYY" in html,
        "SF": "SF" in html,
        "Aaron Judge": "Aaron Judge" in html,
    }

    logging.info(
        "rotowire fetch url=%s status=%s final_url=%s content_length=%s key_checks=%s",
        url,
        resp.status_code,
        resp.url,
        content_len,
        checks,
    )
    return html, {
        "status_code": resp.status_code,
        "final_url": resp.url,
        "content_length": content_len,
        "contains_expected_lineup": checks["Expected Lineup"],
        "contains_starting_pitcher_intel": checks["Starting Pitcher Intel"],
        "contains_nyy": checks["NYY"],
        "contains_sf": checks["SF"],
        "contains_aaron_judge": checks["Aaron Judge"],
    }


def _iter_visible_lines(soup: BeautifulSoup) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()

    for s in soup.stripped_strings:
        t = _normalize_text(s)
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        lines.append(t)

    return lines


def _maybe_team_token(s: str, aliases: dict[str, str]) -> str | None:
    abbr = _extract_team_abbr(s, aliases)
    if abbr is not None:
        return abbr

    for token in re.split(r"\s+|/|-|\|", s):
        abbr = _extract_team_abbr(token, aliases)
        if abbr is not None:
            return abbr

    return None


def _parse_visible_text_lineups(lines: list[str], aliases: dict[str, str]) -> pd.DataFrame:
    """
    Rotowire text usually contains:
      TEAM
      Expected Lineup
      RF Brandon Nimmo L
      LF Wyatt Langford R
      ...
    This parser anchors on the lineup status marker, finds the nearest prior team token,
    then reads forward until it has 9 hitters or hits a boundary.
    """
    rows: list[dict[str, object]] = []
    rows_by_team: dict[str, int] = {}
    rejected_sample: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line not in STATUS_LABELS:
            i += 1
            continue

        team_abbr = None
        for back in range(1, 9):
            j = i - back
            if j < 0:
                break
            team_abbr = _maybe_team_token(lines[j], aliases)
            if team_abbr is not None:
                break

        if team_abbr is None:
            i += 1
            continue

        slot = 0
        j = i + 1
        while j < len(lines) and slot < 9:
            candidate = lines[j]
            low = candidate.lower()

            boundary = (
                candidate in STATUS_LABELS
                or candidate.startswith("Umpire:")
                or "lineup has not been posted yet" in low
                or "starting pitcher intel" in low
                or "pitcher intel" in low
                or re.search(r"\b\d{1,2}:\d{2}\s*(AM|PM)\s*ET\b", candidate, flags=re.IGNORECASE)
            )
            if boundary:
                break

            pos, name, bats = _extract_position_and_name(candidate)
            if name:
                slot += 1
                rows.append(
                    {
                        "batter_team": team_abbr,
                        "player_name": name,
                        "lineup_slot": slot,
                        "position": pos,
                        "bats": bats,
                        "source_text": candidate,
                    }
                )
                rows_by_team[team_abbr] = rows_by_team.get(team_abbr, 0) + 1
            else:
                if len(rejected_sample) < 25:
                    rejected_sample.append(candidate)

            j += 1

        i = j

    logging.info(
        "rotowire visible-text parser rows=%s teams=%s rows_by_team=%s rejected_sample=%s",
        len(rows),
        len(rows_by_team),
        rows_by_team,
        rejected_sample[:15],
    )
    return pd.DataFrame(rows)


def _parse_rotowire_cards(soup: BeautifulSoup, aliases: dict[str, str]) -> pd.DataFrame:
    """
    Secondary parser that tries to work from card-like DOM chunks.
    We still aggressively clean the resulting text before trusting it.
    """
    rows: list[dict[str, object]] = []

    card_selectors = [
        "div.lineup.is-mlb",
        "div.lineup",
        "div[class*='lineup__box']",
        "div[class*='lineup-card']",
        "section[class*='lineup']",
        "div[class*='Lineup']",
    ]

    cards: list[object] = []
    for sel in card_selectors:
        cards.extend(list(soup.select(sel)))

    seen = set()
    uniq_cards = []
    for c in cards:
        cid = id(c)
        if cid not in seen:
            seen.add(cid)
            uniq_cards.append(c)

    raw_nodes_scanned = 0

    for card in uniq_cards:
        card_lines = [_normalize_text(t) for t in card.stripped_strings if _normalize_text(t)]
        if not card_lines:
            continue

        team_candidates = []
        for t in card_lines[:20]:
            abbr = _maybe_team_token(t, aliases)
            if abbr is not None:
                team_candidates.append(abbr)

        if not team_candidates:
            continue

        team_abbr = team_candidates[0]

        slot = 0
        for candidate in card_lines:
            if slot >= 9:
                break

            pos, name, bats = _extract_position_and_name(candidate)
            raw_nodes_scanned += 1
            if not name:
                continue

            slot += 1
            rows.append(
                {
                    "batter_team": team_abbr,
                    "player_name": name,
                    "lineup_slot": slot,
                    "position": pos,
                    "bats": bats,
                    "source_text": candidate,
                }
            )

    logging.info(
        "rotowire card parser raw_nodes_scanned=%s rows=%s game_cards=%s",
        raw_nodes_scanned,
        len(rows),
        len(uniq_cards),
    )
    return pd.DataFrame(rows)


def _dedupe_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["player_name"] = out["player_name"].map(_clean_player_name)
    out = out.dropna(subset=["batter_team", "player_name"]).copy()

    out["batter_team"] = out["batter_team"].astype(str).str.upper()
    out["position"] = out["position"].astype("object")
    out["bats"] = out["bats"].astype("object")

    # keep first valid occurrence within team+slot and team+player
    out = out.sort_values(["batter_team", "lineup_slot", "player_name"]).copy()
    out = out.drop_duplicates(subset=["batter_team", "lineup_slot"], keep="first")
    out = out.drop_duplicates(subset=["batter_team", "player_name"], keep="first")

    # re-number lineup slots cleanly after dedupe
    out = out.sort_values(["batter_team", "lineup_slot", "player_name"]).copy()
    out["lineup_slot"] = out.groupby("batter_team").cumcount() + 1
    out = out[out["lineup_slot"] <= 9].copy()

    # require at least 7 hitters or drop the team entirely
    counts = out.groupby("batter_team").size().to_dict()
    bad_teams = [team for team, n in counts.items() if n < 7]
    if bad_teams:
        logging.warning("rotowire dropping incomplete teams teams=%s counts=%s", bad_teams, counts)
        out = out[~out["batter_team"].isin(bad_teams)].copy()

    final_counts = out.groupby("batter_team").size().to_dict() if len(out) else {}
    logging.info("rotowire final hitter counts by team=%s", final_counts)

    bad_mask = out["player_name"].fillna("").str.contains(
        r"Expected Lineup|Confirmed Lineup|Unknown Lineup|Starting Pitcher Intel|^NONE\b",
        case=False,
        regex=True,
    )
    if bad_mask.any():
        raise ValueError(
            f"Projected lineups still contain junk rows: "
            f"{out.loc[bad_mask, ['batter_team', 'player_name']].head(20).to_dict('records')}"
        )

    return out


def _scrape_rotowire_projected(url: str, aliases: dict[str, str]) -> tuple[pd.DataFrame, str, dict[str, object]]:
    html, fetch_diag = _fetch_rotowire_html(url)
    soup = BeautifulSoup(html, "html.parser")
    lines = _iter_visible_lines(soup)

    game_container_selectors = [
        "div.lineup.is-mlb",
        "div.lineup",
        "div[class*='lineup__box']",
        "div[class*='lineup-card']",
        "section[class*='lineup']",
        "div[class*='Lineup']",
    ]
    game_container_count = len({id(node) for sel in game_container_selectors for node in soup.select(sel)})

    visible_df = _parse_visible_text_lineups(lines, aliases)
    card_df = _parse_rotowire_cards(soup, aliases)

    if len(visible_df) >= len(card_df):
        raw = visible_df.copy()
        parser_used = "visible_text"
    else:
        raw = card_df.copy()
        parser_used = "cards"

    raw = _dedupe_and_validate(raw)

    diag = {
        **fetch_diag,
        "game_container_count": game_container_count,
        "parsed_rows": int(len(raw)),
        "parsed_teams": sorted(raw["batter_team"].dropna().astype(str).unique().tolist()) if len(raw) else [],
        "parser_used": parser_used,
        "visible_contains_expected_lineup": "Expected Lineup" in "\n".join(lines),
        "visible_contains_starting_pitcher_intel": "Starting Pitcher Intel" in "\n".join(lines),
    }
    logging.info(
        "rotowire parser diagnostics parser_used=%s parsed_rows=%s parsed_teams=%s game_container_count=%s",
        parser_used,
        len(raw),
        diag["parsed_teams"],
        game_container_count,
    )
    return raw, html, diag


def main() -> None:
    args = parse_args()
    slate_date = pd.to_datetime(args.date, errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_projected_lineups.log")
    log_header("scripts/live/build_projected_lineups.py", repo_root, config_path, dirs)

    spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(f"Live spine not found: {spine_path}")

    spine = pd.read_parquet(spine_path).copy()
    spine["game_pk"] = pd.to_numeric(spine.get("game_pk"), errors="coerce").astype("Int64")

    team_games = pd.concat(
        [
            spine[["game_pk", "away_team", "home_team"]].rename(
                columns={"away_team": "batter_team", "home_team": "opponent_team"}
            ),
            spine[["game_pk", "home_team", "away_team"]].rename(
                columns={"home_team": "batter_team", "away_team": "opponent_team"}
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    team_games["batter_team"] = team_games["batter_team"].astype(str).str.upper()
    team_games["canonical_team"] = team_games["batter_team"].map(_to_canonical_team)

    aliases = _team_aliases(spine)

    raw, html_used, scrape_diag = _scrape_rotowire_projected(SOURCE_URL, aliases)
    used_url = SOURCE_URL

    if raw.empty:
        logging.warning("rotowire primary URL parsed zero rows; retrying fallback URL=%s", SOURCE_URL_FALLBACK)
        raw, html_used, scrape_diag = _scrape_rotowire_projected(SOURCE_URL_FALLBACK, aliases)
        used_url = SOURCE_URL_FALLBACK

    html_snapshot_path = dirs["logs_dir"] / f"projected_lineups_rotowire_{args.season}_{args.date}.html"
    html_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    html_snapshot_path.write_text(html_used, encoding="utf-8")
    logging.info("rotowire html_snapshot=%s", html_snapshot_path)
    logging.info("rotowire url_used_for_rows=%s parsed_rows=%s", used_url, len(raw))

    raw["canonical_team"] = raw["batter_team"].map(_to_canonical_team)

    scraped_teams = set(raw["batter_team"].dropna().astype(str).str.upper().unique().tolist()) if len(raw) else set()
    scraped_canonical = set(raw["canonical_team"].dropna().astype(str).unique().tolist()) if len(raw) else set()
    slate_teams = set(team_games["batter_team"].dropna().astype(str).str.upper().unique().tolist()) if len(team_games) else set()
    slate_canonical = set(team_games["canonical_team"].dropna().astype(str).unique().tolist()) if len(team_games) else set()

    logging.info(
        "rotowire team diagnostics parsed_raw_teams=%s parsed_canonical_teams=%s slate_raw_teams=%s slate_canonical_teams=%s",
        sorted(scraped_teams),
        sorted(scraped_canonical),
        sorted(slate_teams),
        sorted(slate_canonical),
    )

    filtered = raw[raw["canonical_team"].isin(slate_canonical)].copy()

    if len(raw) > 0 and filtered.empty:
        raise ValueError(
            "Rotowire parsed rows but slate filtering removed all rows. "
            f"parsed_raw_teams={sorted(scraped_teams)} "
            f"parsed_canonical_teams={sorted(scraped_canonical)} "
            f"slate_canonical_teams={sorted(slate_canonical)}"
        )

    out = filtered.merge(
        team_games[["game_pk", "batter_team", "canonical_team"]].drop_duplicates(),
        on="canonical_team",
        how="inner",
        suffixes=("_scraped", ""),
    )

    out["game_date"] = args.date
    out["batter_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["lineup_status"] = "projected"
    out["lineup_source"] = SOURCE_NAME
    out["source_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out = _resolve_player_ids(out, dirs["processed_dir"] / "batter_game_rolling.parquet", slate_date)

    out = out[
        [
            "game_pk",
            "game_date",
            "batter_team",
            "canonical_team",
            "player_name",
            "batter_id",
            "lineup_slot",
            "position",
            "bats",
            "lineup_status",
            "lineup_source",
            "source_timestamp",
        ]
    ].drop_duplicates(subset=["game_pk", "batter_team", "player_name"], keep="first")

    out = out.sort_values(["game_pk", "batter_team", "lineup_slot", "player_name"]).copy()
    out["lineup_slot"] = out.groupby(["game_pk", "batter_team"]).cumcount() + 1
    out = out[out["lineup_slot"] <= 9].copy()

    rows_per_team = out.groupby(["game_pk", "batter_team"]).size().to_dict() if len(out) else {}
    nonnull_position = int(out["position"].notna().sum()) if len(out) else 0
    nonnull_batter_id = int(out["batter_id"].notna().sum()) if len(out) else 0

    logging.info(
        "projected_lineups prewrite total_rows=%s unique_games=%s unique_teams=%s nonnull_position=%s nonnull_batter_id=%s rows_per_team=%s sample_rows=%s",
        len(out),
        int(out["game_pk"].nunique()) if len(out) else 0,
        int(out["batter_team"].nunique()) if len(out) else 0,
        nonnull_position,
        nonnull_batter_id,
        rows_per_team,
        out.head(20).to_dict(orient="records"),
    )

    bad_mask = out["player_name"].fillna("").str.contains(
        r"Expected Lineup|Confirmed Lineup|Unknown Lineup|Starting Pitcher Intel|^NONE\b",
        case=False,
        regex=True,
    )
    if bad_mask.any():
        raise ValueError(
            f"Projected lineups still contain junk rows: "
            f"{out.loc[bad_mask, ['batter_team', 'player_name']].head(20).to_dict('records')}"
        )

    out_path = dirs["processed_dir"] / "live" / f"projected_lineups_{args.season}_{args.date}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    logging.info(
        "projected_lineups rows=%s teams=%s games=%s source=%s parser_used=%s out=%s",
        len(out),
        int(out["batter_team"].nunique()) if len(out) else 0,
        int(out["game_pk"].nunique()) if len(out) else 0,
        used_url,
        scrape_diag.get("parser_used"),
        out_path,
    )
    print(f"projected_lineups_out={out_path}")


if __name__ == "__main__":
    main()
