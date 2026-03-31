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
POS_SET = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}
UI_BLOCKLIST = {"watch now", "tickets", "alerts"}
TEAM_NAME_TO_ABBR = {
    "angels": "LAA",
    "astros": "HOU",
    "athletics": "ATH",
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
    "ATH": "OAKLAND ATHLETICS",
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
    "OAKLAND ATHLETICS": "OAKLAND ATHLETICS",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scrape projected lineups for live hit props.")
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


def _resolve_player_ids(out: pd.DataFrame, batter_path: Path, slate_date: pd.Timestamp) -> pd.DataFrame:
    if not batter_path.exists() or out.empty:
        return out
    batter = pd.read_parquet(batter_path).copy()
    team_col = _pick(list(batter.columns), ["batter_team", "team", "team_abbrev", "team_name", "batting_team"])
    bid_col = _pick(list(batter.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(batter.columns), ["player_name", "batter_name", "name"])
    if team_col is None or bid_col is None or name_col is None:
        return out
    batter["game_date"] = pd.to_datetime(batter.get("game_date"), errors="coerce")
    batter = batter[batter["game_date"] < slate_date].copy()
    if batter.empty:
        return out

    batter["batter_team"] = batter[team_col].astype(str).str.upper()
    batter["batter_id"] = pd.to_numeric(batter[bid_col], errors="coerce").astype("Int64")
    batter["name_norm"] = _norm_name(batter[name_col])
    lookup = (
        batter.sort_values(["game_date"])
        .dropna(subset=["batter_id"])
        .drop_duplicates(subset=["batter_team", "name_norm"], keep="last")[["batter_team", "name_norm", "batter_id"]]
    )
    out["name_norm"] = _norm_name(out["player_name"])
    fill = out["batter_id"].isna()
    if fill.any():
        resolved = out.loc[fill, ["batter_team", "name_norm"]].merge(lookup, on=["batter_team", "name_norm"], how="left")["batter_id"]
        out.loc[fill, "batter_id"] = pd.to_numeric(resolved, errors="coerce").astype("Int64").values
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
        aliases[k] = v
    return aliases


def _extract_team_abbr(team_token: str, aliases: dict[str, str]) -> str | None:
    raw = team_token.strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in aliases:
        return aliases[upper]
    low = raw.lower()
    if low in aliases:
        return aliases[low]
    for k, v in TEAM_NAME_TO_ABBR.items():
        if k in low:
            return v
    return None


def _to_canonical_team(team_token: str) -> str | None:
    tok = str(team_token).strip().upper()
    if not tok:
        return None
    if tok in ABBR_TO_CANONICAL:
        return ABBR_TO_CANONICAL[tok]
    if tok in NAME_TO_CANONICAL:
        return NAME_TO_CANONICAL[tok]
    tok_clean = re.sub(r"[^A-Z ]+", " ", tok)
    tok_clean = re.sub(r"\s+", " ", tok_clean).strip()
    if tok_clean in NAME_TO_CANONICAL:
        return NAME_TO_CANONICAL[tok_clean]
    return None


def _fetch_rotowire_html(url: str) -> tuple[str, dict[str, object]]:
    resp = requests.get(url, timeout=30)
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
    if content_len < 50_000:
        logging.warning("rotowire fetch content appears short content_length=%s", content_len)
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


def _parse_team_from_text(text: str, aliases: dict[str, str]) -> str | None:
    for token in re.split(r"\s+|\||-|/", text):
        abbr = _extract_team_abbr(token.strip(), aliases)
        if abbr:
            return abbr
    return _extract_team_abbr(text, aliases)


def _parse_rotowire_cards(soup: BeautifulSoup, aliases: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    card_selectors = [
        "div.lineup.is-mlb",
        "div.lineup",
        "div[class*='lineup__box']",
        "div[class*='lineup-card']",
        "section[class*='lineup']",
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
    lineup_nodes_found = 0
    for card in uniq_cards:
        card_lines = [t.strip() for t in card.stripped_strings if t and t.strip()]
        if not card_lines:
            continue
        team = _parse_team_from_text(" ".join(card_lines[:12]), aliases)
        if team is None:
            continue
        lineup_nodes = card.select(
            "ul[class*='lineup'] li, ol[class*='lineup'] li, div[class*='lineup__list'] div, div[class*='lineup__player'], table[class*='lineup'] tr"
        )
        raw_nodes_scanned += len(lineup_nodes)
        lineup_nodes_found += len(lineup_nodes)
        slot = 0
        for node in lineup_nodes:
            line = " ".join([s.strip() for s in node.stripped_strings if s and s.strip()])
            if not line:
                continue
            if "starting pitcher" in line.lower() or "pitcher intel" in line.lower() or "era" in line.lower():
                continue
            if slot >= 9:
                break
            slot += 1
            rows.append({"batter_team": team, "player_name": line, "lineup_slot": slot, "position": None})
    logging.info("rotowire parser nodes raw_nodes_scanned=%s lineup_nodes_found=%s game_cards=%s", raw_nodes_scanned, lineup_nodes_found, len(uniq_cards))
    return pd.DataFrame(rows)


def _parse_hitter_row_text(raw_text: str) -> tuple[str | None, str, str | None] | None:
    t = str(raw_text).strip()
    if not t:
        return None
    t_low = t.lower()
    if any(x in t_low for x in UI_BLOCKLIST):
        return None
    if "lineup has not been posted" in t_low:
        return None
    if "era" in t_low:
        return None
    if re.search(r"\(\d{1,3}-\d{1,3}\)", t_low):
        return None
    if re.search(r"\b\d{1,2}:\d{2}\s*(AM|PM)\s*ET\b", t, flags=re.IGNORECASE):
        return None
    if " pm " in f" {t_low} " or " am " in f" {t_low} " or " et" in f" {t_low} ":
        return None
    if re.fullmatch(r"\d+", t):
        return None
    if re.fullmatch(r"[A-Z]{2,4}", t):
        return None

    # remove leading batting-order integer if present
    t = re.sub(r"^\d{1,2}\s+", "", t).strip()
    if not t:
        return None

    pos: str | None = None
    toks = t.split()
    if toks and toks[0].upper() in POS_SET:
        pos = toks[0].upper()
        toks = toks[1:]
    if not toks:
        return None

    bats: str | None = None
    if toks and toks[-1].upper() in {"L", "R", "S"}:
        bats = toks[-1].upper()
        toks = toks[:-1]
    if not toks:
        return None

    name = " ".join(toks).strip()
    if len(name.split()) < 2:
        return None
    if any(x in name.lower() for x in UI_BLOCKLIST):
        return None
    if re.fullmatch(r"\d+", name):
        return None
    if re.fullmatch(r"[A-Z ]{2,}", name) and len(name.split()) <= 3:
        return None
    if _to_canonical_team(name) is not None:
        return None
    return pos, name, bats


def _filter_valid_hitter_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    raw_counts = df.groupby("batter_team").size().to_dict()
    parsed_rows: list[dict[str, object]] = []
    rejected: list[str] = []
    for _, row in df.iterrows():
        team = str(row.get("batter_team", "")).strip().upper()
        pos = str(row.get("position", "")).strip().upper()
        name = str(row.get("player_name", "")).strip()
        bats_existing = str(row.get("bats", "")).strip().upper()
        text = f"{pos} {name}".strip() if pos else name
        parsed = _parse_hitter_row_text(text)
        if parsed is None:
            rejected.append(text)
            continue
        p_pos, p_name, p_bats = parsed
        parsed_rows.append(
            {
                "batter_team": team,
                "position": p_pos,
                "player_name": p_name,
                "bats": p_bats or (bats_existing if bats_existing in {"L", "R", "S"} else None),
            }
        )
    clean = pd.DataFrame(parsed_rows)
    if clean.empty:
        logging.info("rotowire hitter row diagnostics raw_rows_by_team=%s kept_rows_by_team=%s rejected_sample=%s final_rows_by_team=%s", raw_counts, {}, rejected[:10], {})
        return clean

    clean = clean.dropna(subset=["batter_team", "player_name"]).copy()
    clean["position"] = clean["position"].where(clean["position"].isin(POS_SET), None)
    clean["lineup_slot"] = clean.groupby("batter_team").cumcount() + 1
    clean = clean[clean["lineup_slot"] <= 9].copy()
    filtered_counts = clean.groupby("batter_team").size().to_dict()
    bad_teams = [team for team, ct in filtered_counts.items() if ct < 5]
    if bad_teams:
        logging.warning("rotowire dropping teams with <5 valid hitters teams=%s", bad_teams)
        clean = clean[~clean["batter_team"].isin(bad_teams)].copy()
        filtered_counts = clean.groupby("batter_team").size().to_dict() if len(clean) else {}
    final_counts = clean.groupby("batter_team").size().to_dict()

    logging.info(
        "rotowire hitter row diagnostics raw_rows_by_team=%s kept_rows_by_team=%s rejected_sample=%s final_rows_by_team=%s",
        raw_counts,
        filtered_counts,
        rejected[:10],
        final_counts,
    )
    return clean


def _scrape_rotowire_projected(url: str, aliases: dict[str, str]) -> tuple[pd.DataFrame, str, dict[str, object]]:
    html, fetch_diag = _fetch_rotowire_html(url)
    soup = BeautifulSoup(html, "html.parser")
    lines = [s.strip() for s in soup.stripped_strings if s and s.strip()]
    visible_text = "\n".join(lines)

    game_container_selectors = [
        "div.lineup.is-mlb",
        "div.lineup",
        "div[class*='lineup__box']",
        "div[class*='lineup-card']",
        "section[class*='lineup']",
    ]
    team_container_selectors = [
        "div.lineup__team",
        "div[class*='lineup__team']",
        "div[class*='team']",
        "span[class*='team']",
    ]
    game_container_count = len({id(node) for sel in game_container_selectors for node in soup.select(sel)})
    team_container_count = len({id(node) for sel in team_container_selectors for node in soup.select(sel)})

    rows: list[dict[str, object]] = []
    current_team: str | None = None
    pending_pos: str | None = None
    current_slot = 0
    current_game = -1

    matchup_pat = re.compile(r"^[A-Za-z .'-]+ \(\d+-\d+\)\s+[A-Za-z .'-]+ \(\d+-\d+\)$")
    hand_pat = re.compile(r"^[RLS]$")
    status_pat = {"Unknown Lineup", "Expected Lineup", "Confirmed Lineup"}
    team_lineup_row_counts: dict[str, int] = {}
    game_lineup_row_counts: dict[int, int] = {}

    for idx, line in enumerate(lines):
        if matchup_pat.match(line):
            current_game += 1
            current_team = None
            pending_pos = None
            current_slot = 0
            continue
        if line in status_pat:
            # team abbreviation usually appears in the few prior tokens
            for back in range(1, 8):
                j = idx - back
                if j < 0:
                    break
                token = lines[j]
                abbr = _extract_team_abbr(token, aliases)
                if abbr is not None and len(abbr) <= 4:
                    current_team = abbr
                    pending_pos = None
                    current_slot = 0
                    break
            continue

        if current_team is None:
            continue
        if line in POS_SET:
            pending_pos = line
            continue
        if hand_pat.match(line):
            continue
        if line.startswith("The ") and "lineup has not been posted yet" in line:
            current_team = None
            pending_pos = None
            current_slot = 0
            continue
        if line.startswith("Umpire:"):
            current_team = None
            pending_pos = None
            current_slot = 0
            continue
        if line in {"LINE", "O/U"} or line.startswith("Wind ") or " mph " in line:
            continue
        if current_slot >= 9:
            continue

        current_slot += 1
        rows.append(
            {
                "batter_team": current_team,
                "player_name": line,
                "lineup_slot": current_slot,
                "position": pending_pos,
            }
        )
        team_lineup_row_counts[current_team] = team_lineup_row_counts.get(current_team, 0) + 1
        if current_game >= 0:
            game_lineup_row_counts[current_game] = game_lineup_row_counts.get(current_game, 0) + 1
        pending_pos = None

    out = _parse_rotowire_cards(soup, aliases)
    if out.empty:
        out = pd.DataFrame(rows)
    out = _filter_valid_hitter_rows(out)
    out = out.dropna(subset=["batter_team", "player_name"]).copy()
    out["batter_team"] = out["batter_team"].astype(str).str.upper()
    parsed_teams = sorted(out["batter_team"].dropna().astype(str).unique().tolist()) if len(out) else []
    logging.info(
        "rotowire parser diagnostics game_container_count=%s team_container_count=%s parsed_row_count=%s parsed_team_count=%s lineup_rows_by_team=%s lineup_rows_by_game=%s",
        game_container_count,
        team_container_count,
        len(out),
        len(parsed_teams),
        team_lineup_row_counts,
        game_lineup_row_counts,
    )
    logging.info("rotowire parser parsed_teams=%s", parsed_teams)

    diag = {
        **fetch_diag,
        "game_container_count": game_container_count,
        "team_container_count": team_container_count,
        "lineup_rows_by_team": team_lineup_row_counts,
        "lineup_rows_by_game": game_lineup_row_counts,
        "parsed_teams": parsed_teams,
        "visible_contains_expected_lineup": "Expected Lineup" in visible_text,
        "visible_contains_starting_pitcher_intel": "Starting Pitcher Intel" in visible_text,
    }
    return out, html, diag


def main() -> None:
    args = parse_args()
    slate_date = pd.to_datetime(args.date, errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_live_lineups.log")
    log_header("scripts/live/build_live_lineups.py", repo_root, config_path, dirs)

    spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(f"Live spine not found: {spine_path}")
    spine = pd.read_parquet(spine_path).copy()
    spine["game_pk"] = pd.to_numeric(spine.get("game_pk"), errors="coerce").astype("Int64")
    team_games = pd.concat(
        [
            spine[["game_pk", "away_team", "home_team"]].rename(columns={"away_team": "batter_team", "home_team": "opponent_team"}),
            spine[["game_pk", "home_team", "away_team"]].rename(columns={"home_team": "batter_team", "away_team": "opponent_team"}),
        ],
        ignore_index=True,
        sort=False,
    )
    team_games["batter_team"] = team_games["batter_team"].astype(str).str.upper()
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
    logging.info("rotowire parsed teams before slate filtering=%s", sorted(raw["batter_team"].dropna().astype(str).unique().tolist()) if len(raw) else [])

    raw["canonical_team"] = raw["batter_team"].map(_to_canonical_team)
    team_games["canonical_team"] = team_games["batter_team"].map(_to_canonical_team)
    scraped_teams = set(raw["batter_team"].dropna().astype(str).str.upper().unique().tolist()) if len(raw) else set()
    scraped_canonical = set(raw["canonical_team"].dropna().astype(str).unique().tolist()) if len(raw) else set()
    slate_teams = set(team_games["batter_team"].dropna().astype(str).str.upper().unique().tolist()) if len(team_games) else set()
    slate_canonical = set(team_games["canonical_team"].dropna().astype(str).unique().tolist()) if len(team_games) else set()
    unmatched_scraped = sorted(scraped_teams - slate_teams)
    unmatched_canonical = sorted(scraped_canonical - slate_canonical)
    logging.info(
        "rotowire team diagnostics parsed_raw_teams=%s parsed_canonical_teams=%s slate_canonical_teams=%s unmatched_raw_teams=%s unmatched_canonical_teams=%s sample_rows=%s",
        sorted(scraped_teams),
        sorted(scraped_canonical),
        sorted(slate_canonical),
        unmatched_scraped,
        unmatched_canonical,
        raw.head(10).to_dict(orient="records") if len(raw) else [],
    )

    filtered = raw[raw["canonical_team"].isin(slate_canonical)].copy()
    if len(raw) > 0 and filtered.empty:
        raise ValueError(
            "Rotowire parsed rows but canonical slate filtering removed all rows. "
            f"parsed_raw_teams={sorted(scraped_teams)} "
            f"parsed_canonical_teams={sorted(scraped_canonical)} "
            f"slate_canonical_teams={sorted(slate_canonical)} "
            f"unmatched_canonical_teams={unmatched_canonical}"
        )
    out = filtered.merge(
        team_games[["game_pk", "batter_team", "canonical_team"]].drop_duplicates(),
        on="canonical_team",
        how="inner",
        suffixes=("_scraped", ""),
    )
    out["game_date"] = args.date
    out["batter_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["bats"] = pd.NA
    out["lineup_status"] = "projected"
    out["lineup_source"] = SOURCE_NAME
    out["source_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = _resolve_player_ids(out, dirs["processed_dir"] / "batter_game_rolling.parquet", slate_date)

    out = out[
        [
            "game_pk",
            "game_date",
            "batter_team",
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

    out = out.dropna(subset=["player_name"]).copy()
    ui_mask = out["player_name"].str.lower().str.contains("|".join(UI_BLOCKLIST), regex=True, na=False)
    out = out[~ui_mask].copy()
    logging.info(
        "projected_lineups prewrite total_rows=%s unique_games=%s unique_teams=%s sample_rows=%s",
        len(out),
        int(out["game_pk"].nunique()) if len(out) else 0,
        int(out["batter_team"].nunique()) if len(out) else 0,
        out.head(10).to_dict(orient="records"),
    )

    if out.empty:
        logging.warning(
            "projected lineups unavailable; writing empty parquet for fallback usage status_code=%s content_length=%s game_container_count=%s contains_expected_lineup=%s",
            scrape_diag.get("status_code"),
            scrape_diag.get("content_length"),
            scrape_diag.get("game_container_count"),
            scrape_diag.get("contains_expected_lineup"),
        )

    out_path = dirs["processed_dir"] / "live" / f"projected_lineups_{args.season}_{args.date}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    logging.info(
        "projected_lineups rows=%s teams=%s games=%s source=%s out=%s",
        len(out),
        int(out["batter_team"].nunique()) if len(out) else 0,
        int(out["game_pk"].nunique()) if len(out) else 0,
        SOURCE_URL,
        out_path,
    )
    print(f"projected_lineups_out={out_path}")


if __name__ == "__main__":
    main()
