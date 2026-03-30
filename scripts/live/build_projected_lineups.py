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
HAND_SET = {"L", "R", "S"}
UI_BLOCKLIST = {"watch now", "tickets", "alerts"}
LINEUP_START_TOKENS = {"Expected Lineup", "Confirmed Lineup"}
STOP_TOKENS = {
    "Home Run Odds",
    "Starting Pitcher Intel",
    "Umpire:",
    "LINE",
    "O/U",
}
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


def _clean_player_name(text: str) -> str:
    s = str(text or "").strip()
    s = re.sub(r"^\s*NONE\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^\s*\d{1,2}\s+", "", s)
    s = re.sub(r"^\s*(C|1B|2B|3B|SS|LF|CF|RF|DH|SP|RP)\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_bad_name(name: str) -> bool:
    low = name.lower().strip()
    if not low:
        return True
    if low in {x.lower() for x in LINEUP_START_TOKENS}:
        return True
    if any(x in low for x in UI_BLOCKLIST):
        return True
    if "lineup" in low:
        return True
    if "pitcher" in low or "intel" in low:
        return True
    if "era" in low:
        return True
    if re.fullmatch(r"[a-z]{1,4}", low):
        return True
    return False


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
    batter["player_name_clean"] = batter[name_col].astype(str).map(_clean_player_name)
    batter["name_norm"] = _norm_name(batter["player_name_clean"])

    lookup = (
        batter.sort_values(["game_date"])
        .dropna(subset=["batter_id"])
        .drop_duplicates(subset=["batter_team", "name_norm"], keep="last")[["batter_team", "name_norm", "batter_id"]]
    )

    out["name_norm"] = _norm_name(out["player_name"])
    fill = out["batter_id"].isna()
    if fill.any():
        resolved = (
            out.loc[fill, ["batter_team", "name_norm"]]
            .merge(lookup, on=["batter_team", "name_norm"], how="left")["batter_id"]
        )
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


def _parse_team_card(card: BeautifulSoup, team: str) -> list[dict[str, object]]:
    tokens = [t.strip() for t in card.stripped_strings if t and t.strip()]
    rows: list[dict[str, object]] = []
    started = False
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if not started:
            if tok in LINEUP_START_TOKENS:
                started = True
            i += 1
            continue

        if tok in STOP_TOKENS or tok.startswith("Umpire:"):
            break
        if "Home Run Odds" in tok or "Starting Pitcher Intel" in tok:
            break
        if tok.startswith("The ") and "lineup has not been posted yet" in tok:
            break

        if tok in POS_SET:
            pos = tok
            player_name = None
            bats = None

            j = i + 1
            while j < len(tokens):
                nxt = tokens[j].strip()
                if not nxt:
                    j += 1
                    continue
                if nxt in STOP_TOKENS or nxt in POS_SET:
                    break
                if nxt in LINEUP_START_TOKENS:
                    break
                if re.fullmatch(r"\d+", nxt):
                    j += 1
                    continue
                if re.search(r"\bERA\b", nxt, flags=re.IGNORECASE):
                    break
                player_name = _clean_player_name(nxt)
                break

            if player_name and not _is_bad_name(player_name):
                k = j + 1
                while k < len(tokens):
                    nxt2 = tokens[k].strip()
                    if not nxt2:
                        k += 1
                        continue
                    if nxt2 in HAND_SET:
                        bats = nxt2
                    break

                rows.append(
                    {
                        "batter_team": team,
                        "player_name": player_name,
                        "lineup_slot": len(rows) + 1,
                        "position": pos,
                        "bats": bats,
                    }
                )

                if len(rows) >= 9:
                    break

        i += 1

    return rows


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
            "ul[class*='lineup'] li, ol[class*='lineup'] li, "
            "div[class*='lineup__list'] div, div[class*='lineup__player'], "
            "table[class*='lineup'] tr"
        )
        raw_nodes_scanned += len(lineup_nodes)
        lineup_nodes_found += len(lineup_nodes)

        rows.extend(_parse_team_card(card, team))

    logging.info(
        "rotowire parser nodes raw_nodes_scanned=%s lineup_nodes_found=%s game_cards=%s",
        raw_nodes_scanned,
        lineup_nodes_found,
        len(uniq_cards),
    )
    return pd.DataFrame(rows)


def _filter_valid_hitter_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    raw_counts = df.groupby("batter_team").size().to_dict()
    clean = df.copy()
    clean["player_name"] = clean["player_name"].astype(str).map(_clean_player_name)
    clean = clean[~clean["player_name"].map(_is_bad_name)].copy()
    clean = clean.dropna(subset=["batter_team", "player_name"]).copy()
    clean["position"] = clean["position"].where(clean["position"].isin(POS_SET), None)

    clean = clean.sort_values(["batter_team", "lineup_slot"], kind="mergesort")
    clean = clean.drop_duplicates(subset=["batter_team", "lineup_slot"], keep="first")
    clean = clean.groupby("batter_team").head(9).copy()

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
        [],
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

    out = _parse_rotowire_cards(soup, aliases)
    out = _filter_valid_hitter_rows(out)
    out = out.dropna(subset=["batter_team", "player_name"]).copy()
    out["batter_team"] = out["batter_team"].astype(str).str.upper()

    parsed_teams = sorted(out["batter_team"].dropna().astype(str).unique().tolist()) if len(out) else []
    logging.info(
        "rotowire parser diagnostics game_container_count=%s team_container_count=%s parsed_row_count=%s parsed_team_count=%s",
        game_container_count,
        team_container_count,
        len(out),
        len(parsed_teams),
    )
    logging.info("rotowire parser parsed_teams=%s", parsed_teams)

    diag = {
        **fetch_diag,
        "game_container_count": game_container_count,
        "team_container_count": team_container_count,
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

    configure_logging(dirs["logs_dir"] / "build_projected_lineups.log")
    log_header("scripts/live/build_projected_lineups.py", repo_root, config_path, dirs)

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

    raw["canonical_team"] = raw["batter_team"].map(_to_canonical_team)
    team_games["canonical_team"] = team_games["batter_team"].map(_to_canonical_team)

    scraped_teams = set(raw["batter_team"].dropna().astype(str).str.upper().unique().tolist()) if len(raw) else set()
    scraped_canonical = set(raw["canonical_team"].dropna().astype(str).unique().tolist()) if len(raw) else set()
    slate_canonical = set(team_games["canonical_team"].dropna().astype(str).unique().tolist()) if len(team_games) else set()

    logging.info(
        "rotowire team diagnostics parsed_raw_teams=%s parsed_canonical_teams=%s slate_canonical_teams=%s sample_rows=%s",
        sorted(scraped_teams),
        sorted(scraped_canonical),
        sorted(slate_canonical),
        raw.head(10).to_dict(orient="records") if len(raw) else [],
    )

    filtered = raw[raw["canonical_team"].isin(slate_canonical)].copy()
    if len(raw) > 0 and filtered.empty:
        raise ValueError(
            "Rotowire parsed rows but canonical slate filtering removed all rows. "
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

    if "batter_team_scraped" in out.columns:
        out["batter_team"] = out["batter_team_scraped"].fillna(out["batter_team"])
        out = out.drop(columns=["batter_team_scraped"], errors="ignore")

    out["game_date"] = args.date
    out["batter_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["lineup_status"] = "projected"
    out["lineup_source"] = SOURCE_NAME
    out["source_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if "bats" not in out.columns:
        out["bats"] = pd.NA

    out["player_name"] = out["player_name"].map(_clean_player_name)
    out = out[~out["player_name"].map(_is_bad_name)].copy()

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
        "projected_lineups prewrite total_rows=%s unique_games=%s unique_teams=%s resolved_batter_ids=%s sample_rows=%s",
        len(out),
        int(out["game_pk"].nunique()) if len(out) else 0,
        int(out["batter_team"].nunique()) if len(out) else 0,
        int(out["batter_id"].notna().sum()) if len(out) else 0,
        out.head(10).to_dict(orient="records"),
    )

    out_path = dirs["processed_dir"] / "live" / f"projected_lineups_{args.season}_{args.date}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    logging.info(
        "projected_lineups rows=%s teams=%s games=%s resolved_batter_ids=%s source=%s out=%s",
        len(out),
        int(out["batter_team"].nunique()) if len(out) else 0,
        int(out["game_pk"].nunique()) if len(out) else 0,
        int(out["batter_id"].notna().sum()) if len(out) else 0,
        SOURCE_URL,
        out_path,
    )
    print(f"projected_lineups_out={out_path}")


if __name__ == "__main__":
    main()