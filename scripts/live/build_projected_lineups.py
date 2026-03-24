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
POS_SET = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}
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


def _fetch_rotowire_html(date_str: str, html_snapshot_path: Path) -> tuple[str, dict[str, object]]:
    resp = requests.get(SOURCE_URL, params={"date": date_str, "site": "Yahoo"}, timeout=30)
    resp.raise_for_status()
    html = resp.text
    html_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    html_snapshot_path.write_text(html, encoding="utf-8")

    content_len = len(html)
    checks = {
        "Expected Lineup": "Expected Lineup" in html,
        "Starting Pitcher Intel": "Starting Pitcher Intel" in html,
        "NYY": "NYY" in html,
        "SF": "SF" in html,
        "Aaron Judge": "Aaron Judge" in html,
    }
    logging.info(
        "rotowire fetch status=%s final_url=%s content_length=%s key_checks=%s",
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


def _scrape_rotowire_projected(date_str: str, aliases: dict[str, str], html_snapshot_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    html, fetch_diag = _fetch_rotowire_html(date_str, html_snapshot_path=html_snapshot_path)
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

    out = pd.DataFrame(rows)
    out = out.dropna(subset=["batter_team", "player_name"]).copy()
    out["batter_team"] = out["batter_team"].astype(str).str.upper()
    out = out[out.groupby("batter_team").cumcount() < 9].copy()
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
    return out, diag


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

    html_snapshot_path = dirs["logs_dir"] / f"projected_lineups_rotowire_{args.season}_{args.date}.html"
    raw, scrape_diag = _scrape_rotowire_projected(args.date, aliases, html_snapshot_path=html_snapshot_path)
    logging.info("rotowire html_snapshot=%s", html_snapshot_path)
    logging.info("rotowire parsed teams before slate filtering=%s", sorted(raw["batter_team"].dropna().astype(str).unique().tolist()) if len(raw) else [])

    scraped_teams = set(raw["batter_team"].dropna().astype(str).str.upper().unique().tolist()) if len(raw) else set()
    slate_teams = set(team_games["batter_team"].dropna().astype(str).str.upper().unique().tolist()) if len(team_games) else set()
    unmatched_scraped = sorted(scraped_teams - slate_teams)
    unmatched_slate = sorted(slate_teams - scraped_teams)
    logging.info("rotowire team matching unmatched_scraped=%s unmatched_slate=%s", unmatched_scraped, unmatched_slate)

    out = raw.merge(team_games[["game_pk", "batter_team"]].drop_duplicates(), on="batter_team", how="inner")
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
    logging.info(
        "projected_lineups prewrite total_rows=%s unique_games=%s unique_teams=%s sample_rows=%s",
        len(out),
        int(out["game_pk"].nunique()) if len(out) else 0,
        int(out["batter_team"].nunique()) if len(out) else 0,
        out.head(10).to_dict(orient="records"),
    )

    if out.empty:
        raise ValueError(
            "Projected lineup scrape parsed zero rows after filtering. "
            f"status_code={scrape_diag.get('status_code')} "
            f"content_length={scrape_diag.get('content_length')} "
            f"game_container_count={scrape_diag.get('game_container_count')} "
            f"contains_expected_lineup={scrape_diag.get('contains_expected_lineup')} "
            f"contains_nyy={scrape_diag.get('contains_nyy')} "
            f"contains_sf={scrape_diag.get('contains_sf')}"
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
