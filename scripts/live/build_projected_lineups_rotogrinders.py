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

SOURCE_NAME = "rotogrinders"
SOURCE_URL = "https://rotogrinders.com/lineups/mlb"

TEAM_MAP = {
    "Arizona Diamondbacks": "ARIZONA DIAMONDBACKS",
    "Atlanta Braves": "ATLANTA BRAVES",
    "Athletics": "ATHLETICS",
    "Baltimore Orioles": "BALTIMORE ORIOLES",
    "Boston Red Sox": "BOSTON RED SOX",
    "Chicago Cubs": "CHICAGO CUBS",
    "Chicago White Sox": "CHICAGO WHITE SOX",
    "Cincinnati Reds": "CINCINNATI REDS",
    "Cleveland Guardians": "CLEVELAND GUARDIANS",
    "Colorado Rockies": "COLORADO ROCKIES",
    "Detroit Tigers": "DETROIT TIGERS",
    "Houston Astros": "HOUSTON ASTROS",
    "Kansas City Royals": "KANSAS CITY ROYALS",
    "Los Angeles Angels": "LOS ANGELES ANGELS",
    "Los Angeles Dodgers": "LOS ANGELES DODGERS",
    "Miami Marlins": "MIAMI MARLINS",
    "Milwaukee Brewers": "MILWAUKEE BREWERS",
    "Minnesota Twins": "MINNESOTA TWINS",
    "New York Mets": "NEW YORK METS",
    "New York Yankees": "NEW YORK YANKEES",
    "Philadelphia Phillies": "PHILADELPHIA PHILLIES",
    "Pittsburgh Pirates": "PITTSBURGH PIRATES",
    "San Diego Padres": "SAN DIEGO PADRES",
    "San Francisco Giants": "SAN FRANCISCO GIANTS",
    "Seattle Mariners": "SEATTLE MARINERS",
    "St. Louis Cardinals": "ST. LOUIS CARDINALS",
    "Tampa Bay Rays": "TAMPA BAY RAYS",
    "Texas Rangers": "TEXAS RANGERS",
    "Toronto Blue Jays": "TORONTO BLUE JAYS",
    "Washington Nationals": "WASHINGTON NATIONALS",
}

TEAM_ALIASES = {
    "ath": "ATHLETICS",
    "athletics": "ATHLETICS",
    "st louis cardinals": "ST. LOUIS CARDINALS",
    "st. louis cardinals": "ST. LOUIS CARDINALS",
}

POS_TOKEN_RE = re.compile(r"\((L|R|S)\)\s+([A-Z0-9/]+)\b")
SLOT_RE = re.compile(r"^\*?\s*(\d)\s*$")
TIME_RE = re.compile(r"^\*?\s*\d{1,2}:\d{2}\s*[AP]M\s*ET\s*$", flags=re.IGNORECASE)
WEATHER_RE = re.compile(r"(mph|humidity|forecast|precipitation|°)", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build projected lineups from RotoGrinders MLB lineups page")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--url", default=SOURCE_URL)
    p.add_argument("--html-path", type=Path, default=None, help="Optional local HTML file instead of live fetch")
    return p.parse_args()


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    cset = set(cols)
    for c in candidates:
        if c in cset:
            return c
    return None


def _normalize_text(s: object) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_name(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _to_canonical_team(team_token: object) -> str | None:
    tok = _normalize_text(team_token)
    if not tok:
        return None
    if tok in TEAM_MAP:
        return TEAM_MAP[tok]
    low = tok.lower()
    if low in TEAM_ALIASES:
        return TEAM_ALIASES[low]
    return None


def _looks_like_team(line: str) -> bool:
    return _to_canonical_team(line) is not None


def _looks_like_player_line(line: str) -> bool:
    if "(L)" not in line and "(R)" not in line and "(S)" not in line:
        return False
    if " SP " in f" {line} ":
        return False
    return POS_TOKEN_RE.search(line) is not None


def _extract_player_fields(line: str) -> tuple[str | None, str | None, str | None]:
    """
    Example:
    Brandon Nimmo (L) OF $4.2K 9.4 25%
    Josh Smith (L) 2B/SS $3K 7.2 11%
    """
    line = _normalize_text(line)
    m = POS_TOKEN_RE.search(line)
    if not m:
        return None, None, None

    bats = m.group(1).upper()
    pos_token = m.group(2).upper()
    pos = pos_token.split("/")[0]

    name = line[:m.start()].strip()
    name = re.sub(r"\s+$", "", name)
    if not name:
        return None, None, None

    return name, bats, pos


def _build_team_game_map(spine: pd.DataFrame) -> pd.DataFrame:
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
    return team_games.drop_duplicates(subset=["game_pk", "canonical_team"]).copy()


def _resolve_player_ids(out: pd.DataFrame, batter_path: Path, slate_date: pd.Timestamp) -> pd.DataFrame:
    if out.empty or not batter_path.exists():
        return out

    batter = pd.read_parquet(batter_path).copy()
    team_col = _pick(list(batter.columns), ["batter_team", "team", "team_abbrev", "team_name", "batting_team"])
    bid_col = _pick(list(batter.columns), ["batter_id", "player_id"])
    name_col = _pick(list(batter.columns), ["player_name", "batter_name", "name", "batter"])

    if team_col is None or bid_col is None or name_col is None:
        logging.warning("rotogrinders id resolver skipped missing columns batter_cols=%s", sorted(batter.columns))
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


def _load_html(url: str, html_path: Path | None) -> str:
    if html_path is not None:
        if not html_path.exists():
            raise FileNotFoundError(f"html_path not found: {html_path}")
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        logging.info("rotogrinders html loaded path=%s content_length=%s", html_path, len(html))
        return html

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://rotogrinders.com/",
        }
    )
    resp = session.get(url, timeout=45, allow_redirects=True)
    logging.info(
        "rotogrinders fetch url=%s status=%s final_url=%s content_length=%s",
        url,
        resp.status_code,
        resp.url,
        len(resp.text),
    )
    resp.raise_for_status()
    return resp.text


def _extract_visible_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    lines: list[str] = []

    for s in soup.stripped_strings:
        t = _normalize_text(s)
        if not t:
            continue
        lines.append(t)

    return lines


def _parse_rotogrinders_lines(lines: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    i = 0
    while i < len(lines):
        if not TIME_RE.match(lines[i]):
            i += 1
            continue

        teams = []
        j = i + 1
        while j < len(lines) and len(teams) < 2:
            line = lines[j]
            if _looks_like_team(line):
                teams.append(_to_canonical_team(line))
            j += 1

        if len(teams) < 2:
            i += 1
            continue

        away_team, home_team = teams[0], teams[1]
        game_rows = 0

        current_team = None
        current_status = "confirmed"
        pending_slot = None

        k = j
        while k < len(lines):
            line = lines[k]

            if TIME_RE.match(line):
                break

            if _looks_like_team(line):
                team = _to_canonical_team(line)
                if team in {away_team, home_team}:
                    current_team = team
                    current_status = "confirmed"
                    pending_slot = None
                k += 1
                continue

            low = line.lower()

            if "lineup not released" in low:
                current_status = "projected"
                k += 1
                continue

            if WEATHER_RE.search(line) or "o/u" in low or "view forecast" in low:
                k += 1
                continue

            if " SP " in f" {line} ":
                k += 1
                continue

            slot_match = SLOT_RE.match(line)
            if slot_match:
                pending_slot = int(slot_match.group(1))
                k += 1
                continue

            if current_team and pending_slot and _looks_like_player_line(line):
                name, bats, pos = _extract_player_fields(line)
                if name and pos:
                    rows.append(
                        {
                            "canonical_team": current_team,
                            "player_name": name,
                            "lineup_slot": pending_slot,
                            "position": pos,
                            "bats": bats,
                            "lineup_status": current_status,
                            "source_text": line,
                        }
                    )
                    game_rows += 1
                    pending_slot = None

            k += 1

        diagnostics.append(
            {
                "away_team": away_team,
                "home_team": home_team,
                "rows": game_rows,
            }
        )
        i = k

    logging.info("rotogrinders parse diagnostics=%s", diagnostics)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    slate_date = pd.to_datetime(args.date, errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_projected_lineups_rotogrinders.log")
    log_header("scripts/live/build_projected_lineups_rotogrinders.py", repo_root, config_path, dirs)

    spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(f"Live spine not found: {spine_path}")

    spine = pd.read_parquet(spine_path).copy()
    spine["game_pk"] = pd.to_numeric(spine.get("game_pk"), errors="coerce").astype("Int64")
    team_games = _build_team_game_map(spine)
    slate_canonical = set(team_games["canonical_team"].dropna().astype(str).unique().tolist())

    html = _load_html(args.url, args.html_path)

    html_snapshot_path = dirs["logs_dir"] / f"projected_lineups_rotogrinders_{args.season}_{args.date}.html"
    html_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    html_snapshot_path.write_text(html, encoding="utf-8")
    logging.info("rotogrinders html_snapshot=%s", html_snapshot_path)

    lines = _extract_visible_lines(html)
    logging.info("rotogrinders visible lines count=%s", len(lines))
    logging.info("rotogrinders sample lines 260:340=%s", lines[260:340])

    raw = _parse_rotogrinders_lines(lines)
    if raw.empty:
        raise RuntimeError("No lineup rows parsed from RotoGrinders page")

    raw = raw.dropna(subset=["canonical_team", "player_name", "lineup_slot", "position"]).copy()
    raw = raw[raw["position"].isin({"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "OF"})].copy()
    raw = raw.drop_duplicates(subset=["canonical_team", "player_name"], keep="first").copy()
    raw = raw.sort_values(["canonical_team", "lineup_slot", "player_name"]).copy()
    raw = raw.drop_duplicates(subset=["canonical_team", "lineup_slot"], keep="first").copy()

    counts_before = raw.groupby("canonical_team").size().to_dict()
    logging.info("rotogrinders counts before slate filter=%s", counts_before)

    raw = raw[raw["canonical_team"].isin(slate_canonical)].copy()
    if raw.empty:
        raise RuntimeError(
            f"Parsed lineup rows but none matched slate teams. "
            f"parsed={sorted(counts_before.keys())} slate={sorted(slate_canonical)}"
        )

    out = raw.merge(
        team_games[["game_pk", "batter_team", "canonical_team"]].drop_duplicates(),
        on="canonical_team",
        how="inner",
    )

    out["game_date"] = args.date
    out["batter_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
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

    counts = out.groupby(["game_pk", "batter_team"]).size().to_dict() if len(out) else {}
    bad_teams = [k for k, n in counts.items() if n < 7]
    if bad_teams:
        logging.warning("rotogrinders incomplete projected lineup teams=%s counts=%s", bad_teams, counts)

    if out.empty:
        raise RuntimeError("Output dataframe is empty after joining lineups to slate teams")

    if out["position"].notna().sum() == 0:
        raise RuntimeError("Output has zero non-null positions")

    logging.info(
        "rotogrinders projected_lineups prewrite total_rows=%s unique_games=%s unique_teams=%s nonnull_position=%s nonnull_batter_id=%s rows_per_team=%s sample_rows=%s",
        len(out),
        int(out["game_pk"].nunique()),
        int(out["batter_team"].nunique()),
        int(out["position"].notna().sum()),
        int(out["batter_id"].notna().sum()),
        counts,
        out.head(20).to_dict(orient="records"),
    )

    out_path = dirs["processed_dir"] / "live" / f"projected_lineups_{args.season}_{args.date}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    logging.info(
        "rotogrinders projected_lineups rows=%s teams=%s games=%s out=%s",
        len(out),
        int(out["batter_team"].nunique()),
        int(out["game_pk"].nunique()),
        out_path,
    )
    print(f"projected_lineups_out={out_path}")


if __name__ == "__main__":
    main()
