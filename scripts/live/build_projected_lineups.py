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


def _scrape_rotowire_projected(date_str: str, aliases: dict[str, str]) -> pd.DataFrame:
    resp = requests.get(SOURCE_URL, params={"date": date_str, "site": "Yahoo"}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    lines = [s.strip() for s in soup.stripped_strings if s and s.strip()]

    rows: list[dict[str, object]] = []
    current_team: str | None = None
    pending_pos: str | None = None
    current_slot = 0

    matchup_pat = re.compile(r"^[A-Za-z .'-]+ \(\d+-\d+\)\s+[A-Za-z .'-]+ \(\d+-\d+\)$")
    hand_pat = re.compile(r"^[RLS]$")

    for idx, line in enumerate(lines):
        if matchup_pat.match(line):
            current_team = None
            pending_pos = None
            current_slot = 0
            continue
        if line in {"Unknown Lineup", "Expected Lineup", "Confirmed Lineup"}:
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
        pending_pos = None

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No projected lineup hitters parsed from source page.")
    out = out.dropna(subset=["batter_team", "player_name"]).copy()
    out["batter_team"] = out["batter_team"].astype(str).str.upper()
    out = out[out.groupby("batter_team").cumcount() < 9].copy()
    return out


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

    raw = _scrape_rotowire_projected(args.date, aliases)
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
