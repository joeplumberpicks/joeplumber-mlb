import argparse
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROTOWIRE_URL = "https://www.rotowire.com/baseball/daily-lineups.php"

POS_SET = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"}
HAND_SET = {"L", "R", "S"}

LINEUP_START = {"Confirmed Lineup", "Expected Lineup"}
STOP_TOKENS = {
    "Home Run Odds",
    "Starting Pitcher Intel",
    "Umpire:",
    "LINE",
    "O/U",
}


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    cset = set(cols)
    for c in candidates:
        if c in cset:
            return c
    return None


def _clean_player_name(text: str) -> str:
    text = text.replace("NONE", "").strip()
    parts = text.split()
    if parts and parts[0].isdigit():
        parts = parts[1:]
    if parts and parts[0] in POS_SET:
        parts = parts[1:]
    return " ".join(parts).strip()


def _is_bad_name(name: str) -> bool:
    if not name:
        return True
    low = name.lower()
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
