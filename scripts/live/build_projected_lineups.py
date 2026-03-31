from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

SOURCE_NAME = "fangraphs_pasted_text"

TEAM_NICKNAME_TO_CANONICAL = {
    "Angels": "LOS ANGELES ANGELS",
    "Astros": "HOUSTON ASTROS",
    "Athletics": "ATHLETICS",
    "Blue Jays": "TORONTO BLUE JAYS",
    "Braves": "ATLANTA BRAVES",
    "Brewers": "MILWAUKEE BREWERS",
    "Cardinals": "ST. LOUIS CARDINALS",
    "Cubs": "CHICAGO CUBS",
    "Diamondbacks": "ARIZONA DIAMONDBACKS",
    "Dodgers": "LOS ANGELES DODGERS",
    "Giants": "SAN FRANCISCO GIANTS",
    "Guardians": "CLEVELAND GUARDIANS",
    "Mariners": "SEATTLE MARINERS",
    "Marlins": "MIAMI MARLINS",
    "Mets": "NEW YORK METS",
    "Nationals": "WASHINGTON NATIONALS",
    "Orioles": "BALTIMORE ORIOLES",
    "Padres": "SAN DIEGO PADRES",
    "Phillies": "PHILADELPHIA PHILLIES",
    "Pirates": "PITTSBURGH PIRATES",
    "Rangers": "TEXAS RANGERS",
    "Rays": "TAMPA BAY RAYS",
    "Red Sox": "BOSTON RED SOX",
    "Reds": "CINCINNATI REDS",
    "Rockies": "COLORADO ROCKIES",
    "Royals": "KANSAS CITY ROYALS",
    "Tigers": "DETROIT TIGERS",
    "Twins": "MINNESOTA TWINS",
    "White Sox": "CHICAGO WHITE SOX",
    "Yankees": "NEW YORK YANKEES",
}

TEAM_ABBR_TO_CANONICAL = {
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
    "SEA": "SEATTLE MARINERS",
    "STL": "ST. LOUIS CARDINALS",
    "TB": "TAMPA BAY RAYS",
    "TEX": "TEXAS RANGERS",
    "TOR": "TORONTO BLUE JAYS",
    "WSH": "WASHINGTON NATIONALS",
}

HEADER_RE = re.compile(r"^2026\s+(.+?)\s+Lineups$", flags=re.IGNORECASE)
POS_SLOT_RE = re.compile(r"\b(C|1B|2B|3B|SS|LF|CF|RF|DH|OF)\s*\((\d)\)\b", flags=re.IGNORECASE)
VALID_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "OF"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build projected lineups from pasted plain text")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--text-path", type=Path, required=True, help="Path to pasted plain-text lineup file")
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


def _norm_name_value(s: object) -> str:
    s = _normalize_text(s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_name(series: pd.Series) -> pd.Series:
    return series.map(_norm_name_value)


def _to_canonical_team(team_token: object) -> str | None:
    tok = _normalize_text(team_token)
    if not tok:
        return None
    if tok in TEAM_NICKNAME_TO_CANONICAL:
        return TEAM_NICKNAME_TO_CANONICAL[tok]
    up = tok.upper()
    if up in TEAM_ABBR_TO_CANONICAL:
        return TEAM_ABBR_TO_CANONICAL[up]
    return None


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


def _load_batter_lookup(batter_path: Path, slate_date: pd.Timestamp) -> pd.DataFrame:
    batter = pd.read_parquet(batter_path).copy()
    team_col = _pick(list(batter.columns), ["batter_team", "team", "team_abbrev", "team_name", "batting_team"])
    bid_col = _pick(list(batter.columns), ["batter_id", "player_id"])
    name_col = _pick(list(batter.columns), ["player_name", "batter_name", "name", "batter"])

    if team_col is None or bid_col is None or name_col is None:
        raise RuntimeError(f"Could not build batter lookup from columns: {sorted(batter.columns)}")

    batter["game_date"] = pd.to_datetime(batter.get("game_date"), errors="coerce")
    batter = batter[batter["game_date"] < slate_date].copy()
    batter["canonical_team"] = batter[team_col].map(_to_canonical_team)
    batter["batter_id"] = pd.to_numeric(batter[bid_col], errors="coerce").astype("Int64")
    batter["name_norm"] = _norm_name(batter[name_col])

    return (
        batter.sort_values(["game_date"])
        .dropna(subset=["canonical_team"])
        .drop_duplicates(subset=["canonical_team", "name_norm"], keep="last")[["canonical_team", "name_norm", "batter_id"]]
        .copy()
    )


def _resolve_player_ids(out: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    if out.empty:
        return out

    out["name_norm"] = _norm_name(out["player_name"])
    merged = out.merge(
        lookup,
        on=["canonical_team", "name_norm"],
        how="left",
        suffixes=("", "_lk"),
    )
    if "batter_id_lk" in merged.columns:
        merged["batter_id"] = merged["batter_id"].fillna(merged["batter_id_lk"])
        merged = merged.drop(columns=["batter_id_lk"], errors="ignore")
    return merged.drop(columns=["name_norm"], errors="ignore")


def _read_text_lines(text_path: Path) -> list[str]:
    text = text_path.read_text(encoding="utf-8", errors="ignore")
    lines: list[str] = []
    for raw in text.splitlines():
        t = _normalize_text(raw)
        if t:
            lines.append(t)
    return lines


def _extract_lineups(lines: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    i = 0
    while i < len(lines):
        m = HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue

        team_nickname = _normalize_text(m.group(1))
        canonical_team = _to_canonical_team(team_nickname)
        start_i = i + 1

        end_i = len(lines)
        for j in range(start_i, len(lines)):
            if HEADER_RE.match(lines[j]):
                end_i = j
                break

        block = lines[start_i:end_i]
        kept = 0

        k = 0
        while k < len(block) - 4:
            name = block[k]
            role = block[k + 1] if k + 1 < len(block) else ""

            if role not in {str(x) for x in range(1, 10)}:
                k += 1
                continue

            pos = None
            slot = None
            source_text = None

            for w in range(4, 16):
                if k + w >= len(block):
                    break
                cell = block[k + w]
                mm = POS_SLOT_RE.search(cell)
                if mm:
                    pos = mm.group(1).upper()
                    slot = int(mm.group(2))
                    source_text = cell
                    break

            if canonical_team and pos in VALID_POSITIONS and slot in range(1, 10):
                rows.append(
                    {
                        "canonical_team": canonical_team,
                        "player_name": name,
                        "lineup_slot": slot,
                        "position": pos,
                        "bats": None,
                        "lineup_status": "projected",
                        "source_text": source_text,
                    }
                )
                kept += 1
                k += 5
            else:
                k += 1

        diagnostics.append(
            {
                "team": team_nickname,
                "canonical_team": canonical_team,
                "block_len": len(block),
                "kept_rows": kept,
            }
        )

        i = end_i

    logging.info("pasted-text parse diagnostics=%s", diagnostics)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    slate_date = pd.to_datetime(args.date, errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_projected_lineups.log")
    log_header("scripts/live/build_projected_lineups.py", repo_root, config_path, dirs)

    if not args.text_path.exists():
        raise FileNotFoundError(f"text_path not found: {args.text_path}")

    spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(f"Live spine not found: {spine_path}")

    batter_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    if not batter_path.exists():
        raise FileNotFoundError(f"Batter rolling file not found: {batter_path}")

    spine = pd.read_parquet(spine_path).copy()
    spine["game_pk"] = pd.to_numeric(spine.get("game_pk"), errors="coerce").astype("Int64")
    team_games = _build_team_game_map(spine)
    slate_canonical = set(team_games["canonical_team"].dropna().astype(str).unique().tolist())

    lookup = _load_batter_lookup(batter_path, slate_date)
    lines = _read_text_lines(args.text_path)
    logging.info("pasted text lines count=%s", len(lines))
    logging.info("pasted text sample 0:60=%s", lines[:60])

    raw = _extract_lineups(lines)
    if raw.empty:
        raise RuntimeError("No lineup rows parsed from pasted text")

    raw = raw.dropna(subset=["canonical_team", "player_name", "lineup_slot", "position"]).copy()
    raw = raw[raw["position"].isin(VALID_POSITIONS)].copy()
    raw = raw.drop_duplicates(subset=["canonical_team", "player_name"], keep="first").copy()
    raw = raw.sort_values(["canonical_team", "lineup_slot", "player_name"]).copy()
    raw = raw.drop_duplicates(subset=["canonical_team", "lineup_slot"], keep="first").copy()

    counts_before = raw.groupby("canonical_team").size().to_dict()
    logging.info("pasted text counts before slate filter=%s", counts_before)

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

    out = _resolve_player_ids(out, lookup)

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
        logging.warning("incomplete projected lineup teams=%s counts=%s", bad_teams, counts)

    if out.empty:
        raise RuntimeError("Output dataframe is empty after joining lineups to slate teams")

    if out["position"].notna().sum() == 0:
        raise RuntimeError("Output has zero non-null positions")

    logging.info(
        "projected_lineups prewrite total_rows=%s unique_games=%s unique_teams=%s nonnull_position=%s nonnull_batter_id=%s rows_per_team=%s sample_rows=%s",
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
        "projected_lineups rows=%s teams=%s games=%s out=%s",
        len(out),
        int(out["batter_team"].nunique()),
        int(out["game_pk"].nunique()),
        out_path,
    )
    print(f"projected_lineups_out={out_path}")


if __name__ == "__main__":
    main()
