from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

SOURCE_NAME = "fangraphs_rosterresource"

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

POS_SET = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "OF"}
POSITION_SLOT_RE = re.compile(r"\b(C|1B|2B|3B|SS|LF|CF|RF|DH|OF)\s*\((\d)\)\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build projected lineups from FanGraphs saved HTML")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--html-path", type=Path, required=True, help="Saved FanGraphs Lineup Tracker HTML")
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


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        flat = []
        for col in out.columns:
            parts = [_normalize_text(x) for x in col if _normalize_text(x)]
            flat.append(" | ".join(parts))
        out.columns = flat
    else:
        out.columns = [_normalize_text(c) for c in out.columns]
    return out


def _to_canonical_team(team_token: object) -> str | None:
    tok = _normalize_text(team_token)
    if not tok:
        return None
    if tok in TEAM_NICKNAME_TO_CANONICAL:
        return TEAM_NICKNAME_TO_CANONICAL[tok]
    upper = tok.upper()
    if upper in TEAM_ABBR_TO_CANONICAL:
        return TEAM_ABBR_TO_CANONICAL[upper]
    return None


def _parse_position_slot(cell: object) -> tuple[str | None, int | None]:
    text = _normalize_text(cell)
    if not text:
        return None, None
    m = POSITION_SLOT_RE.search(text)
    if not m:
        return None, None
    pos = m.group(1).upper()
    slot = int(m.group(2))
    if pos not in POS_SET:
        return None, None
    if slot not in range(1, 10):
        return None, None
    return pos, slot


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
        logging.warning(
            "fangraphs id resolver skipped missing columns batter_cols=%s",
            sorted(batter.columns),
        )
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


def _select_date_column(df: pd.DataFrame, slate_date: pd.Timestamp) -> str | None:
    mmdd = f"{slate_date.month}/{slate_date.day}"
    weekday_abbrev = slate_date.strftime("%a")

    cols = [_normalize_text(c) for c in df.columns]

    for c in cols:
        if mmdd in c:
            return c

    for c in cols:
        if weekday_abbrev in c and re.search(r"\d{1,2}/\d{1,2}", c):
            return c

    for c in cols:
        if c in {"Name", "Role", "Ovr", "Last 7 Days"}:
            continue
        if re.search(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b", c):
            return c

    return None


def _extract_tables_with_team_names(html: str) -> list[tuple[str, pd.DataFrame]]:
    team_tables: list[tuple[str, pd.DataFrame]] = []

    headings = re.findall(r"2026\s+([A-Za-z ]+?)\s+Lineups", html)
    parsed_tables = pd.read_html(StringIO(html))

    for df in parsed_tables:
        df = _flatten_columns(df)
        cols = list(df.columns)

        name_col = _pick(cols, ["Name"])
        role_col = _pick(cols, ["Role"])
        if name_col is None or role_col is None:
            continue

        name_vals = df[name_col].astype(str).tolist()
        role_vals = df[role_col].astype(str).tolist()

        if not any(v in [str(i) for i in range(1, 10)] for v in role_vals):
            continue

        sample_names = " | ".join(name_vals[:5])

        matched_team = None
        for nickname in headings:
            if nickname not in TEAM_NICKNAME_TO_CANONICAL:
                continue

            # loose heuristic: just keep order of tables by lineup structure
            if matched_team is None:
                matched_team = nickname
                break

        if matched_team is not None:
            team_tables.append((matched_team, df))

    # Safer fallback: assign in page order to known nicknames found in HTML
    if team_tables:
        ordered_unique = []
        seen = set()
        for t in headings:
            if t in TEAM_NICKNAME_TO_CANONICAL and t not in seen:
                ordered_unique.append(t)
                seen.add(t)

        lineup_like_tables = []
        for df in parsed_tables:
            f = _flatten_columns(df)
            cols = list(f.columns)
            name_col = _pick(cols, ["Name"])
            role_col = _pick(cols, ["Role"])
            if name_col is None or role_col is None:
                continue
            role_vals = f[role_col].astype(str).tolist()
            if any(v in [str(i) for i in range(1, 10)] for v in role_vals):
                lineup_like_tables.append(f)

        if len(lineup_like_tables) == len(ordered_unique) and len(ordered_unique) > 0:
            return list(zip(ordered_unique, lineup_like_tables))

    return team_tables


def main() -> None:
    args = parse_args()
    slate_date = pd.to_datetime(args.date, errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_projected_lineups_fangraphs.log")
    log_header("scripts/live/build_projected_lineups_fangraphs.py", repo_root, config_path, dirs)

    if not args.html_path.exists():
        raise FileNotFoundError(f"html_path not found: {args.html_path}")

    html = args.html_path.read_text(encoding="utf-8", errors="ignore")
    logging.info("fangraphs html loaded path=%s content_length=%s", args.html_path, len(html))

    spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(f"Live spine not found: {spine_path}")

    spine = pd.read_parquet(spine_path).copy()
    spine["game_pk"] = pd.to_numeric(spine.get("game_pk"), errors="coerce").astype("Int64")
    team_games = _build_team_game_map(spine)
    slate_canonical = set(team_games["canonical_team"].dropna().astype(str).unique().tolist())

    team_tables = _extract_tables_with_team_names(html)
    logging.info("fangraphs extracted team_tables=%s", len(team_tables))

    if not team_tables:
        raise RuntimeError("No lineup-like tables found in saved FanGraphs HTML")

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    for team_nickname, df in team_tables:
        canonical_team = _to_canonical_team(team_nickname)
        if canonical_team is None:
            diagnostics.append({"team": team_nickname, "reason": "unknown_team"})
            continue

        cols = list(df.columns)
        name_col = _pick(cols, ["Name"])
        role_col = _pick(cols, ["Role"])
        date_col = _select_date_column(df, slate_date)

        if name_col is None or role_col is None or date_col is None:
            diagnostics.append(
                {
                    "team": team_nickname,
                    "canonical_team": canonical_team,
                    "reason": "missing_required_column",
                    "cols": cols,
                }
            )
            continue

        kept = 0
        for _, row in df.iterrows():
            player_name = _normalize_text(row.get(name_col))
            role_raw = _normalize_text(row.get(role_col))
            date_cell = row.get(date_col)

            if not player_name:
                continue
            if role_raw not in {str(i) for i in range(1, 10)}:
                continue

            pos, slot = _parse_position_slot(date_cell)
            if pos is None or slot is None:
                continue

            rows.append(
                {
                    "canonical_team": canonical_team,
                    "player_name": player_name,
                    "lineup_slot": slot,
                    "position": pos,
                    "bats": None,
                    "source_text": _normalize_text(date_cell),
                    "role_raw": role_raw,
                    "source_team_heading": team_nickname,
                }
            )
            kept += 1

        diagnostics.append(
            {
                "team": team_nickname,
                "canonical_team": canonical_team,
                "date_col": date_col,
                "rows": len(df),
                "kept_rows": kept,
            }
        )

    logging.info("fangraphs table diagnostics=%s", diagnostics)

    raw = pd.DataFrame(rows)
    if raw.empty:
        raise RuntimeError("No lineup rows parsed from FanGraphs tables")

    raw = raw.drop_duplicates(subset=["canonical_team", "player_name"], keep="first").copy()
    raw = raw.sort_values(["canonical_team", "lineup_slot", "player_name"]).copy()
    raw = raw.drop_duplicates(subset=["canonical_team", "lineup_slot"], keep="first").copy()

    counts_before = raw.groupby("canonical_team").size().to_dict()
    logging.info("fangraphs counts before slate filter=%s", counts_before)

    raw = raw[raw["canonical_team"].isin(slate_canonical)].copy()
    if raw.empty:
        raise RuntimeError(
            f"FanGraphs parsed lineup rows, but none matched slate teams. "
            f"parsed={sorted(counts_before.keys())} slate={sorted(slate_canonical)}"
        )

    out = raw.merge(
        team_games[["game_pk", "batter_team", "canonical_team"]].drop_duplicates(),
        on="canonical_team",
        how="inner",
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

    counts = out.groupby(["game_pk", "batter_team"]).size().to_dict() if len(out) else {}
    bad_teams = [k for k, n in counts.items() if n < 7]
    if bad_teams:
        logging.warning("fangraphs incomplete projected lineup teams=%s counts=%s", bad_teams, counts)

    if out.empty:
        raise RuntimeError("Output dataframe is empty after joining FanGraphs lineups to slate teams")

    if out["position"].notna().sum() == 0:
        raise RuntimeError("FanGraphs output has zero non-null positions")

    logging.info(
        "fangraphs projected_lineups prewrite total_rows=%s unique_games=%s unique_teams=%s nonnull_position=%s nonnull_batter_id=%s rows_per_team=%s sample_rows=%s",
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
        "fangraphs projected_lineups rows=%s teams=%s games=%s out=%s",
        len(out),
        int(out["batter_team"].nunique()),
        int(out["game_pk"].nunique()),
        out_path,
    )
    print(f"projected_lineups_out={out_path}")


if __name__ == "__main__":
    main()
