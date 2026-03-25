from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reference.build_weather_game import normalize_weather_frame
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

GRADE_THRESHOLDS = [("A+", 0.70), ("A", 0.67), ("A-", 0.64), ("B+", 0.61), ("B", 0.58), ("B-", 0.55), ("C+", 0.52)]
LINEUP_PA_MAP = {1: 4.65, 2: 4.55, 3: 4.45, 4: 4.35, 5: 4.25, 6: 4.10, 7: 3.95, 8: 3.80, 9: 3.70}
LINEUP_JUNK_TOKENS = ["watch", "tickets", "alerts", "lineup has not been posted", "era"]
TEAM_TO_CANONICAL = {
    "ARI": "ARIZONA DIAMONDBACKS",
    "ARIZONA DIAMONDBACKS": "ARIZONA DIAMONDBACKS",
    "ATL": "ATLANTA BRAVES",
    "ATLANTA BRAVES": "ATLANTA BRAVES",
    "BAL": "BALTIMORE ORIOLES",
    "BALTIMORE ORIOLES": "BALTIMORE ORIOLES",
    "BOS": "BOSTON RED SOX",
    "BOSTON RED SOX": "BOSTON RED SOX",
    "CHC": "CHICAGO CUBS",
    "CHICAGO CUBS": "CHICAGO CUBS",
    "CWS": "CHICAGO WHITE SOX",
    "CHW": "CHICAGO WHITE SOX",
    "CHICAGO WHITE SOX": "CHICAGO WHITE SOX",
    "CIN": "CINCINNATI REDS",
    "CINCINNATI REDS": "CINCINNATI REDS",
    "CLE": "CLEVELAND GUARDIANS",
    "CLEVELAND GUARDIANS": "CLEVELAND GUARDIANS",
    "COL": "COLORADO ROCKIES",
    "COLORADO ROCKIES": "COLORADO ROCKIES",
    "DET": "DETROIT TIGERS",
    "DETROIT TIGERS": "DETROIT TIGERS",
    "HOU": "HOUSTON ASTROS",
    "HOUSTON ASTROS": "HOUSTON ASTROS",
    "KC": "KANSAS CITY ROYALS",
    "KCR": "KANSAS CITY ROYALS",
    "KANSAS CITY ROYALS": "KANSAS CITY ROYALS",
    "LAA": "LOS ANGELES ANGELS",
    "ANA": "LOS ANGELES ANGELS",
    "LOS ANGELES ANGELS": "LOS ANGELES ANGELS",
    "LAD": "LOS ANGELES DODGERS",
    "LOS ANGELES DODGERS": "LOS ANGELES DODGERS",
    "MIA": "MIAMI MARLINS",
    "FLA": "MIAMI MARLINS",
    "MIAMI MARLINS": "MIAMI MARLINS",
    "MIL": "MILWAUKEE BREWERS",
    "MILWAUKEE BREWERS": "MILWAUKEE BREWERS",
    "MIN": "MINNESOTA TWINS",
    "MINNESOTA TWINS": "MINNESOTA TWINS",
    "NYM": "NEW YORK METS",
    "NEW YORK METS": "NEW YORK METS",
    "NYY": "NEW YORK YANKEES",
    "NEW YORK YANKEES": "NEW YORK YANKEES",
    "ATH": "OAKLAND ATHLETICS",
    "OAKLAND ATHLETICS": "OAKLAND ATHLETICS",
    "PHI": "PHILADELPHIA PHILLIES",
    "PHILADELPHIA PHILLIES": "PHILADELPHIA PHILLIES",
    "PIT": "PITTSBURGH PIRATES",
    "PITTSBURGH PIRATES": "PITTSBURGH PIRATES",
    "SD": "SAN DIEGO PADRES",
    "SDP": "SAN DIEGO PADRES",
    "SAN DIEGO PADRES": "SAN DIEGO PADRES",
    "SF": "SAN FRANCISCO GIANTS",
    "SFG": "SAN FRANCISCO GIANTS",
    "SAN FRANCISCO GIANTS": "SAN FRANCISCO GIANTS",
    "SEA": "SEATTLE MARINERS",
    "SEATTLE MARINERS": "SEATTLE MARINERS",
    "STL": "ST. LOUIS CARDINALS",
    "SLN": "ST. LOUIS CARDINALS",
    "ST. LOUIS CARDINALS": "ST. LOUIS CARDINALS",
    "ST LOUIS CARDINALS": "ST. LOUIS CARDINALS",
    "TB": "TAMPA BAY RAYS",
    "TBR": "TAMPA BAY RAYS",
    "TAMPA BAY RAYS": "TAMPA BAY RAYS",
    "TEX": "TEXAS RANGERS",
    "TEXAS RANGERS": "TEXAS RANGERS",
    "TOR": "TORONTO BLUE JAYS",
    "TORONTO BLUE JAYS": "TORONTO BLUE JAYS",
    "WSH": "WASHINGTON NATIONALS",
    "WSN": "WASHINGTON NATIONALS",
    "WASHINGTON NATIONALS": "WASHINGTON NATIONALS",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run daily 1+ hit props board v1.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--min-current-games", type=int, default=5)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], cands: list[str]) -> str | None:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def _grade(p: float) -> str:
    for g, t in GRADE_THRESHOLDS:
        if p >= t:
            return g
    return "C"


def _latest_model(models_dir: Path) -> Path:
    c = sorted((models_dir / "hit_prop").glob("hit_prop_logit_*.joblib"))
    if not c:
        raise FileNotFoundError(f"no hit prop model in {models_dir / 'hit_prop'}")
    return c[-1]


def _lineup_conf(slot: pd.Series) -> pd.Series:
    s = pd.to_numeric(slot, errors="coerce")
    out = pd.Series(0.75, index=slot.index, dtype="float64")
    out = out.where(~s.between(1, 5), 1.00)
    out = out.where(~s.between(6, 7), 0.92)
    out = out.where(~s.between(8, 9), 0.85)
    return out


def _is_junk_name(name: str) -> bool:
    s = str(name or "").strip().lower()
    if not s:
        return True
    if any(t in s for t in LINEUP_JUNK_TOKENS):
        return True
    if "(" in s and "-" in s and ")" in s:
        return True
    if " pm " in f" {s} " or " am " in f" {s} " or " et" in f" {s} ":
        return True
    return False


def _normalize_team_canonical(team_val: object) -> str | None:
    tok = str(team_val or "").strip().upper()
    if not tok:
        return None
    return TEAM_TO_CANONICAL.get(tok)


def _validate_lineup_rows(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if df.empty:
        return df
    v = df.copy()
    v["player_name"] = v["player_name"].astype(str)
    v = v[~v["player_name"].map(_is_junk_name)].copy()
    v = v[v["player_name"].str.split().str.len().fillna(0) >= 2].copy()
    per_team = v.groupby("batter_team").size()
    keep_teams = per_team[per_team >= 5].index.astype(str).tolist()
    dropped_teams = sorted(set(v["batter_team"].astype(str).unique().tolist()) - set(keep_teams))
    if dropped_teams:
        logging.warning("lineup validation dropped teams label=%s teams=%s reason=lt5_valid_hitters", label, dropped_teams)
    v = v[v["batter_team"].astype(str).isin(keep_teams)].copy()
    logging.info("lineup validation label=%s input_rows=%s output_rows=%s kept_teams=%s", label, len(df), len(v), sorted(set(keep_teams)))
    return v


def _normalize_live_lineups(
    lu: pd.DataFrame,
    team_games: pd.DataFrame,
    source_label: str,
    default_status: str,
) -> pd.DataFrame:
    if lu.empty:
        return pd.DataFrame(columns=["game_pk", "batter_team", "batter_id", "player_name", "lineup_slot", "lineup_status", "lineup_source"])

    game_pk_col = _pick(list(lu.columns), ["game_pk", "game_id"])
    team_col = _pick(list(lu.columns), ["batter_team", "team", "team_abbrev", "team_name", "batting_team"])
    bid_col = _pick(list(lu.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(lu.columns), ["player_name", "batter_name", "name"])
    slot_col = _pick(list(lu.columns), ["lineup_slot", "bat_order", "batting_order", "lineup_position", "order"])
    status_col = _pick(list(lu.columns), ["lineup_status", "status"])

    if team_col is None:
        logging.warning("lineup source=%s missing team column; skipping", source_label)
        return pd.DataFrame(columns=["game_pk", "batter_team", "batter_id", "player_name", "lineup_slot", "lineup_status", "lineup_source"])

    out = pd.DataFrame(index=lu.index)
    out["game_pk"] = pd.to_numeric(lu[game_pk_col], errors="coerce").astype("Int64") if game_pk_col else pd.Series(pd.NA, index=lu.index, dtype="Int64")
    out["batter_team"] = lu[team_col].astype(str)
    out["batter_id"] = pd.to_numeric(lu[bid_col], errors="coerce").astype("Int64") if bid_col else pd.Series(pd.NA, index=lu.index, dtype="Int64")
    out["player_name"] = lu[name_col].astype(str) if name_col else out["batter_id"].astype(str)
    out["lineup_slot"] = pd.to_numeric(lu[slot_col], errors="coerce") if slot_col else np.nan
    out["lineup_status"] = lu[status_col].astype(str).str.lower() if status_col else default_status
    out["lineup_source"] = source_label

    if out["game_pk"].isna().any():
        team_map = team_games[["game_pk", "batter_team"]].drop_duplicates()
        missing = out["game_pk"].isna()
        if missing.any():
            fill_vals = out.loc[missing, ["batter_team"]].merge(team_map, on="batter_team", how="left")["game_pk"]
            out.loc[missing, "game_pk"] = pd.to_numeric(fill_vals, errors="coerce").astype("Int64").values

    out = out.dropna(subset=["game_pk", "batter_team", "player_name"]).copy()
    out = out.drop_duplicates(subset=["game_pk", "batter_team", "player_name"], keep="last")
    return out


def _build_fallback_lineups(
    batter: pd.DataFrame,
    slate_date: pd.Timestamp,
    season: int,
    team_games: pd.DataFrame,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["game_pk", "batter_team", "batter_id", "player_name", "lineup_slot", "lineup_status", "lineup_source"])
    if batter.empty:
        return empty

    b = batter.copy()
    if "game_date" in b.columns:
        b["game_date"] = pd.to_datetime(b.get("game_date"), errors="coerce")
        b = b[b["game_date"] < slate_date].copy()
    if b.empty:
        return empty

    team_col = _pick(list(b.columns), ["batter_team", "team", "team_abbrev", "team_name", "batting_team"])
    bid_col = _pick(list(b.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(b.columns), ["player_name", "batter_name", "name"])
    pa_col = _pick(list(b.columns), ["bat_pa_per_game_roll15", "pa", "bat_pa_roll15", "bat_ab_per_game_roll15", "ab"])
    old_slot_col = _pick(list(b.columns), ["lineup_slot", "bat_order", "batting_order", "order", "lineup_position"])
    season_col = pd.to_numeric(b.get("season"), errors="coerce") if "season" in b.columns else pd.to_datetime(b.get("game_date"), errors="coerce").dt.year
    b["_season"] = season_col
    if team_col is None or bid_col is None:
        return empty
    available_seasons = sorted(pd.to_numeric(b["_season"], errors="coerce").dropna().astype(int).unique().tolist())
    chosen_season = season if season in available_seasons else (max(available_seasons) if available_seasons else None)
    if chosen_season is not None:
        b = b[pd.to_numeric(b["_season"], errors="coerce") == chosen_season].copy()

    b["batter_team"] = b[team_col].astype(str).str.upper().str.strip()
    b["batter_team_canonical"] = b["batter_team"].map(_normalize_team_canonical)
    b["batter_id"] = pd.to_numeric(b[bid_col], errors="coerce").astype("Int64")
    b["player_name"] = b[name_col].astype(str) if name_col else b["batter_id"].astype(str)
    b["_pa"] = pd.to_numeric(b[pa_col], errors="coerce") if pa_col else np.nan
    b["_old_slot"] = pd.to_numeric(b[old_slot_col], errors="coerce") if old_slot_col else np.nan
    b["game_date"] = pd.to_datetime(b.get("game_date"), errors="coerce")
    b = b[b["batter_id"].notna()].copy()
    b = b.sort_values(["game_date", "_pa"], ascending=[False, False], kind="mergesort")
    b = b.groupby(["batter_team", "batter_id"], as_index=False).head(1)

    slate_team_map = team_games[["game_pk", "batter_team"]].drop_duplicates().copy()
    slate_team_map["batter_team"] = slate_team_map["batter_team"].astype(str).str.upper().str.strip()
    slate_team_map["batter_team_canonical"] = slate_team_map["batter_team"].map(_normalize_team_canonical)
    if "opponent_team" in team_games.columns:
        slate_teams_long = (
            team_games[["batter_team", "opponent_team"]]
            .melt(value_name="slate_team_raw")
            .drop(columns=["variable"], errors="ignore")
        )
    else:
        slate_teams_long = team_games[["batter_team"]].rename(columns={"batter_team": "slate_team_raw"})
    slate_teams_long["slate_team_canonical"] = slate_teams_long["slate_team_raw"].astype(str).str.upper().str.strip().map(_normalize_team_canonical)
    slate_team_canonical = sorted(slate_teams_long["slate_team_canonical"].dropna().unique().tolist())
    logging.info(
        "fallback diagnostics batter_history_rows=%s batter_team_raw_unique=%s batter_team_canonical_unique=%s slate_team_canonical=%s",
        len(b),
        sorted(b["batter_team"].dropna().astype(str).unique().tolist()),
        sorted(b["batter_team_canonical"].dropna().astype(str).unique().tolist()),
        slate_team_canonical,
    )
    b = b[b["batter_team_canonical"].isin(slate_team_canonical)].copy()
    logging.info("fallback diagnostics rows_after_slate_team_filter=%s", len(b))
    if b.empty:
        batter_raw_teams = sorted(batter[team_col].astype(str).str.upper().str.strip().dropna().unique().tolist())
        batter_canonical = sorted(pd.Series(batter_raw_teams).map(_normalize_team_canonical).dropna().unique().tolist())
        mapping_failures = sorted(pd.Series(batter_raw_teams)[pd.Series(batter_raw_teams).map(_normalize_team_canonical).isna()].unique().tolist())
        raise ValueError(
            "Fallback lineup canonical team match produced zero rows. "
            f"raw_batter_teams={batter_raw_teams} canonical_batter_teams={batter_canonical} "
            f"slate_team_canonical={slate_team_canonical} mapping_failures={mapping_failures}"
        )
    b["batter_team"] = b["batter_team_canonical"]
    b = b.sort_values(["batter_team", "game_date", "_pa", "_old_slot"], ascending=[True, False, False, True], kind="mergesort")
    b["lineup_slot"] = b.groupby("batter_team").cumcount() + 1
    b = b[b["lineup_slot"] <= 9].copy()
    per_team = b.groupby("batter_team").size().to_dict() if len(b) else {}
    logging.info("fallback diagnostics rows_selected_per_team=%s", per_team)

    if "game_pk" in b.columns:
        game_col = "game_pk"
    elif "game_id" in b.columns:
        game_col = "game_id"
    else:
        game_col = None

    if game_col is None:
        b["game_pk"] = b["batter_team"].astype(str) + "_" + str(slate_date.date())
        game_col = "game_pk"

    if "game_pk" in team_games.columns and not team_games.empty:
        canonical_game_map = slate_team_map[["game_pk", "batter_team_canonical"]].drop_duplicates().rename(columns={"game_pk": "game_pk_mapped"})
        mapped = b.merge(canonical_game_map, on="batter_team_canonical", how="left")
        mapped["game_pk"] = mapped["game_pk_mapped"].where(mapped["game_pk_mapped"].notna(), mapped[game_col] if game_col in mapped.columns else pd.NA)
        b = mapped.drop(columns=["game_pk_mapped"], errors="ignore")
        game_col = "game_pk"

    b["lineup_slot"] = pd.to_numeric(b.get("lineup_slot"), errors="coerce").fillna(5)
    b["lineup_status"] = "fallback"
    b["lineup_source"] = "fallback"
    cols = [game_col, "batter_team", "batter_id", "player_name", "lineup_slot", "lineup_status", "lineup_source"]
    cols = [c for c in cols if c in b.columns]
    out = b[cols].copy()
    if game_col in out.columns and game_col != "game_pk":
        out = out.rename(columns={game_col: "game_pk"})
    if "game_pk" not in out.columns:
        out["game_pk"] = pd.NA
    out = out[[c for c in ["game_pk", "batter_team", "batter_id", "player_name", "lineup_slot", "lineup_status", "lineup_source"] if c in out.columns]]
    out["batter_id"] = pd.to_numeric(out.get("batter_id"), errors="coerce").astype("Int64")
    out["player_name"] = out["player_name"].astype(str).where(out["player_name"].astype(str).str.strip() != "", out["batter_id"].astype(str))
    non_null_batter_id_count = int(out["batter_id"].notna().sum()) if "batter_id" in out.columns else 0
    unique_batter_id_count = int(out["batter_id"].dropna().nunique()) if "batter_id" in out.columns else 0
    logging.info(
        "fallback diagnostics non_null_batter_id_count=%s unique_batter_id_count=%s",
        non_null_batter_id_count,
        unique_batter_id_count,
    )
    logging.info("fallback diagnostics final_fallback_row_count=%s", len(out))
    return out


def main() -> None:
    args = parse_args()
    slate_date = pd.to_datetime(args.date, errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "run_daily_hit_props_v1.log")
    log_header("scripts/live/run_daily_hit_props_v1.py", repo_root, config_path, dirs)

    spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(
            f"Live spine not found: {spine_path}. Build it first (e.g. run scripts/live/build_spine_from_schedule.py for the requested season/date)."
        )
    bat_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    pit_path = dirs["processed_dir"] / "pitcher_game_rolling.parquet"
    live_dir = dirs["processed_dir"] / "live"
    lineup_path = live_dir / f"lineups_{args.season}_{args.date}.parquet"
    confirmed_lineup_path = live_dir / f"confirmed_lineups_{args.season}_{args.date}.parquet"
    projected_lineup_path = live_dir / f"projected_lineups_{args.season}_{args.date}.parquet"

    spine = pd.read_parquet(spine_path).copy()
    batter = pd.read_parquet(bat_path).copy()
    pitcher = pd.read_parquet(pit_path).copy() if pit_path.exists() else pd.DataFrame()

    parks_path = dirs["reference_dir"] / "parks.parquet"
    parks_dyn_path = dirs["reference_dir"] / "parks_dynamic_2026.parquet"
    weather_live_processed_path = dirs["processed_dir"] / "live" / f"weather_game_{args.season}_{args.date}.parquet"
    weather_live_raw_path = dirs["raw_dir"] / "live" / f"weather_game_{args.season}_{args.date}.parquet"
    weather_fallback_path = dirs["processed_dir"] / "weather_game.parquet"

    game_cols = [c for c in ["game_pk", "away_team", "home_team", "home_sp_id", "away_sp_id", "temperature", "wind_speed", "park_factor_hits_blend", "park_factor_hits_hist", "venue_id", "canonical_park_key", "park_id"] if c in spine.columns]
    g = spine[game_cols].copy()
    g["game_pk"] = pd.to_numeric(g.get("game_pk"), errors="coerce").astype("Int64")
    home_games = g.rename(columns={"home_team": "batter_team", "away_team": "opponent_team", "away_sp_id": "opp_pitcher_id"})
    home_games["home_away"] = 1.0
    away_games = g.rename(columns={"away_team": "batter_team", "home_team": "opponent_team", "home_sp_id": "opp_pitcher_id"})
    away_games["home_away"] = 0.0
    team_games = pd.concat([home_games, away_games], ignore_index=True, sort=False)

    weather_source_path: Path | None = None
    weather_source_kind = "none"
    wg = pd.DataFrame()
    if weather_live_processed_path.exists():
        weather_source_path = weather_live_processed_path
        weather_source_kind = "processed_live"
    elif weather_live_raw_path.exists():
        weather_source_path = weather_live_raw_path
        weather_source_kind = "raw_live"
    elif weather_fallback_path.exists():
        weather_source_path = weather_fallback_path
        weather_source_kind = "processed_fallback"

    if weather_source_path is not None:
        try:
            wg = pd.read_parquet(weather_source_path).copy()
            if weather_source_kind == "raw_live":
                wg = normalize_weather_frame(wg, fallback_season=args.season)
            logging.info("hit_prop live weather_source kind=%s path=%s rows=%s", weather_source_kind, weather_source_path, len(wg))
            if "game_pk" in wg.columns:
                wg["game_pk"] = pd.to_numeric(wg["game_pk"], errors="coerce").astype("Int64")
                keep = [c for c in ["game_pk", "temperature", "wind_speed", "weather_wind_out", "weather_wind_in"] if c in wg.columns]
                if len(keep) > 1:
                    g = g.merge(wg[keep].drop_duplicates(subset=["game_pk"], keep="last"), on="game_pk", how="left", suffixes=("", "_wg"))
                    for c in ["temperature", "wind_speed", "weather_wind_out", "weather_wind_in"]:
                        if f"{c}_wg" in g.columns:
                            g[c] = pd.to_numeric(g.get(c), errors="coerce").fillna(pd.to_numeric(g.get(f"{c}_wg"), errors="coerce"))
                            g = g.drop(columns=[f"{c}_wg"], errors="ignore")
        except Exception:
            logging.exception("hit_prop live optional weather join failed; continuing")
    else:
        logging.warning(
            "hit_prop live optional weather table missing checked processed_live=%s raw_live=%s fallback=%s",
            weather_live_processed_path,
            weather_live_raw_path,
            weather_fallback_path,
        )

    if parks_path.exists():
        try:
            parks = pd.read_parquet(parks_path).copy()
            pkey = _pick(list(parks.columns), ["canonical_park_key", "venue_id", "park_id"])
            if pkey and "park_factor_hits_hist" in parks.columns:
                gkey = _pick(list(g.columns), ["canonical_park_key", "venue_id", "park_id"])
                if gkey:
                    g["_park_join_key"] = g[gkey].astype(str)
                    parks["_park_join_key"] = parks[pkey].astype(str)
                    g = g.merge(parks[["_park_join_key", "park_factor_hits_hist"]].drop_duplicates(subset=["_park_join_key"], keep="last"), on="_park_join_key", how="left", suffixes=("", "_p"))
                    g["park_factor_hits_hist"] = pd.to_numeric(g.get("park_factor_hits_hist"), errors="coerce").fillna(pd.to_numeric(g.get("park_factor_hits_hist_p"), errors="coerce"))
                    g = g.drop(columns=["_park_join_key", "park_factor_hits_hist_p"], errors="ignore")
        except Exception:
            logging.exception("hit_prop live optional parks join failed; continuing")
    else:
        logging.warning("hit_prop live optional parks table missing: %s", parks_path)


    if parks_dyn_path.exists():
        try:
            pdyn = pd.read_parquet(parks_dyn_path).copy()
            pkey = _pick(list(pdyn.columns), ["canonical_park_key", "venue_id", "park_id"])
            gkey = _pick(list(g.columns), ["canonical_park_key", "venue_id", "park_id"])
            if pkey and gkey:
                g["_park_join_key"] = g[gkey].astype(str)
                pdyn["_park_join_key"] = pdyn[pkey].astype(str)
                keep = ["_park_join_key"] + [c for c in ["park_factor_hits_2026_roll", "park_factor_hits_blend"] if c in pdyn.columns]
                g = g.merge(pdyn[keep].drop_duplicates(subset=["_park_join_key"], keep="last"), on="_park_join_key", how="left")
                g = g.drop(columns=["_park_join_key"], errors="ignore")
        except Exception:
            logging.exception("hit_prop live optional dynamic parks join failed; continuing")
    else:
        logging.warning("hit_prop live optional dynamic parks table missing: %s", parks_dyn_path)

    confirmed = pd.DataFrame()
    if confirmed_lineup_path.exists():
        confirmed = _normalize_live_lineups(
            pd.read_parquet(confirmed_lineup_path).copy(),
            team_games=team_games,
            source_label="confirmed_lineups",
            default_status="confirmed",
        )

    projected = pd.DataFrame()
    if projected_lineup_path.exists():
        projected = _normalize_live_lineups(
            pd.read_parquet(projected_lineup_path).copy(),
            team_games=team_games,
            source_label="projected_lineups",
            default_status="projected",
        )
        projected = _validate_lineup_rows(projected, label="projected_lineups")

    if confirmed.empty and projected.empty and lineup_path.exists():
        projected = _normalize_live_lineups(
            pd.read_parquet(lineup_path).copy(),
            team_games=team_games,
            source_label="lineups_compat",
            default_status="projected",
        )
        projected = _validate_lineup_rows(projected, label="lineups_compat")

    confirmed_games = set(pd.to_numeric(confirmed.get("game_pk"), errors="coerce").dropna().astype(int).tolist()) if not confirmed.empty else set()
    projected_use = projected[~pd.to_numeric(projected.get("game_pk"), errors="coerce").astype("Int64").isin(list(confirmed_games))].copy() if not projected.empty else pd.DataFrame()
    selected = pd.concat([confirmed, projected_use], ignore_index=True, sort=False)

    selected_games = set(pd.to_numeric(selected.get("game_pk"), errors="coerce").dropna().astype(int).tolist()) if not selected.empty else set()
    slate_games = set(pd.to_numeric(g.get("game_pk"), errors="coerce").dropna().astype(int).tolist())
    missing_games = sorted(slate_games - selected_games)
    fallback = pd.DataFrame()
    if missing_games:
        if confirmed.empty and projected.empty:
            logging.info("Using fallback lineup builder (no confirmed or projected lineups found)")
        else:
            logging.info("Using fallback lineup builder for games missing both confirmed and projected lineups")
        fallback_all = _build_fallback_lineups(batter=batter, slate_date=slate_date, season=args.season, team_games=team_games)
        if not fallback_all.empty:
            fallback = fallback_all[pd.to_numeric(fallback_all["game_pk"], errors="coerce").astype("Int64").isin(missing_games)].copy()
            selected = pd.concat([selected, fallback], ignore_index=True, sort=False)

    if selected.empty:
        batter_team_col = _pick(list(batter.columns), ["batter_team", "team", "team_abbrev", "team_name", "batting_team"])
        batter_teams_available = (
            sorted(batter[batter_team_col].astype(str).str.upper().dropna().unique().tolist())
            if (batter_team_col is not None and not batter.empty)
            else []
        )
        slate_team_list = sorted(team_games["batter_team"].astype(str).str.upper().dropna().unique().tolist())
        raise ValueError(
            "No lineup candidates were available from confirmed, projected, compatibility, or fallback sources. "
            f"confirmed_rows={len(confirmed)} projected_rows={len(projected_use)} fallback_rows={len(fallback)} "
            f"slate_teams={slate_team_list} batter_teams_available={batter_teams_available}"
        )

    hitter_props_features_path = dirs["marts_dir"] / "hitter_props_features.parquet"
    if not hitter_props_features_path.exists():
        raise FileNotFoundError(f"hitter feature mart not found: {hitter_props_features_path}")
    df_features = pd.read_parquet(hitter_props_features_path).copy()
    df_lineups = selected.copy()
    if "batter_id" not in df_lineups.columns:
        raise ValueError("lineup candidates missing batter_id column before hitter feature join diagnostics")
    if "batter_id" not in df_features.columns:
        raise ValueError("hitter feature mart missing batter_id column")
    df_lineups["batter_id"] = pd.to_numeric(df_lineups["batter_id"], errors="coerce")
    df_features["batter_id"] = pd.to_numeric(df_features["batter_id"], errors="coerce")
    lineup_batter_sample = df_lineups["batter_id"].dropna().head(10).tolist()
    feature_batter_sample = df_features["batter_id"].dropna().head(10).tolist()
    lineup_batter_set = set(df_lineups["batter_id"].dropna().tolist())
    feature_batter_set = set(df_features["batter_id"].dropna().tolist())
    intersection = lineup_batter_set & feature_batter_set
    logging.info(
        "hit_prop live batter_id diagnostics lineup_sample=%s feature_sample=%s intersection_count=%s",
        lineup_batter_sample,
        feature_batter_sample,
        len(intersection),
    )
    if len(intersection) == 0:
        raise ValueError(
            "No overlapping batter_id values between lineups and hitter feature mart. "
            f"lineup_batter_id_sample={lineup_batter_sample} feature_batter_id_sample={feature_batter_sample} "
            f"lineup_batter_id_dtype={df_lineups['batter_id'].dtype} feature_batter_id_dtype={df_features['batter_id'].dtype}"
        )
    feature_season_col = _pick(list(df_features.columns), ["season", "_season"])
    if feature_season_col is not None:
        df_features["_feature_season"] = pd.to_numeric(df_features[feature_season_col], errors="coerce")
    elif "game_date" in df_features.columns:
        df_features["_feature_season"] = pd.to_datetime(df_features["game_date"], errors="coerce").dt.year
    else:
        df_features["_feature_season"] = np.nan
    df_features["_feature_game_date"] = pd.to_datetime(df_features.get("game_date"), errors="coerce")

    df_joined = df_lineups[["batter_id"]].dropna().drop_duplicates().merge(df_features, on=["batter_id"], how="inner")
    if df_joined.empty:
        raise ValueError("No batter_id overlap between lineups and hitter feature mart")
    df_joined["_season_priority"] = np.select(
        [
            pd.to_numeric(df_joined["_feature_season"], errors="coerce") == args.season,
            pd.to_numeric(df_joined["_feature_season"], errors="coerce") == (args.season - 1),
        ],
        [2, 1],
        default=0,
    )
    joined_candidate_row_count = len(df_joined)
    candidate_counts_by_season = pd.to_numeric(df_joined["_feature_season"], errors="coerce").value_counts(dropna=False).to_dict()
    df_joined = df_joined.sort_values(
        ["batter_id", "_season_priority", "_feature_season", "_feature_game_date"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    df_selected_features = df_joined.drop_duplicates(subset=["batter_id"], keep="first").copy()
    selected_current_season_count = int((pd.to_numeric(df_selected_features["_feature_season"], errors="coerce") == args.season).sum())
    selected_previous_season_count = int((pd.to_numeric(df_selected_features["_feature_season"], errors="coerce") == (args.season - 1)).sum())
    selected_older_fallback_count = int(
        (
            (pd.to_numeric(df_selected_features["_feature_season"], errors="coerce") != args.season)
            & (pd.to_numeric(df_selected_features["_feature_season"], errors="coerce") != (args.season - 1))
        ).sum()
    )
    final_selected_scoring_row_count = len(df_selected_features)
    sample_selected = (
        df_selected_features[["batter_id", "_feature_season"]]
        .head(10)
        .to_dict("records")
    )
    logging.info(
        "hit_prop live feature_row_selection joined_candidate_row_count=%s candidate_counts_by_season=%s selected_current_season_count=%s selected_previous_season_count=%s selected_older_fallback_count=%s final_selected_scoring_row_count=%s sample_batter_season=%s",
        joined_candidate_row_count,
        candidate_counts_by_season,
        selected_current_season_count,
        selected_previous_season_count,
        selected_older_fallback_count,
        final_selected_scoring_row_count,
        sample_selected,
    )
    if selected_previous_season_count > 0 or selected_older_fallback_count > 0:
        logging.info(
            "Opening Day / early-season fallback engaged: using most recent historical feature rows where current-season rows are unavailable."
        )
    if final_selected_scoring_row_count == 0:
        raise ValueError("No selected scoring rows after season-priority feature selection")

    selected = selected[pd.to_numeric(selected["batter_id"], errors="coerce").isin(df_selected_features["batter_id"].dropna().unique().tolist())].copy()
    selected_feature_by_batter = df_selected_features.set_index("batter_id")
    df_scoring = selected.copy()

    logging.info(
        "hit_prop live lineup_source_summary confirmed_rows=%s confirmed_games=%s projected_rows=%s projected_games_used=%s fallback_rows=%s fallback_games=%s total_rows=%s total_games=%s",
        len(confirmed),
        len(confirmed_games),
        len(projected),
        int(projected_use["game_pk"].nunique()) if not projected_use.empty else 0,
        len(fallback),
        int(fallback["game_pk"].nunique()) if not fallback.empty else 0,
        len(selected),
        int(selected["game_pk"].nunique()) if not selected.empty else 0,
    )
    logging.info("hit_prop live confirmed_lineup_rows=%s", len(confirmed))
    logging.info("hit_prop live projected_lineup_rows=%s", len(projected_use))
    logging.info("hit_prop live fallback_lineup_rows=%s", len(fallback))
    logging.info("hit_prop live final_lineup_candidate_rows=%s", len(selected))
    if not selected.empty and {"game_pk", "lineup_source"}.issubset(selected.columns):
        per_game = (
            selected[["game_pk", "lineup_source"]]
            .dropna(subset=["game_pk"])
            .astype({"lineup_source": str})
            .groupby("game_pk")["lineup_source"]
            .agg(lambda s: ",".join(sorted(set(s))))
            .to_dict()
        )
        logging.info("hit_prop live lineup_source_by_game=%s", per_game)

    row_count_before_team_join = len(df_scoring)
    df_scoring = df_scoring.merge(team_games, on=["game_pk", "batter_team"], how="left")
    logging.info("hit_prop live post_selection_team_merge row_count_before=%s row_count_after=%s", row_count_before_team_join, len(df_scoring))
    if "park_factor_hits_blend" in df_scoring.columns:
        df_scoring["park_factor_hits_blend"] = pd.to_numeric(df_scoring["park_factor_hits_blend"], errors="coerce")
    if "park_factor_hits_hist" in df_scoring.columns:
        df_scoring["park_factor_hits_hist"] = pd.to_numeric(df_scoring["park_factor_hits_hist"], errors="coerce")
    if "park_factor_hits_blend" not in df_scoring.columns:
        df_scoring["park_factor_hits_blend"] = np.nan
    if "park_factor_hits_hist" in df_scoring.columns:
        df_scoring["park_factor_hits_blend"] = df_scoring["park_factor_hits_blend"].fillna(df_scoring["park_factor_hits_hist"])

    batter["game_date"] = pd.to_datetime(batter.get("game_date"), errors="coerce")
    batter = batter[batter["game_date"] < slate_date].copy()
    b_id_col = _pick(list(batter.columns), ["batter_id", "batter", "player_id"])
    batter["batter_id"] = pd.to_numeric(batter[b_id_col], errors="coerce").astype("Int64")
    season_col = pd.to_numeric(batter.get("season"), errors="coerce") if "season" in batter.columns else batter["game_date"].dt.year
    batter["_season"] = season_col

    # carryover logic by threshold
    grp = batter.groupby("batter_id")
    cur_ct = grp.apply(lambda x: int((pd.to_numeric(x["_season"], errors="coerce") == args.season).sum())).rename("_cur_n")
    cur_latest = batter[pd.to_numeric(batter["_season"], errors="coerce") == args.season].sort_values("game_date").groupby("batter_id").tail(1).set_index("batter_id")
    prev_latest = batter[pd.to_numeric(batter["_season"], errors="coerce") == (args.season - 1)].sort_values("game_date").groupby("batter_id").tail(1).set_index("batter_id")

    row_count_before_cur_merge = len(df_scoring)
    df_scoring = df_scoring.merge(cur_ct.reset_index(), on="batter_id", how="left")
    logging.info("hit_prop live post_selection_batter_merge row_count_before=%s row_count_after=%s", row_count_before_cur_merge, len(df_scoring))
    if df_selected_features["_feature_season"].eq(args.season).any():
        use_prev = pd.to_numeric(df_scoring["_cur_n"], errors="coerce").fillna(0) < args.min_current_games
    else:
        logging.info("Skipping current-season filters (using fallback historical data)")
        use_prev = pd.Series(True, index=df_scoring.index)

    model_path = _latest_model(dirs["models_dir"])
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feats = list(bundle.get("feature_list", []))
    X = pd.DataFrame(index=df_scoring.index, columns=feats, dtype="float64")

    for c in feats:
        if c.startswith("bat_"):
            best = df_scoring["batter_id"].map(selected_feature_by_batter[c]) if (not selected_feature_by_batter.empty and c in selected_feature_by_batter.columns) else pd.Series(np.nan, index=df_scoring.index)
            cur = df_scoring["batter_id"].map(cur_latest[c]) if (not cur_latest.empty and c in cur_latest.columns) else pd.Series(np.nan, index=df_scoring.index)
            prv = df_scoring["batter_id"].map(prev_latest[c]) if (not prev_latest.empty and c in prev_latest.columns) else pd.Series(np.nan, index=df_scoring.index)
            X[c] = pd.to_numeric(best, errors="coerce").where(pd.to_numeric(best, errors="coerce").notna(), np.where(use_prev, prv, cur))
        elif c in df_scoring.columns:
            X[c] = pd.to_numeric(df_scoring[c], errors="coerce")

    if not pitcher.empty:
        pitcher["game_date"] = pd.to_datetime(pitcher.get("game_date"), errors="coerce")
        pitcher = pitcher[pitcher["game_date"] < slate_date].copy()
        p_id_col = _pick(list(pitcher.columns), ["pitcher_id", "pitcher", "player_id", "mlb_id"])
        pitcher["pitcher_id"] = pd.to_numeric(pitcher[p_id_col], errors="coerce").astype("Int64")
        p_season = pd.to_numeric(pitcher.get("season"), errors="coerce") if "season" in pitcher.columns else pitcher["game_date"].dt.year
        pitcher["_season"] = p_season
        p_grp = pitcher.groupby("pitcher_id")
        p_cur_ct = p_grp.apply(lambda x: int((pd.to_numeric(x["_season"], errors="coerce") == args.season).sum())).rename("_p_cur_n")
        p_cur = pitcher[pd.to_numeric(pitcher["_season"], errors="coerce") == args.season].sort_values("game_date").groupby("pitcher_id").tail(1).set_index("pitcher_id")
        p_prev = pitcher[pd.to_numeric(pitcher["_season"], errors="coerce") == (args.season - 1)].sort_values("game_date").groupby("pitcher_id").tail(1).set_index("pitcher_id")
        row_count_before_pitcher_merge = len(df_scoring)
        df_scoring = df_scoring.merge(p_cur_ct.reset_index(), left_on="opp_pitcher_id", right_on="pitcher_id", how="left")
        logging.info("hit_prop live post_selection_pitcher_merge row_count_before=%s row_count_after=%s", row_count_before_pitcher_merge, len(df_scoring))
        use_prev_p = pd.to_numeric(df_scoring["_p_cur_n"], errors="coerce").fillna(0) < args.min_current_games
        for c in feats:
            if c.startswith("pit_"):
                cur = df_scoring["opp_pitcher_id"].map(p_cur[c]) if (not p_cur.empty and c in p_cur.columns) else pd.Series(np.nan, index=df_scoring.index)
                prv = df_scoring["opp_pitcher_id"].map(p_prev[c]) if (not p_prev.empty and c in p_prev.columns) else pd.Series(np.nan, index=df_scoring.index)
                X[c] = np.where(use_prev_p, prv, cur)

    df_scoring["expected_batting_order_pa"] = pd.to_numeric(df_scoring["lineup_slot"], errors="coerce").map(LINEUP_PA_MAP)
    df_scoring["lineup_confidence"] = _lineup_conf(df_scoring["lineup_slot"])

    has_slot = pd.to_numeric(df_scoring.get("lineup_slot"), errors="coerce").notna()
    ab_proxy_series = pd.Series(np.nan, index=X.index, dtype="float64")
    ab_proxy_series.loc[has_slot] = (
        pd.to_numeric(df_scoring.loc[has_slot, "lineup_confidence"], errors="coerce")
        * pd.to_numeric(df_scoring.loc[has_slot, "expected_batting_order_pa"], errors="coerce")
    )

    x_extra = {
        "bat_ab_per_game_roll15": pd.to_numeric(X["bat_ab_per_game_roll15"], errors="coerce") if "bat_ab_per_game_roll15" in X.columns else pd.Series(np.nan, index=X.index, dtype="float64"),
        "bat_pa_per_game_roll15": pd.to_numeric(X["bat_pa_per_game_roll15"], errors="coerce") if "bat_pa_per_game_roll15" in X.columns else pd.Series(np.nan, index=X.index, dtype="float64"),
        "expected_batting_order_pa": pd.to_numeric(df_scoring["expected_batting_order_pa"], errors="coerce"),
        "lineup_confidence": pd.to_numeric(df_scoring["lineup_confidence"], errors="coerce"),
        "expected_ab_proxy": ab_proxy_series,
    }
    extra_df = pd.DataFrame(x_extra, index=X.index)
    overlap = [c for c in extra_df.columns if c in X.columns]
    if overlap:
        X = X.drop(columns=overlap)
    X = pd.concat([X, extra_df], axis=1)

    dup_cols = X.columns[X.columns.duplicated()].tolist()
    if dup_cols:
        logging.warning("hit_prop live duplicate_columns_detected=%s", dup_cols)
        X = X.loc[:, ~X.columns.duplicated(keep="last")].copy()

    # scoring matrix remains aligned to trained features only
    if df_scoring.empty:
        raise ValueError("Scoring dataframe unexpectedly empty after selection stage")
    available_features = [c for c in feats if c in X.columns]
    missing_features = [c for c in feats if c not in X.columns]
    available_feature_count = len(available_features)
    missing_feature_count = len(missing_features)
    if available_feature_count == 0:
        raise ValueError("No model features found in scoring dataframe")
    X_scoring = X[available_features].copy()
    X_scoring = X_scoring.reindex(columns=feats, fill_value=np.nan)
    logging.info(
        "hit_prop live feature_availability expected_feature_count=%s available_feature_count=%s missing_feature_count=%s sample_missing_features=%s",
        len(feats),
        available_feature_count,
        missing_feature_count,
        missing_features[:10],
    )
    logging.info("hit_prop live scoring_dataframe_rows=%s", len(df_scoring))
    if X_scoring.shape[0] == 0:
        raise ValueError("Scoring dataframe unexpectedly empty after selection stage")
    logging.info("hit_prop live X_scoring_shape rows=%s cols=%s", X_scoring.shape[0], X_scoring.shape[1])
    logging.info("hit_prop live scoring_feature_count=%s", len(feats))

    real_slot = int(pd.to_numeric(df_scoring.get("lineup_slot"), errors="coerce").notna().sum()) if "lineup_slot" in df_scoring.columns else 0
    fallback_conf_only = int(((pd.to_numeric(df_scoring.get("lineup_slot"), errors="coerce").isna()) & (pd.to_numeric(df_scoring.get("lineup_confidence"), errors="coerce").notna())).sum()) if "lineup_confidence" in df_scoring.columns else 0
    if "expected_ab_proxy" in X.columns:
        expected_ab_proxy = pd.to_numeric(X["expected_ab_proxy"], errors="coerce")
    else:
        expected_ab_proxy = pd.Series(np.nan, index=X.index, dtype="float64")
    ab_non_null = int(expected_ab_proxy.notna().sum())
    logging.info("hit_prop live lineup_slot_real_count=%s", real_slot)
    logging.info("hit_prop live lineup_confidence_fallback_only_count=%s", fallback_conf_only)
    logging.info("hit_prop live expected_ab_proxy_non_null_count=%s", ab_non_null)

    base_hit_probability = model.predict_proba(X_scoring)[:, 1]
    live_adjusted_hit_probability = np.clip(
        base_hit_probability + 0.015 * (expected_ab_proxy.fillna(4.1) - 4.1),
        0.01,
        0.99,
    )
    avg_adjustment = float(np.mean(live_adjusted_hit_probability - base_hit_probability)) if len(base_hit_probability) else 0.0
    logging.info("hit_prop live average_adjustment_applied=%.6f", avg_adjustment)

    out = df_scoring[[c for c in ["player_name", "batter_team", "opponent_team", "lineup_slot", "expected_batting_order_pa", "lineup_confidence"] if c in df_scoring.columns]].copy()
    out["bat_ab_per_game_roll15"] = pd.to_numeric(X["bat_ab_per_game_roll15"], errors="coerce") if "bat_ab_per_game_roll15" in X.columns else pd.Series(np.nan, index=out.index, dtype="float64")
    out["expected_ab_proxy"] = expected_ab_proxy
    out["base_hit_probability"] = base_hit_probability
    out["live_adjusted_hit_probability"] = live_adjusted_hit_probability
    out["grade"] = out["live_adjusted_hit_probability"].map(_grade)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = dirs["outputs_dir"] / "hit_prop"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hit_prop_board_{args.season}_{args.date}_{ts}.csv"
    out.sort_values("live_adjusted_hit_probability", ascending=False, kind="mergesort").to_csv(out_path, index=False)

    print("\nJOE PLUMBER 1+ HIT BOARD")
    for _, r in out.sort_values("live_adjusted_hit_probability", ascending=False, kind="mergesort").head(30).iterrows():
        print(f"{str(r.get('player_name','')):<24} {100*float(r['live_adjusted_hit_probability']):>5.1f}%  {r['grade']}")
    print(f"\nboard_out={out_path}")


if __name__ == "__main__":
    main()
