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
    if batter.empty:
        return pd.DataFrame(columns=["game_pk", "batter_team", "batter_id", "player_name", "lineup_slot", "lineup_status", "lineup_source"])

    b = batter.copy()
    b["game_date"] = pd.to_datetime(b.get("game_date"), errors="coerce")
    b = b[b["game_date"] < slate_date].copy()
    if b.empty:
        return pd.DataFrame(columns=["game_pk", "batter_team", "batter_id", "player_name", "lineup_slot", "lineup_status", "lineup_source"])

    team_col = _pick(list(b.columns), ["batter_team", "team", "team_abbrev", "team_name", "batting_team"])
    bid_col = _pick(list(b.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(b.columns), ["player_name", "batter_name", "name"])
    score_col = _pick(list(b.columns), ["bat_pa_per_game_roll15", "bat_ab_per_game_roll15", "pa", "ab"])
    season_col = pd.to_numeric(b.get("season"), errors="coerce") if "season" in b.columns else b["game_date"].dt.year
    b["_season"] = season_col
    b = b[pd.to_numeric(b["_season"], errors="coerce") <= season].copy()
    if team_col is None or bid_col is None:
        return pd.DataFrame(columns=["game_pk", "batter_team", "batter_id", "player_name", "lineup_slot", "lineup_status", "lineup_source"])

    b["batter_team"] = b[team_col].astype(str)
    b["batter_id"] = pd.to_numeric(b[bid_col], errors="coerce").astype("Int64")
    b["player_name"] = b[name_col].astype(str) if name_col else b["batter_id"].astype(str)
    b["_score"] = pd.to_numeric(b[score_col], errors="coerce") if score_col else np.nan
    b = b.sort_values(["game_date"]).groupby(["batter_team", "batter_id"], as_index=False).tail(1)
    b = b.sort_values(["batter_team", "_score", "player_name"], ascending=[True, False, True], kind="mergesort")
    b["lineup_slot"] = b.groupby("batter_team").cumcount() + 1
    b = b[b["lineup_slot"] <= 9].copy()
    b = b.merge(team_games[["game_pk", "batter_team"]].drop_duplicates(), on="batter_team", how="inner")
    b["lineup_status"] = "fallback"
    b["lineup_source"] = "heuristic_recent_usage"
    return b[["game_pk", "batter_team", "batter_id", "player_name", "lineup_slot", "lineup_status", "lineup_source"]]


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

    if confirmed.empty and projected.empty and lineup_path.exists():
        projected = _normalize_live_lineups(
            pd.read_parquet(lineup_path).copy(),
            team_games=team_games,
            source_label="lineups_compat",
            default_status="projected",
        )

    confirmed_games = set(pd.to_numeric(confirmed.get("game_pk"), errors="coerce").dropna().astype(int).tolist()) if not confirmed.empty else set()
    projected_use = projected[~pd.to_numeric(projected.get("game_pk"), errors="coerce").astype("Int64").isin(list(confirmed_games))].copy() if not projected.empty else pd.DataFrame()
    selected = pd.concat([confirmed, projected_use], ignore_index=True, sort=False)

    selected_games = set(pd.to_numeric(selected.get("game_pk"), errors="coerce").dropna().astype(int).tolist()) if not selected.empty else set()
    slate_games = set(pd.to_numeric(g.get("game_pk"), errors="coerce").dropna().astype(int).tolist())
    missing_games = sorted(slate_games - selected_games)
    fallback = pd.DataFrame()
    if missing_games:
        fallback_all = _build_fallback_lineups(batter=batter, slate_date=slate_date, season=args.season, team_games=team_games)
        if not fallback_all.empty:
            fallback = fallback_all[pd.to_numeric(fallback_all["game_pk"], errors="coerce").astype("Int64").isin(missing_games)].copy()
            selected = pd.concat([selected, fallback], ignore_index=True, sort=False)

    if selected.empty:
        raise ValueError(
            "No lineup candidates were available from confirmed, projected, compatibility, or fallback sources."
        )

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

    board = selected.merge(team_games, on=["game_pk", "batter_team"], how="inner")
    if "park_factor_hits_blend" in board.columns:
        board["park_factor_hits_blend"] = pd.to_numeric(board["park_factor_hits_blend"], errors="coerce")
    if "park_factor_hits_hist" in board.columns:
        board["park_factor_hits_hist"] = pd.to_numeric(board["park_factor_hits_hist"], errors="coerce")
    if "park_factor_hits_blend" not in board.columns:
        board["park_factor_hits_blend"] = np.nan
    if "park_factor_hits_hist" in board.columns:
        board["park_factor_hits_blend"] = board["park_factor_hits_blend"].fillna(board["park_factor_hits_hist"])

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

    board = board.merge(cur_ct.reset_index(), on="batter_id", how="left")
    use_prev = pd.to_numeric(board["_cur_n"], errors="coerce").fillna(0) < args.min_current_games

    model_path = _latest_model(dirs["models_dir"])
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feats = list(bundle.get("feature_list", []))
    X = pd.DataFrame(index=board.index, columns=feats, dtype="float64")

    for c in feats:
        if c.startswith("bat_"):
            cur = board["batter_id"].map(cur_latest[c]) if (not cur_latest.empty and c in cur_latest.columns) else pd.Series(np.nan, index=board.index)
            prv = board["batter_id"].map(prev_latest[c]) if (not prev_latest.empty and c in prev_latest.columns) else pd.Series(np.nan, index=board.index)
            X[c] = np.where(use_prev, prv, cur)
        elif c in board.columns:
            X[c] = pd.to_numeric(board[c], errors="coerce")

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
        board = board.merge(p_cur_ct.reset_index(), left_on="opp_pitcher_id", right_on="pitcher_id", how="left")
        use_prev_p = pd.to_numeric(board["_p_cur_n"], errors="coerce").fillna(0) < args.min_current_games
        for c in feats:
            if c.startswith("pit_"):
                cur = board["opp_pitcher_id"].map(p_cur[c]) if (not p_cur.empty and c in p_cur.columns) else pd.Series(np.nan, index=board.index)
                prv = board["opp_pitcher_id"].map(p_prev[c]) if (not p_prev.empty and c in p_prev.columns) else pd.Series(np.nan, index=board.index)
                X[c] = np.where(use_prev_p, prv, cur)

    board["expected_batting_order_pa"] = pd.to_numeric(board["lineup_slot"], errors="coerce").map(LINEUP_PA_MAP)
    board["lineup_confidence"] = _lineup_conf(board["lineup_slot"])

    has_slot = pd.to_numeric(board.get("lineup_slot"), errors="coerce").notna()
    ab_proxy_series = pd.Series(np.nan, index=X.index, dtype="float64")
    ab_proxy_series.loc[has_slot] = (
        pd.to_numeric(board.loc[has_slot, "lineup_confidence"], errors="coerce")
        * pd.to_numeric(board.loc[has_slot, "expected_batting_order_pa"], errors="coerce")
    )

    x_extra = {
        "bat_ab_per_game_roll15": pd.to_numeric(X["bat_ab_per_game_roll15"], errors="coerce") if "bat_ab_per_game_roll15" in X.columns else pd.Series(np.nan, index=X.index, dtype="float64"),
        "bat_pa_per_game_roll15": pd.to_numeric(X["bat_pa_per_game_roll15"], errors="coerce") if "bat_pa_per_game_roll15" in X.columns else pd.Series(np.nan, index=X.index, dtype="float64"),
        "expected_batting_order_pa": pd.to_numeric(board["expected_batting_order_pa"], errors="coerce"),
        "lineup_confidence": pd.to_numeric(board["lineup_confidence"], errors="coerce"),
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
    X_scoring = X.reindex(columns=feats, fill_value=np.nan)
    logging.info("hit_prop live X_scoring_shape rows=%s cols=%s", X_scoring.shape[0], X_scoring.shape[1])
    logging.info("hit_prop live scoring_feature_count=%s", len(feats))

    real_slot = int(pd.to_numeric(board.get("lineup_slot"), errors="coerce").notna().sum()) if "lineup_slot" in board.columns else 0
    fallback_conf_only = int(((pd.to_numeric(board.get("lineup_slot"), errors="coerce").isna()) & (pd.to_numeric(board.get("lineup_confidence"), errors="coerce").notna())).sum()) if "lineup_confidence" in board.columns else 0
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

    out = board[[c for c in ["player_name", "batter_team", "opponent_team", "lineup_slot", "expected_batting_order_pa", "lineup_confidence"] if c in board.columns]].copy()
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
