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
    lineup_candidates = [
        dirs["processed_dir"] / "live" / f"projected_lineups_{args.season}_{args.date}.parquet",
        dirs["processed_dir"] / "live" / f"lineups_{args.season}_{args.date}.parquet",
        dirs["processed_dir"] / "live" / f"confirmed_lineups_{args.season}_{args.date}.parquet",
    ]

    spine = pd.read_parquet(spine_path).copy()
    batter = pd.read_parquet(bat_path).copy()
    pitcher = pd.read_parquet(pit_path).copy() if pit_path.exists() else pd.DataFrame()

    parks_path = dirs["reference_dir"] / "parks.parquet"
    weather_path = dirs["processed_dir"] / "weather_game.parquet"

    lineup_path = next((p for p in lineup_candidates if p.exists()), None)
    if lineup_path:
        lu = pd.read_parquet(lineup_path).copy()
    else:
        # fallback pool: top recent batters per team
        team_col = _pick(list(batter.columns), ["batter_team", "bat_team", "batting_team", "team"])
        bid_col = _pick(list(batter.columns), ["batter_id", "batter", "player_id"])
        name_col = _pick(list(batter.columns), ["player_name", "batter_name", "name"])
        score_col = _pick(list(batter.columns), ["bat_pa_roll15", "bat_pa_roll30", "bat_h_roll30"])
        tmp = batter.copy()
        tmp["game_date"] = pd.to_datetime(tmp.get("game_date"), errors="coerce")
        tmp = tmp[tmp["game_date"] < slate_date]
        tmp["_score"] = pd.to_numeric(tmp.get(score_col), errors="coerce") if score_col else 0.0
        lu = (
            tmp.sort_values([team_col, "_score"], ascending=[True, False], kind="mergesort")
            .groupby(team_col)
            .head(9)
            .copy()
        )
        lu["batter_team"] = lu[team_col]
        lu["batter_id"] = pd.to_numeric(lu[bid_col], errors="coerce").astype("Int64")
        lu["player_name"] = lu[name_col] if name_col else lu["batter_id"].astype(str)
        lu["lineup_slot"] = np.nan

    game_cols = [c for c in ["game_pk", "away_team", "home_team", "home_sp_id", "away_sp_id", "temperature", "weather_wind", "park_factor", "park_factor_hits", "venue_id", "canonical_park_key", "park_id"] if c in spine.columns]
    g = spine[game_cols].copy()
    g["game_pk"] = pd.to_numeric(g.get("game_pk"), errors="coerce").astype("Int64")

    if weather_path.exists():
        try:
            wg = pd.read_parquet(weather_path).copy()
            if "game_pk" in wg.columns:
                wg["game_pk"] = pd.to_numeric(wg["game_pk"], errors="coerce").astype("Int64")
                tcol = _pick(list(wg.columns), ["temperature", "temp_f", "game_temp", "weather_temp"])
                wcol = _pick(list(wg.columns), ["weather_wind", "wind_speed", "wind_mph", "wind"])
                keep = ["game_pk"] + ([tcol] if tcol else []) + ([wcol] if wcol else [])
                g = g.merge(wg[keep].drop_duplicates(subset=["game_pk"], keep="last"), on="game_pk", how="left", suffixes=("", "_wg"))
                if tcol:
                    g["temperature"] = pd.to_numeric(g.get("temperature"), errors="coerce").fillna(pd.to_numeric(g.get(f"{tcol}_wg"), errors="coerce"))
                    g = g.drop(columns=[f"{tcol}_wg"], errors="ignore")
                if wcol:
                    g["weather_wind"] = pd.to_numeric(g.get("weather_wind"), errors="coerce").fillna(pd.to_numeric(g.get(f"{wcol}_wg"), errors="coerce"))
                    g = g.drop(columns=[f"{wcol}_wg"], errors="ignore")
        except Exception:
            logging.exception("hit_prop live optional weather join failed; continuing")
    else:
        logging.warning("hit_prop live optional weather table missing: %s", weather_path)

    if parks_path.exists():
        try:
            parks = pd.read_parquet(parks_path).copy()
            pkey = _pick(list(parks.columns), ["canonical_park_key", "venue_id", "park_id"])
            if pkey and "park_factor_hits" in parks.columns:
                gkey = _pick(list(g.columns), ["canonical_park_key", "venue_id", "park_id"])
                if gkey:
                    g["_park_join_key"] = g[gkey].astype(str)
                    parks["_park_join_key"] = parks[pkey].astype(str)
                    g = g.merge(parks[["_park_join_key", "park_factor_hits"]].drop_duplicates(subset=["_park_join_key"], keep="last"), on="_park_join_key", how="left", suffixes=("", "_p"))
                    g["park_factor_hits"] = pd.to_numeric(g.get("park_factor_hits"), errors="coerce").fillna(pd.to_numeric(g.get("park_factor_hits_p"), errors="coerce"))
                    g = g.drop(columns=["_park_join_key", "park_factor_hits_p"], errors="ignore")
        except Exception:
            logging.exception("hit_prop live optional parks join failed; continuing")
    else:
        logging.warning("hit_prop live optional parks table missing: %s", parks_path)

    team_col = _pick(list(lu.columns), ["batter_team", "team", "team_abbrev", "team_name"])
    bid_col = _pick(list(lu.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(lu.columns), ["player_name", "batter_name", "name"])
    slot_col = _pick(list(lu.columns), ["lineup_slot", "bat_order", "batting_order", "lineup_position", "order"])

    lu["batter_team"] = lu[team_col]
    lu["batter_id"] = pd.to_numeric(lu[bid_col], errors="coerce").astype("Int64")
    lu["player_name"] = lu[name_col] if name_col else lu["batter_id"].astype(str)
    lu["lineup_slot"] = pd.to_numeric(lu[slot_col], errors="coerce") if slot_col else np.nan

    # attach games via team membership
    home = g.rename(columns={"home_team": "batter_team", "away_team": "opponent_team", "away_sp_id": "opp_pitcher_id"})
    home["home_away"] = 1.0
    away = g.rename(columns={"away_team": "batter_team", "home_team": "opponent_team", "home_sp_id": "opp_pitcher_id"})
    away["home_away"] = 0.0
    team_games = pd.concat([home, away], ignore_index=True, sort=False)
    board = lu.merge(team_games, on="batter_team", how="inner")

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
