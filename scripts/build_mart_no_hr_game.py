from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header

SUPPRESSION_KEYWORDS = [
    "hr",
    "barrel",
    "hard_hit",
    "launch_angle",
    "launch_speed",
    "fb",
    "gb",
    "air",
    "xbh",
    "iso",
    "slug",
    "contact",
    "whiff",
    "chase",
    "bb",
    "k",
]


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _pick_contains(cols: list[str], terms: list[str]) -> str | None:
    terms_l = [t.lower() for t in terms]
    for c in cols:
        lc = c.lower()
        if any(t in lc for t in terms_l):
            return c
    return None


def _filter_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.copy()
    if "game_date" in out.columns:
        gd = pd.to_datetime(out["game_date"], errors="coerce")
        out = out[gd.dt.year == season].copy()
    elif "season" in out.columns:
        out = out[pd.to_numeric(out["season"], errors="coerce") == season].copy()
    return out


def _load_spine_for_season(dirs: dict[str, Path], season: int) -> pd.DataFrame:
    hist_path = dirs["processed_dir"] / "model_spine_game.parquet"
    hist = _filter_season(read_parquet(hist_path), season)
    logging.info("historical spine path=%s season=%s rows=%s", hist_path.resolve(), season, len(hist))
    if len(hist):
        logging.info("using historical spine for season=%s", season)
        return hist

    live_dir = dirs["processed_dir"] / "live"
    candidates = sorted(live_dir.glob(f"model_spine_game_{season}_*.parquet"))
    if not candidates:
        logging.warning("no live spine candidates found for season=%s in %s", season, live_dir.resolve())
        return hist

    def _rank_live_path(p: Path) -> tuple[pd.Timestamp, float]:
        m = re.search(rf"model_spine_game_{season}_(\d{{4}}-\d{{2}}-\d{{2}})\.parquet$", p.name)
        ts = pd.to_datetime(m.group(1), errors="coerce") if m else pd.NaT
        return (ts if pd.notna(ts) else pd.Timestamp.min, p.stat().st_mtime)

    live_path = max(candidates, key=_rank_live_path)
    live_spine = read_parquet(live_path)
    live_spine = _filter_season(live_spine, season)
    logging.info("using live spine fallback season=%s path=%s rows=%s", season, live_path.resolve(), len(live_spine))
    return live_spine


def _usable_batter_rows(df: pd.DataFrame) -> pd.DataFrame:
    team_col = _pick(df, ["batter_team", "batting_team", "bat_team", "team", "offense_team"])
    if team_col is None or "game_pk" not in df.columns:
        return df.iloc[0:0].copy()
    out = df.copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out[team_col] = out[team_col].astype("string")
    out = out[out["game_pk"].notna() & out[team_col].notna()]
    return out


def _usable_pitcher_rows(df: pd.DataFrame) -> pd.DataFrame:
    pid_col = _pick(df, ["pitcher_id", "pitcher", "mlbam_pitcher_id", "player_id"])
    if pid_col is None or "game_pk" not in df.columns:
        return df.iloc[0:0].copy()
    out = df.copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out[pid_col] = pd.to_numeric(out[pid_col], errors="coerce").astype("Int64")
    out = out[out["game_pk"].notna() & out[pid_col].notna()]
    return out


def _sort_for_latest(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "game_date" in work.columns:
        work["_sort_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    else:
        work["_sort_date"] = pd.NaT
    if "game_pk" in work.columns:
        work["_sort_game_pk"] = pd.to_numeric(work["game_pk"], errors="coerce")
    else:
        work["_sort_game_pk"] = np.nan
    return work


def _select_batter_feature_cols(batter_roll: pd.DataFrame) -> list[str]:
    exclude = {
        "game_pk",
        "season",
        "batter_id",
        "batter",
        "pitcher_id",
        "player_id",
        "total_hr",
        "no_hr_game",
    }
    num_cols = []
    for c in batter_roll.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(batter_roll[c]):
            continue
        lc = c.lower()
        if any(k in lc for k in SUPPRESSION_KEYWORDS) and "team" not in lc:
            num_cols.append(c)
    return num_cols


def _team_rollup_by_game(batter_roll: pd.DataFrame) -> pd.DataFrame:
    team_col = _pick(batter_roll, ["batter_team", "batting_team", "bat_team", "team", "offense_team"])
    if team_col is None or "game_pk" not in batter_roll.columns:
        return pd.DataFrame(columns=["game_pk", "team"])
    feat_cols = _select_batter_feature_cols(batter_roll)
    if not feat_cols:
        return pd.DataFrame(columns=["game_pk", "team"])
    work = batter_roll[["game_pk", team_col] + feat_cols].copy()
    work = work.rename(columns={team_col: "team"})
    agg_spec = {c: ["mean", "max"] for c in feat_cols}
    agg = work.groupby(["game_pk", "team"], dropna=False).agg(agg_spec)
    agg.columns = [f"team_{c}_{m}" for c, m in agg.columns]
    return agg.reset_index()


def _team_rollup_latest_by_team(batter_roll: pd.DataFrame) -> pd.DataFrame:
    team_col = _pick(batter_roll, ["batter_team", "batting_team", "bat_team", "team", "offense_team"])
    batter_col = _pick(batter_roll, ["batter_id", "batter", "player_id"])
    if team_col is None or batter_col is None:
        return pd.DataFrame(columns=["team"])

    feat_cols = _select_batter_feature_cols(batter_roll)
    if not feat_cols:
        return pd.DataFrame(columns=["team"])

    work = _sort_for_latest(batter_roll)
    work[batter_col] = pd.to_numeric(work[batter_col], errors="coerce").astype("Int64")
    work = work[work[batter_col].notna() & work[team_col].notna()].copy()
    work = work.sort_values(["_sort_date", "_sort_game_pk"], ascending=[True, True])
    latest = work.groupby([team_col, batter_col], dropna=False).tail(1)

    team_agg = latest.groupby(team_col, dropna=False)[feat_cols].agg(["mean", "max"])
    team_agg.columns = [f"team_{c}_{m}" for c, m in team_agg.columns]
    team_agg = team_agg.reset_index().rename(columns={team_col: "team"})
    return team_agg


def _starter_features(spine: pd.DataFrame, pitcher_roll: pd.DataFrame, use_latest_per_pitcher: bool = False) -> pd.DataFrame:
    if pitcher_roll.empty:
        return spine

    pr = pitcher_roll.copy()
    pr["game_pk"] = pd.to_numeric(pr.get("game_pk"), errors="coerce").astype("Int64")
    pid_col = _pick(pr, ["pitcher_id", "pitcher", "mlbam_pitcher_id", "player_id"])
    if pid_col is None:
        return spine
    logging.info("starter merge pitcher id column=%s", pid_col)

    pr["pitcher_id"] = pd.to_numeric(pr[pid_col], errors="coerce").astype("Int64")
    exclude = {"game_pk", "pitcher_id", "season", pid_col}
    num_cols: list[str] = []
    for c in pr.columns:
        if c in exclude:
            continue
        series = pr[c] if pd.api.types.is_numeric_dtype(pr[c]) else pd.to_numeric(pr[c], errors="coerce")
        if series.notna().any():
            pr[c] = series
            num_cols.append(c)
    if not num_cols:
        return spine

    out = spine.copy()
    home_sp = _pick(out, ["home_sp_id", "home_starter_id", "home_pitcher_id", "home_starting_pitcher_id", "home_probable_pitcher_id", "home_pitcher"])
    away_sp = _pick(out, ["away_sp_id", "away_starter_id", "away_pitcher_id", "away_starting_pitcher_id", "away_probable_pitcher_id", "away_pitcher"])
    if home_sp is None or away_sp is None:
        return out

    out["home_sp_id"] = pd.to_numeric(out[home_sp], errors="coerce").astype("Int64")
    out["away_sp_id"] = pd.to_numeric(out[away_sp], errors="coerce").astype("Int64")
    starter_ids = pd.concat([out["home_sp_id"], out["away_sp_id"]], ignore_index=True).dropna().nunique()
    logging.info("starter merge distinct starter ids in spine=%s", int(starter_ids))

    if use_latest_per_pitcher:
        slim = _sort_for_latest(pr[["game_pk", "pitcher_id"] + num_cols + (["game_date"] if "game_date" in pr.columns else [])])
        slim = slim.sort_values(["_sort_date", "_sort_game_pk"], ascending=[True, True])
        slim = slim.groupby(["pitcher_id"], dropna=False).tail(1)
        h = slim[["pitcher_id"] + num_cols].rename(columns={c: f"home_sp_{c}" for c in num_cols})
        a = slim[["pitcher_id"] + num_cols].rename(columns={c: f"away_sp_{c}" for c in num_cols})
        out = out.merge(h, left_on="home_sp_id", right_on="pitcher_id", how="left").drop(columns=["pitcher_id"], errors="ignore")
        out = out.merge(a, left_on="away_sp_id", right_on="pitcher_id", how="left").drop(columns=["pitcher_id"], errors="ignore")
        home_pref = [c for c in out.columns if c.startswith("home_sp_")]
        away_pref = [c for c in out.columns if c.startswith("away_sp_")]
        home_matches = int(out[home_pref].notna().any(axis=1).sum()) if home_pref else 0
        away_matches = int(out[away_pref].notna().any(axis=1).sum()) if away_pref else 0
        logging.info("starter merge matched rows (latest-per-pitcher) home=%s away=%s", home_matches, away_matches)
        return out

    if use_latest_per_pitcher:
        slim = _sort_for_latest(pr[["game_pk", "pitcher_id"] + num_cols + (["game_date"] if "game_date" in pr.columns else [])])
        slim = slim.sort_values(["_sort_date", "_sort_game_pk"], ascending=[True, True])
        slim = slim.groupby(["pitcher_id"], dropna=False).tail(1)
        h = slim[["pitcher_id"] + num_cols].rename(columns={c: f"home_sp_{c}" for c in num_cols})
        a = slim[["pitcher_id"] + num_cols].rename(columns={c: f"away_sp_{c}" for c in num_cols})
        out = out.merge(h, left_on="home_sp_id", right_on="pitcher_id", how="left").drop(columns=["pitcher_id"], errors="ignore")
        out = out.merge(a, left_on="away_sp_id", right_on="pitcher_id", how="left").drop(columns=["pitcher_id"], errors="ignore")
        return out

    slim = pr[["game_pk", "pitcher_id"] + num_cols].drop_duplicates(subset=["game_pk", "pitcher_id"])
    h = slim.rename(columns={c: f"home_sp_{c}" for c in num_cols})
    a = slim.rename(columns={c: f"away_sp_{c}" for c in num_cols})
    out = out.merge(h, left_on=["game_pk", "home_sp_id"], right_on=["game_pk", "pitcher_id"], how="left")
    out = out.drop(columns=["pitcher_id"], errors="ignore")
    out = out.merge(a, left_on=["game_pk", "away_sp_id"], right_on=["game_pk", "pitcher_id"], how="left")
    out = out.drop(columns=["pitcher_id"], errors="ignore")
    home_pref = [c for c in out.columns if c.startswith("home_sp_")]
    away_pref = [c for c in out.columns if c.startswith("away_sp_")]
    home_matches = int(out[home_pref].notna().any(axis=1).sum()) if home_pref else 0
    away_matches = int(out[away_pref].notna().any(axis=1).sum()) if away_pref else 0
    logging.info("starter merge matched rows (game+pitcher) home=%s away=%s", home_matches, away_matches)

    if (home_matches == 0 and away_matches == 0) and not out.empty:
        latest = _sort_for_latest(pr[["game_pk", "pitcher_id"] + num_cols + (["game_date"] if "game_date" in pr.columns else [])])
        latest = latest.sort_values(["_sort_date", "_sort_game_pk"], ascending=[True, True]).groupby(["pitcher_id"], dropna=False).tail(1)
        h_latest = latest[["pitcher_id"] + num_cols].rename(columns={c: f"home_sp_{c}" for c in num_cols})
        a_latest = latest[["pitcher_id"] + num_cols].rename(columns={c: f"away_sp_{c}" for c in num_cols})
        out = out.drop(columns=[c for c in out.columns if c.startswith("home_sp_") or c.startswith("away_sp_")], errors="ignore")
        out = out.merge(h_latest, left_on="home_sp_id", right_on="pitcher_id", how="left").drop(columns=["pitcher_id"], errors="ignore")
        out = out.merge(a_latest, left_on="away_sp_id", right_on="pitcher_id", how="left").drop(columns=["pitcher_id"], errors="ignore")
        home_pref = [c for c in out.columns if c.startswith("home_sp_")]
        away_pref = [c for c in out.columns if c.startswith("away_sp_")]
        home_matches = int(out[home_pref].notna().any(axis=1).sum()) if home_pref else 0
        away_matches = int(out[away_pref].notna().any(axis=1).sum()) if away_pref else 0
        logging.info("starter merge fallback matched rows (pitcher-only latest) home=%s away=%s", home_matches, away_matches)
    return out


def _attach_engineered_features(mart: pd.DataFrame) -> pd.DataFrame:
    out = mart.copy()

    def _as_numeric_series(df: pd.DataFrame, value: object, default: float) -> pd.Series:
        if isinstance(value, str) and value in df.columns:
            return pd.to_numeric(df[value], errors="coerce")
        if isinstance(value, pd.Series):
            return pd.to_numeric(value.reindex(df.index), errors="coerce")
        return pd.Series(default, index=df.index, dtype="float64")

    temp_col = _pick_contains(list(out.columns), ["temperature", "temp"])
    wind_speed_col = _pick_contains(list(out.columns), ["wind_speed", "wind_mph", "wind"])
    if temp_col or wind_speed_col:
        temp_v = _as_numeric_series(out, temp_col, 70.0)
        wind_v = _as_numeric_series(out, wind_speed_col, 0.0)
        out["env_temp_wind_interaction"] = temp_v * wind_v

    park_hr_col = _pick_contains(list(out.columns), ["park_hr", "hr_factor", "home_run_factor", "hr_park"])
    wind_out_col = _pick_contains(list(out.columns), ["wind_out", "windout"])
    wind_in_col = _pick_contains(list(out.columns), ["wind_in", "windin"])
    if temp_col or wind_speed_col or park_hr_col or wind_out_col or wind_in_col:
        temp_v = _as_numeric_series(out, temp_col, 70.0)
        wind_v = _as_numeric_series(out, wind_speed_col, 0.0)
        park_v = _as_numeric_series(out, park_hr_col, 1.0)
        wind_out_v = _as_numeric_series(out, wind_out_col, 0.0)
        wind_in_v = _as_numeric_series(out, wind_in_col, 0.0)
        out["combined_park_weather_hr_index"] = (temp_v.fillna(70.0) / 70.0) * (1.0 + wind_v.fillna(0.0) / 20.0) * pd.to_numeric(park_v, errors="coerce").fillna(1.0)
        out["env_hr_suppression_proxy"] = (1.0 / out["combined_park_weather_hr_index"].clip(lower=0.25, upper=5.0)) + (wind_in_v.fillna(0.0) * 0.02) - (wind_out_v.fillna(0.0) * 0.02)

    home_sp_col = _pick_contains([c for c in out.columns if c.startswith("home_sp_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])
    away_sp_col = _pick_contains([c for c in out.columns if c.startswith("away_sp_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])
    home_team_power_col = _pick_contains([c for c in out.columns if c.startswith("home_team_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])
    away_team_power_col = _pick_contains([c for c in out.columns if c.startswith("away_team_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])

    if away_sp_col and home_team_power_col:
        out["starter_hr_suppression_gap_away_vs_home"] = pd.to_numeric(out[away_sp_col], errors="coerce") - pd.to_numeric(out[home_team_power_col], errors="coerce")
    if home_sp_col and away_team_power_col:
        out["starter_hr_suppression_gap_home_vs_away"] = pd.to_numeric(out[home_sp_col], errors="coerce") - pd.to_numeric(out[away_team_power_col], errors="coerce")

    return out


def _attach_engineered_features(mart: pd.DataFrame) -> pd.DataFrame:
    out = mart.copy()

    def _as_numeric_series(df: pd.DataFrame, value: object, default: float) -> pd.Series:
        if isinstance(value, str) and value in df.columns:
            return pd.to_numeric(df[value], errors="coerce")
        if isinstance(value, pd.Series):
            return pd.to_numeric(value.reindex(df.index), errors="coerce")
        return pd.Series(default, index=df.index, dtype="float64")

    temp_col = _pick_contains(list(out.columns), ["temperature", "temp"])
    wind_speed_col = _pick_contains(list(out.columns), ["wind_speed", "wind_mph", "wind"])
    if temp_col or wind_speed_col:
        temp_v = _as_numeric_series(out, temp_col, 70.0)
        wind_v = _as_numeric_series(out, wind_speed_col, 0.0)
        out["env_temp_wind_interaction"] = temp_v * wind_v

    park_hr_col = _pick_contains(list(out.columns), ["park_hr", "hr_factor", "home_run_factor", "hr_park"])
    wind_out_col = _pick_contains(list(out.columns), ["wind_out", "windout"])
    wind_in_col = _pick_contains(list(out.columns), ["wind_in", "windin"])
    if temp_col or wind_speed_col or park_hr_col or wind_out_col or wind_in_col:
        temp_v = _as_numeric_series(out, temp_col, 70.0)
        wind_v = _as_numeric_series(out, wind_speed_col, 0.0)
        park_v = _as_numeric_series(out, park_hr_col, 1.0)
        wind_out_v = _as_numeric_series(out, wind_out_col, 0.0)
        wind_in_v = _as_numeric_series(out, wind_in_col, 0.0)
        out["combined_park_weather_hr_index"] = (temp_v.fillna(70.0) / 70.0) * (1.0 + wind_v.fillna(0.0) / 20.0) * pd.to_numeric(park_v, errors="coerce").fillna(1.0)
        out["env_hr_suppression_proxy"] = (1.0 / out["combined_park_weather_hr_index"].clip(lower=0.25, upper=5.0)) + (wind_in_v.fillna(0.0) * 0.02) - (wind_out_v.fillna(0.0) * 0.02)

    home_sp_col = _pick_contains([c for c in out.columns if c.startswith("home_sp_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])
    away_sp_col = _pick_contains([c for c in out.columns if c.startswith("away_sp_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])
    home_team_power_col = _pick_contains([c for c in out.columns if c.startswith("home_team_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])
    away_team_power_col = _pick_contains([c for c in out.columns if c.startswith("away_team_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])

    if away_sp_col and home_team_power_col:
        out["starter_hr_suppression_gap_away_vs_home"] = pd.to_numeric(out[away_sp_col], errors="coerce") - pd.to_numeric(out[home_team_power_col], errors="coerce")
    if home_sp_col and away_team_power_col:
        out["starter_hr_suppression_gap_home_vs_away"] = pd.to_numeric(out[home_sp_col], errors="coerce") - pd.to_numeric(out[away_team_power_col], errors="coerce")

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build No-HR game feature mart")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_mart_no_hr_game.log")
    log_header("scripts/build_mart_no_hr_game.py", repo_root, config_path, dirs)

    logging.info("requested season=%s", args.season)

    spine = _load_spine_for_season(dirs, args.season)
    spine = spine.copy()
    spine["game_pk"] = pd.to_numeric(spine["game_pk"], errors="coerce").astype("Int64")

    all_batter_roll = read_parquet(dirs["processed_dir"] / "batter_game_rolling.parquet")
    all_pitcher_roll = read_parquet(dirs["processed_dir"] / "pitcher_game_rolling.parquet")

    curr_batter = _usable_batter_rows(_filter_season(all_batter_roll, args.season))
    prev_batter = _usable_batter_rows(_filter_season(all_batter_roll, args.season - 1))
    curr_pitcher = _usable_pitcher_rows(_filter_season(all_pitcher_roll, args.season))
    prev_pitcher = _usable_pitcher_rows(_filter_season(all_pitcher_roll, args.season - 1))

    logging.info(
        "batter rolling usable rows season=%s:%s prior=%s:%s",
        args.season,
        len(curr_batter),
        args.season - 1,
        len(prev_batter),
    )
    logging.info(
        "pitcher rolling usable rows season=%s:%s prior=%s:%s",
        args.season,
        len(curr_pitcher),
        args.season - 1,
        len(prev_pitcher),
    )

    use_batter_fallback = args.season == 2026 and curr_batter.empty and not prev_batter.empty
    use_pitcher_fallback = args.season == 2026 and curr_pitcher.empty and not prev_pitcher.empty

    if args.season == 2026 and curr_batter.empty and prev_batter.empty:
        logging.warning("No usable batter rolling rows found for 2026 or 2025 fallback")
    if args.season == 2026 and curr_pitcher.empty and prev_pitcher.empty:
        logging.warning("No usable pitcher rolling rows found for 2026 or 2025 fallback")

    batter_source = "previous_season_fallback" if use_batter_fallback else "requested_season"
    pitcher_source = "previous_season_fallback" if use_pitcher_fallback else "requested_season"
    batter_snapshot_season = args.season - 1 if use_batter_fallback else args.season
    pitcher_snapshot_season = args.season - 1 if use_pitcher_fallback else args.season

    mart = spine[
        [
            c
            for c in ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key"]
            if c in spine.columns
        ]
    ].copy()

    batter_for_rollup = prev_batter if use_batter_fallback else curr_batter
    if use_batter_fallback:
        team_roll = _team_rollup_latest_by_team(batter_for_rollup)
        if not team_roll.empty and {"home_team", "away_team"}.issubset(mart.columns):
            home_roll = team_roll.rename(columns={c: f"home_{c}" for c in team_roll.columns if c != "team"})
            away_roll = team_roll.rename(columns={c: f"away_{c}" for c in team_roll.columns if c != "team"})
            mart = mart.merge(home_roll, left_on="home_team", right_on="team", how="left").drop(columns=["team"], errors="ignore")
            mart = mart.merge(away_roll, left_on="away_team", right_on="team", how="left").drop(columns=["team"], errors="ignore")
    else:
        team_roll = _team_rollup_by_game(batter_for_rollup)
        if not team_roll.empty and {"home_team", "away_team"}.issubset(mart.columns):
            home_roll = team_roll.rename(columns={c: f"home_{c}" for c in team_roll.columns if c not in {"game_pk", "team"}})
            away_roll = team_roll.rename(columns={c: f"away_{c}" for c in team_roll.columns if c not in {"game_pk", "team"}})
            mart = mart.merge(home_roll, left_on=["game_pk", "home_team"], right_on=["game_pk", "team"], how="left")
            mart = mart.drop(columns=["team"], errors="ignore")
            mart = mart.merge(away_roll, left_on=["game_pk", "away_team"], right_on=["game_pk", "team"], how="left")
            mart = mart.drop(columns=["team"], errors="ignore")

    pitcher_for_merge = prev_pitcher if use_pitcher_fallback else curr_pitcher
    mart = _starter_features(mart, pitcher_for_merge, use_latest_per_pitcher=use_pitcher_fallback)

    weather_path = dirs["processed_dir"] / "weather_game.parquet"
    if weather_path.exists():
        weather = _filter_season(read_parquet(weather_path), args.season)
        if "game_pk" in weather.columns:
            weather["game_pk"] = pd.to_numeric(weather["game_pk"], errors="coerce").astype("Int64")
            weather_cols = [
                c
                for c in weather.columns
                if c == "game_pk" or c.startswith(("temp", "wind", "weather", "humidity", "pressure"))
            ]
            if len(weather_cols) > 1:
                mart = mart.merge(weather[weather_cols].drop_duplicates(subset=["game_pk"]), on="game_pk", how="left")

    parks_path = dirs["processed_dir"] / "parks.parquet"
    if parks_path.exists() and "park_id" in mart.columns:
        parks = read_parquet(parks_path)
        pid = _pick(parks, ["park_id", "venue_id", "id"])
        if pid is not None:
            parks = parks.copy()
            parks["park_id"] = parks[pid]
            park_cols = [c for c in parks.columns if c == "park_id" or "hr" in c.lower() or "park_factor" in c.lower()]
            if len(park_cols) > 1:
                mart = mart.merge(parks[park_cols].drop_duplicates(subset=["park_id"]), on="park_id", how="left")

    target_path = dirs["processed_dir"] / "targets" / "game" / f"targets_no_hr_game_{args.season}.parquet"
    if target_path.exists():
        targets = read_parquet(target_path)
        targets["game_pk"] = pd.to_numeric(targets["game_pk"], errors="coerce").astype("Int64")
        merge_cols = [c for c in ["game_pk", "total_hr", "no_hr_game"] if c in targets.columns]
        mart = mart.merge(targets[merge_cols], on="game_pk", how="left")
    else:
        logging.info("Targets file missing for season=%s (expected for live scoring): %s", args.season, target_path.resolve())

    if "total_hr" not in mart.columns:
        mart["total_hr"] = pd.Series([pd.NA] * len(mart), dtype="Int64")
    if "no_hr_game" not in mart.columns:
        mart["no_hr_game"] = pd.Series([pd.NA] * len(mart), dtype="Int64")
    else:
        mart["no_hr_game"] = pd.to_numeric(mart["no_hr_game"], errors="coerce").astype("Int64")

    mart["batter_feature_source"] = batter_source
    mart["pitcher_feature_source"] = pitcher_source
    mart["fallback_used"] = bool(use_batter_fallback or use_pitcher_fallback)
    mart["feature_snapshot_season"] = np.where(use_batter_fallback or use_pitcher_fallback, args.season - 1, args.season)
    mart["batter_feature_snapshot_season"] = batter_snapshot_season
    mart["pitcher_feature_snapshot_season"] = pitcher_snapshot_season

    mart = _attach_engineered_features(mart)

    out_dir = dirs["processed_dir"] / "marts" / "no_hr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"no_hr_game_features_{args.season}.parquet"
    if out_path.exists() and not args.force:
        logging.info("exists and force=False: %s", out_path.resolve())
    else:
        write_parquet(mart, out_path)

    home_feat_cols = [c for c in mart.columns if c.startswith("home_team_")]
    away_feat_cols = [c for c in mart.columns if c.startswith("away_team_")]
    starter_cols = [c for c in mart.columns if c.startswith("home_sp_") or c.startswith("away_sp_")]

    home_games_with_feats = int(mart[home_feat_cols].notna().any(axis=1).sum()) if home_feat_cols else 0
    away_games_with_feats = int(mart[away_feat_cols].notna().any(axis=1).sum()) if away_feat_cols else 0
    games_with_starter = int(mart[starter_cols].notna().any(axis=1).sum()) if starter_cols else 0

    logging.info(
        "fallback diagnostics requested_season=%s batter_source=%s pitcher_source=%s fallback_used=%s",
        args.season,
        batter_source,
        pitcher_source,
        use_batter_fallback or use_pitcher_fallback,
    )
    logging.info(
        "no_hr_mart rows=%s unique_games=%s games_with_home_features=%s games_with_away_features=%s games_with_starter_features=%s path=%s",
        len(mart),
        int(mart["game_pk"].nunique()) if "game_pk" in mart.columns else 0,
        home_games_with_feats,
        away_games_with_feats,
        games_with_starter,
        out_path.resolve(),
    )
    print(f"no_hr_mart -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
