from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.live.daily_context import (
    build_game_level_lineup_features,
    load_live_spine,
    load_live_weather,
    load_projected_lineups,
    merge_live_context,
    resolve_live_paths,
    run_live_preflight,
    summarize_live_context,
)
from src.utils.logging import configure_logging, log_header
from src.utils.team_ids import normalize_team_abbr

GRADE_THRESHOLDS = [
    ("A+", 0.70),
    ("A", 0.67),
    ("A-", 0.64),
    ("B+", 0.61),
    ("B", 0.58),
    ("B-", 0.55),
    ("C+", 0.53),
]
A_TIER = {"A+", "A", "A-"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run daily Moneyline v1.0 scoring from live spine + carryover features.")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--fallback-season", type=int, default=2025)
    p.add_argument("--auto-build", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-lineups", action="store_true")
    p.add_argument("--skip-weather", action="store_true")
    p.add_argument("--permissive-live-context", action="store_true")
    p.add_argument("--outdir-name", default="moneyline")
    return p.parse_args()


def grade_from_conf(prob: float) -> str:
    p = float(prob)
    for grade, threshold in GRADE_THRESHOLDS:
        if p >= threshold:
            return grade
    return "C"


def american_odds_from_prob(p: float) -> int:
    p = float(p)
    p = min(max(p, 1e-6), 1 - 1e-6)
    if p >= 0.5:
        return int(round(-100.0 * p / (1.0 - p)))
    return int(round(100.0 * (1.0 - p) / p))


def _print_board(df: pd.DataFrame, date_str: str) -> None:
    print(f"\nJOE PLUMBER MONEYLINE BOARD — {date_str}")
    print("---------------------------------------")
    for _, r in df.sort_values("pick_prob", ascending=False, kind="mergesort").iterrows():
        matchup = f"{r['away_team']} @ {r['home_team']}"
        picked_odds = r["home_fair_odds"] if r["pick"] == "HOME" else r["away_fair_odds"]
        print(f"{matchup:<26} {r['pick']:<4} {100*r['pick_prob']:>5.1f}%  {r['grade']:<2}  fair={picked_odds:+d}")
    print("")


def _latest_model(models_dir: Path) -> Path:
    cand = sorted(models_dir.glob("moneyline_sim_*.joblib"))
    if not cand:
        raise FileNotFoundError(f"No moneyline model found in {models_dir}")
    return cand[-1]


def _first_existing(cols: Iterable[str], candidates: list[str]) -> str | None:
    colset = set(cols)
    for c in candidates:
        if c in colset:
            return c
    return None


def _latest_team_offense_rollups(batter_roll: pd.DataFrame, target_date: pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
    if batter_roll.empty:
        return pd.DataFrame(), None

    team_col = _first_existing(
        batter_roll.columns,
        ["batter_team", "batting_team", "bat_team", "offense_team", "team", "team_abbrev", "team_name"],
    )
    if team_col is None:
        logging.warning("moneyline live offense rollups skipped: no team column in batter rolling")
        return pd.DataFrame(), None

    bat = batter_roll.copy()
    bat["game_date"] = pd.to_datetime(bat.get("game_date"), errors="coerce")
    bat = bat[(bat["game_date"].notna()) & (bat["game_date"] < target_date)].copy()
    if bat.empty:
        return pd.DataFrame(), team_col

    bat["team_key"] = bat[team_col].map(normalize_team_abbr)
    bat = bat[bat["team_key"].notna()].copy()
    if bat.empty:
        return pd.DataFrame(), team_col

    exclude = {
        "game_pk",
        "game_date",
        "season",
        "batter_id",
        "batter",
        "pitcher_id",
        "home_team",
        "away_team",
        "team_key",
        team_col,
    }
    metric_cols = [c for c in bat.select_dtypes(include=[np.number]).columns if c not in exclude]
    if not metric_cols:
        return pd.DataFrame(), team_col

    latest_dates = bat.groupby("team_key", as_index=False)["game_date"].max().rename(columns={"game_date": "_max_date"})
    latest = bat.merge(latest_dates, on="team_key", how="inner")
    latest = latest[latest["game_date"] == latest["_max_date"]]
    rollup = latest.groupby("team_key", as_index=True)[metric_cols].mean(numeric_only=True)
    return rollup, team_col


def _latest_pitcher_rollups(pitcher_roll: pd.DataFrame, target_date: pd.Timestamp) -> tuple[pd.DataFrame, str | None]:
    if pitcher_roll.empty:
        return pd.DataFrame(), None

    key_col = _first_existing(pitcher_roll.columns, ["pitcher_id", "pitcher", "player_id", "mlb_id"])
    if key_col is None:
        logging.warning("moneyline live pitcher rollups skipped: no pitcher id column in pitcher rolling")
        return pd.DataFrame(), None

    pit = pitcher_roll.copy()
    pit["game_date"] = pd.to_datetime(pit.get("game_date"), errors="coerce")
    pit = pit[(pit["game_date"].notna()) & (pit["game_date"] < target_date)].copy()
    if pit.empty:
        return pd.DataFrame(), key_col

    pit["pitcher_key"] = pd.to_numeric(pit[key_col], errors="coerce").astype("Int64")
    pit = pit[pit["pitcher_key"].notna()].copy()
    if pit.empty:
        return pd.DataFrame(), key_col

    exclude = {
        "game_pk",
        "game_date",
        "season",
        "pitcher_id",
        "pitcher",
        "player_id",
        "mlb_id",
        "home_team",
        "away_team",
        "pitcher_key",
    }
    metric_cols = [c for c in pit.select_dtypes(include=[np.number]).columns if c not in exclude]
    if not metric_cols:
        return pd.DataFrame(), key_col

    pit = pit.sort_values(["pitcher_key", "game_date"], kind="mergesort")
    latest = pit.groupby("pitcher_key", as_index=False).tail(1).set_index("pitcher_key")
    return latest[metric_cols], key_col


def _pick_metric_name(source_cols: set[str], expected_col: str, prefix: str) -> str | None:
    if expected_col in source_cols:
        return expected_col
    suffix = expected_col[len(prefix) :] if expected_col.startswith(prefix) else expected_col
    candidates = [suffix, f"{prefix}{suffix}"]
    for c in candidates:
        if c in source_cols:
            return c
    return None


def main() -> None:
    args = parse_args()
    date_ts = pd.to_datetime(args.date, format="%Y-%m-%d", errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "run_daily_moneyline_v1.log")
    log_header("scripts/live/run_daily_moneyline_v1.py", repo_root, config_path, dirs)

    run_live_preflight(
        repo_root=repo_root,
        config_path=config_path,
        season=args.season,
        date_str=args.date,
        auto_build=bool(args.auto_build),
        force_spine=True,
        build_lineups=not args.skip_lineups,
        build_weather=not args.skip_weather,
        permissive_live_context=bool(args.permissive_live_context),
    )
    live_paths = resolve_live_paths(config=config, season=args.season, date_str=args.date)
    spine_path = live_paths["live_spine_path"]
    fallback_mart_path = dirs["marts_dir"] / "by_season" / f"moneyline_features_{args.fallback_season}.parquet"
    batter_roll_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    pitcher_roll_path = dirs["processed_dir"] / "pitcher_game_rolling.parquet"
    model_dir = dirs["models_dir"] / "moneyline_sim"

    if not spine_path.exists():
        raise FileNotFoundError(f"Live spine not found: {spine_path}")
    if not fallback_mart_path.exists():
        raise FileNotFoundError(f"Fallback moneyline mart not found: {fallback_mart_path}")

    live_df = load_live_spine(spine_path).copy()
    live_weather = load_live_weather(live_paths["live_weather_path"]) if not args.skip_weather else pd.DataFrame()
    projected_lineups = (
        load_projected_lineups(live_paths["projected_lineups_path"]) if not args.skip_lineups else pd.DataFrame()
    )
    batter_roll_for_lineups = pd.read_parquet(batter_roll_path).copy() if batter_roll_path.exists() else pd.DataFrame()
    lineup_game = build_game_level_lineup_features(projected_lineups, batter_roll_for_lineups, date_ts)
    live_context = merge_live_context(live_df, live_weather, lineup_game)
    smoke = summarize_live_context(live_context)
    print(
        "live_feature_smoke "
        f"games={smoke['games']} "
        f"pct_with_starters={smoke['pct_with_starters']:.2f} "
        f"pct_with_weather={smoke['pct_with_weather']:.2f} "
        f"away_lineup_found={smoke['away_lineup_found']} "
        f"home_lineup_found={smoke['home_lineup_found']}"
    )

    live_df = live_context.copy()
    required = ["game_pk", "game_date", "home_team", "away_team", "home_sp_id", "away_sp_id"]
    missing = [c for c in required if c not in live_df.columns]
    if missing:
        raise ValueError(f"Live spine missing required columns: {missing}")

    fallback_df = pd.read_parquet(fallback_mart_path).copy()
    if "game_date" not in fallback_df.columns:
        raise ValueError(f"Fallback mart missing game_date: {fallback_mart_path}")

    live_df["game_date"] = pd.to_datetime(live_df["game_date"], errors="coerce")
    fallback_df["game_date"] = pd.to_datetime(fallback_df["game_date"], errors="coerce")

    live_df["home_team_key"] = live_df["home_team"].apply(normalize_team_abbr)
    live_df["away_team_key"] = live_df["away_team"].apply(normalize_team_abbr)
    fallback_df["home_team_key"] = fallback_df["home_team"].apply(normalize_team_abbr)
    fallback_df["away_team_key"] = fallback_df["away_team"].apply(normalize_team_abbr)

    batter_roll = pd.read_parquet(batter_roll_path).copy() if batter_roll_path.exists() else pd.DataFrame()
    pitcher_roll = pd.read_parquet(pitcher_roll_path).copy() if pitcher_roll_path.exists() else pd.DataFrame()
    off_roll, off_team_col = _latest_team_offense_rollups(batter_roll, date_ts)
    pit_roll, pit_key_col = _latest_pitcher_rollups(pitcher_roll, date_ts)

    live_df["away_sp_id"] = pd.to_numeric(live_df["away_sp_id"], errors="coerce").astype("Int64")
    live_df["home_sp_id"] = pd.to_numeric(live_df["home_sp_id"], errors="coerce").astype("Int64")

    logging.info("moneyline live unique home_team_keys=%s", live_df["home_team_key"].nunique())
    logging.info("moneyline live unique away_team_keys=%s", live_df["away_team_key"].nunique())
    logging.info("moneyline fallback unique home_team_keys=%s", fallback_df["home_team_key"].nunique())
    logging.info("moneyline fallback unique away_team_keys=%s", fallback_df["away_team_key"].nunique())

    numeric_cols = [
        c
        for c in fallback_df.select_dtypes(include=[np.number]).columns
        if c not in {"game_pk", "season", "target_home_win"}
    ]

    fallback_sorted = fallback_df.sort_values("game_date", kind="mergesort")
    home_latest = (
        fallback_sorted.groupby("home_team_key", dropna=False)
        .tail(1)[["home_team_key"] + numeric_cols]
        .drop_duplicates(subset=["home_team_key"], keep="last")
        .set_index("home_team_key")
    )
    away_latest = (
        fallback_sorted.groupby("away_team_key", dropna=False)
        .tail(1)[["away_team_key"] + numeric_cols]
        .drop_duplicates(subset=["away_team_key"], keep="last")
        .set_index("away_team_key")
    )

    merged = live_df.copy()

    model_path = _latest_model(model_dir)
    model = joblib.load(model_path)

    X = merged.select_dtypes(include=[np.number]).copy()
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        raise ValueError(f"Model missing feature_names_in_: {model_path}")

    X = pd.DataFrame(index=merged.index, columns=expected, dtype="float64")

    off_cols = set(off_roll.columns) if not off_roll.empty else set()
    pit_cols = set(pit_roll.columns) if not pit_roll.empty else set()

    away_off_match = int(live_df["away_team_key"].isin(off_roll.index).sum()) if not off_roll.empty else 0
    home_off_match = int(live_df["home_team_key"].isin(off_roll.index).sum()) if not off_roll.empty else 0
    away_pit_match = int(live_df["away_sp_id"].isin(pit_roll.index).sum()) if not pit_roll.empty else 0
    home_pit_match = int(live_df["home_sp_id"].isin(pit_roll.index).sum()) if not pit_roll.empty else 0

    bat_populated = 0
    pit_populated = 0
    diff_off_populated = 0
    for col in expected:
        if col.startswith("bat_") and off_cols:
            m = _pick_metric_name(off_cols, col, "bat_")
            if m is not None:
                away_v = merged["away_team_key"].map(off_roll[m])
                home_v = merged["home_team_key"].map(off_roll[m])
                X[col] = (pd.to_numeric(away_v, errors="coerce") + pd.to_numeric(home_v, errors="coerce")) / 2.0
                if X[col].notna().any():
                    bat_populated += 1
        elif col.startswith("pit_") and pit_cols:
            m = _pick_metric_name(pit_cols, col, "pit_")
            if m is not None:
                away_v = merged["away_sp_id"].map(pit_roll[m])
                home_v = merged["home_sp_id"].map(pit_roll[m])
                X[col] = (pd.to_numeric(away_v, errors="coerce") + pd.to_numeric(home_v, errors="coerce")) / 2.0
                if X[col].notna().any():
                    pit_populated += 1
        elif col.startswith("diff_off_") and off_cols:
            m = _pick_metric_name(off_cols, col, "diff_off_")
            if m is not None:
                away_v = merged["away_team_key"].map(off_roll[m])
                home_v = merged["home_team_key"].map(off_roll[m])
                X[col] = pd.to_numeric(away_v, errors="coerce") - pd.to_numeric(home_v, errors="coerce")
                if X[col].notna().any():
                    diff_off_populated += 1
        elif col in merged.columns:
            X[col] = pd.to_numeric(merged[col], errors="coerce")

    # fallback mart: fill remaining feature gaps without overwriting rolling-derived values
    home_num_cols = set(home_latest.columns) if not home_latest.empty else set()
    away_num_cols = set(away_latest.columns) if not away_latest.empty else set()

    def _fallback_side_series(side_df: pd.DataFrame, team_keys: pd.Series, side_prefix: str, target_col: str) -> pd.Series:
        if side_df.empty:
            return pd.Series(np.nan, index=team_keys.index, dtype="float64")
        source_col = _pick_metric_name(set(side_df.columns), target_col, side_prefix)
        if source_col is None:
            source_col = _pick_metric_name(set(side_df.columns), target_col, "")
        if source_col is None:
            return pd.Series(np.nan, index=team_keys.index, dtype="float64")
        return pd.to_numeric(team_keys.map(side_df[source_col]), errors="coerce")

    for col in expected:
        mask = X[col].isna()
        if not bool(mask.any()):
            continue

        home_v = _fallback_side_series(home_latest, merged["home_team_key"], "home_", col)
        away_v = _fallback_side_series(away_latest, merged["away_team_key"], "away_", col)

        candidate = None
        if col.startswith("diff_off_"):
            candidate = away_v - home_v
        else:
            avg = (away_v + home_v) / 2.0
            single = pd.Series(np.nan, index=merged.index, dtype="float64")
            direct_home = _pick_metric_name(home_num_cols, col, "") if home_num_cols else None
            direct_away = _pick_metric_name(away_num_cols, col, "") if away_num_cols else None
            if direct_home is not None:
                single = single.fillna(pd.to_numeric(merged["home_team_key"].map(home_latest[direct_home]), errors="coerce"))
            if direct_away is not None:
                single = single.fillna(pd.to_numeric(merged["away_team_key"].map(away_latest[direct_away]), errors="coerce"))
            candidate = avg.fillna(single)

        X.loc[mask, col] = candidate.loc[mask]

    X = X.reindex(columns=expected, fill_value=np.nan)

    logging.info("moneyline live X shape rows=%s cols=%s", X.shape[0], X.shape[1])
    avg_missing = float(X.isna().mean().mean()) if X.shape[1] else 0.0
    logging.info("moneyline live avg_missing_rate=%.6f", avg_missing)

    varied = X.nunique(dropna=False)
    varied_count = int((varied > 1).sum())
    logging.info("moneyline live varied_feature_cols=%s", varied_count)

    top_varied = [(c, int(v)) for c, v in varied.sort_values(ascending=False).head(30).items()]
    logging.info("moneyline live top30_varied_cols=%s", top_varied)

    missing = X.isna().mean().sort_values(ascending=False)
    top_missing = [(c, float(v)) for c, v in missing.head(30).items()]
    logging.info("moneyline live top30_missing_cols=%s", top_missing)

    all_missing_cols = [c for c in X.columns if X[c].isna().all()]
    logging.info(
        "moneyline live all_missing_feature_cols=%s first30=%s",
        len(all_missing_cols),
        all_missing_cols[:30],
    )
    logging.info(
        "moneyline live populated_feature_families bat_cols=%s pit_cols=%s diff_off_cols=%s",
        bat_populated,
        pit_populated,
        diff_off_populated,
    )
    logging.info(
        "moneyline live offense_rolling_coverage away_matches=%s/%s home_matches=%s/%s team_col=%s",
        away_off_match,
        len(live_df),
        home_off_match,
        len(live_df),
        off_team_col,
    )
    logging.info(
        "moneyline live pitcher_rolling_coverage away_matches=%s/%s home_matches=%s/%s pitcher_col=%s",
        away_pit_match,
        len(live_df),
        home_pit_match,
        len(live_df),
        pit_key_col,
    )

    p_home_win_raw = model.predict_proba(X)[:, 1]
    p_home_win = 0.5 + 0.35 * (p_home_win_raw - 0.5)
    p_home_win = np.clip(p_home_win, 0.35, 0.65)
    p_away_win = 1.0 - p_home_win

    logging.info(
        "moneyline live raw_p_home summary min=%.6f mean=%.6f max=%.6f",
        float(np.min(p_home_win_raw)) if len(p_home_win_raw) else float("nan"),
        float(np.mean(p_home_win_raw)) if len(p_home_win_raw) else float("nan"),
        float(np.max(p_home_win_raw)) if len(p_home_win_raw) else float("nan"),
    )
    logging.info(
        "moneyline live calibrated_p_home summary min=%.6f mean=%.6f max=%.6f",
        float(np.min(p_home_win)) if len(p_home_win) else float("nan"),
        float(np.mean(p_home_win)) if len(p_home_win) else float("nan"),
        float(np.max(p_home_win)) if len(p_home_win) else float("nan"),
    )

    base_cols = ["game_date", "season", "game_pk", "away_team", "home_team", "away_sp_id", "home_sp_id", "away_sp_name", "home_sp_name"]
    for c in base_cols:
        if c not in merged.columns:
            merged[c] = pd.NA
    out = merged[base_cols].copy()
    out = out.assign(
        p_home_win_raw=p_home_win_raw,
        p_home_win=p_home_win,
        p_away_win=p_away_win,
    )
    out = out.assign(
        pick=np.where(out["p_home_win"] >= 0.5, "HOME", "AWAY"),
        pick_prob=np.maximum(out["p_home_win"], out["p_away_win"]),
    )
    out["grade"] = out["pick_prob"].map(grade_from_conf)
    out["home_fair_odds"] = out["p_home_win"].map(american_odds_from_prob)
    out["away_fair_odds"] = out["p_away_win"].map(american_odds_from_prob)
    out["has_weather"] = pd.Series(merged.get("has_weather", False)).fillna(False).astype(bool)
    out["has_projected_lineups"] = pd.Series(merged.get("has_projected_lineups", False)).fillna(False).astype(bool)
    for c in [
        "away_lineup_completeness_score",
        "home_lineup_completeness_score",
        "away_lineup_quality_score",
        "home_lineup_quality_score",
    ]:
        out[c] = pd.to_numeric(merged.get(c), errors="coerce")
    out["feature_source_summary"] = "live_spine|carry_rollups|fallback_mart"
    if not args.skip_lineups:
        out["feature_source_summary"] += "|projected_lineups"
    if not args.skip_weather:
        out["feature_source_summary"] += "|live_weather"
    out["model_version"] = model_path.stem
    out["run_date"] = args.date
    out["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    daily_dir = dirs["outputs_dir"] / "daily" / args.outdir_name
    public_dir = daily_dir
    daily_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    daily_path = daily_dir / f"moneyline_board_{args.season}_{args.date}.csv"
    daily_parquet_path = daily_dir / f"moneyline_board_{args.season}_{args.date}.parquet"
    public_path = public_dir / f"moneyline_board_public_{args.season}_{args.date}.csv"

    out.to_csv(daily_path, index=False)
    out.to_parquet(daily_parquet_path, index=False)
    out[out["grade"].isin(A_TIER)].sort_values("pick_prob", ascending=False, kind="mergesort").to_csv(public_path, index=False)

    _print_board(out, args.date)
    logging.info(
        "moneyline daily run complete date=%s season=%s fallback=%s rows=%s model=%s daily=%s public=%s",
        args.date,
        args.season,
        args.fallback_season,
        len(out),
        model_path,
        daily_path,
        public_path,
    )
    print(f"daily_out={daily_path}")
    print(f"daily_out_parquet={daily_parquet_path}")
    print(f"public_out={public_path}")


if __name__ == "__main__":
    main()
