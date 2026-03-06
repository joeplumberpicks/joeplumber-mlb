from __future__ import annotations

import argparse
import logging
import sys
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


def main() -> None:
    args = parse_args()
    date_ts = pd.to_datetime(args.date, format="%Y-%m-%d", errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "run_daily_moneyline_v1.log")
    log_header("scripts/live/run_daily_moneyline_v1.py", repo_root, config_path, dirs)

    spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    fallback_mart_path = dirs["marts_dir"] / "by_season" / f"moneyline_features_{args.fallback_season}.parquet"
    model_dir = dirs["models_dir"] / "moneyline_sim"

    if not spine_path.exists():
        raise FileNotFoundError(f"Live spine not found: {spine_path}")
    if not fallback_mart_path.exists():
        raise FileNotFoundError(f"Fallback moneyline mart not found: {fallback_mart_path}")

    live_df = pd.read_parquet(spine_path).copy()
    required = ["game_pk", "game_date", "home_team", "away_team", "home_sp_id", "away_sp_id", "venue_id"]
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

    logging.info("moneyline live unique home_team_keys=%s", live_df["home_team_key"].nunique())
    logging.info("moneyline live unique away_team_keys=%s", live_df["away_team_key"].nunique())
    logging.info("moneyline fallback unique home_team_keys=%s", fallback_df["home_team_key"].nunique())
    logging.info("moneyline fallback unique away_team_keys=%s", fallback_df["away_team_key"].nunique())

    numeric_cols = [
        c
        for c in fallback_df.select_dtypes(include=[np.number]).columns
        if c not in {"game_pk", "season", "target_home_win"}
    ]

    home_latest = (
        fallback_df.sort_values("game_date", kind="mergesort")
        .groupby("home_team_key", dropna=False)
        .tail(1)[["home_team_key"] + numeric_cols]
        .rename(columns={c: f"home_{c}" for c in numeric_cols})
    )
    away_latest = (
        fallback_df.sort_values("game_date", kind="mergesort")
        .groupby("away_team_key", dropna=False)
        .tail(1)[["away_team_key"] + numeric_cols]
        .rename(columns={c: f"away_{c}" for c in numeric_cols})
    )

    merged = live_df.merge(home_latest, on="home_team_key", how="left")
    merged = merged.merge(away_latest, on="away_team_key", how="left")

    model_path = _latest_model(model_dir)
    model = joblib.load(model_path)

    X = merged.select_dtypes(include=[np.number]).copy()
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        raise ValueError(f"Model missing feature_names_in_: {model_path}")
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

    p_home = model.predict_proba(X)[:, 1]
    out = merged[["game_date", "game_pk", "away_team", "home_team"]].copy()
    out = out.assign(
        p_home_win=p_home,
        p_away_win=(1.0 - p_home),
    )
    out = out.assign(
        pick=np.where(out["p_home_win"] >= 0.5, "HOME", "AWAY"),
        pick_prob=np.maximum(out["p_home_win"], out["p_away_win"]),
    )
    out["grade"] = out["pick_prob"].map(grade_from_conf)
    out["home_fair_odds"] = out["p_home_win"].map(american_odds_from_prob)
    out["away_fair_odds"] = out["p_away_win"].map(american_odds_from_prob)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    daily_dir = dirs["outputs_dir"] / "moneyline_sim" / "v1.0" / "daily"
    public_dir = dirs["outputs_dir"] / "moneyline_sim" / "v1.0" / "public"
    daily_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    daily_path = daily_dir / f"{args.date}_moneyline_predictions.csv"
    public_path = public_dir / f"{args.date}_moneyline_A_tier_picks.csv"

    out.to_csv(daily_path, index=False)
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
    print(f"public_out={public_path}")


if __name__ == "__main__":
    main()
