from __future__ import annotations

"""
Usage:
  python scripts/live/run_daily_moneyline_v1.py --season 2026 --date 2026-04-02
"""

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

MODEL_VERSION = "moneyline_sim_v1.0"
GRADE_THRESHOLDS = [
    ("A+", 0.70),
    ("A", 0.67),
    ("A-", 0.64),
    ("B+", 0.61),
    ("B", 0.58),
    ("B-", 0.55),
    ("C+", 0.53),
]
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+"]


REQUIRED_LIVE_COLS = [
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "home_sp_id",
    "away_sp_id",
    "venue_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily Moneyline v1.0 predictions using live spine + carryover features.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", required=True, help="Scoring date in YYYY-MM-DD")
    parser.add_argument("--fallback-season", type=int, default=2025)
    return parser.parse_args()


def _validate_date(date_str: str) -> pd.Timestamp:
    try:
        return pd.to_datetime(date_str, format="%Y-%m-%d", errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid --date format: {date_str}. Expected YYYY-MM-DD") from exc


def _grade_from_conf(prob: float) -> str:
    p = float(prob)
    for grade, threshold in GRADE_THRESHOLDS:
        if p >= threshold:
            return grade
    return "C+"


def _grade_at_or_above(grade: str, min_grade: str) -> bool:
    rank = {g: i for i, g in enumerate(GRADE_ORDER)}
    if grade not in rank or min_grade not in rank:
        raise ValueError(f"Unknown grade. grade={grade} min_grade={min_grade} supported={GRADE_ORDER}")
    return rank[grade] <= rank[min_grade]


def _american_odds_from_prob(prob: float) -> int:
    p = float(np.clip(prob, 1e-6, 1 - 1e-6))
    if p >= 0.5:
        odds = -100.0 * p / (1.0 - p)
    else:
        odds = 100.0 * (1.0 - p) / p
    return int(np.round(odds))


def _latest_model_path(models_dir: Path) -> Path:
    candidates = sorted((models_dir / "moneyline_sim").glob("moneyline_sim_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No model files found in {(models_dir / 'moneyline_sim').resolve()} matching moneyline_sim_*.joblib")
    return candidates[-1]


def _required_columns(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing from {path}: {missing}")


def _normalize_team_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["home_team_key"] = out["home_team"].map(normalize_team_abbr)
    out["away_team_key"] = out["away_team"].map(normalize_team_abbr)
    return out


def _carryover_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    keep: list[str] = []
    for c in numeric_cols:
        cl = c.lower()
        if cl in {"game_pk", "season", "target_home_win"}:
            continue
        if "id" in cl or cl.endswith("_pk"):
            continue
        keep.append(c)
    return keep


def _build_latest_side_features(fallback: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    fb = fallback.copy()
    fb["game_date"] = pd.to_datetime(fb.get("game_date"), errors="coerce")
    numeric_cols = _carryover_numeric_columns(fb)

    home_latest = (
        fb[["home_team_key", "game_date"] + numeric_cols]
        .dropna(subset=["home_team_key", "game_date"])
        .sort_values(["home_team_key", "game_date"], kind="mergesort")
        .groupby("home_team_key", as_index=False)
        .tail(1)
        .drop(columns=["game_date"])
        .rename(columns={c: f"carry_home_{c}" for c in numeric_cols})
    )

    away_latest = (
        fb[["away_team_key", "game_date"] + numeric_cols]
        .dropna(subset=["away_team_key", "game_date"])
        .sort_values(["away_team_key", "game_date"], kind="mergesort")
        .groupby("away_team_key", as_index=False)
        .tail(1)
        .drop(columns=["game_date"])
        .rename(columns={c: f"carry_away_{c}" for c in numeric_cols})
    )

    return home_latest, away_latest, numeric_cols


def _print_board(df: pd.DataFrame, date_str: str, top_n: int = 15) -> None:
    print(f"\nJOE PLUMBER MONEYLINE BOARD — {date_str}")
    if df is None or df.empty:
        print("(no rows)")
        return
    show = df.sort_values("pick_prob", ascending=False).head(top_n)
    for _, r in show.iterrows():
        away = str(r.get("away_team", ""))
        home = str(r.get("home_team", ""))
        pick = str(r.get("pick", ""))
        conf = 100.0 * float(r.get("pick_prob", 0.0))
        grade = str(r.get("grade", ""))
        fair = int(r.get("picked_side_fair_odds", 0))
        print(f"{away} @ {home}  {pick:<4}  {conf:>5.1f}%  {grade:<2}  fair={fair:+d}")


def main() -> None:
    args = parse_args()
    run_date = _validate_date(args.date)

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=False)
    configure_logging(dirs["logs_dir"] / "run_daily_moneyline_v1.log")
    log_header("scripts/live/run_daily_moneyline_v1.py", repo_root, config_path, dirs)

    live_spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not live_spine_path.exists():
        raise FileNotFoundError(f"Live spine file not found: {live_spine_path}")
    live = pd.read_parquet(live_spine_path)
    _required_columns(live, REQUIRED_LIVE_COLS, live_spine_path)
    live = _normalize_team_keys(live)

    fallback_path = dirs["marts_dir"] / "by_season" / f"moneyline_features_{args.fallback_season}.parquet"
    if not fallback_path.exists():
        raise FileNotFoundError(f"Fallback moneyline mart not found: {fallback_path}")
    fallback = pd.read_parquet(fallback_path)
    _required_columns(fallback, ["game_date", "home_team", "away_team"], fallback_path)
    fallback = _normalize_team_keys(fallback)

    home_latest, away_latest, numeric_cols = _build_latest_side_features(fallback)
    logging.info("Fallback numeric columns used for carryover: %s", len(numeric_cols))

    feat = live.merge(home_latest, on="home_team_key", how="left")
    feat = feat.merge(away_latest, on="away_team_key", how="left")

    model_path = _latest_model_path(dirs["models_dir"])
    model = joblib.load(model_path)
    if not hasattr(model, "feature_names_in_"):
        raise ValueError(f"Loaded model at {model_path} does not provide feature_names_in_")
    expected = [str(c) for c in model.feature_names_in_]

    numeric = feat.select_dtypes(include=[np.number]).copy()
    X = numeric.reindex(columns=expected, fill_value=np.nan)

    p_home = model.predict_proba(X)[:, 1]
    p_away = 1.0 - p_home

    out = feat[["game_date", "game_pk", "away_team", "home_team"]].copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["p_home_win"] = p_home
    out["p_away_win"] = p_away
    out["pick"] = np.where(out["p_home_win"] >= 0.5, "HOME", "AWAY")
    out["pick_prob"] = out[["p_home_win", "p_away_win"]].max(axis=1)
    out["grade"] = out["pick_prob"].map(_grade_from_conf)
    out["home_fair_odds"] = out["p_home_win"].map(_american_odds_from_prob)
    out["away_fair_odds"] = out["p_away_win"].map(_american_odds_from_prob)
    out["picked_side_fair_odds"] = np.where(out["pick"] == "HOME", out["home_fair_odds"], out["away_fair_odds"])

    out = out[pd.to_datetime(out["game_date"], errors="coerce").dt.date == run_date.date()].copy()
    if out.empty:
        logging.error("No rows found in live spine for date=%s", args.date)
        raise SystemExit(1)

    out_dir = dirs["outputs_dir"] / "moneyline_sim" / "v1.0" / "daily"
    pub_dir = dirs["outputs_dir"] / "moneyline_sim" / "v1.0" / "public"
    out_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)

    daily_path = out_dir / f"{args.date}_moneyline_predictions.csv"
    public_path = pub_dir / f"{args.date}_moneyline_A_tier_picks.csv"

    daily_cols = [
        "game_date",
        "game_pk",
        "away_team",
        "home_team",
        "p_home_win",
        "p_away_win",
        "pick",
        "pick_prob",
        "grade",
        "home_fair_odds",
        "away_fair_odds",
    ]
    out[daily_cols].to_csv(daily_path, index=False)

    public = out[out["grade"].map(lambda g: _grade_at_or_above(g, "A-"))].copy()
    public_cols = [
        "game_date",
        "away_team",
        "home_team",
        "pick",
        "pick_prob",
        "grade",
        "home_fair_odds",
        "away_fair_odds",
    ]
    public[public_cols].to_csv(public_path, index=False)

    logging.info(
        "Moneyline v1.0 daily run complete | date=%s season=%s rows=%s model=%s daily_csv=%s public_csv=%s",
        args.date,
        args.season,
        len(out),
        model_path,
        daily_path,
        public_path,
    )

    _print_board(out, args.date)
    print(f"rows={len(out)}")
    print(f"daily_csv={daily_path}")
    print(f"public_csv={public_path}")


if __name__ == "__main__":
    main()
