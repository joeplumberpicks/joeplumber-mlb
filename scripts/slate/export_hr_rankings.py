from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_csv
from src.utils.logging import configure_logging, log_header


def _latest_model(model_dir: Path) -> Path:
    files = list(model_dir.glob("hr_batter_ranker_*.joblib"))
    if not files:
        files = list(model_dir.glob("*.joblib"))
    if not files:
        raise FileNotFoundError(f"No hr_batter_ranker model found in {model_dir.resolve()}")
    return max(files, key=lambda p: (p.stem, p.stat().st_mtime))



def _latest_backtest(backtest_dir: Path) -> Path:
    files = list(backtest_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No hr_batter_ranker backtest metadata found in {backtest_dir.resolve()}")
    return max(files, key=lambda p: (p.stem, p.stat().st_mtime))


def _load_train_features(backtest_dir: Path) -> list[str]:
    meta_path = _latest_backtest(backtest_dir)
    with meta_path.open("r", encoding="utf-8") as fp:
        meta = json.load(fp)

    train_features = meta.get("features")
    if not isinstance(train_features, list) or not train_features:
        raise ValueError(f"Invalid or missing 'features' in backtest metadata: {meta_path.resolve()}")

    logging.info("Using backtest metadata: %s", meta_path.resolve())
    return [str(c) for c in train_features]


def _build_scoring_matrix(df: pd.DataFrame, train_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    missing_cols: list[str] = []
    for col in train_features:
        if col not in df.columns:
            df[col] = 0.0
            missing_cols.append(col)

    X = df[train_features].copy()

    for col in train_features:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return X, missing_cols


def _select_export_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "game_date",
        "game_pk",
        "season",
        "home_team",
        "away_team",
        "park_id",
        "batter_id",
        "batter_team",
        "opp_sp_id",
        "p_hr",
        "rank",
        "tier",
    ]
    for col in ["batter_id", "p_hr", "rank", "tier"]:
        if col not in df.columns:
            df[col] = pd.NA
    return [c for c in preferred if c in df.columns]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export HR batter rankings (Top11/Top25).")
    p.add_argument("--date", type=str, default=None, help="YYYY-MM-DD; defaults to latest game_date in mart")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "export_hr_rankings.log")
    log_header("scripts/slate/export_hr_rankings.py", repo_root, config_path, dirs)

    model_path = _latest_model(dirs["models_dir"] / "hr_batter_ranker")
    logging.info("Using model: %s", model_path.resolve())
    model = load(model_path)

    train_features = _load_train_features(dirs["backtests_dir"] / "hr_batter_ranker")

    mart_path = dirs["marts_dir"] / "hr_batter_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing hr batter mart: {mart_path.resolve()}")

    mart = read_parquet(mart_path)
    if mart.empty:
        raise RuntimeError("hr_batter_features.parquet is empty; cannot export rankings")

    if "game_date" in mart.columns:
        mart["game_date"] = pd.to_datetime(mart["game_date"], errors="coerce")
    else:
        mart["game_date"] = pd.NaT

    if args.date:
        target_date = pd.to_datetime(args.date, errors="coerce")
        if pd.isna(target_date):
            raise ValueError(f"Invalid --date value: {args.date}")
    else:
        target_date = mart["game_date"].max()

    day_df = mart[mart["game_date"] == target_date].copy() if pd.notna(target_date) else mart.copy()
    if day_df.empty:
        raise RuntimeError(f"No rows found in hr_batter_features for date {target_date}")

    X, missing_created = _build_scoring_matrix(day_df, train_features)
    logging.info("Scoring with %s rows and %s features", len(X), len(train_features))
    logging.info("Missing training feature columns created: %s", len(missing_created))
    if missing_created:
        logging.info("Created missing feature columns: %s", missing_created)

    if hasattr(model, "predict_proba"):
        day_df["p_hr"] = model.predict_proba(X)[:, 1]
    else:
        day_df["p_hr"] = model.predict(X)
    logging.info("Unique p_hr after scoring: %s", int(day_df["p_hr"].nunique(dropna=True)))

    day_df = day_df.sort_values("p_hr", ascending=False).reset_index(drop=True)
    day_df["rank"] = np.arange(1, len(day_df) + 1)
    day_df["tier"] = np.where(day_df["rank"] <= 11, "Top11", np.where(day_df["rank"] <= 25, "12-25", "Other"))

    top11 = day_df[day_df["rank"] <= 11].copy()
    top25 = day_df[day_df["rank"] <= 25].copy()

    out_dir = dirs["outputs_dir"] / "HR"
    out_dir.mkdir(parents=True, exist_ok=True)
    top11_path = out_dir / "hr_top11.csv"
    top25_path = out_dir / "hr_top25.csv"

    export_cols = _select_export_columns(day_df)
    write_csv(top11[export_cols], top11_path)
    write_csv(top25[export_cols], top25_path)

    logging.info("Wrote Top11 rows: %s -> %s", len(top11), top11_path.resolve())
    logging.info("Wrote Top25 rows: %s -> %s", len(top25), top25_path.resolve())
    if not top25.empty:
        logging.info(
            "Top25 p_hr summary min/mean/max: %.6f / %.6f / %.6f",
            float(top25["p_hr"].min()),
            float(top25["p_hr"].mean()),
            float(top25["p_hr"].max()),
        )


if __name__ == "__main__":
    main()
