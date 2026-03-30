from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_json
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Elite HR Engine (player-level HR probability model).")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--mart-file", type=str, default="hr_batter_features.parquet")
    parser.add_argument("--target-col", type=str, default="target_hr")
    return parser.parse_args()


def _feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    drop_exact = {
        target_col,
        "game_pk",
        "game_date",
        "season",
        "batter_id",
        "batter_team",
        "home_team",
        "away_team",
        "opp_sp_id",
        "park_id",
        "canonical_park_key",
    }
    cols = []
    for c in df.columns:
        if c in drop_exact:
            continue
        cl = c.lower()
        if cl.endswith("_id") or "game_pk" in cl:
            continue
        cols.append(c)
    return cols


def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return [
        {"bin": int(i + 1), "pred_mean": float(pp), "actual_rate": float(pt)}
        for i, (pt, pp) in enumerate(zip(prob_true, prob_pred))
    ]


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "train_hr_model.log")
    log_header("scripts/train/train_hr_model.py", repo_root, config_path, dirs)

    candidates = [dirs["marts_dir"] / args.mart_file, dirs["marts_dir"] / "by_season" / f"hr_batter_features_2025.parquet"]
    mart_path = next((p for p in candidates if p.exists()), None)
    if mart_path is None:
        raise FileNotFoundError(f"Could not find HR mart. Checked: {[str(p) for p in candidates]}")

    df = read_parquet(mart_path)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        df["season"] = df["game_date"].dt.year
    elif "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
    else:
        raise ValueError("HR mart requires game_date or season for train/test split.")

    if args.target_col not in df.columns:
        raise ValueError(f"Missing {args.target_col} in {mart_path.resolve()}")
    df[args.target_col] = (pd.to_numeric(df[args.target_col], errors="coerce").fillna(0) > 0).astype(int)

    feat_cols = _feature_columns(df, args.target_col)
    logging.info("hr_model feature count before numeric filter: %s", len(feat_cols))
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    kept_cols = [c for c in X.columns if X[c].notna().any()]
    X = X[kept_cols]
    logging.info("hr_model feature count after numeric filter: %s", len(kept_cols))

    y = df[args.target_col].astype(int).to_numpy()
    season = pd.to_numeric(df["season"], errors="coerce")
    train_mask = season.between(2019, 2024, inclusive="both")
    test_mask = season == 2025

    if int(train_mask.sum()) == 0 or int(test_mask.sum()) == 0:
        raise ValueError("Need non-empty split: train(2019-2024), test(2025).")

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    clf.fit(X_imp[train_mask], y[train_mask])
    pred_test = clf.predict_proba(X_imp[test_mask])[:, 1]
    auc_test = float(roc_auc_score(y[test_mask], pred_test))

    baseline = float(y[test_mask].mean())
    cutoff = float(np.quantile(pred_test, 0.9))
    top_decile_mask = pred_test >= cutoff
    top_decile_rate = float(y[test_mask][top_decile_mask].mean()) if top_decile_mask.any() else 0.0
    lift = float(top_decile_rate / baseline) if baseline > 0 else 0.0
    calib = _calibration_bins(y[test_mask], pred_test, n_bins=10)

    logging.info("hr_model AUC test_2025=%.6f baseline_hr_rate=%.6f", auc_test, baseline)
    logging.info("hr_model top_decile_hr_rate=%.6f lift_vs_baseline=%.6f", top_decile_rate, lift)
    logging.info("hr_model calibration_bins=%s", json.dumps(calib))

    # Retrain for live: 2019-2025
    live_mask = season.between(2019, 2025, inclusive="both")
    clf_live = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
    )
    clf_live.fit(X_imp[live_mask], y[live_mask])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_dir = dirs["models_dir"] / "hr_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"hr_model_{timestamp}.joblib"
    dump({"model": clf_live, "imputer": imputer, "features": kept_cols}, model_path)

    report = {
        "status": "TRAINED",
        "mart_path": str(mart_path.resolve()),
        "model_path": str(model_path.resolve()),
        "train_years": "2019-2024",
        "test_year": 2025,
        "retrain_years": "2019-2025",
        "auc_test_2025": auc_test,
        "baseline_hr_rate_test_2025": baseline,
        "top_decile_hr_rate_test_2025": top_decile_rate,
        "lift_vs_baseline_test_2025": lift,
        "calibration_bins_test_2025": calib,
        "feature_count_before": len(feat_cols),
        "feature_count_after": len(kept_cols),
    }
    write_json(report, dirs["backtests_dir"] / "hr_model" / f"{timestamp}.json")
    print(f"Writing to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
