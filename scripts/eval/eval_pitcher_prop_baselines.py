from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

REG_TARGETS = ["target_k", "target_outs"]
LADDERS = {
    "k_ge_4": ("target_k", 4.0),
    "k_ge_5": ("target_k", 5.0),
    "k_ge_6": ("target_k", 6.0),
    "outs_ge_15": ("target_outs", 15.0),
    "outs_ge_18": ("target_outs", 18.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pitcher prop baselines with time split.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--split-date", type=str, required=True)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def _quantile_calibration_bins(y_true: pd.Series, y_prob: np.ndarray, bins: int) -> list[dict[str, float | int]]:
    frame = pd.DataFrame({"y": y_true.astype(float).values, "p": y_prob})
    if frame.empty:
        return []
    frame = frame.sort_values("p").reset_index(drop=True)
    try:
        frame["bin"] = pd.qcut(frame["p"], q=min(bins, len(frame)), labels=False, duplicates="drop")
    except ValueError:
        frame["bin"] = 0
    out: list[dict[str, float | int]] = []
    for b, grp in frame.groupby("bin", dropna=True):
        out.append(
            {
                "bin_index": int(b),
                "n": int(len(grp)),
                "p_mean": float(grp["p"].mean()),
                "y_mean": float(grp["y"].mean()),
                "p_min": float(grp["p"].min()),
                "p_max": float(grp["p"].max()),
            }
        )
    return out


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded_non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c != "game_pk" and not c.startswith("target_")
    ]
    return feature_cols, excluded_non_numeric


def _fit_regression(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], target: str) -> dict:
    X_train = train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = test_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = pd.to_numeric(train_df[target], errors="coerce")
    y_test = pd.to_numeric(test_df[target], errors="coerce")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "mean_y_test": float(y_test.mean()),
        "features_n": int(len(features)),
    }


def _fit_ladder(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], source_target: str, threshold: float, bins: int) -> tuple[dict, list[dict]]:
    X_train = train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = test_df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_train = (pd.to_numeric(train_df[source_target], errors="coerce") >= threshold).astype(int)
    y_test = (pd.to_numeric(test_df[source_target], errors="coerce") >= threshold).astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
    ])
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else None
    metrics = {
        "auc": auc,
        "brier": float(brier_score_loss(y_test, prob)),
        "logloss": float(log_loss(y_test, prob, labels=[0, 1])),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "positive_rate_test": float(y_test.mean()),
        "features_n": int(len(features)),
    }
    cal_bins = _quantile_calibration_bins(y_test.reset_index(drop=True), prob, bins)
    return metrics, cal_bins


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "eval_pitcher_prop_baselines.log")
    log_header("scripts/eval/eval_pitcher_prop_baselines.py", repo_root, config_path, dirs)

    split_date = pd.to_datetime(args.split_date, errors="raise")

    mart_path = dirs["marts_dir"] / "pitcher_game_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart file: {mart_path.resolve()}")

    df = pd.read_parquet(mart_path)
    if "season" in df.columns:
        season_vals = pd.to_numeric(df["season"], errors="coerce")
        df = df[season_vals == args.season].copy()

    if "game_date" not in df.columns:
        raise ValueError("pitcher_game_features missing required column game_date")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).sort_values("game_date").reset_index(drop=True)

    feature_cols, excluded_non_numeric = _feature_columns(df)
    logging.info("Excluded non-numeric columns: %s", excluded_non_numeric)
    if not feature_cols:
        raise ValueError("No numeric features available for pitcher prop baseline evaluation")

    per_target: dict[str, dict] = {}
    calibration_by_target: dict[str, list[dict]] = {}
    rmse_values: list[float] = []

    for target in REG_TARGETS:
        if target not in df.columns:
            logging.warning("Skipping %s regression: missing column", target)
            continue
        tdf = df[pd.to_numeric(df[target], errors="coerce").notna()].copy()
        if tdf.empty:
            logging.warning("Skipping %s regression: no non-null rows", target)
            continue
        train_df = tdf[tdf["game_date"] < split_date].copy()
        test_df = tdf[tdf["game_date"] >= split_date].copy()
        if train_df.empty or test_df.empty:
            logging.warning("Skipping %s regression: empty split train=%s test=%s", target, len(train_df), len(test_df))
            continue
        reg_metrics = _fit_regression(train_df, test_df, feature_cols, target)
        per_target[target] = {"regression_metrics": reg_metrics}
        rmse_values.append(reg_metrics["rmse"])
        print(
            f"pitcher_baseline {target} rmse={reg_metrics['rmse']:.6f} mae={reg_metrics['mae']:.6f} "
            f"n_train={reg_metrics['n_train']} n_test={reg_metrics['n_test']}"
        )

    for ladder_name, (source_target, threshold) in LADDERS.items():
        if source_target not in df.columns:
            logging.warning("Skipping %s: missing source target %s", ladder_name, source_target)
            continue
        tdf = df[pd.to_numeric(df[source_target], errors="coerce").notna()].copy()
        if tdf.empty:
            logging.warning("Skipping %s: no non-null source rows", ladder_name)
            continue
        train_df = tdf[tdf["game_date"] < split_date].copy()
        test_df = tdf[tdf["game_date"] >= split_date].copy()
        if train_df.empty or test_df.empty:
            logging.warning("Skipping %s: empty split train=%s test=%s", ladder_name, len(train_df), len(test_df))
            continue
        metrics, cal_bins = _fit_ladder(train_df, test_df, feature_cols, source_target, threshold, args.bins)
        per_target[ladder_name] = {"metrics": metrics, "calibration_bins": cal_bins}
        calibration_by_target[ladder_name] = cal_bins
        print(
            f"pitcher_baseline {ladder_name} auc={metrics['auc']} brier={metrics['brier']:.6f} "
            f"logloss={metrics['logloss']:.6f} n_train={metrics['n_train']} n_test={metrics['n_test']}"
        )

    overall = {
        "targets_evaluated": int(len(per_target)),
        "avg_rmse_regression": float(np.mean(rmse_values)) if rmse_values else None,
    }

    output = {
        "run_ts": datetime.now(timezone.utc).isoformat(),
        "season": int(args.season),
        "split_date": split_date.strftime("%Y-%m-%d"),
        "model": "StandardScaler+Ridge / StandardScaler+LogisticRegression",
        "features_n": int(len(feature_cols)),
        "metrics": overall,
        "per_target": per_target,
        "calibration_bins": calibration_by_target,
    }

    out_path = dirs["backtests_dir"] / f"pitcher_prop_baselines_{args.season}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(f"pitcher_prop_baselines -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
