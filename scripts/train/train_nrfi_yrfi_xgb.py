from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, safe_mkdir, write_json
from src.utils.logging import configure_logging, log_header

SCRIPT_NAME = "scripts/train/train_nrfi_yrfi_xgb.py"
TARGETS = ["target_nrfi", "target_yrfi"]
DEFAULT_CONFIG = Path("configs/nrfi_train_2019_2025.yaml")
DEFAULT_DROP_COLUMNS = {
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "park_id",
    "park_name",
    "canonical_park_key",
    "season",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NRFI and YRFI XGBoost models.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["eval", "prod"], default="eval")
    return parser.parse_args()


def _as_int_seasons(values: list[Any] | None) -> list[int]:
    if not values:
        return []
    return [int(v) for v in values]


def _load_marts_for_seasons(marts_dir: Path, seasons: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        mart_path = marts_dir / "by_season" / f"nrfi_features_{season}.parquet"
        if not mart_path.exists():
            raise FileNotFoundError(f"Missing mart file for season={season}: {mart_path}")
        df = read_parquet(mart_path)
        if "season" not in df.columns:
            df["season"] = int(season)
        else:
            df["season"] = pd.to_numeric(df["season"], errors="coerce").fillna(int(season)).astype("Int64")
        frames.append(df)

    if not frames:
        raise ValueError("No mart files were loaded.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        raise ValueError("Loaded mart frame is empty after concatenation.")
    return combined


def _build_numeric_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    X = df.copy()
    drop_cols = set(DEFAULT_DROP_COLUMNS)
    drop_cols.update(TARGETS)
    present_drop_cols = [c for c in X.columns if c in drop_cols]
    if present_drop_cols:
        X = X.drop(columns=present_drop_cols)

    # coerce every candidate feature to numeric so numeric-looking strings survive
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # keep only columns with at least one non-null numeric value
    keep_cols = [c for c in X.columns if X[c].notna().any()]
    X = X[keep_cols]

    if X.empty:
        raise ValueError("No numeric feature columns remain after preprocessing.")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    return X, list(X.columns)


def _safe_auc(y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    uniq = pd.Series(y_true).dropna().unique()
    if len(uniq) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _compute_metrics(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, float | None]:
    y_true_arr = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).to_numpy()
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)

    return {
        "brier": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "logloss": float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1])),
        "auc": _safe_auc(pd.Series(y_true_arr), y_prob_arr),
    }


def _compute_calibration_bins(y_true: pd.Series, y_prob: np.ndarray, bins: int = 10) -> list[dict[str, float | int | None]]:
    y_true_arr = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int).to_numpy()
    y_prob_arr = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)

    output: list[dict[str, float | int | None]] = []
    for idx in range(bins):
        low = float(edges[idx])
        high = float(edges[idx + 1])
        if idx < bins - 1:
            mask = (y_prob_arr >= low) & (y_prob_arr < high)
        else:
            mask = (y_prob_arr >= low) & (y_prob_arr <= high)
        count = int(mask.sum())
        avg_pred = float(np.mean(y_prob_arr[mask])) if count else None
        avg_true = float(np.mean(y_true_arr[mask])) if count else None
        output.append(
            {
                "bin": idx,
                "lower": low,
                "upper": high,
                "count": count,
                "avg_pred": avg_pred,
                "avg_true": avg_true,
            }
        )
    return output


def _train_one_target(
    *,
    target: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    feature_cols: list[str],
    xgb_params: dict[str, Any],
) -> tuple[XGBClassifier, dict[str, Any]]:
    y_train = pd.to_numeric(train_df[target], errors="coerce").fillna(0).astype(int)
    X_train = train_df[feature_cols]

    if y_train.nunique() < 2:
        raise ValueError(f"Target {target} has fewer than 2 classes in training data.")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **xgb_params,
    )
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    result: dict[str, Any] = {
        "rows_train": int(len(train_df)),
        "positive_rate_train": float(y_train.mean()) if len(y_train) else None,
        "metrics_train": _compute_metrics(y_train, train_prob),
        "calibration_train": _compute_calibration_bins(y_train, train_prob, bins=10),
    }

    if test_df is not None and not test_df.empty:
        y_test = pd.to_numeric(test_df[target], errors="coerce").fillna(0).astype(int)
        X_test = test_df[feature_cols]
        test_prob = model.predict_proba(X_test)[:, 1]
        result.update(
            {
                "rows_test": int(len(test_df)),
                "positive_rate_test": float(y_test.mean()) if len(y_test) else None,
                "metrics_test": _compute_metrics(y_test, test_prob),
                "calibration_test": _compute_calibration_bins(y_test, test_prob, bins=10),
            }
        )
    else:
        result.update(
            {
                "rows_test": 0,
                "positive_rate_test": None,
                "metrics_test": None,
                "calibration_test": [],
            }
        )

    return model, result


def _resolve_output_paths(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    model_version: str,
    mode: str,
    timestamp: str,
    test_seasons: list[int],
) -> tuple[Path, dict[str, Path]]:
    artifacts_subdir = cfg.get("artifacts_subdir", "data/models/nrfi_xgb")
    backtests_subdir = cfg.get("backtests_subdir", "data/backtests/nrfi_xgb")

    artifacts_dir = (repo_root / artifacts_subdir).resolve()
    backtests_dir = (repo_root / backtests_subdir).resolve()
    safe_mkdir(artifacts_dir)
    safe_mkdir(backtests_dir)

    if mode == "eval" and test_seasons:
        test_tag = f"TEST{''.join(str(s) for s in test_seasons)}"
    elif mode == "eval":
        test_tag = "TEST"
    else:
        test_tag = "PROD"

    model_paths = {
        target: artifacts_dir / f"{model_version}_{target}_{test_tag}_{timestamp}.json" for target in TARGETS
    }
    report_path = backtests_dir / f"{timestamp}.json"
    return report_path, model_paths


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)

    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "train_nrfi_yrfi_xgb.log")
    log_header(SCRIPT_NAME, repo_root, config_path, dirs)

    model_version = str(config.get("model_version", "nrfi_xgb"))
    xgb_params = dict(config.get("xgb_params", {}))

    if args.mode == "eval":
        train_seasons = _as_int_seasons(config.get("train_seasons_default"))
        test_seasons = _as_int_seasons(config.get("test_seasons_default"))
    else:
        train_seasons = _as_int_seasons(config.get("production_seasons_default"))
        test_seasons = []

    if not train_seasons:
        raise ValueError("No train seasons configured.")

    train_df_raw = _load_marts_for_seasons(dirs["marts_dir"], train_seasons)
    test_df_raw = _load_marts_for_seasons(dirs["marts_dir"], test_seasons) if test_seasons else None

    for target in TARGETS:
        if target not in train_df_raw.columns:
            raise ValueError(f"Missing target column in training data: {target}")
        if test_df_raw is not None and not test_df_raw.empty and target not in test_df_raw.columns:
            raise ValueError(f"Missing target column in test data: {target}")

    X_train, feature_cols = _build_numeric_feature_matrix(train_df_raw)
    train_df = train_df_raw.copy()
    for col in feature_cols:
        train_df[col] = X_train[col]

    if test_df_raw is not None and not test_df_raw.empty:
        test_features_frame = pd.DataFrame(index=test_df_raw.index)
        for col in feature_cols:
            if col in test_df_raw.columns:
                test_features_frame[col] = pd.to_numeric(test_df_raw[col], errors="coerce")
            else:
                test_features_frame[col] = np.nan
        test_features_frame = test_features_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_df = test_df_raw.copy()
        for col in feature_cols:
            test_df[col] = test_features_frame[col]
    else:
        test_df = None

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_path, model_paths = _resolve_output_paths(
        repo_root=repo_root,
        cfg=config,
        model_version=model_version,
        mode=args.mode,
        timestamp=timestamp,
        test_seasons=test_seasons,
    )

    report: dict[str, Any] = {
        "timestamp": timestamp,
        "mode": args.mode,
        "model_version": model_version,
        "train_seasons": train_seasons,
        "test_seasons": test_seasons,
        "production_seasons": _as_int_seasons(config.get("production_seasons_default")),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "xgb_params": xgb_params,
        "targets": {},
        "model_paths": {},
    }

    for target in TARGETS:
        logging.info("Training model for %s", target)
        model, target_report = _train_one_target(
            target=target,
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            xgb_params=xgb_params,
        )

        model_path = model_paths[target]
        model.save_model(model_path)
        logging.info("Saved model for %s to %s", target, model_path)

        report["targets"][target] = target_report
        report["model_paths"][target] = str(model_path)

    write_json(report, report_path)
    logging.info("Wrote report to %s", report_path)

    print(f"report_path={report_path}")
    for target in TARGETS:
        print(f"model_{target}={model_paths[target]}")


if __name__ == "__main__":
    main()
