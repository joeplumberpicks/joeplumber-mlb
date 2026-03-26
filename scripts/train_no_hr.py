from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet
from src.utils.logging import configure_logging, log_header


def _parse_seasons(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train No-HR Engine v1")
    p.add_argument("--train-seasons", type=str, required=True, help="Comma-separated seasons, e.g. 2021,2022,2023")
    p.add_argument("--test-seasons", type=str, required=True, help="Comma-separated seasons, e.g. 2024")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _load_marts(dirs: dict[str, Path], seasons: list[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for s in seasons:
        p = dirs["processed_dir"] / "marts" / "no_hr" / f"no_hr_game_features_{s}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"Missing no_hr mart season {s}: {p.resolve()}")
        df = read_parquet(p)
        df["season"] = s
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _prep_X(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str]]:
    drop_cols = {
        "game_pk",
        "game_date",
        target_col,
        "total_hr",
        "home_team",
        "away_team",
        "canonical_park_key",
        "batter_feature_source",
        "pitcher_feature_source",
    }
    feats = [c for c in df.columns if c not in drop_cols]
    X = df[feats].copy()

    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("Int64")
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    keep = [c for c in X.columns if X[c].notna().any()]
    X = X[keep].astype("float32") if keep else pd.DataFrame({"bias": np.zeros(len(df), dtype="float32")})
    return X, list(X.columns)


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p), q=min(bins, max(2, len(np.unique(p)))), duplicates="drop")
    tmp = pd.DataFrame({"y": y, "p": p, "bin": q})
    return tmp.groupby("bin", dropna=False).agg(pred_mean=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size")).reset_index()


def main() -> None:
    args = parse_args()
    train_seasons = _parse_seasons(args.train_seasons)
    test_seasons = _parse_seasons(args.test_seasons)

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "train_no_hr.log")
    log_header("scripts/train_no_hr.py", repo_root, config_path, dirs)

    train_df = _load_marts(dirs, train_seasons)
    test_df = _load_marts(dirs, test_seasons)
    target_col = "no_hr_game"

    y_train = pd.to_numeric(train_df[target_col], errors="coerce")
    y_test = pd.to_numeric(test_df[target_col], errors="coerce")
    train_df = train_df[y_train.notna()].copy()
    test_df = test_df[y_test.notna()].copy()
    y_train = y_train[y_train.notna()].astype("int64").to_numpy()
    y_test = y_test[y_test.notna()].astype("int64").to_numpy()

    logging.info("train rows=%s test rows=%s", len(train_df), len(test_df))
    if "season" in train_df.columns:
        logging.info("train rows by season: %s", train_df["season"].value_counts().sort_index().to_dict())
    if "season" in test_df.columns:
        logging.info("test rows by season: %s", test_df["season"].value_counts().sort_index().to_dict())
    logging.info("train no_hr rate=%.6f", float(np.mean(y_train)) if len(y_train) else float("nan"))
    logging.info("test no_hr rate=%.6f", float(np.mean(y_test)) if len(y_test) else float("nan"))

    X_train, feat_cols = _prep_X(train_df, target_col)
    X_test = test_df.reindex(columns=feat_cols).copy() if feat_cols else pd.DataFrame({"bias": np.zeros(len(test_df), dtype="float32")})
    X_test = X_test.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")

    base_model_name = "logreg"
    try:
        from xgboost import XGBClassifier  # type: ignore

        base = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        )
        base_model_name = "xgboost"
        model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", base)])
    except Exception:
        base = LogisticRegression(max_iter=4000, class_weight="balanced")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", base),
        ])

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    p_test = calibrated.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan")
    brier = float(brier_score_loss(y_test, p_test)) if len(y_test) else float("nan")
    ll = float(log_loss(y_test, p_test, labels=[0, 1])) if len(y_test) else float("nan")
    pred_mean = float(np.mean(p_test)) if len(p_test) else float("nan")
    obs_mean = float(np.mean(y_test)) if len(y_test) else float("nan")
    cal_tbl = _calibration_table(y_test, p_test)

    signature = f"train_{'-'.join(map(str, train_seasons))}_test_{'-'.join(map(str, test_seasons))}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_dir = dirs["models_dir"] / "no_hr"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"no_hr_model_{signature}.pkl"
    dump({"model": calibrated, "features": feat_cols, "target_col": target_col, "signature": signature, "base_model": base_model_name}, model_path)

    feat_path = model_dir / f"no_hr_features_{signature}.txt"
    feat_path.write_text("\n".join(feat_cols) + "\n", encoding="utf-8")

    bt_dir = dirs["backtests_dir"] / "no_hr"
    bt_dir.mkdir(parents=True, exist_ok=True)
    backtest = pd.DataFrame([
        {
            "signature": signature,
            "train_seasons": ",".join(map(str, train_seasons)),
            "test_seasons": ",".join(map(str, test_seasons)),
            "model": base_model_name,
            "auc": auc,
            "brier": brier,
            "logloss": ll,
            "pred_mean": pred_mean,
            "obs_rate": obs_mean,
            "n_test": len(y_test),
        }
    ])
    bt_path = bt_dir / f"no_hr_backtest_{signature}.csv"
    backtest.to_csv(bt_path, index=False)

    cal_path = bt_dir / f"no_hr_calibration_{signature}.csv"
    cal_tbl.to_csv(cal_path, index=False)

    logging.info(
        "no_hr train complete model=%s auc=%.6f brier=%.6f logloss=%.6f pred_mean=%.6f obs_rate=%.6f model_path=%s",
        base_model_name,
        auc,
        brier,
        ll,
        pred_mean,
        obs_mean,
        model_path.resolve(),
    )
    logging.info("features saved path=%s", feat_path.resolve())
    logging.info("calibration_by_decile path=%s", cal_path.resolve())
    print(f"model -> {model_path.resolve()}")
    print(f"backtest -> {bt_path.resolve()}")


if __name__ == "__main__":
    main()
