from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression

from src.utils.checks import print_rowcount
from src.utils.io import read_parquet, write_json


def _prep_X(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str], int, int]:
    X = df[feature_cols].copy()

    # replace inf -> NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_before = int(X.isna().sum().sum())

    kept: list[str] = []
    dropped: list[str] = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            kept.append(col)
            continue

        coerced = pd.to_numeric(X[col], errors="coerce")
        if coerced.notna().any():
            X[col] = coerced
            kept.append(col)
        else:
            dropped.append(col)

    if dropped:
        X = X.drop(columns=dropped)

    if len(X.columns) > 0:
        med = X.median(numeric_only=True)
        X = X.fillna(med)
    X = X.fillna(0)

    if len(X.columns) > 0:
        X = X.astype("float32")

    missing_after = int(X.isna().sum().sum())
    return X, list(X.columns), dropped, missing_before, missing_after


def run_training_scaffold(engine: str, mart_file: str, target_col: str, dirs: dict[str, Path]) -> dict[str, Any]:
    mart_path = dirs["marts_dir"] / mart_file
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart for {engine}: {mart_path.resolve()}")

    mart_df = read_parquet(mart_path)
    print_rowcount(f"mart_{engine}", mart_df)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = dirs["backtests_dir"] / engine / f"{timestamp}.json"

    if mart_df.empty:
        report = {"status": "SKIPPED_EMPTY_MART", "engine": engine, "mart": str(mart_path.resolve())}
        write_json(report, report_path)
        print(f"Writing to: {report_path.resolve()}")
        return report

    if target_col not in mart_df.columns:
        report = {"status": "SKIPPED_NO_TARGET", "engine": engine, "missing_target": target_col}
        write_json(report, report_path)
        print(f"Writing to: {report_path.resolve()}")
        return report

    train_df = mart_df.dropna(subset=[target_col]).copy()
    if train_df.empty or train_df[target_col].nunique() < 2:
        report = {"status": "SKIPPED_INSUFFICIENT_TARGET", "engine": engine}
        write_json(report, report_path)
        print(f"Writing to: {report_path.resolve()}")
        return report

    drop_feature_cols = {target_col, "game_pk", "game_date"}
    feature_cols = [c for c in train_df.columns if c not in drop_feature_cols]

    logging.info("%s features before preprocessing: %s", engine, len(feature_cols))
    X, final_features, dropped_cols, missing_before, missing_after = _prep_X(train_df, feature_cols)

    if dropped_cols:
        logging.info("%s dropped non-numeric columns: %s", engine, dropped_cols)

    logging.info("%s features after preprocessing: %s", engine, len(final_features))
    logging.info("%s missing values before impute: %s", engine, missing_before)
    logging.info("%s missing values after impute: %s", engine, missing_after)

    if X.empty:
        X = pd.DataFrame({"baseline_feature": range(len(train_df))}, dtype="float32")
        final_features = list(X.columns)

    y = pd.to_numeric(train_df[target_col], errors="coerce")
    y = y.fillna(0).astype(int)
    if y.nunique() < 2:
        report = {"status": "SKIPPED_INSUFFICIENT_TARGET", "engine": engine}
        write_json(report, report_path)
        print(f"Writing to: {report_path.resolve()}")
        return report

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    model_path = dirs["models_dir"] / engine / f"{engine}_{timestamp}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    print(f"Writing to: {model_path.resolve()}")

    report = {
        "status": "TRAINED",
        "engine": engine,
        "rows": int(len(train_df)),
        "features": final_features,
        "model_path": str(model_path.resolve()),
    }
    write_json(report, report_path)
    print(f"Writing to: {report_path.resolve()}")
    return report
