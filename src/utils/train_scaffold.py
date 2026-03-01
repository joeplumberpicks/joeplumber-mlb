from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.checks import print_rowcount
from src.utils.io import read_parquet, write_json


def _prep_X(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    X = df[feature_cols].copy()

    # replace inf -> NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    dropped: list[str] = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue

        coerced = pd.to_numeric(X[col], errors="coerce")
        if coerced.notna().any():
            X[col] = coerced
        else:
            dropped.append(col)

    if dropped:
        X = X.drop(columns=dropped)

    if len(X.columns) > 0:
        X = X.astype("float32")

    return X, list(X.columns), dropped


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

    target_numeric = pd.to_numeric(mart_df[target_col], errors="coerce")
    valid_target_mask = target_numeric.notna()
    train_df = mart_df.loc[valid_target_mask].copy()
    y = target_numeric.loc[valid_target_mask].astype("int64")

    logging.info("%s rows kept after dropping NaN target: %s/%s", engine, len(train_df), len(mart_df))

    if train_df.empty or y.nunique() < 2:
        report = {"status": "SKIPPED_INSUFFICIENT_TARGET", "engine": engine}
        write_json(report, report_path)
        print(f"Writing to: {report_path.resolve()}")
        return report

    drop_feature_cols = {target_col, "game_pk", "game_date"}
    feature_cols = [c for c in train_df.columns if c not in drop_feature_cols]

    logging.info("%s features before preprocessing: %s", engine, len(feature_cols))
    X, final_features, dropped_cols = _prep_X(train_df, feature_cols)

    if dropped_cols:
        logging.info("%s dropped non-numeric columns: %s", engine, dropped_cols)

    missing_before = int(X.isna().sum().sum())

    logging.info("%s features after preprocessing: %s", engine, len(final_features))
    logging.info("%s missing values before impute: %s", engine, missing_before)

    if X.empty:
        X = pd.DataFrame({"baseline_feature": np.zeros(len(train_df), dtype="float32")})
        final_features = list(X.columns)

    positive_rate = float((y == 1).mean()) if len(y) else 0.0
    logging.info("%s positive target rate: %.6f", engine, positive_rate)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(X, y)

    imputed = model.named_steps["imputer"].transform(X)
    missing_after = int(np.isnan(imputed).sum())
    logging.info("%s missing values after impute: %s", engine, missing_after)

    coef = getattr(model.named_steps["logreg"], "coef_", None)
    if coef is not None:
        logging.info("%s coefficient norm: %.8f", engine, float(np.linalg.norm(coef)))

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
