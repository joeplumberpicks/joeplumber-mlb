from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression

from src.utils.checks import print_rowcount
from src.utils.io import read_parquet, write_json


def run_training_scaffold(engine: str, mart_file: str, target_col: str, dirs: dict[str, Path]) -> None:
    mart_path = dirs["marts_dir"] / mart_file
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart for {engine}: {mart_path.resolve()}")

    mart_df = read_parquet(mart_path)
    print_rowcount(f"mart_{engine}", mart_df)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = dirs["backtests_dir"] / engine / f"{timestamp}.json"

    if mart_df.empty:
        write_json({"status": "SKIPPED_EMPTY_MART", "engine": engine, "mart": str(mart_path.resolve())}, report_path)
        print(f"Writing to: {report_path.resolve()}")
        return

    if target_col not in mart_df.columns:
        write_json({"status": "SKIPPED_NO_TARGET", "engine": engine, "missing_target": target_col}, report_path)
        print(f"Writing to: {report_path.resolve()}")
        return

    train_df = mart_df.dropna(subset=[target_col]).copy()
    if train_df.empty or train_df[target_col].nunique() < 2:
        write_json({"status": "SKIPPED_INSUFFICIENT_TARGET", "engine": engine}, report_path)
        print(f"Writing to: {report_path.resolve()}")
        return

    X = train_df.select_dtypes(include=["number"]).drop(columns=[target_col], errors="ignore")
    if X.empty:
        X = pd.DataFrame({"baseline_feature": range(len(train_df))})
    y = train_df[target_col].astype(int)

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    model_path = dirs["models_dir"] / engine / f"{engine}_{timestamp}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    print(f"Writing to: {model_path.resolve()}")

    report = {
        "status": "TRAINED",
        "engine": engine,
        "rows": int(len(train_df)),
        "features": list(X.columns),
        "model_path": str(model_path.resolve()),
    }
    write_json(report, report_path)
    print(f"Writing to: {report_path.resolve()}")
