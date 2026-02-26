from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header


REQUIRED_COLS = ["game_date", "target_home_win"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate moneyline logistic-regression baseline with time split.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--split-date", type=str, required=True)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def _calibration_bins(y_true: pd.Series, y_prob: np.ndarray, bins: int) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(y_prob, edges[1:-1], right=False)

    rows: list[dict[str, float | int]] = []
    for i in range(bins):
        mask = idx == i
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin": i,
                "bin_low": float(edges[i]),
                "bin_high": float(edges[i + 1]),
                "count": count,
                "avg_pred": float(np.mean(y_prob[mask])),
                "emp_rate": float(np.mean(y_true[mask])),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "eval_moneyline_baseline.log")
    log_header("scripts/eval/eval_moneyline_baseline.py", repo_root, config_path, dirs)

    split_date = pd.to_datetime(args.split_date, errors="raise")

    mart_path = dirs["marts_dir"] / "moneyline_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart file: {mart_path.resolve()}")

    df = pd.read_parquet(mart_path)
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in moneyline mart: {missing_cols}")

    if "season" in df.columns:
        season_vals = pd.to_numeric(df["season"], errors="coerce")
        df = df[season_vals == args.season].copy()

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["target_home_win"] = pd.to_numeric(df["target_home_win"], errors="coerce")
    df = df.dropna(subset=["game_date", "target_home_win"]).copy()
    df = df.sort_values("game_date").reset_index(drop=True)

    train_df = df[df["game_date"] < split_date].copy()
    test_df = df[df["game_date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Invalid split produced empty partition(s): n_train={len(train_df)}, n_test={len(test_df)}. "
            f"Check --season/--split-date."
        )

    exclude = {"target_home_win", "game_date", "game_pk"}
    feature_cols = [
        c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns available for moneyline baseline.")

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_train = train_df["target_home_win"].astype(int)
    y_test = test_df["target_home_win"].astype(int)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    auc: float | None = None
    if y_test.nunique() > 1:
        auc = float(roc_auc_score(y_test, y_prob))

    metrics = {
        "auc": auc,
        "brier": float(brier_score_loss(y_test, y_prob)),
        "logloss": float(log_loss(y_test, y_prob, labels=[0, 1])),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "positive_rate_test": float(y_test.mean()),
        "features_n": int(len(feature_cols)),
    }

    output = {
        "run_ts": datetime.now(timezone.utc).isoformat(),
        "season": int(args.season),
        "split_date": split_date.strftime("%Y-%m-%d"),
        "model": "LogisticRegression_baseline",
        "features_n": int(len(feature_cols)),
        "metrics": metrics,
        "calibration_bins": _calibration_bins(y_test.reset_index(drop=True), y_prob, args.bins),
    }

    out_path = dirs["backtests_dir"] / f"moneyline_baseline_{args.season}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    print(
        "moneyline_baseline "
        f"auc={metrics['auc']} brier={metrics['brier']:.6f} logloss={metrics['logloss']:.6f} "
        f"n_train={metrics['n_train']} n_test={metrics['n_test']} -> {out_path.resolve()}"
    )


if __name__ == "__main__":
    main()
