from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.props.hitter_prop_common import filter_season_range, select_safe_numeric_features
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

TARGET = "target_tb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TB prop Poisson model.")
    p.add_argument("--train-start", type=int, required=True)
    p.add_argument("--train-end", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "train_tb_prop_model.log")
    log_header("scripts/train/train_tb_prop_model.py", repo_root, config_path, dirs)

    mart_path = dirs["marts_dir"] / "tb_prop_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart: {mart_path}")

    df = pd.read_parquet(mart_path).copy()
    if TARGET not in df.columns:
        raise ValueError("Missing target_tb in mart")

    df = filter_season_range(df, args.train_start, args.train_end)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[y.notna()].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce").clip(lower=0)

    excluded = {"game_pk", "batter_id", "opp_pitcher_id", "season", TARGET}
    features, unsafe_features, dropped_all_null = select_safe_numeric_features(df, excluded=excluded)
    logging.info("tb_prop train dropped_unsafe_features_n=%s features=%s", len(unsafe_features), unsafe_features)
    logging.info("tb_prop train dropped_all_null_features_n=%s features=%s", len(dropped_all_null), dropped_all_null)
    if not features:
        raise ValueError("No safe numeric non-null features available for training")

    X = df[features].replace([np.inf, -np.inf], np.nan)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=0.35, max_iter=1000)),
        ]
    )
    model.fit(X, y)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = dirs["models_dir"] / "tb_prop"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"tb_prop_poisson_{ts}.joblib"

    bundle = {
        "model": model,
        "feature_list": features,
        "target": TARGET,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "trained_at": ts,
        "n_rows": int(len(df)),
        "model_family": "poisson_regression",
        "derived_live_probability": "P(TB>=2) via Poisson(lambda=expected_tb)",
        "dropped_unsafe_features": unsafe_features,
        "dropped_all_null_features": dropped_all_null,
    }
    joblib.dump(bundle, model_path)

    meta_path = out_dir / f"tb_prop_poisson_{ts}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "model"}, f, indent=2)

    logging.info("trained tb_prop model rows=%s features=%s path=%s", len(df), len(features), model_path)
    print(f"model_out={model_path}")


if __name__ == "__main__":
    main()
