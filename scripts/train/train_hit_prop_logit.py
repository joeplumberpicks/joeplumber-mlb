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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

TARGET = "target_hit_1_plus"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v1 hit prop logistic model.")
    p.add_argument("--train-start", type=int, required=True)
    p.add_argument("--train-end", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _season_filter(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    season = pd.to_numeric(df.get("season"), errors="coerce")
    if season.notna().any():
        return df[(season >= start) & (season <= end)].copy()
    gd = pd.to_datetime(df.get("game_date"), errors="coerce")
    return df[(gd.dt.year >= start) & (gd.dt.year <= end)].copy()


def _drop_all_null_numeric_features(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    keep: list[str] = []
    dropped: list[str] = []
    for c in features:
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            keep.append(c)
        else:
            dropped.append(c)
    return keep, dropped


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "train_hit_prop_logit.log")
    log_header("scripts/train/train_hit_prop_logit.py", repo_root, config_path, dirs)

    mart_path = dirs["marts_dir"] / "hit_prop_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart: {mart_path}")
    df = pd.read_parquet(mart_path).copy()

    if TARGET not in df.columns:
        if "target_hit1p" in df.columns:
            df[TARGET] = pd.to_numeric(df["target_hit1p"], errors="coerce")
        elif "target_hit_1p" in df.columns:
            df[TARGET] = pd.to_numeric(df["target_hit_1p"], errors="coerce")
    if TARGET not in df.columns:
        raise ValueError("No target_hit_1_plus in mart")

    df = _season_filter(df, args.train_start, args.train_end)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[y.notna()].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce").astype(int)

    excluded = {"game_pk", "batter_id", "opp_pitcher_id", "season", TARGET, "target_hit1p", "target_hit_1p"}
    numeric_features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded]
    feature_stats: list[tuple[str, int, float]] = []
    for c in numeric_features:
        nn = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
        rate = float(nn / max(1, len(df)))
        feature_stats.append((c, nn, rate))
    logging.info("hit_prop train candidate_numeric_features_n=%s", len(numeric_features))
    logging.info("hit_prop train candidate_feature_non_null_stats=%s", feature_stats)

    features, dropped_all_null = _drop_all_null_numeric_features(df, numeric_features)
    logging.info("hit_prop train dropped_all_null_features_n=%s features=%s", len(dropped_all_null), dropped_all_null)
    if not features:
        raise ValueError("No numeric non-null features available for training")
    surviving_stats = [(c, int(pd.to_numeric(df[c], errors="coerce").notna().sum()), float(pd.to_numeric(df[c], errors="coerce").notna().mean())) for c in features]
    logging.info("hit_prop train surviving_features=%s", features)
    logging.info("hit_prop train surviving_feature_non_null_stats=%s", surviving_stats)

    X = df[features].replace([np.inf, -np.inf], np.nan)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
        ]
    )
    model.fit(X, y)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = dirs["models_dir"] / "hit_prop"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"hit_prop_logit_{ts}.joblib"

    bundle = {
        "model": model,
        "feature_list": features,
        "target": TARGET,
        "train_start": args.train_start,
        "train_end": args.train_end,
        "trained_at": ts,
        "n_rows": int(len(df)),
        "dropped_all_null_features": dropped_all_null,
    }
    joblib.dump(bundle, model_path)

    coef_vals = model.named_steps["clf"].coef_[0]
    if len(coef_vals) != len(features):
        raise ValueError(f"Coefficient length mismatch coef={len(coef_vals)} features={len(features)}")
    coef = pd.DataFrame({"feature": features, "coef": coef_vals})
    coef_path = out_dir / f"hit_prop_logit_{ts}_coefficients.csv"
    coef.to_csv(coef_path, index=False)

    meta_path = out_dir / f"hit_prop_logit_{ts}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "model"}, f, indent=2)

    logging.info("trained hit_prop model rows=%s features=%s path=%s", len(df), len(features), model_path)
    print(f"model_out={model_path}")


if __name__ == "__main__":
    main()
