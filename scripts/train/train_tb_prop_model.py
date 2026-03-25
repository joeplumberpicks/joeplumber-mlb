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
from sklearn.metrics import mean_poisson_deviance

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.props.hitter_prop_common import (
    LIVE_ONLY_FEATURE_NAMES,
    classify_feature_sets,
    coerce_lineup_slot_numeric,
    filter_season_range,
    select_safe_numeric_features,
)
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
    df["lineup_slot_numeric"] = coerce_lineup_slot_numeric(df)
    y = pd.to_numeric(df[TARGET], errors="coerce")
    df = df[y.notna()].copy()
    y = pd.to_numeric(df[TARGET], errors="coerce").clip(lower=0)

    excluded = {"game_pk", "batter_id", "opp_pitcher_id", "season", TARGET}
    numeric_before = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in excluded]
    features, unsafe_features, dropped_all_null = select_safe_numeric_features(df, excluded=excluded, exclude_live_only=True)
    historical_feats, live_only_feats_present = classify_feature_sets(features)
    retained_engineered = [c for c in features if c.startswith("tb_") or c.startswith("lineup_bucket_") or c.endswith("_x_volume") or c.endswith("_x_expected_ab") or c.endswith("_x_park_factor")]
    null_rates = {c: float(pd.to_numeric(df[c], errors="coerce").isna().mean()) for c in numeric_before}
    top_missing = sorted(null_rates.items(), key=lambda kv: kv[1], reverse=True)[:15]
    logging.info("tb_prop train feature_count_before_filter=%s", len(numeric_before))
    logging.info("tb_prop train feature_count_after_filter=%s", len(features))
    live_only_excluded = [c for c in numeric_before if c in LIVE_ONLY_FEATURE_NAMES]
    logging.info("tb_prop train dropped_unsafe_features_n=%s features=%s", len(unsafe_features), unsafe_features)
    logging.info("tb_prop train dropped_live_only_features_n=%s features=%s", len(live_only_excluded), live_only_excluded)
    logging.info("tb_prop train dropped_all_null_features_n=%s features=%s", len(dropped_all_null), dropped_all_null)
    logging.info("tb_prop train retained_engineered_extras_n=%s features=%s", len(retained_engineered), retained_engineered)
    logging.info("tb_prop train top_missing_numeric_features=%s", top_missing)
    if not features:
        raise ValueError("No safe numeric non-null features available for training")

    X = df[features].replace([np.inf, -np.inf], np.nan)
    seasons = pd.to_numeric(df.get("season"), errors="coerce")
    if not seasons.notna().any():
        seasons = pd.to_datetime(df.get("game_date"), errors="coerce").dt.year
    val_mask = seasons == args.train_end
    if int(val_mask.sum()) < 500:
        val_mask = pd.Series(False, index=df.index)
        val_mask.iloc[int(0.8 * len(df)) :] = True
    train_mask = ~val_mask
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        train_mask = pd.Series(True, index=df.index)
        val_mask = pd.Series(True, index=df.index)

    alphas = [0.1, 0.35, 0.75, 1.0]
    best_alpha = alphas[0]
    best_dev = float("inf")
    alpha_scores: dict[str, float] = {}
    for alpha in alphas:
        candidate = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("poisson", PoissonRegressor(alpha=alpha, max_iter=1000)),
            ]
        )
        candidate.fit(X.loc[train_mask], y.loc[train_mask])
        pred_val = np.clip(candidate.predict(X.loc[val_mask]), 1e-8, None)
        dev = float(mean_poisson_deviance(y.loc[val_mask], pred_val))
        alpha_scores[str(alpha)] = dev
        if dev < best_dev:
            best_dev = dev
            best_alpha = alpha
    logging.info("tb_prop train alpha_grid_scores=%s chosen_alpha=%s", alpha_scores, best_alpha)

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poisson", PoissonRegressor(alpha=best_alpha, max_iter=1000)),
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
        "chosen_alpha": best_alpha,
        "alpha_grid_scores_poisson_deviance": alpha_scores,
        "derived_live_probability": "P(TB>=2) via Poisson(lambda=expected_tb)",
        "historical_feature_count": int(len(historical_feats)),
        "live_only_features_excluded": live_only_excluded,
        "live_only_features_excluded_count": int(len(live_only_excluded)),
        "dropped_unsafe_features": unsafe_features,
        "dropped_all_null_features": dropped_all_null,
        "retained_engineered_extras": retained_engineered,
    }
    joblib.dump(bundle, model_path)

    coef = pd.DataFrame(
        {
            "feature": features,
            "coef": model.named_steps["poisson"].coef_,
        }
    )
    coef["abs_coef"] = coef["coef"].abs()
    top_coef = coef.sort_values("abs_coef", ascending=False).head(20)
    logging.info("tb_prop train top_abs_coefficients=%s", top_coef.to_dict("records"))
    coef_path = out_dir / f"tb_prop_poisson_{ts}_coefficients.csv"
    coef.sort_values("abs_coef", ascending=False).to_csv(coef_path, index=False)

    meta_path = out_dir / f"tb_prop_poisson_{ts}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "model"}, f, indent=2)

    logging.info("trained tb_prop model rows=%s features=%s path=%s", len(df), len(features), model_path)
    print(f"model_out={model_path}")


if __name__ == "__main__":
    main()
