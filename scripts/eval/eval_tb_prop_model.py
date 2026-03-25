from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.props.hitter_prop_common import (
    LIVE_ONLY_FEATURE_NAMES,
    classify_feature_sets,
    coerce_lineup_slot_numeric,
    filter_season_range,
    poisson_prob_at_least,
    season_series,
    select_safe_numeric_features,
)
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

TARGET = "target_tb"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TB prop model with season split.")
    p.add_argument("--train-start", type=int, required=True)
    p.add_argument("--train-end", type=int, required=True)
    p.add_argument("--test-season", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _cal_bins(y_tb2: pd.Series, p_tb2: pd.Series, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y_tb2, "p": p_tb2}).sort_values("p").reset_index(drop=True)
    try:
        d["bin"] = pd.qcut(d["p"], q=min(bins, max(1, len(d))), labels=False, duplicates="drop")
    except Exception:
        d["bin"] = 0
    return d.groupby("bin", as_index=False).agg(
        n=("y", "size"),
        p_mean=("p", "mean"),
        y_mean=("y", "mean"),
        p_min=("p", "min"),
        p_max=("p", "max"),
    )


def _qstats(arr: np.ndarray) -> dict[str, float]:
    if len(arr) == 0:
        return {}
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "eval_tb_prop_model.log")
    log_header("scripts/eval/eval_tb_prop_model.py", repo_root, config_path, dirs)

    mart_path = dirs["marts_dir"] / "tb_prop_features.parquet"
    df = pd.read_parquet(mart_path).copy()
    df["lineup_slot_numeric"] = coerce_lineup_slot_numeric(df)
    if TARGET not in df.columns:
        raise ValueError("Missing target_tb in mart")

    s = season_series(df)
    train = filter_season_range(df, args.train_start, args.train_end)
    test = df[s == args.test_season].copy()

    train = train[pd.to_numeric(train[TARGET], errors="coerce").notna()].copy()
    test = test[pd.to_numeric(test[TARGET], errors="coerce").notna()].copy()
    y_train = pd.to_numeric(train[TARGET], errors="coerce").clip(lower=0)
    y_test = pd.to_numeric(test[TARGET], errors="coerce").clip(lower=0)

    excluded = {"game_pk", "batter_id", "opp_pitcher_id", "season", TARGET}
    feats, unsafe_features, dropped_all_null = select_safe_numeric_features(train, excluded=excluded, exclude_live_only=True)
    historical_feats, _ = classify_feature_sets(feats)
    live_only_excluded = [c for c in train.columns if c in LIVE_ONLY_FEATURE_NAMES and pd.api.types.is_numeric_dtype(train[c])]
    logging.info("tb_prop eval dropped_unsafe_features_n=%s features=%s", len(unsafe_features), unsafe_features)
    logging.info("tb_prop eval dropped_live_only_features_n=%s features=%s", len(live_only_excluded), live_only_excluded)
    logging.info("tb_prop eval dropped_all_null_features_n=%s features=%s", len(dropped_all_null), dropped_all_null)
    if not feats:
        raise ValueError("No safe numeric non-null features available for evaluation")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=0.35, max_iter=1000)),
    ])
    X_train = train[feats].replace([np.inf, -np.inf], np.nan)
    X_test = test[feats].replace([np.inf, -np.inf], np.nan)
    pipe.fit(X_train, y_train)

    expected_tb = np.clip(pipe.predict(X_test), 1e-8, None)
    tb2_prob = poisson_prob_at_least(expected_tb, threshold=2)
    naive_mu = float(np.clip(y_train.mean(), 1e-8, None))
    naive_expected_tb = np.full(shape=len(y_test), fill_value=naive_mu, dtype=float)
    naive_tb2_prob = poisson_prob_at_least(naive_expected_tb, threshold=2)

    scored = test.copy()
    scored["expected_tb"] = expected_tb
    scored["tb_2_plus_probability"] = tb2_prob
    scored["target_tb_2_plus"] = (pd.to_numeric(scored[TARGET], errors="coerce") >= 2).astype(int)

    top_cut = np.quantile(tb2_prob, 0.9) if len(tb2_prob) else 1.0
    top_decile = scored[scored["tb_2_plus_probability"] >= top_cut]
    naive_top_cut = np.quantile(naive_tb2_prob, 0.9) if len(naive_tb2_prob) else 1.0
    naive_top_decile = scored[pd.Series(naive_tb2_prob, index=scored.index) >= naive_top_cut]
    model_dev = float(mean_poisson_deviance(y_test, expected_tb))
    naive_dev = float(mean_poisson_deviance(y_test, naive_expected_tb))
    top_decile_rate = float(pd.to_numeric(top_decile["target_tb_2_plus"], errors="coerce").mean()) if len(top_decile) else None
    naive_top_decile_rate = float(pd.to_numeric(naive_top_decile["target_tb_2_plus"], errors="coerce").mean()) if len(naive_top_decile) else None

    metrics = {
        "mae": float(mean_absolute_error(y_test, expected_tb)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, expected_tb))),
        "mean_poisson_deviance": model_dev,
        "mean_poisson_deviance_naive": naive_dev,
        "mean_poisson_deviance_lift_vs_naive": float(naive_dev - model_dev),
        "tb_2_plus_rate_test": float((y_test >= 2).mean()),
        "top_decile_tb_2_plus_rate": top_decile_rate,
        "top_decile_tb_2_plus_rate_naive": naive_top_decile_rate,
        "top_decile_tb_2_plus_rate_lift_vs_naive": (top_decile_rate - naive_top_decile_rate) if (top_decile_rate is not None and naive_top_decile_rate is not None) else None,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_count": int(len(feats)),
        "historical_feature_count": int(len(historical_feats)),
        "live_only_features_excluded_count": int(len(live_only_excluded)),
        "live_only_features_excluded": live_only_excluded,
        "historical_eval_excludes_live_only": True,
        "expected_tb_quantiles": _qstats(expected_tb),
        "tb2_probability_quantiles": _qstats(np.asarray(tb2_prob, dtype=float)),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = dirs["backtests_dir"] / "tb_prop"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_tb_prop_{ts}.json"
    cal_path = out_dir / f"eval_tb_prop_{ts}_tb2_calibration.csv"
    scored_path = out_dir / f"eval_tb_prop_{ts}_scored_test.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "train_start": args.train_start,
                "train_end": args.train_end,
                "test_season": args.test_season,
                "dropped_unsafe_features": unsafe_features,
                "dropped_all_null_features": dropped_all_null,
            },
            f,
            indent=2,
        )
    _cal_bins(scored["target_tb_2_plus"], scored["tb_2_plus_probability"]).to_csv(cal_path, index=False)
    scored.to_csv(scored_path, index=False)

    logging.info("eval_tb_prop complete metrics=%s", metrics)
    logging.info("eval_tb_prop board_separation expected_tb=%s tb2_probability=%s", metrics["expected_tb_quantiles"], metrics["tb2_probability_quantiles"])
    print(f"eval_json={json_path}")


if __name__ == "__main__":
    main()
