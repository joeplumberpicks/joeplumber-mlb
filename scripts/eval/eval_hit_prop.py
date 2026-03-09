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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

TARGET = "target_hit_1_plus"


SAFE_ENGINEERED_COLS = {
    "lineup_slot", "expected_batting_order_pa", "lineup_confidence", "bat_ab_per_game_roll15", "bat_pa_per_game_roll15", "expected_ab_proxy", "park_factor_hits", "temperature", "weather_wind"
}
ROLL_SUFFIXES = ("_roll3", "_roll7", "_roll15", "_roll30")


def _is_safe_feature(col: str) -> bool:
    return col in SAFE_ENGINEERED_COLS or col.endswith(ROLL_SUFFIXES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate hit prop v1 model with season split.")
    p.add_argument("--train-start", type=int, required=True)
    p.add_argument("--train-end", type=int, required=True)
    p.add_argument("--test-season", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _season(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df.get("season"), errors="coerce")
    if s.notna().any():
        return s
    return pd.to_datetime(df.get("game_date"), errors="coerce").dt.year


def _drop_all_null_numeric_features(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    keep: list[str] = []
    dropped: list[str] = []
    for c in features:
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            keep.append(c)
        else:
            dropped.append(c)
    return keep, dropped


def _cal_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "p": p}).sort_values("p").reset_index(drop=True)
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


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "eval_hit_prop.log")
    log_header("scripts/eval/eval_hit_prop.py", repo_root, config_path, dirs)

    mart_path = dirs["marts_dir"] / "hit_prop_features.parquet"
    df = pd.read_parquet(mart_path).copy()
    if TARGET not in df.columns:
        if "target_hit1p" in df.columns:
            df[TARGET] = pd.to_numeric(df["target_hit1p"], errors="coerce")
        elif "target_hit_1p" in df.columns:
            df[TARGET] = pd.to_numeric(df["target_hit_1p"], errors="coerce")
    if TARGET not in df.columns:
        raise ValueError("Missing target_hit_1_plus in mart")

    season = _season(df)
    train = df[(season >= args.train_start) & (season <= args.train_end)].copy()
    test = df[season == args.test_season].copy()

    train = train[pd.to_numeric(train[TARGET], errors="coerce").notna()].copy()
    test = test[pd.to_numeric(test[TARGET], errors="coerce").notna()].copy()
    y_train = pd.to_numeric(train[TARGET], errors="coerce").astype(int)
    y_test = pd.to_numeric(test[TARGET], errors="coerce").astype(int)

    excluded = {"game_pk", "batter_id", "opp_pitcher_id", "season", TARGET, "target_hit1p", "target_hit_1p"}
    numeric_features = [c for c in train.columns if pd.api.types.is_numeric_dtype(train[c]) and c not in excluded]
    unsafe_features = [c for c in numeric_features if not _is_safe_feature(c)]
    if unsafe_features:
        logging.info("hit_prop eval leakage_guard dropped_unsafe_features_n=%s features=%s", len(unsafe_features), unsafe_features)
    numeric_features = [c for c in numeric_features if _is_safe_feature(c)]
    feature_stats: list[tuple[str, int, float]] = []
    for c in numeric_features:
        nn = int(pd.to_numeric(train[c], errors="coerce").notna().sum())
        rate = float(nn / max(1, len(train)))
        feature_stats.append((c, nn, rate))
    logging.info("hit_prop eval candidate_numeric_features_n=%s", len(numeric_features))
    logging.info("hit_prop eval candidate_feature_non_null_stats=%s", feature_stats)

    feats, dropped_all_null = _drop_all_null_numeric_features(train, numeric_features)
    logging.info("hit_prop eval dropped_all_null_features_n=%s features=%s", len(dropped_all_null), dropped_all_null)
    logging.info("hit_prop eval final_feature_count=%s", len(feats))
    logging.info("hit_prop eval surviving_features=%s", feats)
    surviving_stats = [(c, int(pd.to_numeric(train[c], errors="coerce").notna().sum()), float(pd.to_numeric(train[c], errors="coerce").notna().mean())) for c in feats]
    logging.info("hit_prop eval surviving_feature_non_null_stats=%s", surviving_stats)
    opp_feats = ["lineup_slot", "expected_batting_order_pa", "lineup_confidence", "bat_ab_per_game_roll15", "bat_pa_per_game_roll15", "expected_ab_proxy"]
    opp_survival = {f: (f in feats) for f in opp_feats}
    logging.info("hit_prop eval opportunity_feature_survival=%s", opp_survival)
    if not feats:
        raise ValueError("No numeric non-null features available for evaluation")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, solver="lbfgs")),
    ])
    X_train = train[feats].replace([np.inf, -np.inf], np.nan)
    X_test = test[feats].replace([np.inf, -np.inf], np.nan)
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]

    scored = test.copy()
    scored["hit_probability"] = p
    top_cut = np.quantile(p, 0.9) if len(p) else 1.0
    top_decile = scored[scored["hit_probability"] >= top_cut]
    metrics = {
        "auc": float(roc_auc_score(y_test, p)) if y_test.nunique() > 1 else None,
        "brier": float(brier_score_loss(y_test, p)),
        "logloss": float(log_loss(y_test, p, labels=[0, 1])),
        "top_decile_hit_rate": float(pd.to_numeric(top_decile[TARGET], errors="coerce").mean()) if len(top_decile) else None,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_count": int(len(feats)),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = dirs["backtests_dir"] / "hit_prop"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"eval_hit_prop_{ts}.json"
    cal_path = out_dir / f"eval_hit_prop_{ts}_calibration.csv"
    scored_path = out_dir / f"eval_hit_prop_{ts}_scored_test.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "train_start": args.train_start,
            "train_end": args.train_end,
            "test_season": args.test_season,
            "dropped_all_null_features": dropped_all_null,
        }, f, indent=2)
    _cal_bins(y_test.to_numpy(), p).to_csv(cal_path, index=False)
    scored.to_csv(scored_path, index=False)

    logging.info("eval_hit_prop complete metrics=%s", metrics)
    print(f"eval_json={json_path}")


if __name__ == "__main__":
    main()
