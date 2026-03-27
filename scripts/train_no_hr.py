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
    p.add_argument(
        "--allow-missing-train-seasons",
        action="store_true",
        help="Allow training when requested train seasons have zero labeled rows.",
    )
    p.add_argument(
        "--allow-missing-test-seasons",
        action="store_true",
        help="Allow training when requested test seasons have zero labeled rows.",
    )
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
    always_drop = {
        "no_hr_game",
        "total_hr",
        "game_pk",
        "game_date",
        "home_sp_id",
        "away_sp_id",
        "home_team",
        "away_team",
        "park_name",
        "canonical_park_key",
        target_col,
    }
    explicit_allow = {
        "season",
        "temperature",
        "wind_speed",
        "wind_direction",
        "combined_park_weather_hr_index",
        "env_hr_suppression_proxy",
        "starter_hr_suppression_gap_away_vs_home",
        "starter_hr_suppression_gap_home_vs_away",
    }
    roll_tokens = ("_roll3", "_roll7", "_roll15", "_roll30")
    trusted_stat_family_tokens = (
        "whiff_rate",
        "contact_rate",
        "chase_rate",
        "swing_rate",
        "zone_swing_rate",
        "launch_speed",
        "launch_angle",
        "release_speed",
        "release_spin_rate",
        "barrel",
        "hard_hit",
        "hardhit",
        "fly",
        "fly_ball",
        "flyball",
        "fb",
        "fb_rate",
        "air",
        "air_ball",
        "airball",
        "air_rate",
        "pulled",
        "pulled_air",
        "pull_air",
        "iso",
        "slug",
        "slg",
        "xbh",
    )
    safe_engineered_tokens = (
        "hr_danger",
        "suppression",
        "env_interaction",
        "vs_",
        "top3",
        "danger_score",
    )
    banned_substrings = (
        "game_pk",
        "game_id",
        "game_date",
        "_id",
        "lineup_id",
    )

    initial_feats = [c for c in df.columns if c not in always_drop]
    dropped_cols: list[str] = []
    allow_kept_cols: list[str] = []
    for c in initial_feats:
        lc = c.lower()
        is_roll = any(tok in lc for tok in roll_tokens)
        is_explicit_allow = c in explicit_allow
        has_trusted_stat_family = any(tok in lc for tok in trusted_stat_family_tokens)
        is_safe_engineered = any(tok in lc for tok in safe_engineered_tokens)
        is_banned_roll = any(tok in lc for tok in banned_substrings)
        is_weather_prefield = lc.startswith("weather_")

        if (
            ((is_roll and has_trusted_stat_family) and not is_banned_roll)
            or is_explicit_allow
            or is_weather_prefield
            or is_safe_engineered
        ):
            allow_kept_cols.append(c)
        else:
            dropped_cols.append(c)

    assert isinstance(allow_kept_cols, list)
    assert isinstance(dropped_cols, list)

    X = df[allow_kept_cols].copy()

    for c in X.columns:
        if pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].astype("Int64")
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    keep = [c for c in X.columns if X[c].notna().any()]
    X = X[keep].astype("float32") if keep else pd.DataFrame({"bias": np.zeros(len(df), dtype="float32")})

    logging.info("feature selection initial feature count=%s", len(initial_feats))
    logging.info("feature selection kept count=%s", len(keep))
    logging.info("feature selection dropped count=%s", len(dropped_cols))
    logging.info("feature selection first 50 kept columns=%s", keep[:50])
    logging.info("feature selection first 50 dropped columns=%s", dropped_cols[:50])
    logging.info(
        "engineered features kept count=%s sample=%s",
        len([c for c in allow_kept_cols if any(tok in c.lower() for tok in safe_engineered_tokens)]),
        [c for c in allow_kept_cols if any(tok in c.lower() for tok in safe_engineered_tokens)][:10],
    )
    return X, list(X.columns)


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p), q=min(bins, max(2, len(np.unique(p)))), duplicates="drop")
    tmp = pd.DataFrame({"y": y, "p": p, "bin": q})
    return tmp.groupby("bin", dropna=False).agg(pred_mean=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size")).reset_index()


def _scan_and_drop_leakage_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    corr_threshold: float = 0.98,
    auc_threshold: float = 0.98,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    y_series = pd.Series(y_train, index=X_train.index)
    leaked_features: list[str] = []

    for c in X_train.columns:
        s = pd.to_numeric(X_train[c], errors="coerce")
        valid = s.notna() & y_series.notna()
        if valid.sum() < 2:
            continue

        y_valid = y_series[valid]
        s_valid = s[valid]

        corr = float(s_valid.corr(y_valid))
        if not np.isfinite(corr):
            corr = 0.0

        single_auc = float("nan")
        if y_valid.nunique() > 1 and s_valid.nunique() > 1:
            try:
                single_auc = float(roc_auc_score(y_valid, s_valid))
            except Exception:
                single_auc = float("nan")
        if not np.isfinite(single_auc):
            single_auc = 0.5

        if abs(corr) > corr_threshold or single_auc > auc_threshold:
            logging.warning('LEAKAGE WARNING: Feature "%s" AUC=%.6f correlation=%.6f -> DROPPED', c, single_auc, corr)
            leaked_features.append(c)

    if leaked_features:
        X_train = X_train.drop(columns=leaked_features, errors="ignore")
        X_test = X_test.drop(columns=leaked_features, errors="ignore")

    logging.info("leakage scan removed features=%s", len(leaked_features))
    logging.info("feature count after leakage scan=%s", X_train.shape[1])
    return X_train, X_test, leaked_features


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
    if "season" in train_df.columns:
        labeled_train_seasons = set(pd.to_numeric(train_df["season"], errors="coerce").dropna().astype(int).tolist())
        missing_labeled = [s for s in train_seasons if s not in labeled_train_seasons]
        if missing_labeled:
            logging.warning("requested train seasons with zero labeled rows: %s", missing_labeled)
            if not args.allow_missing_train_seasons:
                raise ValueError(
                    f"Requested train seasons have zero labeled rows: {missing_labeled}. "
                    "Rebuild targets and marts for those seasons, or rerun with --allow-missing-train-seasons."
                )
    if "season" in test_df.columns:
        labeled_test_seasons = set(pd.to_numeric(test_df["season"], errors="coerce").dropna().astype(int).tolist())
        missing_labeled_test = [s for s in test_seasons if s not in labeled_test_seasons]
        if missing_labeled_test:
            logging.warning("requested test seasons with zero labeled rows: %s", missing_labeled_test)
            if not args.allow_missing_test_seasons:
                raise ValueError(
                    f"Requested test seasons have zero labeled rows: {missing_labeled_test}. "
                    "Rebuild targets and marts for those seasons, or rerun with --allow-missing-test-seasons."
                )
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
    X_train, X_test, leaked_features = _scan_and_drop_leakage_features(X_train, X_test, y_train)
    feat_cols = list(X_train.columns)
    if not feat_cols:
        raise ValueError("All features were removed after leakage scan. Investigate feature engineering.")

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
    if np.isfinite(auc) and auc > 0.90:
        logging.warning("Model performance unusually high — possible residual leakage (auc=%.6f)", auc)
    if np.isfinite(auc) and auc > 0.97:
        raise ValueError("Model AUC exceeds 0.97 — leakage still present. Investigate feature engineering.")

    signature = f"train_{'-'.join(map(str, train_seasons))}_test_{'-'.join(map(str, test_seasons))}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_dir = dirs["models_dir"] / "no_hr"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"no_hr_model_{signature}.pkl"
    dump(
        {
            "model": calibrated,
            "features": feat_cols,
            "target_col": target_col,
            "signature": signature,
            "base_model": base_model_name,
            "leakage_dropped_features": leaked_features,
        },
        model_path,
    )

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
