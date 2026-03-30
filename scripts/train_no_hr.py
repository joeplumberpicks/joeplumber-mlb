from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import brier_score_loss, log_loss

logger = logging.getLogger(__name__)

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
    core_features = [
        "home_team_hitter_hr_danger_score_mean",
        "home_team_hitter_hr_danger_score_max",
        "away_team_hitter_hr_danger_score_mean",
        "away_team_hitter_hr_danger_score_max",
        "home_team_launch_speed_max_roll15_top3_mean",
        "away_team_launch_speed_max_roll15_top3_mean",
        "home_sp_hr_rate_roll15",
        "away_sp_hr_rate_roll15",
        "home_team_launch_angle_mean_roll15_mean",
        "away_team_launch_angle_mean_roll15_mean",
        "temperature",
        "wind_speed",
    ]

    feature_data: dict[str, pd.Series] = {}
    available: list[str] = []
    missing: list[str] = []

    for c in core_features:
        if c in {"home_sp_hr_rate_roll15", "away_sp_hr_rate_roll15"}:
            continue
        if c in df.columns:
            feature_data[c] = pd.to_numeric(df[c], errors="coerce")
            available.append(c)
        else:
            missing.append(c)

    home_sp_candidates = ["home_sp_hr_rate_roll15", "home_sp_hr_roll15", "home_sp_hr_allowed_roll15"]
    away_sp_candidates = ["away_sp_hr_rate_roll15", "away_sp_hr_roll15", "away_sp_hr_allowed_roll15"]
    home_sp_source = next((c for c in home_sp_candidates if c in df.columns), None)
    away_sp_source = next((c for c in away_sp_candidates if c in df.columns), None)

    if home_sp_source is not None:
        feature_data["home_sp_hr_rate_roll15"] = pd.to_numeric(df[home_sp_source], errors="coerce")
        available.append("home_sp_hr_rate_roll15")
    else:
        missing.append("home_sp_hr_rate_roll15")
    if away_sp_source is not None:
        feature_data["away_sp_hr_rate_roll15"] = pd.to_numeric(df[away_sp_source], errors="coerce")
        available.append("away_sp_hr_rate_roll15")
    else:
        missing.append("away_sp_hr_rate_roll15")

    X = pd.DataFrame(feature_data, index=df.index)
    X = X.fillna(0.0).astype("float32")

    logger.info("feature selection requested core count=%s", len(core_features))
    logger.info("feature selection available core count=%s", len(available))
    logger.info("feature selection missing core columns=%s", missing)
    return X, available


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(pd.Series(p), q=min(bins, max(2, len(np.unique(p)))), duplicates="drop")
    tmp = pd.DataFrame({"y": y, "p": p, "bin": q})
    return tmp.groupby("bin", dropna=False).agg(pred_mean=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size")).reset_index()


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
    baseline_rate = float(train_df[target_col].mean())
    logger.info(f"baseline_no_hr_rate={baseline_rate:.4f}")

    logging.info("train rows=%s test rows=%s", len(train_df), len(test_df))
    if "season" in train_df.columns:
        logging.info("train rows by season: %s", train_df["season"].value_counts().sort_index().to_dict())
    if "season" in test_df.columns:
        logging.info("test rows by season: %s", test_df["season"].value_counts().sort_index().to_dict())
    logging.info("train no_hr rate=%.6f", float(np.mean(y_train)) if len(y_train) else float("nan"))
    logging.info("test no_hr rate=%.6f", float(np.mean(y_test)) if len(y_test) else float("nan"))

    X_train, feat_cols = _prep_X(train_df, target_col)
    X_test = test_df.reindex(columns=feat_cols).copy() if feat_cols else pd.DataFrame(index=test_df.index)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    if not feat_cols:
        raise ValueError("No core features available in marts. Rebuild features with required no-hr columns.")

    baseline = baseline_rate
    y_train_delta = y_train - baseline
    y_test_actual = y_test.copy()

    try:
        from xgboost import XGBRegressor  # type: ignore

        model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    except Exception:
        raise RuntimeError("xgboost is required for baseline+delta no-hr training.")

    model.fit(X_train, y_train_delta)
    pred_delta = model.predict(X_test)
    pred_prob = baseline + pred_delta
    pred_prob = np.clip(pred_prob, 0.01, 0.50)

    brier = float(brier_score_loss(y_test_actual, pred_prob)) if len(y_test_actual) else float("nan")
    ll = float(log_loss(y_test_actual, pred_prob, labels=[0, 1])) if len(y_test_actual) else float("nan")
    pred_mean = float(np.mean(pred_prob)) if len(pred_prob) else float("nan")
    obs_mean = float(np.mean(y_test_actual)) if len(y_test_actual) else float("nan")
    cal_tbl = _calibration_table(y_test_actual, pred_prob)
    logger.info(f"pred_prob_mean={pred_prob.mean():.4f}")
    logger.info(f"obs_rate={y_test_actual.mean():.4f}")

    q90 = float(np.quantile(pred_prob, 0.9))
    q10 = float(np.quantile(pred_prob, 0.1))
    top_decile_rate = float(y_test_actual[pred_prob >= q90].mean()) if len(y_test_actual) else float("nan")
    bot_decile_rate = float(y_test_actual[pred_prob <= q10].mean()) if len(y_test_actual) else float("nan")
    logger.info("top_decile_no_hr_rate=%.4f bottom_decile_no_hr_rate=%.4f", top_decile_rate, bot_decile_rate)

    signature = f"train_{'-'.join(map(str, train_seasons))}_test_{'-'.join(map(str, test_seasons))}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    model_dir = dirs["models_dir"] / "no_hr"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"no_hr_model_{signature}.pkl"
    dump(
        {
            "model": model,
            "features": feat_cols,
            "target_col": target_col,
            "signature": signature,
            "base_model": "xgboost_reg_delta",
            "baseline_no_hr_rate": baseline,
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
            "model": "xgboost_reg_delta",
            "brier": brier,
            "logloss": ll,
            "pred_mean": pred_mean,
            "obs_rate": obs_mean,
            "baseline_no_hr_rate": baseline,
            "top_decile_no_hr_rate": top_decile_rate,
            "bottom_decile_no_hr_rate": bot_decile_rate,
            "n_test": len(y_test),
        }
    ])
    bt_path = bt_dir / f"no_hr_backtest_{signature}.csv"
    backtest.to_csv(bt_path, index=False)

    cal_path = bt_dir / f"no_hr_calibration_{signature}.csv"
    cal_tbl.to_csv(cal_path, index=False)

    board = pd.DataFrame(
        {
            "game_id": test_df["game_pk"].values if "game_pk" in test_df.columns else np.arange(len(test_df)),
            "home_team": test_df["home_team"].values if "home_team" in test_df.columns else pd.Series([pd.NA] * len(test_df)),
            "away_team": test_df["away_team"].values if "away_team" in test_df.columns else pd.Series([pd.NA] * len(test_df)),
            "no_hr_prob": pred_prob,
            "delta_vs_baseline": pred_prob - baseline,
        }
    )
    board_path = bt_dir / f"no_hr_board_{signature}.csv"
    board.to_csv(board_path, index=False)

    logging.info(
        "no_hr train complete model=%s brier=%.6f logloss=%.6f pred_mean=%.6f obs_rate=%.6f model_path=%s",
        "xgboost_reg_delta",
        brier,
        ll,
        pred_mean,
        obs_mean,
        model_path.resolve(),
    )
    logging.info("features saved path=%s", feat_path.resolve())
    logging.info("calibration_by_decile path=%s", cal_path.resolve())
    logging.info("board path=%s", board_path.resolve())
    print(f"model -> {model_path.resolve()}")
    print(f"backtest -> {bt_path.resolve()}")


if __name__ == "__main__":
    main()
