from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.models.calibration import NumpyPlattScaler
from src.models.numpy_logreg import NumpyLogisticRegression

ID_EXCLUDE = {"game_date", "game_pk", "home_team", "away_team", "home_sp_id", "away_sp_id"}


@dataclass
class TrainArtifacts:
    feature_columns: list[str]
    model_type: str
    calibrator_method: str
    model_path: Path
    calibrator_path: Path
    eval_path: Path
    metadata_path: Path


def _safe_auc(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(p, dtype=float)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    sum_ranks_pos = ranks[pos].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _log_loss(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def _select_feature_columns(features: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in features.columns:
        if col in ID_EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(features[col]):
            cols.append(col)
    return cols


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    if n < 5:
        raise ValueError("Need at least 5 rows to perform 80/20 time split")
    split_idx = max(1, int(np.floor(n * 0.8)))
    split_idx = min(split_idx, n - 1)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def _fit_base_model(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[object, str]:
    try:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        )
        model.fit(X_train, y_train)
        return model, "xgboost"
    except Exception:
        pass

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("logreg", LogisticRegression(max_iter=1000)),
            ]
        )
        model.fit(X_train, y_train)
        return model, "sklearn_logreg"
    except Exception:
        pass

    model = NumpyLogisticRegression(lr=0.05, epochs=3000, reg_lambda=1e-3)
    model.fit(X_train.to_numpy(dtype=float), y_train)
    return model, "numpy_logreg"


def _predict_raw(model: object, model_type: str, X: pd.DataFrame) -> np.ndarray:
    if model_type == "numpy_logreg":
        return model.predict_proba(X.to_numpy(dtype=float))[:, 1]
    return model.predict_proba(X)[:, 1]


def _fit_calibrator(y_val: np.ndarray, p_raw: np.ndarray) -> tuple[object, str]:
    p_raw = np.clip(p_raw.astype(float), 1e-6, 1 - 1e-6)

    try:
        from sklearn.isotonic import IsotonicRegression

        unique_scores = np.unique(np.round(p_raw, 6))
        if len(unique_scores) >= 10 and len(np.unique(y_val)) >= 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_raw, y_val)
            return iso, "isotonic"
    except Exception:
        pass

    platt = NumpyPlattScaler(lr=0.05, epochs=3000, reg_lambda=1e-3)
    platt.fit(p_raw, y_val)
    return platt, "platt_numpy"


def _apply_calibrator(calibrator: object, method: str, p_raw: np.ndarray) -> np.ndarray:
    p_raw = np.clip(p_raw.astype(float), 1e-6, 1 - 1e-6)
    return np.clip(calibrator.predict(p_raw), 0.0, 1.0)


def _calibration_deciles(y_true: np.ndarray, p_cal: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true.astype(int), "p_cal": p_cal.astype(float)})
    frame["decile"] = pd.qcut(frame["p_cal"], q=10, labels=False, duplicates="drop")
    return (
        frame.groupby("decile", observed=False)
        .agg(n=("y_true", "size"), mean_pred=("p_cal", "mean"), observed_rate=("y_true", "mean"))
        .reset_index()
        .sort_values("decile", kind="mergesort")
        .reset_index(drop=True)
    )


def run_smoke_test(logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger(__name__)
    rng = np.random.default_rng(123)
    X = rng.normal(size=(40, 6))
    X[rng.integers(0, 40, size=6), rng.integers(0, 6, size=6)] = np.nan
    y = (0.8 * np.nan_to_num(X[:, 0], nan=0.0) - 0.5 * np.nan_to_num(X[:, 1], nan=0.0) > 0).astype(int)

    model = NumpyLogisticRegression(lr=0.05, epochs=500, reg_lambda=1e-3).fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert np.all(p >= 0.0) and np.all(p <= 1.0)

    cal = NumpyPlattScaler(lr=0.05, epochs=500, reg_lambda=1e-3).fit(p, y)
    p_cal = cal.predict(p)
    assert np.all(p_cal >= 0.0) and np.all(p_cal <= 1.0)
    log.info("Moneyline smoke test passed")


def train_moneyline_model(
    season: int,
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    model_dir: Path,
    *,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> TrainArtifacts:
    log = logger or logging.getLogger(__name__)

    req_feat = {"game_date", "game_pk"}
    req_tgt = {"game_date", "game_pk", "home_win"}
    if not req_feat.issubset(features_df.columns):
        raise ValueError(f"features missing required columns: {sorted(req_feat - set(features_df.columns))}")
    if not req_tgt.issubset(targets_df.columns):
        raise ValueError(f"targets missing required columns: {sorted(req_tgt - set(targets_df.columns))}")

    feats = features_df.copy()
    targs = targets_df.copy()
    feats["game_date"] = pd.to_datetime(feats["game_date"], errors="coerce").dt.normalize()
    targs["game_date"] = pd.to_datetime(targs["game_date"], errors="coerce").dt.normalize()
    feats["game_pk"] = pd.to_numeric(feats["game_pk"], errors="coerce").astype("Int64")
    targs["game_pk"] = pd.to_numeric(targs["game_pk"], errors="coerce").astype("Int64")
    targs["home_win"] = pd.to_numeric(targs["home_win"], errors="coerce").astype("Int64")

    merged = feats.merge(targs[["game_date", "game_pk", "home_win"]], on=["game_date", "game_pk"], how="inner")
    merged = merged.dropna(subset=["game_date", "game_pk", "home_win"]).copy()
    merged = merged.loc[merged["game_date"].dt.year == int(season)].copy()

    feature_cols = _select_feature_columns(merged.drop(columns=["home_win"]))
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions")

    train_df, val_df = _time_split(merged)
    X_train = train_df[feature_cols]
    y_train = train_df["home_win"].astype(int).to_numpy()
    X_val = val_df[feature_cols]
    y_val = val_df["home_win"].astype(int).to_numpy()

    model, model_type = _fit_base_model(X_train, y_train)
    p_raw = _predict_raw(model, model_type, X_val)
    calibrator, cal_method = _fit_calibrator(y_val, p_raw)
    p_cal = _apply_calibrator(calibrator, cal_method, p_raw)

    auc = _safe_auc(y_val, p_cal)
    ll = _log_loss(y_val, p_cal)
    brier = _brier(y_val, p_cal)
    decile_table = _calibration_deciles(y_val, p_cal)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    calibrator_path = model_dir / "calibrator.pkl"
    eval_path = Path("data/outputs") / f"ml_eval_{season}.csv"
    metadata_path = model_dir / "metadata.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    for p in [model_path, calibrator_path, eval_path, metadata_path]:
        if p.exists() and not force:
            raise FileExistsError(f"Output already exists (use --force to overwrite): {p}")

    with model_path.open("wb") as f:
        pickle.dump({"model": model, "feature_columns": feature_cols, "model_type": model_type}, f)
    with calibrator_path.open("wb") as f:
        pickle.dump({"calibrator": calibrator, "method": cal_method}, f)

    eval_df = val_df[["game_date", "game_pk"]].copy()
    eval_df["p_home_win_raw"] = p_raw
    eval_df["p_home_win_cal"] = p_cal
    eval_df["y_true"] = y_val
    eval_df = eval_df.sort_values(["game_date", "game_pk"], kind="mergesort")
    eval_df.to_csv(eval_path, index=False)

    metadata = {
        "season": season,
        "target": "home_win",
        "model_type": model_type,
        "calibrator_method": cal_method,
        "n_train": int(len(train_df)),
        "n_valid": int(len(val_df)),
        "roc_auc": auc,
        "log_loss": ll,
        "brier_score": brier,
        "feature_columns": feature_cols,
        "calibration_deciles": decile_table.to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log.info("Moneyline metrics: roc_auc=%.4f log_loss=%.4f brier=%.4f", auc, ll, brier)
    log.info("Calibration deciles:\n%s", decile_table.to_string(index=False))
    log.info("Saved model=%s calibrator=%s eval=%s", model_path, calibrator_path, eval_path)

    return TrainArtifacts(
        feature_columns=feature_cols,
        model_type=model_type,
        calibrator_method=cal_method,
        model_path=model_path,
        calibrator_path=calibrator_path,
        eval_path=eval_path,
        metadata_path=metadata_path,
    )


def train_moneyline_from_paths(
    season: int,
    features_path: Path,
    targets_path: Path,
    model_dir: Path,
    *,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> TrainArtifacts:
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features parquet: {features_path}")
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing targets parquet: {targets_path}")

    features_df = pd.read_parquet(features_path, engine="pyarrow")
    targets_df = pd.read_parquet(targets_path, engine="pyarrow")
    return train_moneyline_model(
        season=season,
        features_df=features_df,
        targets_df=targets_df,
        model_dir=model_dir,
        force=force,
        logger=logger,
    )
