from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.models.numpy_ridge import NumpyRidgeRegression

ID_EXCLUDE = {"game_date", "game_pk", "home_team", "away_team", "home_sp_id", "away_sp_id", "starter_id"}


@dataclass
class TrainArtifacts:
    feature_columns: list[str]
    model_type: str
    model_path: Path
    metadata_path: Path


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _select_feature_columns(features: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in features.columns:
        if col in ID_EXCLUDE:
            continue
        low = col.lower()
        if low.endswith("_sp_id") or "starter" in low and low.endswith("_id"):
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
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
        )
        model.fit(X_train, y_train)
        return model, "xgboost_regressor"
    except Exception:
        pass

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("hgb", HistGradientBoostingRegressor(random_state=42)),
            ]
        )
        model.fit(X_train, y_train)
        return model, "sklearn_hgb"
    except Exception:
        pass

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("ridge", Ridge(alpha=1.0, random_state=42)),
            ]
        )
        model.fit(X_train, y_train)
        return model, "sklearn_ridge"
    except Exception:
        pass

    model = NumpyRidgeRegression(alpha=1.0)
    model.fit(X_train.to_numpy(dtype=float), y_train)
    return model, "numpy_ridge"


def _predict(model: object, model_type: str, X: pd.DataFrame) -> np.ndarray:
    if model_type == "numpy_ridge":
        return model.predict(X.to_numpy(dtype=float))
    return np.asarray(model.predict(X), dtype=float)


def _reliability_deciles(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true.astype(float), "y_pred": y_pred.astype(float)})
    frame["decile"] = pd.qcut(frame["y_pred"], q=10, labels=False, duplicates="drop")
    return (
        frame.groupby("decile", observed=False)
        .agg(n=("y_true", "size"), avg_pred=("y_pred", "mean"), avg_actual=("y_true", "mean"))
        .reset_index()
        .sort_values("decile", kind="mergesort")
        .reset_index(drop=True)
    )


def run_smoke_test(logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger(__name__)
    rng = np.random.default_rng(202)
    X = rng.normal(size=(50, 7))
    X[rng.integers(0, 50, size=8), rng.integers(0, 7, size=8)] = np.nan
    y = 4.0 + 1.2 * np.nan_to_num(X[:, 0], nan=0.0) - 0.7 * np.nan_to_num(X[:, 1], nan=0.0) + rng.normal(0, 0.2, 50)

    model = NumpyRidgeRegression(alpha=1.0).fit(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    log.info("Totals smoke test passed")


def train_totals_model(
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
    req_tgt = {"game_date", "game_pk", "total_runs"}
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
    targs["total_runs"] = pd.to_numeric(targs["total_runs"], errors="coerce")

    merged = feats.merge(targs[["game_date", "game_pk", "total_runs"]], on=["game_date", "game_pk"], how="inner")
    merged = merged.dropna(subset=["game_date", "game_pk", "total_runs"]).copy()
    merged = merged.loc[merged["game_date"].dt.year == int(season)].copy()

    feature_cols = _select_feature_columns(merged.drop(columns=["total_runs"]))
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions")

    train_df, val_df = _time_split(merged)
    X_train = train_df[feature_cols]
    y_train = train_df["total_runs"].to_numpy(dtype=float)
    X_val = val_df[feature_cols]
    y_val = val_df["total_runs"].to_numpy(dtype=float)

    model, model_type = _fit_base_model(X_train, y_train)
    pred_val = _predict(model, model_type, X_val)

    rmse = _rmse(y_val, pred_val)
    mae = _mae(y_val, pred_val)
    r2 = _r2(y_val, pred_val)
    rel_table = _reliability_deciles(y_val, pred_val)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    metadata_path = model_dir / "metadata.json"
    eval_path = Path("data/outputs") / f"totals_eval_{season}.csv"
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    for p in [model_path, metadata_path, eval_path]:
        if p.exists() and not force:
            raise FileExistsError(f"Output already exists (use --force to overwrite): {p}")

    with model_path.open("wb") as f:
        pickle.dump({"model": model, "feature_columns": feature_cols, "model_type": model_type}, f)

    eval_df = val_df[["game_date", "game_pk"]].copy()
    eval_df["total_runs_pred"] = pred_val
    eval_df["y_true"] = y_val
    eval_df = eval_df.sort_values(["game_date", "game_pk"], kind="mergesort")
    eval_df.to_csv(eval_path, index=False)

    metadata = {
        "season": season,
        "target": "total_runs",
        "model_type": model_type,
        "feature_columns": feature_cols,
        "n_train": int(len(train_df)),
        "n_valid": int(len(val_df)),
        "train_start": str(train_df["game_date"].min().date()),
        "train_end": str(train_df["game_date"].max().date()),
        "valid_start": str(val_df["game_date"].min().date()),
        "valid_end": str(val_df["game_date"].max().date()),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "reliability_deciles": rel_table.to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log.info("Totals metrics: rmse=%.4f mae=%.4f r2=%.4f", rmse, mae, r2)
    log.info("Totals reliability deciles:\n%s", rel_table.to_string(index=False))
    log.info("Saved model=%s eval=%s", model_path, eval_path)

    return TrainArtifacts(
        feature_columns=feature_cols,
        model_type=model_type,
        model_path=model_path,
        metadata_path=metadata_path,
    )


def train_totals_from_paths(
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
    return train_totals_model(
        season=season,
        features_df=features_df,
        targets_df=targets_df,
        model_dir=model_dir,
        force=force,
        logger=logger,
    )
