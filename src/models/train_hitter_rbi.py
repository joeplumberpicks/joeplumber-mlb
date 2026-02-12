from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.models.numpy_ridge import NumpyRidgeRegression

ID_EXCLUDE = {"game_date", "game_pk", "batter_id", "team", "opponent_team", "park"}
TARGET_CANDIDATES = ["rbi"]


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
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["game_date", "game_pk", "batter_id"], kind="mergesort").reset_index(drop=True)
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

        model = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("hgb", HistGradientBoostingRegressor(random_state=42))])
        model.fit(X_train, y_train)
        return model, "sklearn_hgb"
    except Exception:
        pass

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline

        model = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("ridge", Ridge(alpha=1.0))])
        model.fit(X_train, y_train)
        return model, "sklearn_ridge"
    except Exception:
        pass

    return NumpyRidgeRegression(alpha=1.0).fit(X_train.to_numpy(dtype=float), y_train), "numpy_ridge"


def _predict(model: object, model_type: str, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X.to_numpy(dtype=float)) if model_type == "numpy_ridge" else np.asarray(model.predict(X), dtype=float)


def _reliability_deciles(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true.astype(float), "y_pred": y_pred.astype(float)})
    frame["decile"] = pd.qcut(frame["y_pred"], q=10, labels=False, duplicates="drop")
    return (frame.groupby("decile", observed=False)
        .agg(n=("y_true", "size"), avg_pred=("y_pred", "mean"), avg_actual=("y_true", "mean"))
        .reset_index().sort_values("decile", kind="mergesort").reset_index(drop=True))


def _resolve_target_col(targets_df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in targets_df.columns:
            return c
    raise ValueError(f"targets missing Hitter RBI target column; expected one of {TARGET_CANDIDATES}")


def run_smoke_test(logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger(__name__)
    rng = np.random.default_rng(52)
    X = rng.normal(size=(40, 6))
    X[rng.integers(0, 40, 5), rng.integers(0, 6, 5)] = np.nan
    y = 0.9 + 0.6 * np.nan_to_num(X[:, 0], nan=0.0) + rng.normal(0, 0.2, 40)
    preds = NumpyRidgeRegression(alpha=1.0).fit(X, y).predict(X)
    assert np.isfinite(preds).all()
    log.info("Hitter RBI smoke test passed")


def train_hitter_rbi_model(
    season: int,
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    model_dir: Path,
    *,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> TrainArtifacts:
    log = logger or logging.getLogger(__name__)
    req_feat = {"game_date", "game_pk", "batter_id"}
    if not req_feat.issubset(features_df.columns):
        raise ValueError(f"features missing required columns: {sorted(req_feat - set(features_df.columns))}")

    target_col = _resolve_target_col(targets_df)
    req_tgt = {"game_date", "game_pk", "batter_id", target_col}
    if not req_tgt.issubset(targets_df.columns):
        raise ValueError(f"targets missing required columns: {sorted(req_tgt - set(targets_df.columns))}")

    feats, targs = features_df.copy(), targets_df.copy()
    for d in [feats, targs]:
        d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce").dt.normalize()
        d["game_pk"] = pd.to_numeric(d["game_pk"], errors="coerce").astype("Int64")
        d["batter_id"] = pd.to_numeric(d["batter_id"], errors="coerce").astype("Int64")
    targs[target_col] = pd.to_numeric(targs[target_col], errors="coerce")

    merged = feats.merge(targs[["game_date", "game_pk", "batter_id", target_col]], on=["game_date", "game_pk", "batter_id"], how="inner")
    merged = merged.dropna(subset=["game_date", "game_pk", "batter_id", target_col]).copy()
    merged = merged.loc[merged["game_date"].dt.year == int(season)].copy()

    feature_cols = _select_feature_columns(merged.drop(columns=[target_col]))
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions")

    train_df, val_df = _time_split(merged)
    model, model_type = _fit_base_model(train_df[feature_cols], train_df[target_col].to_numpy(dtype=float))
    pred_val = _predict(model, model_type, val_df[feature_cols])
    y_val = val_df[target_col].to_numpy(dtype=float)

    rmse, mae, r2 = _rmse(y_val, pred_val), _mae(y_val, pred_val), _r2(y_val, pred_val)
    rel = _reliability_deciles(y_val, pred_val)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path, metadata_path = model_dir / "model.pkl", model_dir / "metadata.json"
    eval_path = Path("data/outputs") / f"hitter_rbi_eval_{season}.csv"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    for p in [model_path, metadata_path, eval_path]:
        if p.exists() and not force:
            raise FileExistsError(f"Output already exists (use --force to overwrite): {p}")

    with model_path.open("wb") as f:
        pickle.dump({"model": model, "feature_columns": feature_cols, "model_type": model_type}, f)

    eval_df = val_df[["game_date", "game_pk", "batter_id"]].copy()
    eval_df["rbi_pred"] = pred_val
    eval_df["y_true"] = y_val
    eval_df.sort_values(["game_date", "game_pk", "batter_id"], kind="mergesort").to_csv(eval_path, index=False)

    metadata_path.write_text(json.dumps({
        "season": season, "target": target_col, "model_type": model_type,
        "feature_columns": feature_cols, "n_train": int(len(train_df)), "n_valid": int(len(val_df)),
        "rmse": rmse, "mae": mae, "r2": r2, "reliability_deciles": rel.to_dict(orient="records")
    }, indent=2), encoding="utf-8")

    log.info("Hitter RBI metrics: rmse=%.4f mae=%.4f r2=%.4f", rmse, mae, r2)
    log.info("Hitter RBI reliability deciles:\n%s", rel.to_string(index=False))
    log.info("Saved model=%s eval=%s", model_path, eval_path)
    return TrainArtifacts(feature_columns=feature_cols, model_type=model_type, model_path=model_path, metadata_path=metadata_path)


def train_hitter_rbi_from_paths(
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
    return train_hitter_rbi_model(
        season=season,
        features_df=pd.read_parquet(features_path, engine="pyarrow"),
        targets_df=pd.read_parquet(targets_path, engine="pyarrow"),
        model_dir=model_dir,
        force=force,
        logger=logger,
    )
