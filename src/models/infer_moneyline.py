from __future__ import annotations

import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def _predict_raw(model: object, model_type: str, X: pd.DataFrame) -> np.ndarray:
    if model_type == "numpy_logreg":
        return model.predict_proba(X.to_numpy(dtype=float))[:, 1]
    return model.predict_proba(X)[:, 1]


def _apply_calibrator(calibrator: object, p_raw: np.ndarray) -> np.ndarray:
    p_raw = np.clip(p_raw.astype(float), 1e-6, 1 - 1e-6)
    return np.clip(calibrator.predict(p_raw), 0.0, 1.0)


def infer_moneyline_from_paths(
    season: int,
    features_path: Path,
    model_dir: Path,
    output_path: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features parquet: {features_path}")

    model_path = model_dir / "model.pkl"
    calibrator_path = model_dir / "calibrator.pkl"
    if not model_path.exists() or not calibrator_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found in {model_dir}. Expected {model_path.name} and {calibrator_path.name}"
        )

    with model_path.open("rb") as f:
        model_art = pickle.load(f)
    with calibrator_path.open("rb") as f:
        cal_art = pickle.load(f)

    model = model_art["model"]
    model_type: str = model_art.get("model_type", "unknown")
    feature_columns: list[str] = model_art["feature_columns"]
    calibrator = cal_art["calibrator"]

    features = pd.read_parquet(features_path, engine="pyarrow")
    if "game_date" not in features.columns or "game_pk" not in features.columns:
        raise ValueError("features must contain game_date and game_pk")

    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce").dt.normalize()
    features["game_pk"] = pd.to_numeric(features["game_pk"], errors="coerce").astype("Int64")

    features = features.loc[features["game_date"].dt.year == int(season)].copy()
    if start:
        features = features.loc[features["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        features = features.loc[features["game_date"] <= pd.to_datetime(end).normalize()].copy()

    missing_features = [c for c in feature_columns if c not in features.columns]
    if missing_features:
        raise ValueError(f"features missing columns required by trained model: {missing_features}")

    X = features[feature_columns]
    p_raw = _predict_raw(model, model_type, X)
    p_cal = _apply_calibrator(calibrator, p_raw)

    out = features[["game_date", "game_pk"]].copy()
    out["p_home_win_cal"] = p_cal
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info("wrote_moneyline_preds season=%s rows=%d path=%s", season, len(out), output_path)
    return out
