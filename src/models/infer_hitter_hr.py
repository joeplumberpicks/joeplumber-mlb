from __future__ import annotations

import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def _predict_raw(model: object, model_type: str, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X.to_numpy(dtype=float))[:, 1] if model_type == "numpy_logreg" else model.predict_proba(X)[:, 1]


def _apply_calibrator(calibrator: object, p_raw: np.ndarray) -> np.ndarray:
    p_raw = np.clip(p_raw.astype(float), 1e-6, 1 - 1e-6)
    return np.clip(calibrator.predict(p_raw), 0.0, 1.0)


def infer_hitter_hr_from_paths(
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
    cal_path = model_dir / "calibrator.pkl"
    if not model_path.exists() or not cal_path.exists():
        raise FileNotFoundError(f"Model artifacts not found in {model_dir}. Expected model.pkl and calibrator.pkl")

    with model_path.open("rb") as f:
        model_art = pickle.load(f)
    with cal_path.open("rb") as f:
        cal_art = pickle.load(f)

    features = pd.read_parquet(features_path, engine="pyarrow")
    for c in ["game_date", "game_pk", "batter_id"]:
        if c not in features.columns:
            raise ValueError(f"features must contain {c}")

    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce").dt.normalize()
    features["game_pk"] = pd.to_numeric(features["game_pk"], errors="coerce").astype("Int64")
    features["batter_id"] = pd.to_numeric(features["batter_id"], errors="coerce").astype("Int64")

    features = features.loc[features["game_date"].dt.year == int(season)].copy()
    if start:
        features = features.loc[features["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        features = features.loc[features["game_date"] <= pd.to_datetime(end).normalize()].copy()

    feat_cols: list[str] = model_art["feature_columns"]
    missing = [c for c in feat_cols if c not in features.columns]
    if missing:
        raise ValueError(f"features missing columns required by trained model: {missing}")

    p_raw = _predict_raw(model_art["model"], model_art.get("model_type", "unknown"), features[feat_cols])
    p_cal = _apply_calibrator(cal_art["calibrator"], p_raw)

    out = features[["game_date", "game_pk", "batter_id"]].copy()
    out["p_hr_cal"] = p_cal
    out = out.sort_values(["game_date", "game_pk", "batter_id"], kind="mergesort").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info("wrote_hitter_hr_preds season=%s rows=%d path=%s", season, len(out), output_path)
    return out
