from __future__ import annotations

import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def _predict(model: object, model_type: str, X: pd.DataFrame) -> np.ndarray:
    if model_type == "numpy_ridge":
        return model.predict(X.to_numpy(dtype=float))
    return np.asarray(model.predict(X), dtype=float)


def infer_totals_from_paths(
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
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found in {model_dir}: expected {model_path.name}")

    with model_path.open("rb") as f:
        model_art = pickle.load(f)

    model = model_art["model"]
    model_type: str = model_art.get("model_type", "unknown")
    feature_columns: list[str] = model_art["feature_columns"]

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
    preds = _predict(model, model_type, X)

    out = features[["game_date", "game_pk"]].copy()
    out["total_runs_pred"] = preds
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info("wrote_totals_preds season=%s rows=%d path=%s", season, len(out), output_path)
    return out
