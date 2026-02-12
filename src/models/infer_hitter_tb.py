from __future__ import annotations

import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


def _predict(model: object, model_type: str, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X.to_numpy(dtype=float)) if model_type == "numpy_ridge" else np.asarray(model.predict(X), dtype=float)


def infer_hitter_tb_from_paths(
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
        art = pickle.load(f)

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

    feature_columns: list[str] = art["feature_columns"]
    missing = [c for c in feature_columns if c not in features.columns]
    if missing:
        raise ValueError(f"features missing columns required by trained model: {missing}")

    out = features[["game_date", "game_pk", "batter_id"]].copy()
    out["tb_pred"] = _predict(art["model"], art.get("model_type", "unknown"), features[feature_columns])
    out = out.sort_values(["game_date", "game_pk", "batter_id"], kind="mergesort").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info("wrote_hitter_tb_preds season=%s rows=%d path=%s", season, len(out), output_path)
    return out
