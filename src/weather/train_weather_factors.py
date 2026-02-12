from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from src.models.calibration import NumpyPlattScaler
from src.models.numpy_logreg import NumpyLogisticRegression
from src.models.numpy_ridge import NumpyRidgeRegression

WX_FEATURES = [
    "temp_f",
    "humidity",
    "dewpoint_f",
    "pressure_mb",
    "wind_speed_mph",
    "wind_dir_deg",
    "precip_in",
    "air_density_proxy",
    "wind_out_to_cf_mph",
    "roof_closed_flag",
]


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    if n < 10:
        raise ValueError("Need at least 10 rows for weather factor training")
    split = min(max(int(np.floor(n * 0.8)), 1), n - 1)
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def _fit_regressor(X: pd.DataFrame, y: np.ndarray) -> tuple[object, str]:
    try:
        from sklearn.linear_model import Ridge
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        mdl = Pipeline([("imp", SimpleImputer(strategy="median")), ("ridge", Ridge(alpha=1.0, random_state=42))])
        mdl.fit(X, y)
        return mdl, "sklearn_ridge"
    except Exception:
        mdl = NumpyRidgeRegression(alpha=1.0).fit(X.to_numpy(dtype=float), y)
        return mdl, "numpy_ridge"


def _fit_classifier(X: pd.DataFrame, y: np.ndarray) -> tuple[object, str]:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        mdl = Pipeline([("imp", SimpleImputer(strategy="median")), ("lr", LogisticRegression(max_iter=1000))])
        mdl.fit(X, y)
        return mdl, "sklearn_logistic"
    except Exception:
        mdl = NumpyLogisticRegression(lr=0.05, epochs=2000, reg_lambda=1.0).fit(X.to_numpy(dtype=float), y)
        return mdl, "numpy_logreg"


def _predict_reg(m: object, mtype: str, X: pd.DataFrame) -> np.ndarray:
    if mtype == "numpy_ridge":
        return m.predict(X.to_numpy(dtype=float))
    return np.asarray(m.predict(X), dtype=float)


def _predict_clf(m: object, mtype: str, X: pd.DataFrame) -> np.ndarray:
    if mtype == "numpy_logreg":
        return np.asarray(m.predict_proba(X.to_numpy(dtype=float))[:, 1], dtype=float)
    return np.asarray(m.predict_proba(X)[:, 1], dtype=float)


def derive_hr_count_by_game(season: int, hitter_targets_path: Path, events_path: Path) -> pd.DataFrame:
    if hitter_targets_path.exists():
        h = pd.read_parquet(hitter_targets_path, engine="pyarrow")
        req = {"game_pk", "hr"}
        if req.issubset(h.columns):
            h["game_pk"] = pd.to_numeric(h["game_pk"], errors="coerce").astype("Int64")
            h["hr"] = pd.to_numeric(h["hr"], errors="coerce")
            g = h.groupby("game_pk", observed=False)["hr"].sum(min_count=1).reset_index(name="hr_count_game")
            return g

    if events_path.exists():
        e = pd.read_parquet(events_path, engine="pyarrow")
        if {"game_pk", "events"}.issubset(e.columns):
            e["game_pk"] = pd.to_numeric(e["game_pk"], errors="coerce").astype("Int64")
            e["hr_count_game"] = (e["events"].astype("string").str.lower() == "home_run").astype(int)
            return e.groupby("game_pk", observed=False)["hr_count_game"].sum().reset_index()

    raise FileNotFoundError(
        "Unable to derive hr_count_game: expected targets_hitter_game parquet with hr column or events_pa.parquet with events column"
    )


def run_smoke_test(logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger(__name__)
    rng = np.random.default_rng(9)
    n = 80
    df = pd.DataFrame(
        {
            "game_date": pd.date_range("2024-03-28", periods=n, freq="D"),
            "game_pk": np.arange(1, n + 1),
            "temp_f": rng.normal(70, 10, n),
            "humidity": rng.uniform(25, 85, n),
            "pressure_mb": rng.normal(1010, 8, n),
            "wind_speed_mph": rng.uniform(0, 20, n),
            "wind_dir_deg": rng.uniform(0, 360, n),
            "precip_in": rng.uniform(0, 0.2, n),
            "air_density_proxy": rng.normal(3.6, 0.2, n),
            "wind_out_to_cf_mph": rng.normal(0, 8, n),
            "roof_closed_flag": rng.integers(0, 2, n),
        }
    )
    df["hr_count_game"] = np.clip(1.5 + 0.03 * (df["temp_f"] - 70) + 0.02 * df["wind_out_to_cf_mph"] + rng.normal(0, 0.4, n), 0, None)
    df["total_runs"] = np.clip(8 + 0.06 * (df["temp_f"] - 70) + 0.04 * df["wind_out_to_cf_mph"] + rng.normal(0, 1.0, n), 1, None)
    p = 1 / (1 + np.exp(-(0.03 * (df["temp_f"] - 70) + 0.06 * df["wind_out_to_cf_mph"] / 10)))
    df["yrfi"] = (rng.uniform(0, 1, n) < p).astype(int)

    out = train_weather_factors(
        season=2024,
        weather_df=df[["game_date", "game_pk", *[c for c in WX_FEATURES if c in df.columns]]],
        targets_game_df=df[["game_date", "game_pk", "total_runs", "yrfi"]],
        hr_game_df=df[["game_pk", "hr_count_game"]],
        model_dir=Path("data/models/weather_factors_smoke"),
        output_path=Path("data/processed/weather_factors_game_2024_smoke.parquet"),
        force=True,
        logger=log,
    )
    assert out["wx_hr_mult"].between(0.5, 1.5).all()
    assert out["wx_runs_delta"].between(-3.0, 3.0).all()
    assert out["wx_yrfi_delta"].between(-0.5, 0.5).all()
    log.info("weather_factors smoke test passed")


def train_weather_factors(
    season: int,
    weather_df: pd.DataFrame,
    targets_game_df: pd.DataFrame,
    hr_game_df: pd.DataFrame,
    model_dir: Path,
    output_path: Path,
    *,
    provider: str = "visualcrossing",
    force: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)
    w = weather_df.copy()
    t = targets_game_df.copy()
    h = hr_game_df.copy()

    keep_weather = [c for c in ["game_date", "game_pk", *WX_FEATURES] if c in w.columns]
    w = w[keep_weather].copy()

    w["game_date"] = pd.to_datetime(w["game_date"], errors="coerce").dt.normalize()
    t["game_date"] = pd.to_datetime(t["game_date"], errors="coerce").dt.normalize()
    for df in [w, t, h]:
        df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")

    merged = w.merge(t[["game_date", "game_pk", "total_runs", "yrfi"]], on=["game_date", "game_pk"], how="inner")
    merged = merged.merge(h[["game_pk", "hr_count_game"]], on="game_pk", how="left")
    merged = merged.loc[merged["game_date"].dt.year == int(season)].copy()
    merged = merged.dropna(subset=["total_runs", "yrfi", "hr_count_game"]).copy()

    wx_features = [c for c in WX_FEATURES if c in merged.columns]
    if not wx_features:
        raise ValueError("No weather feature columns available for weather factor training")

    train_df, val_df = _time_split(merged)
    X_train = train_df[wx_features]
    X_val = val_df[wx_features]

    hr_model, hr_type = _fit_regressor(X_train, train_df["hr_count_game"].to_numpy(dtype=float))
    totals_model, totals_type = _fit_regressor(X_train, train_df["total_runs"].to_numpy(dtype=float))
    yrfi_model, yrfi_type = _fit_classifier(X_train, train_df["yrfi"].to_numpy(dtype=float))

    hr_val = _predict_reg(hr_model, hr_type, X_val)
    totals_val = _predict_reg(totals_model, totals_type, X_val)
    yrfi_raw = _predict_clf(yrfi_model, yrfi_type, X_val)

    y_val = val_df["yrfi"].to_numpy(dtype=float)
    platt = NumpyPlattScaler(lr=0.1, epochs=1500, reg_lambda=1.0).fit(yrfi_raw, y_val)
    yrfi_cal = platt.predict(yrfi_raw)

    base_hr = float(train_df["hr_count_game"].mean())
    base_runs = float(train_df["total_runs"].mean())
    base_yrfi = float(np.clip(train_df["yrfi"].mean(), 1e-6, 1 - 1e-6))

    weather_all = w.loc[w["game_date"].dt.year == int(season)].copy()
    X_all = weather_all[wx_features]
    hr_all = _predict_reg(hr_model, hr_type, X_all)
    totals_all = _predict_reg(totals_model, totals_type, X_all)
    yrfi_all_raw = _predict_clf(yrfi_model, yrfi_type, X_all)
    yrfi_all = platt.predict(yrfi_all_raw)

    wx = weather_all[["game_date", "game_pk"]].copy()
    wx["wx_hr_mult"] = np.clip(hr_all / max(base_hr, 1e-6), 0.5, 1.5)
    wx["wx_runs_delta"] = np.clip(totals_all - base_runs, -3.0, 3.0)
    wx["wx_yrfi_delta"] = np.clip(yrfi_all - base_yrfi, -0.5, 0.5)
    wx = wx.sort_values(["game_date", "game_pk"], kind="mergesort").drop_duplicates(subset=["game_pk"], keep="first")

    model_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output exists (use --force): {output_path}")

    artifacts = {
        "hr_model.pkl": {"model": hr_model, "model_type": hr_type, "features": wx_features},
        "totals_model.pkl": {"model": totals_model, "model_type": totals_type, "features": wx_features},
        "yrfi_model.pkl": {"model": yrfi_model, "model_type": yrfi_type, "features": wx_features},
        "yrfi_calibrator.pkl": {"calibrator": platt},
    }
    for name, obj in artifacts.items():
        path = model_dir / name
        if path.exists() and not force:
            raise FileExistsError(f"Artifact exists (use --force): {path}")
        with path.open("wb") as f:
            pickle.dump(obj, f)

    meta = {
        "season": season,
        "provider": provider,
        "features": wx_features,
        "n_train": int(len(train_df)),
        "n_valid": int(len(val_df)),
        "hr_model_type": hr_type,
        "totals_model_type": totals_type,
        "yrfi_model_type": yrfi_type,
    }
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    wx.to_parquet(output_path, index=False, engine="pyarrow")

    log.info("Weather factors trained rows train=%d val=%d", len(train_df), len(val_df))
    log.info("Saved weather factors table rows=%d path=%s", len(wx), output_path)
    return wx.reset_index(drop=True)


def train_weather_factors_from_paths(
    season: int,
    weather_path: Path,
    targets_game_path: Path,
    model_dir: Path,
    output_path: Path,
    *,
    events_path: Path,
    hitter_targets_path: Path,
    provider: str = "visualcrossing",
    force: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    if not weather_path.exists():
        raise FileNotFoundError(f"Missing weather features parquet: {weather_path}")
    if not targets_game_path.exists():
        raise FileNotFoundError(f"Missing targets game parquet: {targets_game_path}")

    weather_df = pd.read_parquet(weather_path, engine="pyarrow")
    targets_df = pd.read_parquet(targets_game_path, engine="pyarrow")
    hr_df = derive_hr_count_by_game(season, hitter_targets_path=hitter_targets_path, events_path=events_path)
    return train_weather_factors(
        season=season,
        weather_df=weather_df,
        targets_game_df=targets_df,
        hr_game_df=hr_df,
        model_dir=model_dir,
        output_path=output_path,
        provider=provider,
        force=force,
        logger=logger,
    )
