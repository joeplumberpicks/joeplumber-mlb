from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_csv
from src.utils.logging import configure_logging, log_header
from src.utils.train_scaffold import _prep_X


def _latest_model(model_dir: Path) -> Path:
    files = list(model_dir.glob("hr_batter_ranker_*.joblib"))
    if not files:
        files = list(model_dir.glob("*.joblib"))
    if not files:
        raise FileNotFoundError(f"No hr_batter_ranker model found in {model_dir.resolve()}")
    return max(files, key=lambda p: (p.stem, p.stat().st_mtime))


def _build_scoring_matrix(df: pd.DataFrame, model: object) -> tuple[pd.DataFrame, list[str], list[str]]:
    feature_cols = [c for c in df.columns if c not in {"target_hr", "game_pk", "game_date"}]
    X, final_features, dropped_cols, _, _ = _prep_X(df, feature_cols)

    if X.empty:
        X = pd.DataFrame(index=df.index, data={"baseline_feature": np.zeros(len(df), dtype="float32")})
        final_features = list(X.columns)

    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features:
        for col in model_features:
            if col not in X.columns:
                X[col] = 0.0
        X = X.reindex(columns=model_features, fill_value=0.0)
        final_features = model_features

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    return X, final_features, dropped_cols


def _select_export_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "game_date",
        "game_pk",
        "season",
        "home_team",
        "away_team",
        "park_id",
        "batter_id",
        "batter_team",
        "opp_sp_id",
        "p_hr",
        "rank",
        "tier",
    ]
    for col in ["batter_id", "p_hr", "rank", "tier"]:
        if col not in df.columns:
            df[col] = pd.NA
    return [c for c in preferred if c in df.columns]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export HR batter rankings (Top11/Top25).")
    p.add_argument("--date", type=str, default=None, help="YYYY-MM-DD; defaults to latest game_date in mart")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "export_hr_rankings.log")
    log_header("scripts/slate/export_hr_rankings.py", repo_root, config_path, dirs)

    model_path = _latest_model(dirs["models_dir"] / "hr_batter_ranker")
    logging.info("Using model: %s", model_path.resolve())
    model = load(model_path)

    mart_path = dirs["marts_dir"] / "hr_batter_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing hr batter mart: {mart_path.resolve()}")

    mart = read_parquet(mart_path)
    if mart.empty:
        raise RuntimeError("hr_batter_features.parquet is empty; cannot export rankings")

    if "game_date" in mart.columns:
        mart["game_date"] = pd.to_datetime(mart["game_date"], errors="coerce")
    else:
        mart["game_date"] = pd.NaT

    if args.date:
        target_date = pd.to_datetime(args.date, errors="coerce")
        if pd.isna(target_date):
            raise ValueError(f"Invalid --date value: {args.date}")
    else:
        target_date = mart["game_date"].max()

    day_df = mart[mart["game_date"] == target_date].copy() if pd.notna(target_date) else mart.copy()
    if day_df.empty:
        raise RuntimeError(f"No rows found in hr_batter_features for date {target_date}")

    X, used_features, dropped_cols = _build_scoring_matrix(day_df, model)
    if dropped_cols:
        logging.info("Dropped non-numeric columns for scoring: %s", dropped_cols)
    logging.info("Scoring with %s rows and %s features", len(X), len(used_features))

    if hasattr(model, "predict_proba"):
        day_df["p_hr"] = model.predict_proba(X)[:, 1]
    else:
        day_df["p_hr"] = model.predict(X)

    day_df = day_df.sort_values("p_hr", ascending=False).reset_index(drop=True)
    day_df["rank"] = np.arange(1, len(day_df) + 1)
    day_df["tier"] = np.where(day_df["rank"] <= 11, "Top11", np.where(day_df["rank"] <= 25, "12-25", "Other"))

    top11 = day_df[day_df["rank"] <= 11].copy()
    top25 = day_df[day_df["rank"] <= 25].copy()

    out_dir = dirs["outputs_dir"] / "HR"
    out_dir.mkdir(parents=True, exist_ok=True)
    top11_path = out_dir / "hr_top11.csv"
    top25_path = out_dir / "hr_top25.csv"

    export_cols = _select_export_columns(day_df)
    write_csv(top11[export_cols], top11_path)
    write_csv(top25[export_cols], top25_path)

    logging.info("Wrote Top11 rows: %s -> %s", len(top11), top11_path.resolve())
    logging.info("Wrote Top25 rows: %s -> %s", len(top25), top25_path.resolve())
    if not top25.empty:
        logging.info(
            "Top25 p_hr summary min/mean/max: %.6f / %.6f / %.6f",
            float(top25["p_hr"].min()),
            float(top25["p_hr"].mean()),
            float(top25["p_hr"].max()),
        )


if __name__ == "__main__":
    main()
