from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import ensure_drive_mounted, resolve_data_dirs
from src.utils.io import read_parquet, write_csv
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export season moneyline fair odds from baseline model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    return parser.parse_args()


def prob_to_ml(p: float) -> int:
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)

    ensure_drive_mounted()
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "export_moneyline_fair_odds.log")
    log_header("scripts/slate/export_moneyline_fair_odds.py", repo_root, config_path, dirs)

    mart_path = dirs["marts_dir"] / "moneyline_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart file: {mart_path.resolve()}")

    df = read_parquet(mart_path)
    if "season" in df.columns:
        season_vals = pd.to_numeric(df["season"], errors="coerce")
        df = df[season_vals == args.season].copy()
    if df.empty:
        raise ValueError(f"No rows found for season={args.season} in {mart_path.resolve()}")

    y = pd.to_numeric(df.get("target_home_win"), errors="coerce")

    output_cols = [c for c in ["game_date", "home_team", "away_team"] if c in df.columns]
    out = df[output_cols].copy() if output_cols else pd.DataFrame(index=df.index)

    exclude = {"target_home_win", "game_pk", "park_id", "season"}
    feature_cols = [
        c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns available in moneyline_features.parquet")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    train_mask = y.notna()
    if not bool(train_mask.any()):
        raise ValueError("No non-null target_home_win rows available to train baseline classifier")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=10000)),
    ])
    model.fit(X.loc[train_mask], y.loc[train_mask].astype(int))

    p_home = model.predict_proba(X)[:, 1]
    out["p_home_win"] = p_home
    out["fair_ml_home"] = out["p_home_win"].map(prob_to_ml)
    out["fair_ml_away"] = (1.0 - out["p_home_win"]).map(prob_to_ml)

    final_cols = ["game_date", "home_team", "away_team", "p_home_win", "fair_ml_home", "fair_ml_away"]
    for col in final_cols:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[final_cols]

    out_path = dirs["outputs_dir"] / f"moneyline_fair_odds_{args.season}.csv"
    write_csv(out, out_path)
    logging.info("Wrote fair odds: %s (rows=%s)", out_path.resolve(), len(out))


if __name__ == "__main__":
    main()
