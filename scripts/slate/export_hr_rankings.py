from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_csv, write_parquet
from src.utils.logging import configure_logging, log_header


def _latest_model(model_dir: Path) -> Path:
    files = sorted(model_dir.glob("*.joblib"))
    if not files:
        raise FileNotFoundError(f"No hr_batter_ranker model found in {model_dir.resolve()}")
    return files[-1]


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

    mart_path = dirs["marts_dir"] / "hr_batter_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing hr batter mart: {mart_path.resolve()}")

    mart = read_parquet(mart_path)
    if mart.empty:
        raise RuntimeError("hr_batter_features.parquet is empty; cannot export rankings")

    mart["game_date"] = pd.to_datetime(mart.get("game_date"), errors="coerce")
    if args.date:
        target_date = pd.to_datetime(args.date)
    else:
        target_date = mart["game_date"].max()
    day_df = mart[mart["game_date"] == target_date].copy()
    if day_df.empty:
        raise RuntimeError(f"No rows found in hr_batter_features for date {target_date.date()}")

    drop_cols = {"target_hr", "game_pk", "game_date"}
    feature_cols = [c for c in day_df.columns if c not in drop_cols]
    X = day_df[feature_cols].copy()
    X = X.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], 0)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0).astype("float32")

    model_path = _latest_model(dirs["models_dir"] / "hr_batter_ranker")
    model = load(model_path)
    if hasattr(model, "predict_proba"):
        day_df["pHR"] = model.predict_proba(X)[:, 1]
    else:
        day_df["pHR"] = model.predict(X)

    day_df = day_df.sort_values("pHR", ascending=False)
    date_tag = target_date.strftime("%Y-%m-%d")
    out_dir = dirs["outputs_dir"] / "hr_rankings"
    full_parquet = out_dir / f"hr_rankings_{date_tag}.parquet"
    top11_csv = out_dir / f"hr_rankings_top11_{date_tag}.csv"
    top25_csv = out_dir / f"hr_rankings_top25_{date_tag}.csv"

    print(f"Writing to: {full_parquet.resolve()}")
    write_parquet(day_df, full_parquet)
    print(f"Writing to: {top11_csv.resolve()}")
    write_csv(day_df.head(11), top11_csv)
    print(f"Writing to: {top25_csv.resolve()}")
    write_csv(day_df.head(25), top25_csv)


if __name__ == "__main__":
    main()
