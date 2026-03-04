from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict No-HR game probabilities")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _tier(p: float) -> str:
    if p >= 0.75:
        return "A"
    if p >= 0.70:
        return "B"
    if p >= 0.65:
        return "C"
    return "D"


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "predict_no_hr.log")
    log_header("scripts/predict_no_hr.py", repo_root, config_path, dirs)

    bundle = load(args.model_path)
    model = bundle["model"]
    features = bundle.get("features", [])

    mart_path = dirs["processed_dir"] / "marts" / "no_hr" / f"no_hr_game_features_{args.season}.parquet"
    mart = read_parquet(mart_path)

    X = mart[features].copy() if features else pd.DataFrame(index=mart.index)
    if X.empty:
        X = pd.DataFrame({"bias": [0.0] * len(mart)})
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    p_no_hr = model.predict_proba(X)[:, 1]

    out = mart[[c for c in ["game_pk", "game_date", "home_team", "away_team"] if c in mart.columns]].copy()
    out["p_no_hr"] = p_no_hr
    out["tier"] = out["p_no_hr"].apply(_tier)

    out_dir = dirs["outputs_dir"] / "no_hr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"no_hr_predictions_{args.season}.parquet"
    if out_path.exists() and not args.force:
        logging.info("exists and force=False: %s", out_path.resolve())
    else:
        write_parquet(out, out_path)

    logging.info("no_hr_predictions rows=%s p_min=%.6f p_max=%.6f path=%s", len(out), float(out["p_no_hr"].min()) if len(out) else 0.0, float(out["p_no_hr"].max()) if len(out) else 0.0, out_path.resolve())
    print(f"no_hr_predictions -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
