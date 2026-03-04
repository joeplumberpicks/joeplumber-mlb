from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_csv
from src.utils.logging import configure_logging, log_header

ENGINES = {
    "hr_ranker": "hr_features.parquet",
    "nrfi_xgb": "nrfi_features.parquet",
    "moneyline_sim": "moneyline_features.parquet",
    "hitter_props": "hitter_props_features.parquet",
    "pitcher_props": "pitcher_props_features.parquet",
}

def latest_model(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    models = sorted(model_dir.glob("*.joblib"))
    return models[-1] if models else None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run slate scoring scaffold.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "run_slate.log")
    log_header("scripts/slate/run_slate.py", repo_root, config_path, dirs)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for engine, mart_file in ENGINES.items():
        mart_path = dirs["marts_dir"] / mart_file
        if mart_path.exists():
            mart_df = read_parquet(mart_path)
        else:
            mart_df = pd.DataFrame(columns=["game_pk", "game_date", "season"])

        out_df = mart_df.copy()
        model_path = latest_model(dirs["models_dir"] / engine)

        if model_path is None or out_df.empty:
            out_df["score"] = pd.NA
            out_df["model_missing"] = model_path is None
        else:
            model = load(model_path)
            X = out_df.select_dtypes(include=["number"])
            if X.empty:
                out_df["score"] = 0.5
            else:
                if hasattr(model, "predict_proba"):
                    out_df["score"] = model.predict_proba(X)[:, 1]
                else:
                    out_df["score"] = model.predict(X)
            out_df["model_missing"] = False

        output_path = dirs["outputs_dir"] / engine / f"slate_{engine}_{ts}.csv"
        print(f"Writing to: {output_path.resolve()}")
        print(f"Row count [{engine}_slate]: {len(out_df):,}")
        write_csv(out_df, output_path)

if __name__ == "__main__":
    main()
