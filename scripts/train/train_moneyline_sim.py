from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header
from src.utils.io import read_parquet
from src.utils.train_scaffold import run_training_scaffold

ENGINE = "moneyline_sim"
MART_FILE = "moneyline_features.parquet"
TARGET_COL = "target_home_win"


def _preflight_moneyline_target(dirs: dict[str, Path]) -> None:
    mart_path = dirs["marts_dir"] / MART_FILE
    if not mart_path.exists():
        return
    mart_df = read_parquet(mart_path)
    if TARGET_COL in mart_df.columns and mart_df[TARGET_COL].isna().all():
        msg = (
            f"{MART_FILE} has {TARGET_COL} all-null. Rebuild marts to merge processed targets: "
            f"python scripts/build_marts.py --season 2024 --force"
        )
        logging.error(msg)
        raise ValueError(msg)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train moneyline scaffold model.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / f"train_{ENGINE}.log")
    log_header("scripts/train/train_moneyline_sim.py", repo_root, config_path, dirs)
    _preflight_moneyline_target(dirs)
    run_training_scaffold(ENGINE, MART_FILE, TARGET_COL, dirs)

if __name__ == "__main__":
    main()
