from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header
from src.utils.train_scaffold import run_training_scaffold

ENGINE = "hitter_props"
MART_FILE = "hitter_props_features.parquet"
TARGET_COL = "target_hitter_prop"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hitter props scaffold model.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / f"train_{ENGINE}.log")
    log_header("scripts/train/train_hitter_props.py", repo_root, config_path, dirs)
    run_training_scaffold(ENGINE, MART_FILE, TARGET_COL, dirs)

if __name__ == "__main__":
    main()
