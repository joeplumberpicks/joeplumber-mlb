from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.rolling import build_rolling_features
from src.utils.checks import require_files
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rolling feature tables with shift(1) leakage prevention.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_features_rolling.log")
    log_header("scripts/build_features_rolling.py", repo_root, config_path, dirs)

    spine_path = dirs["processed_dir"] / "model_spine_game.parquet"
    require_files([spine_path], "rolling_features")

    windows = config.get("rolling_windows", [3, 7, 15, 30])
    shift_n = int(config.get("leakage_shift", 1))
    build_rolling_features(dirs, windows=windows, shift_n=shift_n)


if __name__ == "__main__":
    main()
