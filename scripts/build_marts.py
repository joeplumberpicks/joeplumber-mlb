from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.marts.build_hr_batter_features import build_hr_batter_features
from src.marts.build_marts import build_marts
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature marts from model spine and rolling tables.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_marts.log")
    log_header("scripts/build_marts.py", repo_root, config_path, dirs)
    print(f"Args: season={args.season}, start={args.start}, end={args.end}, force={args.force}")

    build_marts(dirs, season=args.season)
    if args.season is not None:
        build_hr_batter_features(dirs=dirs, season=args.season, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
