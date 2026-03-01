from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.hitter_props import build_batter_boxscore
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build batter boxscore table from StatsAPI game boxscores.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_batter_boxscore.log")
    log_header("scripts/build_batter_boxscore.py", repo_root, config_path, dirs)
    out_path = build_batter_boxscore(dirs=dirs, season=args.season, force=args.force)
    print(f"batter_boxscore -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
