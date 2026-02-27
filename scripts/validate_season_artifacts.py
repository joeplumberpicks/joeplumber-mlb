from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.checks import require_files
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate season artifacts and quick quality checks.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _nonnull_pct(df, col: str) -> float:
    if col not in df.columns or df.empty:
        return 0.0
    return float(df[col].notna().mean() * 100.0)


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "validate_season_artifacts.log")
    log_header("scripts/validate_season_artifacts.py", repo_root, config_path, dirs)

    required = [
        dirs["processed_dir"] / "by_season" / f"games_{args.season}.parquet",
        dirs["processed_dir"] / "by_season" / f"pa_{args.season}.parquet",
        dirs["processed_dir"] / "by_season" / f"batter_game_{args.season}.parquet",
        dirs["processed_dir"] / "by_season" / f"pitcher_game_{args.season}.parquet",
        dirs["processed_dir"] / "batter_game_rolling.parquet",
        dirs["processed_dir"] / "pitcher_game_rolling.parquet",
        dirs["processed_dir"] / "model_spine_game.parquet",
    ]
    require_files(required, f"season_validation_{args.season}")

    games = read_parquet(required[0])
    batter_game = read_parquet(required[2])
    pitcher_game = read_parquet(required[3])
    batter_roll = read_parquet(required[4])
    pitcher_roll = read_parquet(required[5])

    print(f"Row count [games_{args.season}]: {len(games):,}")
    print(f"Row count [batter_game_{args.season}]: {len(batter_game):,}")
    print(f"Row count [pitcher_game_{args.season}]: {len(pitcher_game):,}")
    print(f"Row count [batter_game_rolling]: {len(batter_roll):,}")
    print(f"Row count [pitcher_game_rolling]: {len(pitcher_roll):,}")
    print(f"park_id non-null %: {_nonnull_pct(games, 'park_id'):.2f}%")
    print(f"game_date non-null %: {_nonnull_pct(games, 'game_date'):.2f}%")

    if batter_game.empty or pitcher_game.empty or batter_roll.empty or pitcher_roll.empty:
        raise RuntimeError("Validation failed: one or more critical feature tables are empty.")


if __name__ == "__main__":
    main()
