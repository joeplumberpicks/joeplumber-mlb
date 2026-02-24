from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.spine.build_spine import build_model_spine, build_spine_for_season
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed season tables and model spine scaffold.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()


def _pct_filled(df, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    return float(df[col].notna().mean() * 100.0)


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_spine.log")
    log_header("scripts/build_spine.py", repo_root, config_path, dirs)
    print(f"Args: season={args.season}, start={args.start}, end={args.end}, force={args.force}, allow_partial={args.allow_partial}")

    outputs = build_spine_for_season(args.season, dirs, force=args.force)
    seasons = sorted(set(config.get("seasons_default", []) + [args.season]))
    model_spine_path = build_model_spine(dirs, seasons)

    parks_df = read_parquet(outputs["parks"]) if outputs["parks"].exists() else None
    parks_rows = len(parks_df) if parks_df is not None else 0
    print(f"Row count [parks_{args.season}]: {parks_rows:,}")

    spine_df = read_parquet(model_spine_path)
    print(f"% games with park_id filled: {_pct_filled(spine_df, 'park_id'):.2f}%")
    print(f"% games with venue_id filled: {_pct_filled(spine_df, 'venue_id'):.2f}%")


if __name__ == "__main__":
    main()
