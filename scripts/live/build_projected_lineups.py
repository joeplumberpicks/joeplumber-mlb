from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs




def _write_empty_projected_lineups(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame(
        columns=[
            "game_pk",
            "game_date",
            "batter_team",
            "canonical_team",
            "player_name",
            "batter_id",
            "lineup_slot",
            "position",
            "bats",
            "lineup_status",
            "lineup_source",
            "source_timestamp",
        ]
    )
    empty.to_parquet(out_path, index=False)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical projected lineups artifact for same-day slate.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--source", choices=["rotogrinders", "fangraphs"], default="rotogrinders")
    p.add_argument("--html-path", type=Path, default=None, help="Optional local HTML snapshot.")
    p.add_argument("--url", default=None, help="Optional override URL for rotogrinders source.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    if args.source == "rotogrinders":
        cmd = [
            sys.executable,
            "scripts/live/build_projected_lineups_rotogrinders.py",
            "--season",
            str(args.season),
            "--date",
            args.date,
            "--config",
            str(config_path),
        ]
        if args.html_path is not None:
            cmd.extend(["--html-path", str(args.html_path)])
        if args.url is not None:
            cmd.extend(["--url", str(args.url)])
    else:
        if args.html_path is None:
            raise ValueError("--html-path is required when --source=fangraphs")
        cmd = [
            sys.executable,
            "scripts/live/build_projected_lineups_fangraphs.py",
            "--season",
            str(args.season),
            "--date",
            args.date,
            "--config",
            str(config_path),
            "--html-path",
            str(args.html_path),
        ]

    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)

    out_path = dirs["processed_dir"] / "live" / f"projected_lineups_{args.season}_{args.date}.parquet"
    if proc.returncode != 0:
        logging.warning(
            "Projected lineup build failed (exit=%s). Writing empty projected lineups parquet and continuing: %s",
            proc.returncode,
            " ".join(cmd),
        )
        _write_empty_projected_lineups(out_path)

    print(f"canonical_projected_lineups_out={out_path}")


if __name__ == "__main__":
    main()
