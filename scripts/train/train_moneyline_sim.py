from __future__ import annotations

import argparse
import logging
import re
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
FILTERED_MART_FILE = "moneyline_features_train_filtered.parquet"


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


def _identifier_like_columns(df_cols: list[str]) -> list[str]:
    exact_drop = {
        "game_pk",
        "game_date",
        "season",
        "venue_id",
        "park_id",
        "home_sp_id",
        "away_sp_id",
        "home_team_id",
        "away_team_id",
        "home_team",
        "away_team",
        "canonical_park_key",
    }

    dropped: list[str] = []
    for c in df_cols:
        lc = c.lower()
        if c in exact_drop:
            dropped.append(c)
            continue
        if "game_pk" in lc:
            dropped.append(c)
            continue
        if lc.endswith("_id"):
            dropped.append(c)
            continue
        if "batter_id" in lc or "pitcher_id" in lc:
            dropped.append(c)
            continue
        if re.match(r".*game_pk_roll\d+$", lc):
            dropped.append(c)
            continue
        if re.match(r".*_id_roll\d+$", lc):
            dropped.append(c)
            continue

    # keep order, unique
    return list(dict.fromkeys(dropped))

def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / f"train_{ENGINE}.log")
    log_header("scripts/train/train_moneyline_sim.py", repo_root, config_path, dirs)
    _preflight_moneyline_target(dirs)

    mart_path = dirs["marts_dir"] / MART_FILE
    mart_df = read_parquet(mart_path)
    drop_cols = _identifier_like_columns(list(mart_df.columns))
    logging.info(
        "moneyline identifier filter dropped_cols=%s first30=%s",
        len(drop_cols),
        drop_cols[:30],
    )

    filtered = mart_df.drop(columns=drop_cols, errors="ignore")
    filtered_path = dirs["marts_dir"] / FILTERED_MART_FILE
    filtered.to_parquet(filtered_path, index=False)

    run_training_scaffold(ENGINE, FILTERED_MART_FILE, TARGET_COL, dirs)

if __name__ == "__main__":
    main()
