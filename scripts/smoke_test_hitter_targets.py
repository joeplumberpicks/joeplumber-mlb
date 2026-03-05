from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test hitter target artifacts and rates.")
    p.add_argument("--season", type=int, default=2024)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pct(series: pd.Series) -> float:
    return float(series.notna().mean()) if len(series) else 0.0


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "smoke_test_hitter_targets.log")
    log_header("scripts/smoke_test_hitter_targets.py", repo_root, config_path, dirs)

    box_path = dirs["processed_dir"] / "by_season" / f"batter_boxscore_{args.season}.parquet"
    tgt_path = dirs["processed_dir"] / f"targets_hitter_props_{args.season}.parquet"
    mart_path = dirs["marts_dir"] / "hitter_props_features.parquet"

    box = read_parquet(box_path)
    tgt = read_parquet(tgt_path)
    mart = read_parquet(mart_path)

    print(f"rows batter_boxscore_{args.season}: {len(box):,}")
    print(f"rows targets_hitter_props_{args.season}: {len(tgt):,}")
    print(f"rows hitter_props_features: {len(mart):,}")

    for col in ["target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]:
        if col not in tgt.columns:
            print(f"MISSING {col} in targets")
            continue
        print(f"targets {col} nonnull={_pct(tgt[col]):.4f} prevalence={float(pd.to_numeric(tgt[col], errors='coerce').fillna(0).mean()):.4f}")

    for col in ["target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]:
        if col not in mart.columns:
            print(f"MISSING {col} in hitter_props_features")
            continue
        print(f"mart {col} nonnull={_pct(mart[col]):.4f} prevalence={float(pd.to_numeric(mart[col], errors='coerce').fillna(0).mean()):.4f}")


if __name__ == "__main__":
    main()
