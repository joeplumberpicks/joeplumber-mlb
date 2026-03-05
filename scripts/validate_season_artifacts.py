from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.paths import target_input_candidates
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


def _nonnull(df: pd.DataFrame, col: str) -> float:
    return float(df[col].notna().mean()) if col in df.columns and len(df) else 0.0


def _find_target(processed_dir: Path, market: str, season: int) -> Path | None:
    for p in target_input_candidates(processed_dir, market, season):
        if p.exists():
            return p
    return None


def _validate_binary(df: pd.DataFrame, col: str, allow_warn_up_to: float = 0.05) -> None:
    if col not in df.columns:
        raise RuntimeError(f"Missing binary target column: {col}")
    null_rate = float(df[col].isna().mean()) if len(df) else 0.0
    print(f"{col} null_rate={null_rate:.4f}")
    if null_rate > allow_warn_up_to:
        raise RuntimeError(f"Validation failed: {col} null_rate {null_rate:.4f} > {allow_warn_up_to:.2f}")


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
    ]
    require_files(required, f"season_validation_{args.season}")

    markets = ["moneyline", "nrfi", "hr_batter", "hitter_props", "pitcher_props"]
    targets: dict[str, pd.DataFrame] = {}
    for m in markets:
        p = _find_target(dirs["processed_dir"], m, args.season)
        if p is None:
            raise RuntimeError(f"Missing target file for market={m} season={args.season}")
        targets[m] = read_parquet(p)
        print(f"target[{m}] rows={len(targets[m]):,} path={p}")

    marts_dir = dirs["marts_dir"] / "by_season"
    mart_paths = {
        "moneyline": marts_dir / f"moneyline_features_{args.season}.parquet",
        "nrfi": marts_dir / f"nrfi_features_{args.season}.parquet",
        "hitter_props": marts_dir / f"hitter_props_features_{args.season}.parquet",
        "pitcher_props": marts_dir / f"pitcher_props_features_{args.season}.parquet",
        "hr_batter": marts_dir / f"hr_batter_features_{args.season}.parquet",
    }
    require_files(list(mart_paths.values()), f"marts_validation_{args.season}")

    money = read_parquet(mart_paths["moneyline"])
    nrfi = read_parquet(mart_paths["nrfi"])
    hrb = read_parquet(mart_paths["hr_batter"])
    hit = read_parquet(mart_paths["hitter_props"])
    pit = read_parquet(mart_paths["pitcher_props"])

    _validate_binary(money, "target_home_win")
    _validate_binary(nrfi, "target_nrfi")
    _validate_binary(nrfi, "target_yrfi")
    _validate_binary(hrb, "target_hr")
    for c in ["target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]:
        _validate_binary(hit, c)
    for c in ["target_k", "target_outs", "target_bb"]:
        if c not in pit.columns:
            raise RuntimeError(f"Missing pitcher target column: {c}")

    print("moneyline key alignment", money["game_pk"].nunique(), targets["moneyline"]["game_pk"].nunique())
    if {"game_pk", "batter_id"}.issubset(hrb.columns):
        print("hr_batter key alignment", hrb[["game_pk", "batter_id"]].drop_duplicates().shape[0])


if __name__ == "__main__":
    main()
