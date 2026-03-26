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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate No-HR artifacts")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _validate_diag_sources(df: pd.DataFrame, name: str) -> None:
    valid = {"requested_season", "previous_season_fallback", "2026", "2025_fallback"}
    for col in ["batter_feature_source", "pitcher_feature_source"]:
        if col in df.columns:
            bad = sorted(set(df[col].dropna().astype(str)) - valid)
            if bad:
                raise ValueError(f"{name} {col} invalid values: {bad}")
    if "fallback_used" in df.columns:
        bad = sorted(set(df["fallback_used"].dropna().astype(str)) - {"True", "False", "0", "1"})
        if bad:
            raise ValueError(f"{name} fallback_used invalid values: {bad}")


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    t_path = dirs["processed_dir"] / "targets" / "game" / f"targets_no_hr_game_{args.season}.parquet"
    m_path = dirs["processed_dir"] / "marts" / "no_hr" / f"no_hr_game_features_{args.season}.parquet"
    p_path = dirs["outputs_dir"] / "no_hr" / f"no_hr_predictions_{args.season}.parquet"

    for p in [m_path, p_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p.resolve()}")

    live_mode = args.season == 2026 and not t_path.exists()
    if not live_mode and not t_path.exists():
        raise FileNotFoundError(f"Missing artifact: {t_path.resolve()}")

    t = read_parquet(t_path) if t_path.exists() else pd.DataFrame()
    m = read_parquet(m_path)
    pr = read_parquet(p_path)

    if t_path.exists():
        _require_cols(t, ["game_pk", "total_hr", "no_hr_game"], "targets")
    _require_cols(m, ["game_pk", "no_hr_game"], "mart")
    _require_cols(pr, ["game_pk", "p_no_hr", "tier"], "predictions")

    if t_path.exists():
        t_null = float(t["no_hr_game"].isna().mean()) if len(t) else 0.0
        if t_null > 0.01:
            raise ValueError(f"Target null rate too high targets={t_null:.4f}")

    if not live_mode:
        m_null = float(m["no_hr_game"].isna().mean()) if len(m) else 0.0
        if m_null > 0.01:
            raise ValueError(f"Mart target null rate too high mart={m_null:.4f}")

    if len(pr):
        if bool((pr["p_no_hr"] < 0).any()) or bool((pr["p_no_hr"] > 1).any()):
            raise ValueError("p_no_hr values outside [0,1]")

    _validate_diag_sources(m, "mart")
    _validate_diag_sources(pr, "predictions")

    if live_mode:
        print(
            "validate_no_hr_artifacts LIVE_MODE_OK "
            f"season={args.season} targets_present=False mart_rows={len(m)} pred_rows={len(pr)}"
        )
    else:
        print(
            "validate_no_hr_artifacts OK "
            f"season={args.season} target_rows={len(t)} mart_rows={len(m)} pred_rows={len(pr)}"
        )


if __name__ == "__main__":
    main()
