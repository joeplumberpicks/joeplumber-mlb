from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.paths import target_output_path
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build moneyline targets by season")
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

    configure_logging(dirs["logs_dir"] / "build_targets_moneyline.log")
    log_header("scripts/build_targets_moneyline.py", repo_root, config_path, dirs)

    games_path = dirs["processed_dir"] / "by_season" / f"games_{args.season}.parquet"
    games = read_parquet(games_path)

    home_col = _pick(games, ["home_score", "post_home_score", "final_home_score"])
    away_col = _pick(games, ["away_score", "post_away_score", "final_away_score"])
    if home_col is None or away_col is None:
        raise ValueError(f"Cannot find score columns in games_{args.season}. Available={sorted(games.columns)}")

    out = games[[c for c in ["game_pk", "game_date", "home_team", "away_team", home_col, away_col] if c in games.columns]].copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    hs = pd.to_numeric(out[home_col], errors="coerce")
    aw = pd.to_numeric(out[away_col], errors="coerce")
    out["target_home_win"] = pd.NA
    out.loc[hs > aw, "target_home_win"] = 1
    out.loc[hs < aw, "target_home_win"] = 0
    out["target_home_win"] = pd.to_numeric(out["target_home_win"], errors="coerce").astype("Int64")

    tie_count = int(((hs == aw) & hs.notna() & aw.notna()).sum())
    out = out[["game_pk", "game_date", "home_team", "away_team", "target_home_win"]].drop_duplicates(subset=["game_pk"])

    out_path = target_output_path(dirs["processed_dir"], "moneyline", args.season)
    if out_path.exists() and not args.force:
        logging.info("exists and force=False: %s", out_path.resolve())
    else:
        write_parquet(out, out_path)

    logging.info(
        "targets_moneyline rows=%s unique_games=%s date_min=%s date_max=%s tie_count=%s null_rate=%.4f pos_rate=%.4f path=%s",
        len(out),
        int(out["game_pk"].nunique()) if "game_pk" in out.columns else 0,
        pd.to_datetime(out.get("game_date"), errors="coerce").min() if len(out) else pd.NaT,
        pd.to_datetime(out.get("game_date"), errors="coerce").max() if len(out) else pd.NaT,
        tie_count,
        float(out["target_home_win"].isna().mean()) if len(out) else 0.0,
        float(pd.to_numeric(out["target_home_win"], errors="coerce").fillna(0).mean()) if len(out) else 0.0,
        out_path.resolve(),
    )
    print(f"targets_moneyline -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
