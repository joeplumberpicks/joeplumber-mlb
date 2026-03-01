from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import ensure_drive_mounted, resolve_data_dirs
from src.utils.io import read_csv, write_csv
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily moneyline picks from fair-odds output.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--season", type=int, required=True)
    return parser.parse_args()


def _confidence_tier(p: pd.Series) -> pd.Series:
    return np.where(
        p >= 0.66,
        "A",
        np.where(p >= 0.60, "B", np.where(p >= 0.55, "C", "D")),
    )


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)

    ensure_drive_mounted()
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "export_moneyline_picks.log")
    log_header("scripts/slate/export_moneyline_picks.py", repo_root, config_path, dirs)

    fair_path = dirs["outputs_dir"] / f"moneyline_fair_odds_{args.season}.csv"
    if not fair_path.exists():
        raise FileNotFoundError(f"Missing fair odds input: {fair_path.resolve()}")

    df = read_csv(fair_path)
    required = ["game_date", "home_team", "away_team", "p_home_win", "fair_ml_home", "fair_ml_away"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in fair-odds input: {missing}")

    out = df.copy()
    out["p_home_win"] = pd.to_numeric(out["p_home_win"], errors="coerce")
    out["fair_ml_home"] = pd.to_numeric(out["fair_ml_home"], errors="coerce")
    out["fair_ml_away"] = pd.to_numeric(out["fair_ml_away"], errors="coerce")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.dropna(subset=["game_date", "p_home_win", "fair_ml_home", "fair_ml_away"]).copy()

    out["p_away_win"] = 1.0 - out["p_home_win"]
    is_home_pick = out["p_home_win"] >= 0.5
    out["pick_team"] = np.where(is_home_pick, out["home_team"], out["away_team"])
    out["pick_prob"] = np.where(is_home_pick, out["p_home_win"], out["p_away_win"])
    out["fair_ml_pick"] = np.where(is_home_pick, out["fair_ml_home"], out["fair_ml_away"])
    out["fair_ml_opponent"] = np.where(is_home_pick, out["fair_ml_away"], out["fair_ml_home"])
    out["win_margin"] = out["pick_prob"] - 0.5

    out["confidence_tier"] = _confidence_tier(out["pick_prob"])

    out = out.sort_values(["game_date", "pick_prob"], ascending=[True, False]).copy()
    out["rank"] = out.groupby("game_date")["pick_prob"].rank(method="first", ascending=False).astype(int)

    out["game_date"] = out["game_date"].dt.strftime("%Y-%m-%d")

    final_cols = [
        "game_date",
        "home_team",
        "away_team",
        "p_home_win",
        "p_away_win",
        "pick_team",
        "pick_prob",
        "win_margin",
        "confidence_tier",
        "fair_ml_pick",
        "fair_ml_opponent",
        "rank",
    ]
    out = out[final_cols]

    out_path = dirs["outputs_dir"] / f"moneyline_picks_{args.season}.csv"
    top5_path = dirs["outputs_dir"] / f"moneyline_card_top5_{args.season}.csv"
    ab_path = dirs["outputs_dir"] / f"moneyline_card_ab_{args.season}.csv"

    top5 = out[out["rank"] <= 5].copy()
    ab_only = out[out["confidence_tier"].isin(["A", "B"])].copy()

    write_csv(out, out_path)
    write_csv(top5, top5_path)
    write_csv(ab_only, ab_path)

    tier_counts = out["confidence_tier"].value_counts(dropna=False).to_dict()
    logging.info("Wrote moneyline picks: %s (rows=%s)", out_path.resolve(), len(out))
    logging.info("Wrote moneyline top5 card: %s (rows=%s)", top5_path.resolve(), len(top5))
    logging.info("Wrote moneyline A/B card: %s (rows=%s)", ab_path.resolve(), len(ab_only))
    logging.info("confidence_tier counts: %s", tier_counts)


if __name__ == "__main__":
    main()
