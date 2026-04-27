#!/usr/bin/env python3
from __future__ import annotations
from datetime import datetime

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.pa_outcome_model import load_pa_outcome_artifact
from src.sim.pa_simulator_fast import simulate_single_game_fast, summarize_sim_results
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


# ------------------------
# ARGUMENTS
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # NEW (for slate runner)
    parser.add_argument("--away-batters-file", type=str)
    parser.add_argument("--home-batters-file", type=str)
    parser.add_argument("--away-pitcher-id", type=int)
    parser.add_argument("--home-pitcher-id", type=int)

    # OLD (keep compatibility)
    parser.add_argument("--away-lineup", type=str)
    parser.add_argument("--home-lineup", type=str)
    parser.add_argument("--away-starter", type=str)
    parser.add_argument("--home-starter", type=str)

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/project.yaml")

    parser.add_argument("--out-csv", type=str, default=None)

    return parser.parse_args()


# ------------------------
# HELPERS
# ------------------------
def _read_table(path):
    p = Path(path)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    raise ValueError("Unsupported file type")


def load_starter_from_id(player_id, season, date, dirs):
    """
    Pull starter row from live pitcher features
    """
    path = Path(dirs["data_dir"]) / "processed" / "live" / f"pitcher_game_rolling_{season}_{date}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing pitcher features: {path}")

    df = pd.read_parquet(path)
    row = df[df["player_id"] == player_id]

    if row.empty:
        raise ValueError(f"No pitcher row for id {player_id}")

    return row.iloc[0]


def build_lineup_df(path):
    df = _read_table(path)
    return df


# ------------------------
# MAIN
# ------------------------
def main():
    args = parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    artifact = load_pa_outcome_artifact(args.model_path)
    rng = np.random.default_rng(args.seed)

    # ------------------------
    # LINEUPS
    # ------------------------
    if args.away_batters_file:
        lineup_away = build_lineup_df(args.away_batters_file)
        lineup_home = build_lineup_df(args.home_batters_file)
    else:
        lineup_away = _read_table(args.away_lineup)
        lineup_home = _read_table(args.home_lineup)

    # ------------------------
    # STARTERS
    # ------------------------
    if args.away_pitcher_id:
        season = config["seasons_default"][-1]
        date = datetime.today().strftime("%Y-%m-%d")

        starter_away = load_starter_from_id(args.away_pitcher_id, season, date, dirs)
        starter_home = load_starter_from_id(args.home_pitcher_id, season, date, dirs)
    else:
        starter_away = _read_table(args.away_starter).iloc[0]
        starter_home = _read_table(args.home_starter).iloc[0]

    # ------------------------
    # SIM LOOP
    # ------------------------
    results = []
    start = time.time()

    for i in range(args.n_sims):
        res = simulate_single_game_fast(
            artifact=artifact,
            lineup_away=lineup_away,
            lineup_home=lineup_home,
            starter_away=starter_away,
            starter_home=starter_home,
            bullpen_away=None,
            bullpen_home=None,
            rng=rng,
        )
        results.append(res)

    summary = summarize_sim_results(results)

    # ------------------------
    # SAVE CLEAN OUTPUT
    # ------------------------
    output = {
        "home_win_pct": summary["home_win_pct"],
        "away_win_pct": summary["away_win_pct"],
        "p_nrfi": summary["p_nrfi"],
        "p_yrfi": summary["p_yrfi"],
        "mean_total_runs": summary["mean_total_runs"],
    }

    out_path = Path(args.out_csv) if args.out_csv else Path("sim_summary.csv")
    pd.DataFrame([output]).to_csv(out_path, index=False)

    print("\n=== SIM SUMMARY ===")
    print(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
