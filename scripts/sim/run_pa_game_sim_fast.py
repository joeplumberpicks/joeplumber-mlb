#!/usr/bin/env python3
from __future__ import annotations

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast PA-based game simulation.")
    parser.add_argument("--away-lineup", type=str, required=True)
    parser.add_argument("--home-lineup", type=str, required=True)
    parser.add_argument("--away-pitcher", type=str, required=True)
    parser.add_argument("--home-pitcher", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--n-sims", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--out-json", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    return parser.parse_args()


def _read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type: {p}")


def _load_pitcher_row(path: str) -> pd.Series:
    df = _read_table(path)
    if len(df) != 1:
        raise ValueError(f"Pitcher input must contain exactly 1 row: {path}")
    return df.iloc[0]


def main() -> None:
    args = parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    outputs_dir = Path(dirs["outputs_dir"]) / "pa_sims"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    lineup_away = _read_table(args.away_lineup)
    lineup_home = _read_table(args.home_lineup)
    pitcher_away = _load_pitcher_row(args.away_pitcher)
    pitcher_home = _load_pitcher_row(args.home_pitcher)

    artifact = load_pa_outcome_artifact(args.model_path)
    rng = np.random.default_rng(args.seed)

    results = []
    started = time.time()

    print("========================================")
    print("JOE PLUMBER FAST PA GAME SIM")
    print("========================================")
    print(f"n_sims={args.n_sims}")
    print(f"model_path={args.model_path}")
    print("")

    for i in range(args.n_sims):
        res = simulate_single_game_fast(
            artifact=artifact,
            lineup_away=lineup_away,
            lineup_home=lineup_home,
            pitcher_away=pitcher_away,
            pitcher_home=pitcher_home,
            rng=rng,
            max_innings=9,
            extra_innings_cap=12,
            return_pa_log=False,
        )
        results.append(res)

        if (i + 1) % max(1, min(100, args.n_sims // 10 if args.n_sims >= 10 else 1)) == 0:
            elapsed = time.time() - started
            print(f"completed_sims={i+1}/{args.n_sims} elapsed_sec={elapsed:.2f}")

    sim_df = pd.DataFrame(results)
    summary = summarize_sim_results(results)
    summary["elapsed_seconds"] = float(time.time() - started)

    out_csv = Path(args.out_csv) if args.out_csv else outputs_dir / "pa_game_sim_results_fast.csv"
    out_json = Path(args.out_json) if args.out_json else outputs_dir / "pa_game_sim_summary_fast.json"

    sim_df.to_csv(out_csv, index=False)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("")
    print(json.dumps(summary, indent=2))
    print("")
    print(f"results_out={out_csv}")
    print(f"summary_out={out_json}")


if __name__ == "__main__":
    main()
