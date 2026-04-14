#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.pa_outcome_model import load_pa_outcome_artifact, predict_pa_outcome_proba
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PA probability sanity on sample rows.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--mart-path", type=str, default=None)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--n-rows", type=int, default=10)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    mart_path = Path(args.mart_path) if args.mart_path else Path(dirs["marts_dir"]) / "pa_outcome" / "pa_outcome_features.parquet"
    df = pd.read_parquet(mart_path)
    df["season"] = pd.to_numeric(df["season"], errors="coerce")

    artifact = load_pa_outcome_artifact(args.model_path)

    sample = df[df["season"] == args.season].copy()
    if sample.empty:
        raise ValueError(f"No rows found for season={args.season}")

    keep_preview = [c for c in ["game_date", "batter_id", "pitcher_id", "inning", "outs_before_pa", "base_state_before", "pa_outcome_target"] if c in sample.columns]
    sample = sample.sample(min(args.n_rows, len(sample)), random_state=42).reset_index(drop=True)

    proba = predict_pa_outcome_proba(artifact, sample)
    out = pd.concat([sample[keep_preview], proba], axis=1)

    print("========================================")
    print("JOE PLUMBER PA PROBABILITY SANITY CHECK")
    print("========================================")
    print(out.to_string(index=False))

    print("")
    print("=== MEAN PROBS ===")
    means = {
        c: float(proba[f"p_{c}"].mean())
        for c in ["out", "walk_hbp", "single", "double", "triple", "home_run"]
    }
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main()
