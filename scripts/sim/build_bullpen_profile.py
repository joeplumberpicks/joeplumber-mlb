#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bullpen profile from pitcher IDs.")
    parser.add_argument("--team", type=str, required=True)
    parser.add_argument("--game-date", type=str, required=True)
    parser.add_argument("--pitcher-ids", type=str, required=True, help="Comma-separated reliever pitcher_ids")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def _parse_ids(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    mart_path = Path(dirs["marts_dir"]) / "pa_outcome" / "pa_outcome_features.parquet"
    out_dir = Path(dirs["outputs_dir"]) / "pa_sim_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    mart = pd.read_parquet(mart_path)
    mart["game_date"] = pd.to_datetime(mart["game_date"], errors="coerce")
    mart["pitcher_id"] = pd.to_numeric(mart["pitcher_id"], errors="coerce").astype("Int64")

    as_of_date = pd.Timestamp(args.game_date)
    pitcher_ids = _parse_ids(args.pitcher_ids)

    pitcher_rows = []
    for pid in pitcher_ids:
        sub = mart[(mart["pitcher_id"] == pid) & (mart["game_date"] < as_of_date)].sort_values("game_date")
        if sub.empty:
            raise ValueError(f"No historical mart row found for reliever pitcher_id={pid}")

        row = sub.iloc[-1]
        keep_cols = [c for c in row.index if c.startswith("pit_")]
        clean = row[keep_cols].to_dict()
        clean["pitcher_id"] = pid
        pitcher_rows.append(clean)

    reliever_df = pd.DataFrame(pitcher_rows)

    numeric_cols = [c for c in reliever_df.columns if c.startswith("pit_")]
    bullpen_profile = reliever_df[numeric_cols].mean(axis=0, numeric_only=True).to_dict()
    bullpen_profile["pitcher_id"] = -999
    bullpen_profile["fielding_team"] = args.team
    bullpen_profile["bullpen_pitcher_count"] = len(pitcher_ids)

    out_path = out_dir / f"{args.team}_{args.game_date}_bullpen.csv"
    pd.DataFrame([bullpen_profile]).to_csv(out_path, index=False)

    print("========================================")
    print("JOE PLUMBER BULLPEN PROFILE BUILD")
    print("========================================")
    print(f"team={args.team}")
    print(f"reliever_count={len(pitcher_ids)}")
    print(f"out={out_path}")


if __name__ == "__main__":
    main()
