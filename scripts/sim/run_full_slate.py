#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(cmd: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    print("\n[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slate-csv", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--game-date", required=True)
    parser.add_argument("--n-sims", type=int, default=100)
    parser.add_argument("--only-sim-ready", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    slate = pd.read_csv(args.slate_csv)

    out_root = Path("/content/drive/MyDrive/joeplumber-mlb/data/outputs/pa_sims") / args.game_date
    input_root = Path("/content/drive/MyDrive/joeplumber-mlb/data/outputs/pa_sim_inputs") / args.game_date

    out_root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(parents=True, exist_ok=True)

    summaries = []

    for _, row in slate.iterrows():
        game_pk = int(row["game_pk"])
        away = row["away_team"]
        home = row["home_team"]

        print(f"\n=== {game_pk}: {away} @ {home} ===")

        try:
            game_input_dir = input_root / f"game_{game_pk}"
            game_output_dir = out_root / f"game_{game_pk}"
            game_input_dir.mkdir(parents=True, exist_ok=True)
            game_output_dir.mkdir(parents=True, exist_ok=True)

            away_lineup = game_input_dir / "away_lineup.csv"
            home_lineup = game_input_dir / "home_lineup.csv"
            away_starter = game_input_dir / "away_starter.csv"
            home_starter = game_input_dir / "home_starter.csv"

            away_ids = [int(x) for x in str(row["away_batters"]).split(",") if str(x).strip()]
            home_ids = [int(x) for x in str(row["home_batters"]).split(",") if str(x).strip()]

            pd.DataFrame({
                "batter_id": away_ids,
                "player_id": away_ids,
                "lineup_slot": list(range(1, len(away_ids) + 1)),
                "batting_order": list(range(1, len(away_ids) + 1)),
            }).to_csv(away_lineup, index=False)

            pd.DataFrame({
                "batter_id": home_ids,
                "player_id": home_ids,
                "lineup_slot": list(range(1, len(home_ids) + 1)),
                "batting_order": list(range(1, len(home_ids) + 1)),
            }).to_csv(home_lineup, index=False)

            pd.DataFrame([{
                "pitcher_id": int(row["away_pitcher_id"]),
                "player_id": int(row["away_pitcher_id"]),
            }]).to_csv(away_starter, index=False)

            pd.DataFrame([{
                "pitcher_id": int(row["home_pitcher_id"]),
                "player_id": int(row["home_pitcher_id"]),
            }]).to_csv(home_starter, index=False)

            out_csv = game_output_dir / "sim_results.csv"

            run_cmd([
                sys.executable,
                "scripts/sim/run_pa_game_sim_fast.py",
                "--away-lineup", str(away_lineup),
                "--home-lineup", str(home_lineup),
                "--away-starter", str(away_starter),
                "--home-starter", str(home_starter),
                "--model-path", args.model_path,
                "--n-sims", str(args.n_sims),
                "--out-csv", str(out_csv),
            ])

            if out_csv.exists():
                df = pd.read_csv(out_csv)
                if len(df):
                    summary = df.iloc[0].to_dict()
                    summary["game_pk"] = game_pk
                    summary["away_team"] = away
                    summary["home_team"] = home
                    summaries.append(summary)

        except Exception as e:
            print(f"[ERROR] {game_pk}: {e}")
            if not args.continue_on_error:
                raise

    if summaries:
        final = pd.DataFrame(summaries)
        out_path = out_root / f"slate_sim_results_{args.game_date}.csv"
        final.to_csv(out_path, index=False)

        print("\nSaved slate sim results:")
        print(out_path)
        print(final.to_string(index=False))
    else:
        print("\nNo completed simulations.")


if __name__ == "__main__":
    main()
