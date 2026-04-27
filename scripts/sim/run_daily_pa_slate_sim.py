#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import subprocess
import pandas as pd
from pathlib import Path

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
LIVE_DIR = DATA_DIR / "processed" / "live"
OUTPUT_DIR = DATA_DIR / "outputs" / "pa_sims"


# -------------------------
# SAFE SUBPROCESS RUNNER
# -------------------------
def run_cmd(cmd):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)

    print(f"\n[RUN] {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, env=env)


# -------------------------
# LOADERS
# -------------------------
def load_schedule(season, date):
    path = LIVE_DIR / f"schedule_{season}_{date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing schedule: {path}")
    return pd.read_parquet(path)


def load_lineups(season, date):
    proj_path = LIVE_DIR / f"projected_lineups_{season}_{date}.parquet"
    conf_path = LIVE_DIR / f"confirmed_lineups_{season}_{date}.parquet"

    df_proj = pd.read_parquet(proj_path) if proj_path.exists() else None
    df_conf = pd.read_parquet(conf_path) if conf_path.exists() else None

    return df_proj, df_conf


def load_pitchers(season, date):
    path = LIVE_DIR / f"starting_pitchers_{season}_{date}.parquet"
    if not path.exists():
        raise FileNotFoundError("Missing starting pitchers file")
    return pd.read_parquet(path)


# -------------------------
# LINEUP SELECTION
# -------------------------
def choose_lineup(df_proj, df_conf, game_pk, team):
    if df_conf is not None:
        conf = df_conf[(df_conf.game_pk == game_pk) & (df_conf.team == team)]
        if len(conf) >= 9:
            return conf.sort_values("batting_order")

    if df_proj is not None:
        proj = df_proj[(df_proj.game_pk == game_pk) & (df_proj.team == team)]
        if len(proj) >= 9:
            return proj.sort_values("batting_order")

    return None


def build_lineup_csv(df, path):
    df[["player_id"]].to_csv(path, index=False)


# -------------------------
# MAIN
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--n-sims", type=int, default=500)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--refresh-rotowire", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    season = args.season
    date = args.date

    # -------------------------
    # STEP 1: ROTOWIRE REFRESH
    # -------------------------
    if args.refresh_rotowire:
        print("\n[STEP] Refreshing Rotowire...\n")

        run_cmd([
            "python", "scripts/live/build_projected_lineups_rotowire.py",
            "--season", str(season), "--date", date
        ])

        run_cmd([
            "python", "scripts/live/build_confirmed_lineups_rotowire.py",
            "--season", str(season), "--date", date
        ])

        run_cmd([
            "python", "scripts/live/build_starting_pitchers_rotowire.py",
            "--season", str(season), "--date", date
        ])

    # -------------------------
    # STEP 2: LOAD DATA
    # -------------------------
    schedule = load_schedule(season, date)
    df_proj, df_conf = load_lineups(season, date)
    pitchers = load_pitchers(season, date)

    out_dir = OUTPUT_DIR / f"{season}_{date}"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # -------------------------
    # STEP 3: LOOP GAMES
    # -------------------------
    for _, game in schedule.iterrows():
        game_pk = game["game_pk"]
        home_team = game["home_team"]
        away_team = game["away_team"]

        print(f"\n=== GAME {game_pk} | {away_team} @ {home_team} ===")

        try:
            home_lineup = choose_lineup(df_proj, df_conf, game_pk, home_team)
            away_lineup = choose_lineup(df_proj, df_conf, game_pk, away_team)

            if home_lineup is None or away_lineup is None:
                print("[SKIP] Missing lineup")
                continue

            home_pitcher = pitchers[
                (pitchers.game_pk == game_pk) & (pitchers.team == home_team)
            ]
            away_pitcher = pitchers[
                (pitchers.game_pk == game_pk) & (pitchers.team == away_team)
            ]

            if home_pitcher.empty or away_pitcher.empty:
                print("[SKIP] Missing pitcher")
                continue

            home_pid = int(home_pitcher.iloc[0]["player_id"])
            away_pid = int(away_pitcher.iloc[0]["player_id"])

            # TEMP FILES
            tmp_dir = out_dir / f"game_{game_pk}"
            tmp_dir.mkdir(exist_ok=True)

            home_csv = tmp_dir / "home_lineup.csv"
            away_csv = tmp_dir / "away_lineup.csv"
            sim_csv = tmp_dir / "sim_summary.csv"

            build_lineup_csv(home_lineup, home_csv)
            build_lineup_csv(away_lineup, away_csv)

            # RUN SIM
            run_cmd([
                "python", "scripts/sim/run_pa_game_sim_fast.py",
                "--home-batters-file", str(home_csv),
                "--away-batters-file", str(away_csv),
                "--home-pitcher-id", str(home_pid),
                "--away-pitcher-id", str(away_pid),
                "--model-path", args.model_path,
                "--n-sims", str(args.n_sims),
                "--out-csv", str(sim_csv)
            ])

            if not sim_csv.exists():
                print("[WARN] No sim output")
                continue

            df = pd.read_csv(sim_csv)
            df["game_pk"] = game_pk
            df["home_team"] = home_team
            df["away_team"] = away_team

            results.append(df)

        except Exception as e:
            print(f"[ERROR] {e}")
            if not args.continue_on_error:
                raise

    # -------------------------
    # STEP 4: SAVE SLATE
    # -------------------------
    if results:
        final = pd.concat(results, ignore_index=True)
        out_path = out_dir / "slate_sim_results.csv"
        final.to_csv(out_path, index=False)

        print("\n==============================")
        print(f"SLATE COMPLETE: {out_path}")
        print("==============================\n")
    else:
        print("\n[INFO] No games processed")


if __name__ == "__main__":
    main()
