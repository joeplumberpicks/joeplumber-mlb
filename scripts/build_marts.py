#!/usr/bin/env python3
"""
Build model-ready mart tables from processed spine + events.

Outputs (to data/processed):
- game_runs.parquet
- batter_game.parquet
- pitcher_game.parquet
- team_game.parquet
- model_spine_game.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.drive import ensure_drive_mounted, resolve_data_dirs
from src.utils.io import load_config, read_parquet, write_parquet


# -----------------------------
# Args
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MLB modeling marts.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


# -----------------------------
# Builders
# -----------------------------
def build_game_runs(events_pa: pd.DataFrame) -> pd.DataFrame:
    runs = (
        events_pa.groupby(["game_pk", "season", "game_date"], as_index=False)
        .agg(
            runs_home=("runs_home", "max"),
            runs_away=("runs_away", "max"),
        )
    )
    return runs


def build_batter_game(events_pa: pd.DataFrame) -> pd.DataFrame:
    batter = (
        events_pa.groupby(["game_pk", "season", "game_date", "batter_id"], as_index=False)
        .agg(
            pa=("pa_id", "count"),
            hits=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            bb=("is_bb", "sum"),
            so=("is_so", "sum"),
            rbi=("rbi", "sum"),
        )
    )
    return batter


def build_pitcher_game(events_pa: pd.DataFrame) -> pd.DataFrame:
    pitcher = (
        events_pa.groupby(["game_pk", "season", "game_date", "pitcher_id"], as_index=False)
        .agg(
            batters_faced=("pa_id", "count"),
            hits_allowed=("is_hit", "sum"),
            hr_allowed=("is_hr", "sum"),
            bb_allowed=("is_bb", "sum"),
            so=("is_so", "sum"),
            rbi_allowed=("rbi", "sum"),
        )
    )
    return pitcher


def build_team_game(events_pa: pd.DataFrame) -> pd.DataFrame:
    team = (
        events_pa.groupby(["game_pk", "season", "game_date", "batting_team_id"], as_index=False)
        .agg(
            pa=("pa_id", "count"),
            hits=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            bb=("is_bb", "sum"),
            so=("is_so", "sum"),
            rbi=("rbi", "sum"),
        )
        .rename(columns={"batting_team_id": "team_id"})
    )
    return team


def build_model_spine_game(games: pd.DataFrame, game_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical per-game modeling spine.
    """

    # 🔑 FIX: normalize merge keys
    games = games.copy()
    game_runs = game_runs.copy()

    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    game_runs["game_date"] = pd.to_datetime(game_runs["game_date"], errors="coerce").dt.normalize()

    spine = games.merge(
        game_runs,
        on=["game_pk", "season", "game_date"],
        how="inner",
    )

    return spine


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()

    ensure_drive_mounted()
    config = load_config()
    dirs = resolve_data_dirs(config)
    processed = dirs["processed"]

    # Inputs
    games = read_parquet(processed / "games.parquet")
    events_pa = read_parquet(processed / "events_pa.parquet")

    # Outputs
    out_game_runs = processed / "game_runs.parquet"
    out_batter = processed / "batter_game.parquet"
    out_pitcher = processed / "pitcher_game.parquet"
    out_team = processed / "team_game.parquet"
    out_spine = processed / "model_spine_game.parquet"

    if not args.force:
        for p in [out_game_runs, out_batter, out_pitcher, out_team, out_spine]:
            if p.exists():
                raise FileExistsError(f"{p.name} exists. Re-run with --force to overwrite.")

    # Build marts
    game_runs = build_game_runs(events_pa)
    batter_game = build_batter_game(events_pa)
    pitcher_game = build_pitcher_game(events_pa)
    team_game = build_team_game(events_pa)
    model_spine_game = build_model_spine_game(games, game_runs)

    # Write
    write_parquet(game_runs, out_game_runs)
    write_parquet(batter_game, out_batter)
    write_parquet(pitcher_game, out_pitcher)
    write_parquet(team_game, out_team)
    write_parquet(model_spine_game, out_spine)

    print("Marts written:")
    print(f"- {out_game_runs}")
    print(f"- {out_batter}")
    print(f"- {out_pitcher}")
    print(f"- {out_team}")
    print(f"- {out_spine}")


if __name__ == "__main__":
    main()
