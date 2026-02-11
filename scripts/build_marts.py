#!/usr/bin/env python3
"""
Build model marts from processed spine + events tables.

Inputs (from data/processed):
- games.parquet
- game_runs.parquet
- events_pa.parquet

Outputs (to data/processed):
- model_spine_game.parquet
- batter_game.parquet
- pitcher_game.parquet
- team_game.parquet
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# --- Make `from src...` imports work when running as a script in Colab/CLI ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.drive import ensure_drive_mounted, resolve_data_dirs
from src.utils.io import load_config, read_parquet, write_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MLB marts (game/team/batter/pitcher) from processed spine + events.")
    p.add_argument("--season", type=int, required=True, help="Season year (e.g. 2024)")
    p.add_argument("--start", type=str, default=None, help="Optional start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="Optional end date YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="Overwrite existing output marts")
    return p.parse_args()


def _validate_date(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD; got {value!r}") from exc
    return value


def _normalize_game_date(df: pd.DataFrame, col: str = "game_date") -> pd.DataFrame:
    """Coerce to datetime64[ns] midnight to keep merges stable."""
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def _filter_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty or "game_date" not in df.columns:
        return df
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    if start:
        out = out[out["game_date"] >= pd.to_datetime(start).normalize()]
    if end:
        out = out[out["game_date"] <= pd.to_datetime(end).normalize()]
    return out.reset_index(drop=True)


def build_model_spine_game(games: pd.DataFrame, game_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Game-level modeling spine: games + game_runs.
    """

    # ---- The REAL FIX: normalize merge keys on BOTH sides ----
    games = _normalize_game_date(games, "game_date")
    game_runs = _normalize_game_date(game_runs, "game_date")

    # Normalize merge keys
    games = games.copy()
    game_runs = game_runs.copy()

    # Ensure season exists and is int-like
    if "season" in games.columns:
        games["season"] = pd.to_numeric(games["season"], errors="coerce").astype("Int64")
    if "season" in game_runs.columns:
        game_runs["season"] = pd.to_numeric(game_runs["season"], errors="coerce").astype("Int64")

    # Merge (inner keeps only games with run data)
    spine = games.merge(game_runs, on=["game_pk", "season", "game_date"], how="inner")

    # Basic cleanup
    spine = spine.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    return spine


def build_team_game(model_spine_game: pd.DataFrame) -> pd.DataFrame:
    """
    Team-game rows for moneyline/spread/total style features.
    Produces TWO rows per game: home and away.
    """
    g = model_spine_game

    needed = [
        "game_pk",
        "season",
        "game_date",
        "home_team_id",
        "away_team_id",
        "home_team_name",
        "away_team_name",
        "home_runs",
        "away_runs",
    ]
    missing = [c for c in needed if c not in g.columns]
    if missing:
        raise ValueError(f"model_spine_game missing columns needed for team_game: {missing}")

    home = pd.DataFrame(
        {
            "game_pk": g["game_pk"],
            "season": g["season"],
            "game_date": g["game_date"],
            "team_side": "home",
            "team_id": g["home_team_id"],
            "team_name": g["home_team_name"],
            "opp_team_id": g["away_team_id"],
            "opp_team_name": g["away_team_name"],
            "runs_for": g["home_runs"],
            "runs_against": g["away_runs"],
        }
    )
    away = pd.DataFrame(
        {
            "game_pk": g["game_pk"],
            "season": g["season"],
            "game_date": g["game_date"],
            "team_side": "away",
            "team_id": g["away_team_id"],
            "team_name": g["away_team_name"],
            "opp_team_id": g["home_team_id"],
            "opp_team_name": g["home_team_name"],
            "runs_for": g["away_runs"],
            "runs_against": g["home_runs"],
        }
    )

    team_game = pd.concat([home, away], ignore_index=True)
    team_game["run_diff"] = pd.to_numeric(team_game["runs_for"], errors="coerce") - pd.to_numeric(
        team_game["runs_against"], errors="coerce"
    )
    return team_game


def build_batter_game(events_pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Batter-game mart from PA-level events.
    Requires events_pa columns at minimum:
      - game_pk
      - batter_id
      - batter_name (optional)
      - is_hit (0/1)
      - is_hr (0/1)
      - is_bb (0/1)
      - is_2b (0/1)  (if present)
      - rbi (if present)
      - runs (if present)
    """
    e = events_pa.copy()

    required = ["game_pk", "batter_id"]
    missing = [c for c in required if c not in e.columns]
    if missing:
        raise ValueError(f"events_pa missing required columns for batter_game: {missing}")

    # Safe defaults if upstream didn't create these yet
    for col in ["is_hit", "is_hr", "is_bb", "is_2b", "rbi", "runs", "pa"]:
        if col not in e.columns:
            e[col] = 0

    # Ensure numeric
    for col in ["is_hit", "is_hr", "is_bb", "is_2b", "rbi", "runs", "pa"]:
        e[col] = pd.to_numeric(e[col], errors="coerce").fillna(0).astype(int)

    agg = (
        e.groupby(["game_pk", "batter_id"], as_index=False)
        .agg(
            pa=("pa", "sum"),
            hits=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            bb=("is_bb", "sum"),
            doubles=("is_2b", "sum"),
            rbi=("rbi", "sum"),
            runs=("runs", "sum"),
        )
        .reset_index(drop=True)
    )

    # Attach game_date/season for rolling windows later
    gcols = ["game_pk", "season", "game_date"]
    g = games[gcols].copy()
    g = _normalize_game_date(g, "game_date")

    batter_game = agg.merge(g, on="game_pk", how="left")

    # Optional name columns if present
    if "batter_name" in e.columns:
        names = e[["batter_id", "batter_name"]].dropna().drop_duplicates("batter_id")
        batter_game = batter_game.merge(names, on="batter_id", how="left")

    return batter_game


def build_pitcher_game(events_pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Pitcher-game mart from PA-level events.
    Requires events_pa columns at minimum:
      - game_pk
      - pitcher_id
      - is_k (0/1) (if present)
      - is_bb (0/1)
      - runs_allowed (if present)
      - outs (if present)
      - batters_faced (if present)
    """
    e = events_pa.copy()

    required = ["game_pk", "pitcher_id"]
    missing = [c for c in required if c not in e.columns]
    if missing:
        raise ValueError(f"events_pa missing required columns for pitcher_game: {missing}")

    for col in ["is_k", "is_bb", "runs_allowed", "outs", "batters_faced"]:
        if col not in e.columns:
            e[col] = 0

    for col in ["is_k", "is_bb", "runs_allowed", "outs", "batters_faced"]:
        e[col] = pd.to_numeric(e[col], errors="coerce").fillna(0).astype(int)

    agg = (
        e.groupby(["game_pk", "pitcher_id"], as_index=False)
        .agg(
            batters_faced=("batters_faced", "sum"),
            outs=("outs", "sum"),
            k=("is_k", "sum"),
            bb=("is_bb", "sum"),
            runs_allowed=("runs_allowed", "sum"),
        )
        .reset_index(drop=True)
    )

    gcols = ["game_pk", "season", "game_date"]
    g = games[gcols].copy()
    g = _normalize_game_date(g, "game_date")

    pitcher_game = agg.merge(g, on="game_pk", how="left")

    if "pitcher_name" in e.columns:
        names = e[["pitcher_id", "pitcher_name"]].dropna().drop_duplicates("pitcher_id")
        pitcher_game = pitcher_game.merge(names, on="pitcher_id", how="left")

    return pitcher_game


def main() -> None:
    args = parse_args()
    start = _validate_date(args.start, "--start")
    end = _validate_date(args.end, "--end")
    if start and end and start > end:
        raise ValueError("--start must be <= --end")

    config = load_config()
    ensure_drive_mounted()
    dirs = resolve_data_dirs(config)
    processed = dirs["processed"]

    # ---- Only guard REAL output marts (NOT game_runs.parquet, NOT events_pa.parquet) ----
    outputs = [
        processed / "model_spine_game.parquet",
        processed / "team_game.parquet",
        processed / "batter_game.parquet",
        processed / "pitcher_game.parquet",
    ]
    if not args.force:
        for p in outputs:
            if p.exists():
                raise FileExistsError(f"{p.name} exists. Re-run with --force to overwrite.")

    # Inputs
    games_path = processed / "games.parquet"
    game_runs_path = processed / "game_runs.parquet"
    events_pa_path = processed / "events_pa.parquet"

    for p in [games_path, game_runs_path, events_pa_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    games = read_parquet(games_path)
    game_runs = read_parquet(game_runs_path)
    events_pa = read_parquet(events_pa_path)

    # Normalize + date filter
    games = _normalize_game_date(games, "game_date")
    game_runs = _normalize_game_date(game_runs, "game_date")
    games = _filter_date_range(games, start, end)
    game_runs = _filter_date_range(game_runs, start, end)

    # Build marts
    model_spine_game = build_model_spine_game(games, game_runs).sort_values(["game_pk"], kind="mergesort")
    team_game = build_team_game(model_spine_game).sort_values(["game_pk", "team_side"], kind="mergesort")

    # Batter/pitcher marts from events_pa; attach game_date/season from games
    batter_game = build_batter_game(events_pa, games).sort_values(["game_pk", "batter_id"], kind="mergesort")
    pitcher_game = build_pitcher_game(events_pa, games).sort_values(["game_pk", "pitcher_id"], kind="mergesort")

    # Write outputs
    write_parquet(model_spine_game, processed / "model_spine_game.parquet")
    write_parquet(team_game, processed / "team_game.parquet")
    write_parquet(batter_game, processed / "batter_game.parquet")
    write_parquet(pitcher_game, processed / "pitcher_game.parquet")

    print(f"model_spine_game rows: {len(model_spine_game)}")
    print(f"team_game rows: {len(team_game)}")
    print(f"batter_game rows: {len(batter_game)}")
    print(f"pitcher_game rows: {len(pitcher_game)}")
    print(f"Wrote marts to: {processed}")


if __name__ == "__main__":
    main()
