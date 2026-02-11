#!/usr/bin/env python3
"""
Build marts (Milestone 2A):
- model_spine_game.parquet  (one row per game)
- team_game.parquet         (two rows per game: home/away)
- batter_game.parquet       (player-game aggregation; stub if events not present)
- pitcher_game.parquet      (player-game aggregation; stub if events not present)

This script is Colab-safe:
- Adds repo-root to sys.path so `import src...` works when run as `python scripts/...`
- Does NOT try to mount Drive interactively (drive should be mounted in notebook)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# --- Ensure `src` is importable when running scripts directly ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.drive import ensure_drive_mounted, resolve_data_dirs
from src.utils.io import load_config, read_parquet, write_parquet


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MLB marts (model spine + team/player game tables).")
    p.add_argument("--season", type=int, required=True, help="Season year (e.g. 2024)")
    p.add_argument("--start", type=str, default=None, help="Optional start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="Optional end date YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
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
    """
    Normalize game_date to pandas datetime64[ns] at midnight.
    Works whether source is object/date/datetime.
    """
    out = df.copy()
    if col in out.columns:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def _maybe_filter_dates(games: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if games.empty:
        return games
    g = games.copy()
    g = _normalize_game_date(g, "game_date")
    if start:
        s = pd.to_datetime(start).normalize()
        g = g[g["game_date"] >= s]
    if end:
        e = pd.to_datetime(end).normalize()
        g = g[g["game_date"] <= e]
    return g.reset_index(drop=True)


# ----------------------------
# Builders
# ----------------------------
def build_model_spine_game(games: pd.DataFrame, game_runs: pd.DataFrame) -> pd.DataFrame:
    """
    One row per game with runs + ids/names.
    Ensures merge keys are consistent and creates home_runs/away_runs aliases.
    """

    games2 = _normalize_game_date(games, "game_date")
    runs2 = _normalize_game_date(game_runs, "game_date")

    # ---- Normalize merge keys (fixes datetime64 vs object merge error) ----
    # Ensure same dtype on both sides:
    games2["season"] = pd.to_numeric(games2["season"], errors="coerce").astype("Int64")
    runs2["season"] = pd.to_numeric(runs2["season"], errors="coerce").astype("Int64")

    # Merge on minimal stable keys
    spine = games2.merge(
        runs2,
        on=["game_pk", "season", "game_date"],
        how="left",
        validate="one_to_one",
    )

    # ---- Column normalization for downstream compatibility ----
    # If game_runs provides *_final, alias to home_runs/away_runs expected by older code.
    if "home_runs" not in spine.columns:
        if "home_runs_final" in spine.columns:
            spine["home_runs"] = pd.to_numeric(spine["home_runs_final"], errors="coerce")
        else:
            spine["home_runs"] = pd.NA

    if "away_runs" not in spine.columns:
        if "away_runs_final" in spine.columns:
            spine["away_runs"] = pd.to_numeric(spine["away_runs_final"], errors="coerce")
        else:
            spine["away_runs"] = pd.NA

    # total runs
    if "total_runs" not in spine.columns:
        if "total_runs_final" in spine.columns:
            spine["total_runs"] = pd.to_numeric(spine["total_runs_final"], errors="coerce")
        else:
            spine["total_runs"] = pd.to_numeric(spine["home_runs"], errors="coerce") + pd.to_numeric(
                spine["away_runs"], errors="coerce"
            )

    # first inning runs (NRFI/YRFI target helpers)
    if "home_runs_1st" not in spine.columns:
        spine["home_runs_1st"] = spine.get("home_runs_1st", pd.NA)
    if "away_runs_1st" not in spine.columns:
        spine["away_runs_1st"] = spine.get("away_runs_1st", pd.NA)
    if "total_runs_1st" not in spine.columns:
        spine["total_runs_1st"] = spine.get("total_runs_1st", pd.NA)

    # keep order consistent
    sort_cols = [c for c in ["game_date", "game_pk"] if c in spine.columns]
    if sort_cols:
        spine = spine.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return spine


def build_team_game(model_spine_game: pd.DataFrame) -> pd.DataFrame:
    """
    Two rows per game: home and away.
    Requires: team ids/names + home_runs/away_runs in model_spine_game.
    """

    required = [
        "game_pk",
        "season",
        "game_date",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "home_runs",
        "away_runs",
    ]
    missing = [c for c in required if c not in model_spine_game.columns]
    if missing:
        raise ValueError(f"model_spine_game missing columns needed for team_game: {missing}")

    m = model_spine_game.copy()

    # Numeric runs
    m["home_runs"] = pd.to_numeric(m["home_runs"], errors="coerce")
    m["away_runs"] = pd.to_numeric(m["away_runs"], errors="coerce")

    home = pd.DataFrame(
        {
            "game_pk": m["game_pk"],
            "season": m["season"],
            "game_date": m["game_date"],
            "team_side": "home",
            "team_id": m["home_team_id"],
            "team_name": m["home_team_name"],
            "opponent_id": m["away_team_id"],
            "opponent_name": m["away_team_name"],
            "is_home": 1,
            "runs_scored": m["home_runs"],
            "runs_allowed": m["away_runs"],
            "runs_1st_scored": pd.to_numeric(m.get("home_runs_1st", pd.NA), errors="coerce"),
            "runs_1st_allowed": pd.to_numeric(m.get("away_runs_1st", pd.NA), errors="coerce"),
        }
    )

    away = pd.DataFrame(
        {
            "game_pk": m["game_pk"],
            "season": m["season"],
            "game_date": m["game_date"],
            "team_side": "away",
            "team_id": m["away_team_id"],
            "team_name": m["away_team_name"],
            "opponent_id": m["home_team_id"],
            "opponent_name": m["home_team_name"],
            "is_home": 0,
            "runs_scored": m["away_runs"],
            "runs_allowed": m["home_runs"],
            "runs_1st_scored": pd.to_numeric(m.get("away_runs_1st", pd.NA), errors="coerce"),
            "runs_1st_allowed": pd.to_numeric(m.get("home_runs_1st", pd.NA), errors="coerce"),
        }
    )

    team_game = pd.concat([home, away], ignore_index=True)

    team_game["run_diff"] = pd.to_numeric(team_game["runs_scored"], errors="coerce") - pd.to_numeric(
        team_game["runs_allowed"], errors="coerce"
    )
    team_game["win"] = (team_game["run_diff"] > 0).astype("Int64")

    team_game = team_game.sort_values(["game_pk", "team_side"], kind="mergesort").reset_index(drop=True)
    return team_game


def build_batter_game_stub(model_spine_game: pd.DataFrame) -> pd.DataFrame:
    """
    Stub batter_game until you wire in event-level parsing.
    Exists so downstream rolling scripts don't crash on missing file.
    """
    cols = [
        "game_pk",
        "season",
        "game_date",
        "batter_id",
        "batting_team_id",
        "pa",
        "ab",
        "h",
        "hr",
        "rbi",
        "bb",
        "so",
        "tb",
        "2b",
        "3b",
        "1b",
    ]
    return pd.DataFrame(columns=cols)


def build_pitcher_game_stub(model_spine_game: pd.DataFrame) -> pd.DataFrame:
    """
    Stub pitcher_game until you wire in event-level parsing.
    """
    cols = [
        "game_pk",
        "season",
        "game_date",
        "pitcher_id",
        "pitching_team_id",
        "bf",
        "ip_outs",
        "k",
        "bb",
        "er",
        "r",
        "h_allowed",
        "hr_allowed",
    ]
    return pd.DataFrame(columns=cols)


def _guard_overwrite(paths: list[Path], force: bool) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not force:
        names = ", ".join(p.name for p in existing)
        raise FileExistsError(f"Outputs already exist ({names}). Re-run with --force to overwrite.")
    if existing and force:
        for p in existing:
            p.unlink()


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    args = parse_args()
    start = _validate_date(args.start, "--start")
    end = _validate_date(args.end, "--end")
    if start and end and start > end:
        raise ValueError("--start must be <= --end")

    config = load_config()
    if int(args.season) not in set(config.get("seasons", [])):
        raise ValueError(f"Season {args.season} is not listed in config/project.yaml seasons")

    ensure_drive_mounted()
    dirs = resolve_data_dirs(config)
    processed = dirs["processed"]

    games_path = processed / "games.parquet"
    runs_path = processed / "game_runs.parquet"

    if not games_path.exists():
        raise FileNotFoundError(f"Missing {games_path}. Run scripts/build_spine.py first.")
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing {runs_path}. Run scripts/build_events.py first (to produce game_runs).")

    games = read_parquet(games_path)
    game_runs = read_parquet(runs_path)

    # Filter date range if requested
    games = _maybe_filter_dates(games, start, end)
    # Keep game_runs consistent too
    game_runs = _normalize_game_date(game_runs, "game_date")
    if start:
        s = pd.to_datetime(start).normalize()
        game_runs = game_runs[game_runs["game_date"] >= s]
    if end:
        e = pd.to_datetime(end).normalize()
        game_runs = game_runs[game_runs["game_date"] <= e]
    game_runs = game_runs.reset_index(drop=True)

    # Build marts
    model_spine_game = build_model_spine_game(games, game_runs)

    # Outputs
    out_model_spine = processed / "model_spine_game.parquet"
    out_team_game = processed / "team_game.parquet"
    out_batter_game = processed / "batter_game.parquet"
    out_pitcher_game = processed / "pitcher_game.parquet"

    _guard_overwrite(
        [out_model_spine, out_team_game, out_batter_game, out_pitcher_game],
        force=bool(args.force),
    )

    team_game = build_team_game(model_spine_game)

    # For now these are stubs until event parsing produces batter/pitcher aggs
    batter_game = build_batter_game_stub(model_spine_game)
    pitcher_game = build_pitcher_game_stub(model_spine_game)

    write_parquet(model_spine_game, out_model_spine)
    write_parquet(team_game, out_team_game)
    write_parquet(batter_game, out_batter_game)
    write_parquet(pitcher_game, out_pitcher_game)

    print(f"model_spine_game rows: {len(model_spine_game)}")
    print(f"team_game rows: {len(team_game)}")
    print(f"batter_game rows: {len(batter_game)} (stub)")
    print(f"pitcher_game rows: {len(pitcher_game)} (stub)")
    print(f"Wrote marts to: {processed}")


if __name__ == "__main__":
    main()
