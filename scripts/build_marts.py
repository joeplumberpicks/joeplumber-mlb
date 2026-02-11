#!/usr/bin/env python3
"""
Build marts (model tables) from processed spine + events.

Writes to: <drive_root>/data/processed/
- model_spine_game.parquet
- team_game.parquet
- batter_game.parquet
- pitcher_game.parquet

Requires (from prior steps):
- games.parquet
- game_runs.parquet
- events_pa.parquet  (for batter/pitcher marts; minimal stubs created if missing cols)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# --- sys.path bootstrap so `import src...` always works in Colab ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.drive import ensure_drive_mounted, resolve_data_dirs
from src.utils.io import load_config, read_parquet, write_parquet


# -------------------------
# Helpers
# -------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    return p.parse_args()


def _validate_date(s: Optional[str], name: str) -> Optional[str]:
    if s is None:
        return None
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"{name} must be YYYY-MM-DD, got {s!r}") from e
    return s


def _norm_date_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    return df


def _coalesce_numeric(df: pd.DataFrame, candidates: Iterable[str], out: str) -> pd.DataFrame:
    """
    Create `out` from first existing candidate column in `candidates`.
    Cast to numeric (float -> int where possible).
    """
    df = df.copy()
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        df[out] = pd.NA
        return df

    df[out] = pd.to_numeric(df[found], errors="coerce")
    return df


def _coalesce_text(df: pd.DataFrame, candidates: Iterable[str], out: str) -> pd.DataFrame:
    df = df.copy()
    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break
    if found is None:
        df[out] = pd.NA
        return df
    df[out] = df[found]
    return df


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _check_overwrite(paths: list[Path], force: bool) -> None:
    if force:
        return
    for p in paths:
        if p.exists():
            raise FileExistsError(f"{p.name} exists. Re-run with --force to overwrite.")


# -------------------------
# Core builders
# -------------------------
def build_model_spine_game(games: pd.DataFrame, game_runs: pd.DataFrame) -> pd.DataFrame:
    """
    One row per game with core game context + final runs.

    Robust to different run column names in game_runs.
    """
    games = games.copy()
    game_runs = game_runs.copy()

    # Normalize merge keys
    games = _norm_date_col(games, "game_date")
    game_runs = _norm_date_col(game_runs, "game_date")

    # Standardize team columns from games (different versions name these differently)
    games = _coalesce_text(games, ["home_team_name", "home_team"], "home_team_name")
    games = _coalesce_text(games, ["away_team_name", "away_team"], "away_team_name")
    games = _coalesce_numeric(games, ["home_team_id", "home_id"], "home_team_id")
    games = _coalesce_numeric(games, ["away_team_id", "away_id"], "away_team_id")

    # Standardize runs columns from game_runs (handle multiple possible names)
    game_runs = _coalesce_numeric(
        game_runs,
        ["home_runs", "home_score", "home_team_runs", "homeRuns", "home_runs_final"],
        "home_runs",
    )
    game_runs = _coalesce_numeric(
        game_runs,
        ["away_runs", "away_score", "away_team_runs", "awayRuns", "away_runs_final"],
        "away_runs",
    )

    # Merge on game_pk first (most stable); include season/date when present
    merge_keys = ["game_pk"]
    for k in ["season", "game_date"]:
        if k in games.columns and k in game_runs.columns:
            merge_keys.append(k)

    spine = games.merge(game_runs, on=merge_keys, how="left", suffixes=("", "_runs"))

    # If game_runs had no season/date overlap, we still want runs by game_pk
    # Ensure home_runs/away_runs exist after merge
    if "home_runs" not in spine.columns:
        spine["home_runs"] = pd.NA
    if "away_runs" not in spine.columns:
        spine["away_runs"] = pd.NA

    # Keep only the columns we care about (plus anything else you want later)
    keep = []
    for c in [
        "game_pk",
        "season",
        "game_date",
        "status",
        "venue_id",
        "venue_name",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "home_runs",
        "away_runs",
    ]:
        if c in spine.columns:
            keep.append(c)

    spine = spine[keep].copy()

    # Required columns for downstream marts
    _require_cols(
        spine,
        ["game_pk", "game_date", "home_team_id", "away_team_id", "home_runs", "away_runs"],
        "model_spine_game",
    )

    spine = spine.sort_values(["game_pk"], kind="mergesort").reset_index(drop=True)
    return spine


def build_team_game(model_spine_game: pd.DataFrame) -> pd.DataFrame:
    """
    Two rows per game: home + away.
    Requires home_runs/away_runs to exist.
    """
    m = model_spine_game.copy()

    # Ensure numeric
    m["home_runs"] = pd.to_numeric(m["home_runs"], errors="coerce")
    m["away_runs"] = pd.to_numeric(m["away_runs"], errors="coerce")

    needed = [
        "game_pk",
        "game_date",
        "home_team_id",
        "away_team_id",
        "home_runs",
        "away_runs",
    ]
    _require_cols(m, needed, "model_spine_game for team_game")

    # Names optional
    if "home_team_name" not in m.columns:
        m["home_team_name"] = pd.NA
    if "away_team_name" not in m.columns:
        m["away_team_name"] = pd.NA

    home = pd.DataFrame(
        {
            "game_pk": m["game_pk"],
            "game_date": m["game_date"],
            "team_side": "home",
            "team_id": m["home_team_id"],
            "team_name": m["home_team_name"],
            "runs_for": m["home_runs"],
            "runs_against": m["away_runs"],
            "is_home": 1,
        }
    )

    away = pd.DataFrame(
        {
            "game_pk": m["game_pk"],
            "game_date": m["game_date"],
            "team_side": "away",
            "team_id": m["away_team_id"],
            "team_name": m["away_team_name"],
            "runs_for": m["away_runs"],
            "runs_against": m["home_runs"],
            "is_home": 0,
        }
    )

    out = pd.concat([home, away], ignore_index=True)
    out = out.sort_values(["game_pk", "team_side"], kind="mergesort").reset_index(drop=True)
    return out


def build_batter_game(events_pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal batter_game mart. Uses events_pa if columns exist; otherwise creates a safe stub.
    """
    if events_pa is None or events_pa.empty:
        return pd.DataFrame(columns=["game_pk", "batter_id", "batting_team_id", "pa"])

    df = events_pa.copy()

    # Try to standardize expected columns
    # We'll proceed even if some are missing.
    keys = [c for c in ["game_pk", "batter_id", "batting_team_id"] if c in df.columns]
    if "game_pk" not in df.columns or "batter_id" not in df.columns:
        return pd.DataFrame(columns=["game_pk", "batter_id", "batting_team_id", "pa"])

    if "batting_team_id" not in df.columns:
        df["batting_team_id"] = pd.NA

    # Metrics that might exist
    if "is_hit" not in df.columns:
        df["is_hit"] = 0
    if "is_hr" not in df.columns:
        df["is_hr"] = 0
    if "is_bb" not in df.columns:
        df["is_bb"] = 0
    if "rbi" not in df.columns:
        df["rbi"] = 0

    agg = (
        df.groupby(["game_pk", "batter_id", "batting_team_id"], dropna=False)
        .agg(
            pa=("game_pk", "size"),
            hits=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            bb=("is_bb", "sum"),
            rbi=("rbi", "sum"),
        )
        .reset_index()
    )

    # Attach game_date if available
    if "game_date" in games.columns:
        g = _norm_date_col(games, "game_date")[["game_pk", "game_date"]].drop_duplicates("game_pk")
        agg = agg.merge(g, on="game_pk", how="left")

    return agg


def build_pitcher_game(events_pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal pitcher_game mart. Uses events_pa if columns exist; otherwise safe stub.
    """
    if events_pa is None or events_pa.empty:
        return pd.DataFrame(columns=["game_pk", "pitcher_id", "bf"])

    df = events_pa.copy()
    if "game_pk" not in df.columns or "pitcher_id" not in df.columns:
        return pd.DataFrame(columns=["game_pk", "pitcher_id", "bf"])

    if "is_k" not in df.columns:
        df["is_k"] = 0
    if "is_bb" not in df.columns:
        df["is_bb"] = 0

    agg = (
        df.groupby(["game_pk", "pitcher_id"], dropna=False)
        .agg(
            bf=("game_pk", "size"),
            k=("is_k", "sum"),
            bb=("is_bb", "sum"),
        )
        .reset_index()
    )

    if "game_date" in games.columns:
        g = _norm_date_col(games, "game_date")[["game_pk", "game_date"]].drop_duplicates("game_pk")
        agg = agg.merge(g, on="game_pk", how="left")

    return agg


# -------------------------
# Main
# -------------------------
def main() -> None:
    args = _parse_args()
    start = _validate_date(args.start, "--start")
    end = _validate_date(args.end, "--end")

    config = load_config()
    ensure_drive_mounted()
    dirs = resolve_data_dirs(config)
    processed: Path = dirs["processed"]

    # Inputs
    games_path = processed / "games.parquet"
    runs_path = processed / "game_runs.parquet"
    events_path = processed / "events_pa.parquet"

    if not games_path.exists():
        raise FileNotFoundError(f"Missing {games_path}. Run build_spine.py first.")
    if not runs_path.exists():
        raise FileNotFoundError(f"Missing {runs_path}. Run build_events.py first.")
    if not events_path.exists():
        print(f"WARNING: {events_path} not found. batter_game/pitcher_game will be stubs.")

    games = read_parquet(games_path)
    game_runs = read_parquet(runs_path)
    events_pa = read_parquet(events_path) if events_path.exists() else pd.DataFrame()

    # Optional slice by date (if provided)
    if start or end:
        games = _norm_date_col(games, "game_date")
        if start:
            games = games[games["game_date"] >= pd.to_datetime(start)]
        if end:
            games = games[games["game_date"] <= pd.to_datetime(end)]

    # Outputs
    out_model = processed / "model_spine_game.parquet"
    out_team = processed / "team_game.parquet"
    out_batter = processed / "batter_game.parquet"
    out_pitcher = processed / "pitcher_game.parquet"

    _check_overwrite([out_model, out_team, out_batter, out_pitcher], force=args.force)

    model_spine_game = build_model_spine_game(games, game_runs)
    team_game = build_team_game(model_spine_game)
    batter_game = build_batter_game(events_pa, games)
    pitcher_game = build_pitcher_game(events_pa, games)

    write_parquet(model_spine_game, out_model)
    write_parquet(team_game, out_team)
    write_parquet(batter_game, out_batter)
    write_parquet(pitcher_game, out_pitcher)

    print(f"model_spine_game rows: {len(model_spine_game)}")
    print(f"team_game rows: {len(team_game)}")
    print(f"batter_game rows: {len(batter_game)}")
    print(f"pitcher_game rows: {len(pitcher_game)}")
    print(f"Wrote marts to: {processed}")


if __name__ == "__main__":
    main()
