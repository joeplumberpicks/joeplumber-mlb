#!/usr/bin/env python3
"""
Build marts (model-ready tables) from processed spine + events outputs.

Outputs written to Drive-rooted processed dir:
- game_runs.parquet (pass-through / normalized)
- model_spine_game.parquet
- team_game.parquet
- batter_game.parquet
- pitcher_game.parquet

Fixes:
- Normalizes game_date merge keys
- Maps home_runs_final/away_runs_final -> home_runs/away_runs
- Adds --force overwrite
- Adds sys.path bootstrap so `from src...` works when running as a script
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# --- sys.path bootstrap so `src` imports work in Colab/script execution ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.drive import ensure_drive_mounted, resolve_data_dirs
from src.utils.io import load_config, read_parquet, write_parquet


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build marts from processed spine + events.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (optional)")
    p.add_argument("--windows", type=str, default="3,7,15,30", help="Reserved for future feature windows")
    p.add_argument("--force", action="store_true", help="Overwrite existing parquet outputs")
    return p.parse_args()


def _validate_date(s: Optional[str], label: str) -> Optional[str]:
    if s is None:
        return None
    try:
        datetime.strptime(s, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"{label} must be YYYY-MM-DD; got {s!r}") from e
    return s


def _normalize_game_date(df: pd.DataFrame, col: str = "game_date") -> pd.Series:
    # normalize() -> midnight datetime64[ns], reliable merge key
    return pd.to_datetime(df[col], errors="coerce").dt.normalize()


def _maybe_slice_games(games: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if games.empty:
        return games
    g = games.copy()
    g["_game_date_dt"] = _normalize_game_date(g, "game_date")
    if start:
        g = g[g["_game_date_dt"] >= pd.to_datetime(start)]
    if end:
        g = g[g["_game_date_dt"] <= pd.to_datetime(end)]
    return g.drop(columns=["_game_date_dt"], errors="ignore").reset_index(drop=True)


def _check_overwrite(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path.name} exists. Re-run with --force to overwrite.")


# -----------------------------
# Builders
# -----------------------------
def build_model_spine_game(games: pd.DataFrame, game_runs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge games + game_runs into a single game-level spine table for modeling.
    Ensures columns required downstream exist.
    """
    if games.empty or game_runs.empty:
        return pd.DataFrame()

    # Normalize merge keys
    games = games.copy()
    game_runs = game_runs.copy()

    games["game_date"] = _normalize_game_date(games, "game_date")
    game_runs["game_date"] = _normalize_game_date(game_runs, "game_date")

    # Merge
    spine = games.merge(game_runs, on=["game_pk", "season", "game_date"], how="inner")

    # ---- FIX: map *_final -> expected home_runs/away_runs ----
    if "home_runs_final" in spine.columns:
        spine["home_runs"] = pd.to_numeric(spine["home_runs_final"], errors="coerce")
    elif "home_runs" in spine.columns:
        spine["home_runs"] = pd.to_numeric(spine["home_runs"], errors="coerce")
    else:
        spine["home_runs"] = pd.NA

    if "away_runs_final" in spine.columns:
        spine["away_runs"] = pd.to_numeric(spine["away_runs_final"], errors="coerce")
    elif "away_runs" in spine.columns:
        spine["away_runs"] = pd.to_numeric(spine["away_runs"], errors="coerce")
    else:
        spine["away_runs"] = pd.NA

    # Total runs convenience
    spine["total_runs"] = pd.to_numeric(spine.get("total_runs_final", spine["home_runs"] + spine["away_runs"]), errors="coerce")

    # First inning fields (keep whichever exist)
    for c in ["home_runs_1st", "away_runs_1st", "total_runs_1st"]:
        if c not in spine.columns:
            spine[c] = pd.NA

    # Clean types
    spine["game_pk"] = pd.to_numeric(spine["game_pk"], errors="coerce").astype("Int64")
    spine["season"] = pd.to_numeric(spine["season"], errors="coerce").astype("Int64")

    return spine


def build_team_game(model_spine_game: pd.DataFrame) -> pd.DataFrame:
    """
    Two rows per game: home + away. Robust to run column naming differences:
    - accepts home_runs/away_runs OR home_runs_final/away_runs_final
    """
    if model_spine_game.empty:
        return pd.DataFrame()

    g = model_spine_game.copy()

    # ---- FIX: support *_final naming ----
    if "home_runs" not in g.columns and "home_runs_final" in g.columns:
        g["home_runs"] = pd.to_numeric(g["home_runs_final"], errors="coerce")
    if "away_runs" not in g.columns and "away_runs_final" in g.columns:
        g["away_runs"] = pd.to_numeric(g["away_runs_final"], errors="coerce")

    # Also handle 1st inning if needed
    if "home_runs_1st" not in g.columns and "home_runs_first" in g.columns:
        g["home_runs_1st"] = pd.to_numeric(g["home_runs_first"], errors="coerce")
    if "away_runs_1st" not in g.columns and "away_runs_first" in g.columns:
        g["away_runs_1st"] = pd.to_numeric(g["away_runs_first"], errors="coerce")

    required = ["game_pk", "season", "game_date", "home_team_id", "away_team_id", "home_runs", "away_runs"]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise ValueError(f"model_spine_game missing columns needed for team_game: {missing}")

    # Ensure these exist (safe defaults)
    if "home_runs_1st" not in g.columns:
        g["home_runs_1st"] = pd.NA
    if "away_runs_1st" not in g.columns:
        g["away_runs_1st"] = pd.NA

    home = pd.DataFrame(
        {
            "game_pk": g["game_pk"],
            "season": g["season"],
            "game_date": g["game_date"],
            "team_side": "home",
            "team_id": g["home_team_id"],
            "opponent_team_id": g["away_team_id"],
            "runs_final": g["home_runs"],
            "runs_1st": g["home_runs_1st"],
        }
    )

    away = pd.DataFrame(
        {
            "game_pk": g["game_pk"],
            "season": g["season"],
            "game_date": g["game_date"],
            "team_side": "away",
            "team_id": g["away_team_id"],
            "opponent_team_id": g["home_team_id"],
            "runs_final": g["away_runs"],
            "runs_1st": g["away_runs_1st"],
        }
    )

    out = pd.concat([home, away], ignore_index=True)
    out["is_home"] = (out["team_side"] == "home").astype(int)
    return out


def build_batter_game(events_pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Batter-game aggregation. This is a backbone table used later for rolling features and hitter props.
    If your events schema is richer, we’ll expand this later — but this creates the required file now.
    """
    if events_pa.empty:
        return pd.DataFrame(columns=["game_pk", "batter_id", "pa", "h", "hr", "bb", "so", "rbi"])

    e = events_pa.copy()

    # Flexible column detection
    batter_col = "batter_id" if "batter_id" in e.columns else ("batter" if "batter" in e.columns else None)
    if batter_col is None:
        return pd.DataFrame(columns=["game_pk", "batter_id", "pa", "h", "hr", "bb", "so", "rbi"])

    # Try to infer outcome flags if present; otherwise safe zeros
    def _col_or_zero(name: str) -> pd.Series:
        return pd.to_numeric(e[name], errors="coerce").fillna(0) if name in e.columns else pd.Series([0] * len(e))

    # Basic counts
    e["pa"] = 1
    e["h"] = _col_or_zero("is_hit")
    e["hr"] = _col_or_zero("is_hr")
    e["bb"] = _col_or_zero("is_bb")
    e["so"] = _col_or_zero("is_so")
    e["rbi"] = _col_or_zero("rbi")

    # If event_type exists, we can improve flags a bit
    if "event_type" in e.columns:
        et = e["event_type"].astype(str).str.lower()
        # Only set if your schema didn’t already include flags
        if "is_bb" not in e.columns:
            e["bb"] = et.isin(["walk", "intent_walk", "hit_by_pitch"]).astype(int)
        if "is_hr" not in e.columns:
            e["hr"] = et.eq("home_run").astype(int)
        if "is_hit" not in e.columns:
            e["h"] = et.isin(["single", "double", "triple", "home_run"]).astype(int)
        if "is_so" not in e.columns:
            e["so"] = et.isin(["strikeout", "strikeout_double_play"]).astype(int)

    out = (
        e.groupby(["game_pk", batter_col], as_index=False)
        .agg(pa=("pa", "sum"), h=("h", "sum"), hr=("hr", "sum"), bb=("bb", "sum"), so=("so", "sum"), rbi=("rbi", "sum"))
        .rename(columns={batter_col: "batter_id"})
    )

    # Bring season/game_date for rolling windows
    if not games.empty and "game_pk" in games.columns:
        g = games[["game_pk", "season", "game_date"]].copy()
        g["game_date"] = _normalize_game_date(g, "game_date")
        out = out.merge(g, on="game_pk", how="left")

    return out


def build_pitcher_game(events_pa: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Pitcher-game aggregation. Backbone table for pitcher props.
    """
    if events_pa.empty:
        return pd.DataFrame(columns=["game_pk", "pitcher_id", "bf", "so", "bb", "hr", "outs_on_play"])

    e = events_pa.copy()

    pitcher_col = "pitcher_id" if "pitcher_id" in e.columns else ("pitcher" if "pitcher" in e.columns else None)
    if pitcher_col is None:
        return pd.DataFrame(columns=["game_pk", "pitcher_id", "bf", "so", "bb", "hr", "outs_on_play"])

    def _col_or_zero(name: str) -> pd.Series:
        return pd.to_numeric(e[name], errors="coerce").fillna(0) if name in e.columns else pd.Series([0] * len(e))

    e["bf"] = 1
    e["outs_on_play"] = _col_or_zero("outs_on_play")

    e["so"] = _col_or_zero("is_so")
    e["bb"] = _col_or_zero("is_bb")
    e["hr"] = _col_or_zero("is_hr")

    if "event_type" in e.columns:
        et = e["event_type"].astype(str).str.lower()
        if "is_bb" not in e.columns:
            e["bb"] = et.isin(["walk", "intent_walk", "hit_by_pitch"]).astype(int)
        if "is_hr" not in e.columns:
            e["hr"] = et.eq("home_run").astype(int)
        if "is_so" not in e.columns:
            e["so"] = et.isin(["strikeout", "strikeout_double_play"]).astype(int)

    out = (
        e.groupby(["game_pk", pitcher_col], as_index=False)
        .agg(bf=("bf", "sum"), so=("so", "sum"), bb=("bb", "sum"), hr=("hr", "sum"), outs_on_play=("outs_on_play", "sum"))
        .rename(columns={pitcher_col: "pitcher_id"})
    )

    if not games.empty and "game_pk" in games.columns:
        g = games[["game_pk", "season", "game_date"]].copy()
        g["game_date"] = _normalize_game_date(g, "game_date")
        out = out.merge(g, on="game_pk", how="left")

    return out


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    args = parse_args()
    start = _validate_date(args.start, "--start")
    end = _validate_date(args.end, "--end")

    if start and end and start > end:
        raise ValueError("--start must be <= --end")

    config = load_config()
    ensure_drive_mounted()
    dirs = resolve_data_dirs(config)
    processed: Path = dirs["processed"]

    # Inputs
    games_path = processed / "games.parquet"
    events_path = processed / "events_pa.parquet"
    runs_path = processed / "game_runs.parquet"

    if not games_path.exists():
        raise FileNotFoundError(f"Missing required input: {games_path}")
    if not runs_path.exists():
        raise FileNotFoundError(
            f"Missing required input: {runs_path}\n"
            f"Run: python scripts/build_events.py --season {args.season} --start ... --end ... --max_games ..."
        )

    games = read_parquet(games_path)
    games = _maybe_slice_games(games, start, end)

    # Events are optional for batter/pitcher-game backbone (but recommended)
    events_pa = read_parquet(events_path) if events_path.exists() else pd.DataFrame()

    # game_runs is required for model_spine_game/team_game
    game_runs = read_parquet(runs_path)

    # If slice is used, align game_runs to those games
    if not games.empty and "game_pk" in games.columns:
        game_runs = game_runs[game_runs["game_pk"].isin(games["game_pk"])].reset_index(drop=True)

    # Outputs
    out_game_runs = processed / "game_runs.parquet"
    out_model_spine = processed / "model_spine_game.parquet"
    out_team_game = processed / "team_game.parquet"
    out_batter_game = processed / "batter_game.parquet"
    out_pitcher_game = processed / "pitcher_game.parquet"

    for p in [out_game_runs, out_model_spine, out_team_game, out_batter_game, out_pitcher_game]:
        _check_overwrite(p, args.force)

    # Write normalized game_runs (pass-through)
    write_parquet(game_runs, out_game_runs)

    # Build marts
    model_spine_game = build_model_spine_game(games, game_runs).sort_values(["game_pk"], kind="mergesort")
    if model_spine_game.empty:
        raise ValueError("model_spine_game is empty after merge. Check date range / inputs.")

    team_game = build_team_game(model_spine_game).sort_values(["game_pk", "team_side"], kind="mergesort")

    batter_game = build_batter_game(events_pa, games).sort_values(["game_pk", "batter_id"], kind="mergesort") if not events_pa.empty else pd.DataFrame()
    pitcher_game = build_pitcher_game(events_pa, games).sort_values(["game_pk", "pitcher_id"], kind="mergesort") if not events_pa.empty else pd.DataFrame()

    # Write outputs
    write_parquet(model_spine_game, out_model_spine)
    write_parquet(team_game, out_team_game)

    # Always write these files so rolling features can run (even if empty)
    if batter_game is None or batter_game.empty:
        batter_game = pd.DataFrame(columns=["game_pk", "season", "game_date", "batter_id", "pa", "h", "hr", "bb", "so", "rbi"])
    if pitcher_game is None or pitcher_game.empty:
        pitcher_game = pd.DataFrame(columns=["game_pk", "season", "game_date", "pitcher_id", "bf", "so", "bb", "hr", "outs_on_play"])

    write_parquet(batter_game, out_batter_game)
    write_parquet(pitcher_game, out_pitcher_game)

    print(f"model_spine_game rows: {len(model_spine_game)}")
    print(f"team_game rows: {len(team_game)}")
    print(f"batter_game rows: {len(batter_game)}")
    print(f"pitcher_game rows: {len(pitcher_game)}")
    print(f"Wrote marts to: {processed}")


if __name__ == "__main__":
    main()
