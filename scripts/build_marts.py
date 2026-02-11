#!/usr/bin/env python3
"""
Build marts from processed spine + events.

Writes to /data/processed (Drive-rooted via config/project.yaml):
- model_spine_game.parquet
- team_game.parquet
- batter_game.parquet
- pitcher_game.parquet

Requires:
- games.parquet (from build_spine.py)
- events_pa.parquet (from build_events.py)
Optional:
- game_runs.parquet (from build_events.py) – will be used if present
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# --- Make `src.*` imports work when executed as a script in Colab ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.drive import ensure_drive_mounted, resolve_data_dirs
from src.utils.io import load_config, read_parquet, write_parquet


# -------------------------
# Helpers
# -------------------------
def _as_date_norm(s: pd.Series) -> pd.Series:
    """Datetime normalize to midnight; coercing errors to NaT."""
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _maybe_overwrite(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path.name} exists. Re-run with --force to overwrite.")


# -------------------------
# Core marts builders
# -------------------------
def build_model_spine_game(games: pd.DataFrame, game_runs: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge spine games with run outcomes (final/1st) if available.
    Ensures `home_runs`, `away_runs`, `total_runs` exist for downstream team_game.
    """
    games = games.copy()
    games["game_date"] = _as_date_norm(games["game_date"])

    spine = games

    if game_runs is not None and not game_runs.empty:
        gr = game_runs.copy()
        if "game_date" in gr.columns:
            gr["game_date"] = _as_date_norm(gr["game_date"])
        else:
            # allow merge on game_pk only if no date available
            pass

        # Normalize merge keys for stability
        merge_keys = ["game_pk"]
        if "season" in spine.columns and "season" in gr.columns:
            merge_keys.append("season")
        if "game_date" in spine.columns and "game_date" in gr.columns:
            merge_keys.append("game_date")

        spine = spine.merge(gr, on=merge_keys, how="left")

    # Ensure canonical run columns exist (what team_game expects)
    # Prefer final columns if present
    if "home_runs_final" in spine.columns and "away_runs_final" in spine.columns:
        spine["home_runs"] = spine["home_runs_final"]
        spine["away_runs"] = spine["away_runs_final"]
        spine["total_runs"] = spine.get("total_runs_final", spine["home_runs"] + spine["away_runs"])
    else:
        # If no run info exists, keep placeholders (still lets pipeline run)
        spine["home_runs"] = spine.get("home_runs", pd.NA)
        spine["away_runs"] = spine.get("away_runs", pd.NA)
        spine["total_runs"] = spine.get("total_runs", pd.NA)

    return spine


def build_team_game(model_spine_game: pd.DataFrame) -> pd.DataFrame:
    """
    Two rows per game: home + away team. Requires home_runs/away_runs.
    """
    needed = [
        "game_pk",
        "season",
        "game_date",
        "home_team_id",
        "home_team_name",
        "away_team_id",
        "away_team_name",
        "home_runs",
        "away_runs",
        "total_runs",
    ]
    _require(model_spine_game, needed, "model_spine_game")

    g = model_spine_game.copy()

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
            "runs_scored": g["home_runs"],
            "runs_allowed": g["away_runs"],
            "total_runs": g["total_runs"],
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
            "runs_scored": g["away_runs"],
            "runs_allowed": g["home_runs"],
            "total_runs": g["total_runs"],
        }
    )

    out = pd.concat([home, away], ignore_index=True)
    return out


def build_batter_game(events_pa: pd.DataFrame) -> pd.DataFrame:
    """
    Batter-game aggregation from PA-level events.
    Creates a robust minimal schema for rolling features.
    """
    batter_col = _safe_col(events_pa, ["batter_id", "batter", "batterId"])
    game_col = _safe_col(events_pa, ["game_pk", "gamePk", "game_id"])
    if batter_col is None or game_col is None:
        raise ValueError("events_pa must include game_pk and batter_id columns (or recognizable variants).")

    df = events_pa.copy()
    df.rename(columns={game_col: "game_pk", batter_col: "batter_id"}, inplace=True)

    # event type column (optional but helpful)
    ev_col = _safe_col(df, ["event_type", "event", "eventType", "play_event"])
    if ev_col is None:
        df["event_type_norm"] = ""
    else:
        df["event_type_norm"] = df[ev_col].astype(str).str.lower()

    # Define common flags (best-effort; works even if event strings vary)
    et = df["event_type_norm"]
    df["is_pa"] = 1
    df["is_bb"] = et.str.contains("walk")
    df["is_so"] = et.str.contains("strikeout")
    df["is_hbp"] = et.str.contains("hit_by_pitch|hbp")
    df["is_single"] = et.str.contains("single")
    df["is_double"] = et.str.contains("double")
    df["is_triple"] = et.str.contains("triple")
    df["is_hr"] = et.str.contains("home_run|homered|home run|hr")

    # Hits & TB (approx)
    df["is_hit"] = df["is_single"] | df["is_double"] | df["is_triple"] | df["is_hr"]
    df["tb"] = (
        df["is_single"].astype(int) * 1
        + df["is_double"].astype(int) * 2
        + df["is_triple"].astype(int) * 3
        + df["is_hr"].astype(int) * 4
    )

    # AB proxy: count PA that are not BB/HBP (best-effort)
    df["is_ab"] = (~df["is_bb"] & ~df["is_hbp"]).astype(int)

    agg = (
        df.groupby(["game_pk", "batter_id"], as_index=False)
        .agg(
            pa=("is_pa", "sum"),
            ab=("is_ab", "sum"),
            h=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            bb=("is_bb", "sum"),
            so=("is_so", "sum"),
            hbp=("is_hbp", "sum"),
            tb=("tb", "sum"),
        )
        .sort_values(["game_pk", "batter_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return agg


def build_pitcher_game(events_pa: pd.DataFrame) -> pd.DataFrame:
    """
    Pitcher-game aggregation from PA-level events.
    Minimal schema for rolling pitcher features.
    """
    pitcher_col = _safe_col(events_pa, ["pitcher_id", "pitcher", "pitcherId"])
    game_col = _safe_col(events_pa, ["game_pk", "gamePk", "game_id"])
    if pitcher_col is None or game_col is None:
        raise ValueError("events_pa must include game_pk and pitcher_id columns (or recognizable variants).")

    df = events_pa.copy()
    df.rename(columns={game_col: "game_pk", pitcher_col: "pitcher_id"}, inplace=True)

    ev_col = _safe_col(df, ["event_type", "event", "eventType", "play_event"])
    if ev_col is None:
        df["event_type_norm"] = ""
    else:
        df["event_type_norm"] = df[ev_col].astype(str).str.lower()

    et = df["event_type_norm"]
    df["bf"] = 1
    df["so"] = et.str.contains("strikeout").astype(int)
    df["bb"] = et.str.contains("walk").astype(int)
    df["hbp"] = et.str.contains("hit_by_pitch|hbp").astype(int)
    df["hr"] = et.str.contains("home_run|homered|home run|hr").astype(int)
    df["h"] = (et.str.contains("single|double|triple|home_run|homered|home run|hr")).astype(int)

    agg = (
        df.groupby(["game_pk", "pitcher_id"], as_index=False)
        .agg(
            batters_faced=("bf", "sum"),
            h=("h", "sum"),
            hr=("hr", "sum"),
            bb=("bb", "sum"),
            so=("so", "sum"),
            hbp=("hbp", "sum"),
        )
        .sort_values(["game_pk", "pitcher_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return agg


# -------------------------
# CLI / main
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build marts (model spine + team + batter/pitcher game tables).")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--start", type=str, default=None, help="Optional YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="Optional YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="Overwrite existing parquet outputs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # validate date args
    if args.start:
        datetime.strptime(args.start, "%Y-%m-%d")
    if args.end:
        datetime.strptime(args.end, "%Y-%m-%d")
    if args.start and args.end and args.start > args.end:
        raise ValueError("--start must be <= --end")

    config = load_config()
    ensure_drive_mounted()
    dirs = resolve_data_dirs(config)
    processed = dirs["processed"]

    games = read_parquet(processed / "games.parquet")
    # filter by season and optional date window
    if "season" in games.columns:
        games = games[games["season"].astype(int) == int(args.season)].copy()

    if args.start or args.end:
        games = games.copy()
        games["game_date"] = _as_date_norm(games["game_date"])
        if args.start:
            games = games[games["game_date"] >= pd.to_datetime(args.start).normalize()]
        if args.end:
            games = games[games["game_date"] <= pd.to_datetime(args.end).normalize()]

    # events_pa is required for batter/pitcher game
    events_pa = read_parquet(processed / "events_pa.parquet")

    # game_runs optional
    game_runs_path = processed / "game_runs.parquet"
    game_runs = read_parquet(game_runs_path) if game_runs_path.exists() else None

    model_spine_game = build_model_spine_game(games, game_runs).sort_values(["game_pk"], kind="mergesort")
    team_game = build_team_game(model_spine_game).sort_values(["game_pk", "team_side"], kind="mergesort")

    batter_game = build_batter_game(events_pa)
    pitcher_game = build_pitcher_game(events_pa)

    # Output paths
    out_model = processed / "model_spine_game.parquet"
    out_team = processed / "team_game.parquet"
    out_batter = processed / "batter_game.parquet"
    out_pitcher = processed / "pitcher_game.parquet"

    for p in (out_model, out_team, out_batter, out_pitcher):
        _maybe_overwrite(p, args.force)

    write_parquet(model_spine_game, out_model)
    write_parquet(team_game, out_team)
    write_parquet(batter_game, out_batter)
    write_parquet(pitcher_game, out_pitcher)

    print(f"model_spine_game columns: {list(model_spine_game.columns)}")
    print(f"model_spine_game rows: {len(model_spine_game)}")
    print(f"team_game rows: {len(team_game)}")
    print(f"batter_game rows: {len(batter_game)}")
    print(f"pitcher_game rows: {len(pitcher_game)}")
    print(f"Wrote marts to: {processed}")


if __name__ == "__main__":
    main()
