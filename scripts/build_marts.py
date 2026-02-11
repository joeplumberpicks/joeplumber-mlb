#!/usr/bin/env python3
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

SIGNATURE = "✅ build_marts.py v2 (writes batter_game + pitcher_game)"


def _as_date_norm(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MLB marts.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--start", type=str, default=None, help="Optional YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="Optional YYYY-MM-DD")
    p.add_argument("--force", action="store_true", help="Overwrite existing parquet outputs.")
    return p.parse_args()


def _validate_date(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{label} must be YYYY-MM-DD; got {value!r}") from exc
    return value


def build_model_spine_game(games: pd.DataFrame, game_runs: pd.DataFrame | None) -> pd.DataFrame:
    games = games.copy()
    games["game_date"] = _as_date_norm(games["game_date"])

    spine = games

    if game_runs is not None and not game_runs.empty:
        gr = game_runs.copy()
        gr["game_date"] = _as_date_norm(gr["game_date"])

        spine = spine.merge(gr, on=["game_pk", "season", "game_date"], how="left")

    # Normalize run columns for downstream
    if "home_runs_final" in spine.columns and "away_runs_final" in spine.columns:
        spine["home_runs"] = spine["home_runs_final"]
        spine["away_runs"] = spine["away_runs_final"]
        spine["total_runs"] = spine.get("total_runs_final", spine["home_runs"] + spine["away_runs"])
    else:
        # fallback
        spine["home_runs"] = spine.get("home_runs", pd.NA)
        spine["away_runs"] = spine.get("away_runs", pd.NA)
        spine["total_runs"] = spine.get("total_runs", pd.NA)

    return spine


def build_team_game(model_spine_game: pd.DataFrame) -> pd.DataFrame:
    g = model_spine_game.copy()

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
        "total_runs",
    ]
    missing = [c for c in required if c not in g.columns]
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

    return pd.concat([home, away], ignore_index=True)


def build_batter_game(events_pa: pd.DataFrame) -> pd.DataFrame:
    df = events_pa.copy()
    df["game_date"] = _as_date_norm(df["game_date"])

    required = ["game_pk", "season", "game_date", "batter_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"events_pa missing required columns for batter_game: {missing}")

    # Ensure boolean/int columns exist
    for c in ["is_hit", "is_1b", "is_2b", "is_3b", "is_hr", "is_bb", "is_hbp", "is_so", "is_rbi", "runs_on_play"]:
        if c not in df.columns:
            df[c] = 0

    df["pa"] = 1
    df["ab"] = ((df["is_bb"] == 0) & (df["is_hbp"] == 0)).astype(int)
    df["h"] = df["is_hit"].astype(int)

    df["tb"] = (
        df["is_1b"].astype(int) * 1
        + df["is_2b"].astype(int) * 2
        + df["is_3b"].astype(int) * 3
        + df["is_hr"].astype(int) * 4
    )

    out = (
        df.groupby(["game_pk", "season", "game_date", "batter_id"], as_index=False)
        .agg(
            pa=("pa", "sum"),
            ab=("ab", "sum"),
            h=("h", "sum"),
            tb=("tb", "sum"),
            hr=("is_hr", "sum"),
            bb=("is_bb", "sum"),
            so=("is_so", "sum"),
            hbp=("is_hbp", "sum"),
            rbi=("is_rbi", "sum"),
            runs_on_play=("runs_on_play", "sum"),
        )
        .sort_values(["game_pk", "batter_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def build_pitcher_game(events_pa: pd.DataFrame) -> pd.DataFrame:
    df = events_pa.copy()
    df["game_date"] = _as_date_norm(df["game_date"])

    required = ["game_pk", "season", "game_date", "pitcher_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"events_pa missing required columns for pitcher_game: {missing}")

    for c in ["is_hit", "is_hr", "is_bb", "is_hbp", "is_so", "outs_on_play", "runs_on_play"]:
        if c not in df.columns:
            df[c] = 0

    df["bf"] = 1

    out = (
        df.groupby(["game_pk", "season", "game_date", "pitcher_id"], as_index=False)
        .agg(
            batters_faced=("bf", "sum"),
            h=("is_hit", "sum"),
            hr=("is_hr", "sum"),
            bb=("is_bb", "sum"),
            so=("is_so", "sum"),
            hbp=("is_hbp", "sum"),
            outs=("outs_on_play", "sum"),
            runs=("runs_on_play", "sum"),
        )
        .sort_values(["game_pk", "pitcher_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


def main() -> None:
    args = parse_args()
    print(SIGNATURE)

    start = _validate_date(args.start, "--start")
    end = _validate_date(args.end, "--end")
    if start and end and start > end:
        raise ValueError("--start must be <= --end")

    config = load_config()
    ensure_drive_mounted()
    dirs = resolve_data_dirs(config)
    processed = dirs["processed"]

    games = read_parquet(processed / "games.parquet").copy()
    games["game_date"] = _as_date_norm(games["game_date"])
    games = games[games["season"].astype(int) == int(args.season)]
    if start:
        games = games[games["game_date"] >= pd.to_datetime(start).normalize()]
    if end:
        games = games[games["game_date"] <= pd.to_datetime(end).normalize()]
    games = games.reset_index(drop=True)

    events_pa = read_parquet(processed / "events_pa.parquet").copy()
    events_pa["game_date"] = _as_date_norm(events_pa["game_date"])
    events_pa = events_pa[events_pa["season"].astype(int) == int(args.season)]
    if start:
        events_pa = events_pa[events_pa["game_date"] >= pd.to_datetime(start).normalize()]
    if end:
        events_pa = events_pa[events_pa["game_date"] <= pd.to_datetime(end).normalize()]
    events_pa = events_pa.reset_index(drop=True)

    game_runs_path = processed / "game_runs.parquet"
    game_runs = read_parquet(game_runs_path) if game_runs_path.exists() else None

    model_spine_game = build_model_spine_game(games, game_runs).sort_values(["game_pk"], kind="mergesort")
    team_game = build_team_game(model_spine_game).sort_values(["game_pk", "team_side"], kind="mergesort")
    batter_game = build_batter_game(events_pa)
    pitcher_game = build_pitcher_game(events_pa)

    out_model = processed / "model_spine_game.parquet"
    out_team = processed / "team_game.parquet"
    out_batter = processed / "batter_game.parquet"
    out_pitcher = processed / "pitcher_game.parquet"

    if args.force:
        for pth in (out_model, out_team, out_batter, out_pitcher):
            if pth.exists():
                pth.unlink()
    else:
        for pth in (out_model, out_team, out_batter, out_pitcher):
            if pth.exists():
                raise FileExistsError(f"{pth.name} exists. Re-run with --force to overwrite.")

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
