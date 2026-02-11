#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.utils.drive import resolve_data_dirs
from src.utils.io import load_config, read_parquet, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aggregated marts from events.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--windows", type=str, default=None)
    return parser.parse_args()


def filter_games(games: pd.DataFrame, season: int, start: str | None, end: str | None) -> pd.DataFrame:
    out = games.loc[games["season"] == season].copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if start:
        out = out.loc[out["game_date"] >= pd.to_datetime(start)]
    if end:
        out = out.loc[out["game_date"] <= pd.to_datetime(end)]
    return out.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)


def build_batter_game(events: pd.DataFrame) -> pd.DataFrame:
    grouped = events.groupby(["game_pk", "batter_id"], dropna=False, as_index=False).agg(
        pa=("pa_id", "count"),
        hits=("is_hit", "sum"),
        **{
            "1b": ("is_1b", "sum"),
            "2b": ("is_2b", "sum"),
            "3b": ("is_3b", "sum"),
        },
        hr=("is_hr", "sum"),
        bb=("is_bb", "sum"),
        hbp=("is_hbp", "sum"),
        so=("is_so", "sum"),
        rbi=("is_rbi", lambda x: x.fillna(0).astype(int).sum()),
    )
    grouped["xbh"] = grouped["2b"] + grouped["3b"] + grouped["hr"]
    grouped["tb"] = grouped["1b"] + 2 * grouped["2b"] + 3 * grouped["3b"] + 4 * grouped["hr"]
    return grouped


def build_pitcher_game(events: pd.DataFrame) -> pd.DataFrame:
    grouped = events.groupby(["game_pk", "pitcher_id"], dropna=False, as_index=False).agg(
        batters_faced=("pa_id", "count"),
        so=("is_so", "sum"),
        bb=("is_bb", "sum"),
        hbp=("is_hbp", "sum"),
        hits_allowed=("is_hit", "sum"),
        hr_allowed=("is_hr", "sum"),
        runs_allowed=("runs_on_play", "sum"),
        outs_recorded=("outs_on_play", "sum"),
    )
    grouped["outs_recorded"] = grouped["outs_recorded"].clip(lower=0)
    return grouped


def build_team_game(events: pd.DataFrame, game_runs: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    offense = events.groupby(["game_pk", "batting_team_id"], as_index=False).agg(
        pa=("pa_id", "count"),
        hits=("is_hit", "sum"),
        hr=("is_hr", "sum"),
        bb=("is_bb", "sum"),
        so=("is_so", "sum"),
        **{"2b": ("is_2b", "sum"), "3b": ("is_3b", "sum")},
        tb=("is_1b", "sum"),
    )
    offense["xbh"] = offense["2b"] + offense["3b"] + offense["hr"]
    offense["tb"] = offense["tb"] + 2 * offense["2b"] + 3 * offense["3b"] + 4 * offense["hr"]
    offense = offense.rename(columns={"batting_team_id": "team_id"})

    game_team_runs = []
    for _, g in game_runs.iterrows():
        game_team_runs.append(
            {
                "game_pk": int(g["game_pk"]),
                "team_id": int(games.loc[games["game_pk"] == g["game_pk"], "home_team_id"].iloc[0]),
                "runs_scored": int(g["home_runs_final"]),
                "runs_allowed": int(g["away_runs_final"]),
            }
        )
        game_team_runs.append(
            {
                "game_pk": int(g["game_pk"]),
                "team_id": int(games.loc[games["game_pk"] == g["game_pk"], "away_team_id"].iloc[0]),
                "runs_scored": int(g["away_runs_final"]),
                "runs_allowed": int(g["home_runs_final"]),
            }
        )
    runs_df = pd.DataFrame(game_team_runs)

    team_game = runs_df.merge(offense, on=["game_pk", "team_id"], how="left")
    fill_zero = ["pa", "hits", "hr", "bb", "so", "2b", "3b", "xbh", "tb"]
    team_game[fill_zero] = team_game[fill_zero].fillna(0).astype(int)
    return team_game


def build_model_spine_game(games: pd.DataFrame, game_runs: pd.DataFrame) -> pd.DataFrame:
    spine = games.merge(game_runs, on=["game_pk", "season", "game_date"], how="inner")
    spine["y_nrfi"] = (spine["total_runs_1st"] == 0).astype(int)
    spine["y_yrfi"] = (spine["total_runs_1st"] > 0).astype(int)
    spine["home_win"] = (spine["home_runs_final"] > spine["away_runs_final"]).astype(int)
    spine["run_diff"] = spine["home_runs_final"] - spine["away_runs_final"]
    return spine


def validate_pk(df: pd.DataFrame, keys: list[str], name: str) -> None:
    if df.empty:
        return
    if df[keys].isna().any().any():
        raise ValueError(f"{name} has nulls in PK columns {keys}")
    if df.duplicated(subset=keys).any():
        raise ValueError(f"{name} has duplicate PK rows for {keys}")


def main() -> None:
    args = parse_args()
    config = load_config()
    dirs = resolve_data_dirs(config)
    processed = dirs["processed"]

    games = read_parquet(processed / "games.parquet")
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").dt.date
    games = filter_games(games, args.season, args.start, args.end)

    events = read_parquet(processed / "events_pa.parquet")
    events["game_date"] = pd.to_datetime(events["game_date"], errors="coerce").dt.date
    events = events.loc[events["game_pk"].isin(games["game_pk"])].copy()

    game_runs = read_parquet(processed / "game_runs.parquet")
    game_runs["game_date"] = pd.to_datetime(game_runs["game_date"], errors="coerce").dt.date
    game_runs = game_runs.loc[game_runs["game_pk"].isin(games["game_pk"])].copy()

    batter_game = build_batter_game(events).sort_values(["game_pk", "batter_id"], kind="mergesort")
    pitcher_game = build_pitcher_game(events).sort_values(["game_pk", "pitcher_id"], kind="mergesort")
    team_game = build_team_game(events, game_runs, games).sort_values(["game_pk", "team_id"], kind="mergesort")
    model_spine_game = build_model_spine_game(games, game_runs).sort_values(["game_pk"], kind="mergesort")

    batter_game = batter_game.drop_duplicates(subset=["game_pk", "batter_id"], keep="first").reset_index(drop=True)
    pitcher_game = pitcher_game.drop_duplicates(subset=["game_pk", "pitcher_id"], keep="first").reset_index(drop=True)
    team_game = team_game.drop_duplicates(subset=["game_pk", "team_id"], keep="first").reset_index(drop=True)
    model_spine_game = model_spine_game.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    validate_pk(batter_game, ["game_pk", "batter_id"], "batter_game")
    validate_pk(pitcher_game, ["game_pk", "pitcher_id"], "pitcher_game")
    validate_pk(team_game, ["game_pk", "team_id"], "team_game")
    validate_pk(model_spine_game, ["game_pk"], "model_spine_game")

    print(f"batter_game rows: {len(batter_game)}")
    print(f"pitcher_game rows: {len(pitcher_game)}")
    print(f"team_game rows: {len(team_game)}")
    print(f"model_spine_game rows: {len(model_spine_game)}")

    write_parquet(batter_game, processed / "batter_game.parquet")
    write_parquet(pitcher_game, processed / "pitcher_game.parquet")
    write_parquet(team_game, processed / "team_game.parquet")
    write_parquet(model_spine_game, processed / "model_spine_game.parquet")


if __name__ == "__main__":
    main()
