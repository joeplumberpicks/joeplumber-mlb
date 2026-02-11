#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from tqdm import tqdm

from src.utils.drive import resolve_data_dirs
from src.utils.io import load_config, read_parquet, write_parquet
from src.utils.mlb_api import get_game_feed, safe_json_get


EVENT_MAP = {
    "single": "single",
    "double": "double",
    "triple": "triple",
    "home_run": "home_run",
    "walk": "walk",
    "intent_walk": "intent_walk",
    "hit_by_pitch": "hit_by_pitch",
    "strikeout": "strikeout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build event-level MLB tables from game feeds.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--max_games", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalize_event_type(event_type: str | None) -> str:
    if event_type is None:
        return "unknown"
    return EVENT_MAP.get(event_type, event_type)


def filter_games(games: pd.DataFrame, season: int, start: str | None, end: str | None, max_games: int | None) -> pd.DataFrame:
    out = games.loc[games["season"] == season].copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if start:
        out = out.loc[out["game_date"] >= pd.to_datetime(start)]
    if end:
        out = out.loc[out["game_date"] <= pd.to_datetime(end)]

    out = out.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    if max_games is not None:
        out = out.head(max_games).copy()
    return out


def parse_events_for_game(game_row: pd.Series, feed: dict) -> list[dict]:
    plays = safe_json_get(feed, ["liveData", "plays", "allPlays"], default=[]) or []
    events: list[dict] = []

    for play in plays:
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        result = play.get("result", {})
        count = play.get("count", {})

        at_bat_index = about.get("atBatIndex")
        if at_bat_index is None:
            continue

        half = (about.get("halfInning") or "").lower()
        inning = about.get("inning")
        outs_before = count.get("outs")

        outs_on_play = result.get("outs")
        if outs_on_play is None and outs_before is not None:
            outs_after = about.get("outs")
            if outs_after is not None:
                outs_on_play = max(0, int(outs_after) - int(outs_before))

        batting_team_id = (
            game_row["away_team_id"] if half == "top" else game_row["home_team_id"]
        )
        fielding_team_id = (
            game_row["home_team_id"] if half == "top" else game_row["away_team_id"]
        )

        event_type_raw = result.get("eventType")
        event_type = normalize_event_type(event_type_raw)

        rbi_val = result.get("rbi")
        is_rbi = pd.NA if rbi_val is None else int(int(rbi_val) > 0)

        row = {
            "game_pk": int(game_row["game_pk"]),
            "game_date": pd.to_datetime(game_row["game_date"]).date(),
            "season": int(game_row["season"]),
            "home_team_id": int(game_row["home_team_id"]),
            "away_team_id": int(game_row["away_team_id"]),
            "batting_team_id": int(batting_team_id),
            "fielding_team_id": int(fielding_team_id),
            "inning": int(inning) if inning is not None else pd.NA,
            "inning_half": half,
            "outs_before": int(outs_before) if outs_before is not None else pd.NA,
            "outs_on_play": int(outs_on_play) if outs_on_play is not None else 0,
            "runs_on_play": int(result.get("rbi", 0) or 0),
            "batter_id": matchup.get("batter", {}).get("id"),
            "pitcher_id": matchup.get("pitcher", {}).get("id"),
            "event_type": event_type,
            "is_hit": int(event_type in {"single", "double", "triple", "home_run"}),
            "is_1b": int(event_type == "single"),
            "is_2b": int(event_type == "double"),
            "is_3b": int(event_type == "triple"),
            "is_hr": int(event_type == "home_run"),
            "is_bb": int(event_type in {"walk", "intent_walk"}),
            "is_hbp": int(event_type == "hit_by_pitch"),
            "is_so": int(event_type == "strikeout"),
            "is_rbi": is_rbi,
            "pa_id": f"{int(game_row['game_pk'])}_{int(at_bat_index)}",
        }
        events.append(row)

    return events


def build_game_runs(games: pd.DataFrame, events: pd.DataFrame, feed_cache: dict[int, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for _, game in games.iterrows():
        game_pk = int(game["game_pk"])
        feed = feed_cache.get(game_pk, {})

        line = safe_json_get(feed, ["liveData", "linescore", "teams"], default={}) or {}
        home_runs_final = safe_json_get(line, ["home", "runs"], default=None)
        away_runs_final = safe_json_get(line, ["away", "runs"], default=None)

        innings = safe_json_get(feed, ["liveData", "linescore", "innings"], default=[]) or []
        home_runs_1st = None
        away_runs_1st = None
        if innings:
            first = innings[0]
            home_runs_1st = safe_json_get(first, ["home", "runs"], default=None)
            away_runs_1st = safe_json_get(first, ["away", "runs"], default=None)

        if home_runs_1st is None or away_runs_1st is None:
            g1 = events.loc[(events["game_pk"] == game_pk) & (events["inning"] == 1)]
            away_runs_1st = int(g1.loc[g1["inning_half"] == "top", "runs_on_play"].sum())
            home_runs_1st = int(g1.loc[g1["inning_half"] == "bottom", "runs_on_play"].sum())

        if home_runs_final is None or away_runs_final is None:
            ge = events.loc[events["game_pk"] == game_pk]
            away_runs_final = int(ge.loc[ge["batting_team_id"] == int(game["away_team_id"]), "runs_on_play"].sum())
            home_runs_final = int(ge.loc[ge["batting_team_id"] == int(game["home_team_id"]), "runs_on_play"].sum())

        rows.append(
            {
                "game_pk": game_pk,
                "season": int(game["season"]),
                "game_date": pd.to_datetime(game["game_date"]).date(),
                "home_runs_final": int(home_runs_final),
                "away_runs_final": int(away_runs_final),
                "total_runs_final": int(home_runs_final) + int(away_runs_final),
                "home_runs_1st": int(home_runs_1st),
                "away_runs_1st": int(away_runs_1st),
                "total_runs_1st": int(home_runs_1st) + int(away_runs_1st),
            }
        )

    return pd.DataFrame(rows)


def validate_events(events: pd.DataFrame) -> None:
    required = [
        "game_pk", "game_date", "season", "home_team_id", "away_team_id", "batting_team_id",
        "fielding_team_id", "inning", "inning_half", "outs_before", "outs_on_play", "runs_on_play",
        "batter_id", "pitcher_id", "event_type", "is_hit", "is_1b", "is_2b", "is_3b", "is_hr",
        "is_bb", "is_hbp", "is_so", "is_rbi", "pa_id",
    ]
    missing = set(required) - set(events.columns)
    if missing:
        raise ValueError(f"events_pa missing columns: {sorted(missing)}")

    if events["pa_id"].duplicated().any():
        raise ValueError("events_pa duplicate pa_id found after dedupe")


def validate_game_runs(game_runs: pd.DataFrame) -> None:
    required = [
        "game_pk", "season", "game_date", "home_runs_final", "away_runs_final", "total_runs_final",
        "home_runs_1st", "away_runs_1st", "total_runs_1st",
    ]
    missing = set(required) - set(game_runs.columns)
    if missing:
        raise ValueError(f"game_runs missing columns: {sorted(missing)}")

    if game_runs["game_pk"].duplicated().any():
        raise ValueError("game_runs duplicate game_pk found after dedupe")


def main() -> None:
    args = parse_args()
    config = load_config()
    dirs = resolve_data_dirs(config)
    processed_dir = dirs["processed"]

    events_path = processed_dir / "events_pa.parquet"
    game_runs_path = processed_dir / "game_runs.parquet"
    if (events_path.exists() or game_runs_path.exists()) and not args.force:
        raise FileExistsError("Output parquet exists. Re-run with --force to overwrite.")

    games = read_parquet(processed_dir / "games.parquet")
    games = filter_games(games, args.season, args.start, args.end, args.max_games)
    if games.empty:
        raise ValueError("No games matched filters. Build spine first or adjust season/date filters.")

    all_events: list[dict] = []
    feed_cache: dict[int, dict] = {}

    for _, game in tqdm(games.iterrows(), total=len(games), desc="Fetching game feeds"):
        game_pk = int(game["game_pk"])
        feed = get_game_feed(game_pk)
        feed_cache[game_pk] = feed
        all_events.extend(parse_events_for_game(game, feed))

    events = pd.DataFrame(all_events)
    if events.empty:
        events = pd.DataFrame(
            columns=[
                "game_pk", "game_date", "season", "home_team_id", "away_team_id", "batting_team_id",
                "fielding_team_id", "inning", "inning_half", "outs_before", "outs_on_play", "runs_on_play",
                "batter_id", "pitcher_id", "event_type", "is_hit", "is_1b", "is_2b", "is_3b", "is_hr",
                "is_bb", "is_hbp", "is_so", "is_rbi", "pa_id",
            ]
        )

    if not events.empty:
        events = events.sort_values(["pa_id", "game_date", "inning", "outs_before"], kind="mergesort")
        events = events.drop_duplicates(subset=["pa_id"], keep="first").reset_index(drop=True)

    game_runs = build_game_runs(games, events, feed_cache)
    if not game_runs.empty:
        game_runs = game_runs.sort_values(["game_pk", "game_date"], kind="mergesort")
        game_runs = game_runs.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    validate_events(events)
    validate_game_runs(game_runs)

    print(f"events_pa rows: {len(events)}")
    print(f"game_runs rows: {len(game_runs)}")

    write_parquet(events, events_path)
    write_parquet(game_runs, game_runs_path)
    print(f"Wrote: {events_path}")
    print(f"Wrote: {game_runs_path}")


if __name__ == "__main__":
    main()
