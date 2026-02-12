#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.io import load_config, read_parquet, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical model_spine_game parquet with starter IDs.")
    parser.add_argument("--season", type=int, required=True, help="MLB season year, e.g. 2024")
    parser.add_argument("--start", type=str, default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Optional end date YYYY-MM-DD")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path (default: data/processed/model_spine_game.parquet)",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _resolve_processed_dir(config: dict) -> Path:
    return REPO_ROOT / config.get("paths", {}).get("processed", "data/processed")


def _filter_dates(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    if start:
        out = out[out["game_date"] >= pd.to_datetime(start).normalize()]
    if end:
        out = out[out["game_date"] <= pd.to_datetime(end).normalize()]
    return out.reset_index(drop=True)


def _derive_starters_from_existing_spine(spine_path: Path) -> pd.DataFrame | None:
    if not spine_path.exists():
        return None

    existing = read_parquet(spine_path)
    required = {
        "game_pk",
        "game_date",
        "home_team_name",
        "away_team_name",
        "home_sp_id",
        "away_sp_id",
    }
    if not required.issubset(existing.columns):
        return None

    out = existing[
        ["game_pk", "game_date", "home_team_name", "away_team_name", "home_sp_id", "away_sp_id"]
    ].copy()
    out = out.rename(columns={"home_team_name": "home_team", "away_team_name": "away_team"})
    out["home_sp_id"] = pd.to_numeric(out["home_sp_id"], errors="coerce").astype("Int64")
    out["away_sp_id"] = pd.to_numeric(out["away_sp_id"], errors="coerce").astype("Int64")
    return out


def _derive_starters_from_events(games: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    expected_events_cols = {"game_pk", "pitcher_id", "fielding_team_id", "inning", "outs_before", "pa_id"}
    missing = expected_events_cols - set(events.columns)
    if missing:
        raise ValueError(
            "Cannot derive starter IDs: events_pa.parquet is missing columns "
            f"{sorted(missing)}. Expected columns: {sorted(expected_events_cols)}"
        )

    ev = events.copy()
    ev["game_pk"] = pd.to_numeric(ev["game_pk"], errors="coerce").astype("Int64")
    ev["pitcher_id"] = pd.to_numeric(ev["pitcher_id"], errors="coerce").astype("Int64")
    ev["fielding_team_id"] = pd.to_numeric(ev["fielding_team_id"], errors="coerce").astype("Int64")
    ev["inning"] = pd.to_numeric(ev["inning"], errors="coerce")
    ev["outs_before"] = pd.to_numeric(ev["outs_before"], errors="coerce")

    ev = ev.sort_values(["game_pk", "fielding_team_id", "inning", "outs_before", "pa_id"], kind="mergesort")
    starters = ev.dropna(subset=["game_pk", "fielding_team_id", "pitcher_id"]).drop_duplicates(
        subset=["game_pk", "fielding_team_id"], keep="first"
    )

    home = games[["game_pk", "home_team_id"]].copy()
    home["home_team_id"] = pd.to_numeric(home["home_team_id"], errors="coerce").astype("Int64")
    home = home.merge(
        starters[["game_pk", "fielding_team_id", "pitcher_id"]],
        left_on=["game_pk", "home_team_id"],
        right_on=["game_pk", "fielding_team_id"],
        how="left",
    )
    home = home.rename(columns={"pitcher_id": "home_sp_id"})[["game_pk", "home_sp_id"]]

    away = games[["game_pk", "away_team_id"]].copy()
    away["away_team_id"] = pd.to_numeric(away["away_team_id"], errors="coerce").astype("Int64")
    away = away.merge(
        starters[["game_pk", "fielding_team_id", "pitcher_id"]],
        left_on=["game_pk", "away_team_id"],
        right_on=["game_pk", "fielding_team_id"],
        how="left",
    )
    away = away.rename(columns={"pitcher_id": "away_sp_id"})[["game_pk", "away_sp_id"]]

    out = games.merge(home, on="game_pk", how="left").merge(away, on="game_pk", how="left")
    out["home_sp_id"] = pd.to_numeric(out["home_sp_id"], errors="coerce").astype("Int64")
    out["away_sp_id"] = pd.to_numeric(out["away_sp_id"], errors="coerce").astype("Int64")
    return out


def main() -> None:
    setup_logging()
    log = logging.getLogger("build_model_spine_game")
    args = parse_args()

    config = load_config()
    processed_dir = _resolve_processed_dir(config)

    games_path = processed_dir / "games.parquet"
    events_path = processed_dir / "events_pa.parquet"
    existing_spine_path = processed_dir / "model_spine_game.parquet"

    output_path = Path(args.output) if args.output else (processed_dir / "model_spine_game.parquet")

    log.info("resolved_games_path=%s", games_path)
    log.info("resolved_events_path=%s", events_path)
    log.info("resolved_existing_spine_path=%s", existing_spine_path)
    log.info("resolved_output_path=%s", output_path)

    if not games_path.exists():
        raise FileNotFoundError(f"Missing required input: {games_path}")

    games = read_parquet(games_path)
    required_game_cols = {"game_pk", "season", "game_date", "home_team_name", "away_team_name", "home_team_id", "away_team_id"}
    missing_game = required_game_cols - set(games.columns)
    if missing_game:
        raise ValueError(f"games.parquet missing columns: {sorted(missing_game)}")

    games = games.loc[pd.to_numeric(games["season"], errors="coerce") == int(args.season)].copy()
    games = games.rename(columns={"home_team_name": "home_team", "away_team_name": "away_team"})
    games = _filter_dates(games, args.start, args.end)

    log.info("games_loaded=%d", len(games))
    if games.empty:
        raise ValueError("No games found for requested season/date filters")

    spine_base = games[["game_pk", "game_date", "home_team", "away_team", "home_team_id", "away_team_id"]].copy()

    from_existing = _derive_starters_from_existing_spine(existing_spine_path)
    if from_existing is not None:
        merged = spine_base.merge(from_existing[["game_pk", "home_sp_id", "away_sp_id"]], on="game_pk", how="left")
    else:
        if not events_path.exists():
            raise FileNotFoundError(
                "Cannot derive starter IDs. Missing events_pa.parquet and existing model_spine_game with starter columns. "
                f"Expected one of: {events_path} or {existing_spine_path} containing home_sp_id/away_sp_id"
            )
        events = read_parquet(events_path)
        events = events.loc[pd.to_numeric(events.get("season"), errors="coerce") == int(args.season)].copy() if "season" in events.columns else events
        merged = _derive_starters_from_events(spine_base, events)

    out = merged[["game_date", "game_pk", "home_team", "away_team", "home_sp_id", "away_sp_id"]].copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    missing_home_pct = float(out["home_sp_id"].isna().mean() * 100.0) if len(out) else 0.0
    missing_away_pct = float(out["away_sp_id"].isna().mean() * 100.0) if len(out) else 0.0

    write_parquet(out, output_path)

    log.info("games_written=%d", len(out))
    log.info("pct_missing_home_sp_id=%.2f", missing_home_pct)
    log.info("pct_missing_away_sp_id=%.2f", missing_away_pct)


if __name__ == "__main__":
    main()
