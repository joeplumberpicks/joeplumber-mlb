from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.weather.stadiums import load_park_overrides, load_stadium_reference, resolve_park_for_game


def _resolve_games(games: pd.DataFrame, season: int) -> pd.DataFrame:
    required = {"game_date", "game_pk", "home_team", "away_team"}
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"games parquet missing required columns: {sorted(missing)}")

    out = games.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["game_date", "game_pk", "home_team", "away_team"]).copy()
    out = out.loc[out["game_date"].dt.year == int(season)].copy()
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first")
    return out.reset_index(drop=True)


def _stadium_lookup_by_park_id(stadiums: pd.DataFrame) -> pd.DataFrame:
    s = stadiums.copy()
    s = s.sort_values(["season_start", "season_end"], ascending=[False, True], kind="mergesort")
    return s.drop_duplicates(subset=["park_id"], keep="first")


def build_park_game(
    season: int,
    start: str | None = None,
    end: str | None = None,
    games_path: Path = Path("data/processed/games.parquet"),
    stadiums_path: Path = Path("data/reference/mlb_stadiums.csv"),
    overrides_path: Path = Path("data/reference/park_overrides.csv"),
    output_path: Path = Path("data/processed/park_game.parquet"),
    allow_partial: bool = False,
    max_games: int | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build deterministic park reference table per game_pk."""
    log = logger or logging.getLogger(__name__)
    if not games_path.exists():
        raise FileNotFoundError(f"Missing games parquet: {games_path}")

    games = pd.read_parquet(games_path, engine="pyarrow")
    games = _resolve_games(games, season=season)
    if start:
        games = games.loc[games["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        games = games.loc[games["game_date"] <= pd.to_datetime(end).normalize()].copy()
    if max_games is not None:
        games = games.head(int(max_games)).copy()

    stadiums = load_stadium_reference(stadiums_path, logger=log)
    overrides = load_park_overrides(overrides_path)

    resolved = games.copy()
    park_ids: list[str | None] = []
    sources: list[str] = []
    for _, row in resolved.iterrows():
        park_id, source = resolve_park_for_game(row, int(row["game_date"].year), stadiums, overrides, logger=log)
        park_ids.append(park_id)
        sources.append(source)
    resolved["park_id"] = park_ids
    resolved["park_resolution_source"] = sources

    counts = resolved["park_resolution_source"].value_counts(dropna=False).to_dict()
    unresolved = resolved["park_id"].isna()
    unresolved_count = int(unresolved.sum())
    total = len(resolved)
    unresolved_rate = unresolved_count / total if total else 0.0

    log.info(
        "park_resolution total_games=%d override=%d explicit_venue=%d team_season=%d unresolved=%d",
        total,
        int(counts.get("override", 0)),
        int(counts.get("explicit_venue", 0)),
        int(counts.get("team_season", 0)),
        unresolved_count,
    )

    if unresolved_count > 0:
        sample = resolved.loc[unresolved, ["game_pk", "home_team", "game_date"]].head(10)
        log.warning("park_resolution_unresolved_sample:\n%s", sample.to_string(index=False))

    if unresolved_rate > 0.05 and not allow_partial:
        raise SystemExit(
            f"Unresolved park_id rate {unresolved_rate:.2%} exceeds 5%. "
            "Fix data/reference/mlb_stadiums.csv or add data/reference/park_overrides.csv entries."
        )

    lookup = _stadium_lookup_by_park_id(stadiums)
    out = resolved.merge(
        lookup[
            [
                "park_id",
                "stadium_name",
                "city",
                "state",
                "lat",
                "lon",
                "timezone",
                "roof_type",
                "cf_bearing_deg",
            ]
        ],
        on="park_id",
        how="left",
    )

    out = out[
        [
            "game_date",
            "game_pk",
            "park_id",
            "stadium_name",
            "city",
            "state",
            "lat",
            "lon",
            "timezone",
            "roof_type",
            "cf_bearing_deg",
            "park_resolution_source",
        ]
    ].copy()

    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_park_game season=%s rows=%d path=%s", season, len(out), output_path)
    return out
