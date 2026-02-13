from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.paths import processed_dir, reference_dir
from src.weather.stadiums import load_park_overrides, load_stadium_reference, normalize_team_abbr, resolve_park_for_game

MLB_TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Athletics": "ATH",
    "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


HOME_TEAM_CANDIDATES = ["home_team", "home_team_name", "home_name"]
AWAY_TEAM_CANDIDATES = ["away_team", "away_team_name", "away_name"]



def _canonicalize_team_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    mapped = MLB_TEAM_NAME_TO_ABBR.get(text)
    if mapped is not None:
        text = mapped
    return normalize_team_abbr(text.upper())



def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((c for c in candidates if c in df.columns), None)



def _prepare_spine(spine: pd.DataFrame) -> pd.DataFrame:
    out = spine.copy()
    date_col = _first_existing_column(out, ["game_date", "date", "game_dt"])
    gpk_col = _first_existing_column(out, ["game_pk", "game_id", "mlb_game_id"])
    home_col = _first_existing_column(out, HOME_TEAM_CANDIDATES)
    away_col = _first_existing_column(out, AWAY_TEAM_CANDIDATES)

    if date_col is not None and date_col != "game_date":
        out = out.rename(columns={date_col: "game_date"})
    if gpk_col is not None and gpk_col != "game_pk":
        out = out.rename(columns={gpk_col: "game_pk"})
    if home_col is not None and home_col != "home_team":
        out = out.rename(columns={home_col: "home_team"})
    if away_col is not None and away_col != "away_team":
        out = out.rename(columns={away_col: "away_team"})

    if "game_pk" not in out.columns:
        return pd.DataFrame(columns=["game_pk", "game_date", "home_team", "away_team"])

    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce").dt.normalize()
    out["home_team"] = out.get("home_team", pd.Series(pd.NA, index=out.index)).map(_canonicalize_team_value)
    out["away_team"] = out.get("away_team", pd.Series(pd.NA, index=out.index)).map(_canonicalize_team_value)
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first")
    return out[["game_pk", "game_date", "home_team", "away_team"]].reset_index(drop=True)



def _resolve_games(games: pd.DataFrame, season: int, spine: pd.DataFrame | None = None) -> pd.DataFrame:
    out = games.copy()
    date_col = _first_existing_column(out, ["game_date", "date", "game_dt"])
    gpk_col = _first_existing_column(out, ["game_pk", "game_id", "mlb_game_id"])
    home_col = _first_existing_column(out, HOME_TEAM_CANDIDATES)
    away_col = _first_existing_column(out, AWAY_TEAM_CANDIDATES)

    if date_col is not None and date_col != "game_date":
        out = out.rename(columns={date_col: "game_date"})
    if gpk_col is not None and gpk_col != "game_pk":
        out = out.rename(columns={gpk_col: "game_pk"})
    if home_col is not None and home_col != "home_team":
        out = out.rename(columns={home_col: "home_team"})
    if away_col is not None and away_col != "away_team":
        out = out.rename(columns={away_col: "away_team"})

    if home_col is None:
        out["home_team"] = pd.NA
    if away_col is None:
        out["away_team"] = pd.NA

    if "game_pk" not in out.columns:
        raise ValueError("games parquet missing required column: game_pk")

    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce").dt.normalize()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["home_team"] = out["home_team"].map(_canonicalize_team_value)
    out["away_team"] = out["away_team"].map(_canonicalize_team_value)

    if spine is not None:
        sp = _prepare_spine(spine)
        out = out.merge(sp, on=["game_pk"], how="left", suffixes=("", "_sp"))
        if "home_team_sp" in out.columns:
            out["home_team"] = out["home_team"].fillna(out["home_team_sp"])
        if "away_team_sp" in out.columns:
            out["away_team"] = out["away_team"].fillna(out["away_team_sp"])

    if out["home_team"].isna().all():
        raise ValueError(f"games parquet missing all home team candidates: {HOME_TEAM_CANDIDATES}")
    if out["away_team"].isna().all():
        raise ValueError(f"games parquet missing all away team candidates: {AWAY_TEAM_CANDIDATES}")

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
    games_path: Path = processed_dir() / "games.parquet",
    stadiums_path: Path = reference_dir() / "mlb_stadiums.csv",
    overrides_path: Path = reference_dir() / "park_overrides.csv",
    output_path: Path = processed_dir() / "park_game.parquet",
    allow_partial: bool = False,
    max_games: int | None = None,
    spine_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build deterministic park reference table per game_pk."""
    log = logger or logging.getLogger(__name__)
    if not games_path.exists():
        raise FileNotFoundError(f"Missing games parquet: {games_path}")

    if spine_path is None:
        spine_path = processed_dir() / "model_spine_game.parquet"
    spine_df = pd.read_parquet(spine_path, engine="pyarrow") if spine_path.exists() else None

    games = pd.read_parquet(games_path, engine="pyarrow")
    games = _resolve_games(games, season=season, spine=spine_df)
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
            "home_team",
            "away_team",
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
