from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.parks.park_identity import load_park_overrides, resolve_park_for_game
from src.utils.checks import print_rowcount, require_columns
from src.utils.io import read_parquet, write_parquet
from src.utils.team_normalize import canonical_team_abbr

GAMES_COLUMNS = [
    "game_pk",
    "game_date",
    "home_team",
    "away_team",
    "home_sp_id",
    "away_sp_id",
    "park_id",
    "venue_id",
    "park_name",
    "canonical_park_key",
    "season",
]
PA_COLUMNS = ["game_pk", "pa_id", "batter_id", "pitcher_id", "event_type", "season"]
WEATHER_COLUMNS = ["game_pk", "temperature_f", "wind_mph", "wind_dir", "season"]
PARK_COLUMNS = ["park_id", "venue_id", "park_name", "lat", "lon", "roofType", "tz", "season"]
PITCHER_ID_CANDIDATES = ["pitcher", "pitcher_id", "mlbam_pitcher_id", "player_id"]
PITCHING_TEAM_CANDIDATES = ["pitching_team", "defense_team", "fielding_team"]
INNING_CANDIDATES = ["inning"]
INNING_HALF_CANDIDATES = ["inning_topbot", "topbot", "inning_half"]
PITCH_SEQUENCE_CANDIDATES = ["pitch_number", "pitch_num", "pitch_seq", "pitch_index"]


def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalize_pa_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.copy()
    if "batter_id" not in out.columns and "batter" in out.columns:
        out["batter_id"] = out["batter"]
    if "pitcher_id" not in out.columns and "pitcher" in out.columns:
        out["pitcher_id"] = out["pitcher"]
    if "pa_id" not in out.columns:
        if "at_bat_number" in out.columns and "game_pk" in out.columns:
            out["pa_id"] = out["game_pk"].astype(str) + "-" + out["at_bat_number"].astype(str)
        else:
            out["pa_id"] = pd.RangeIndex(start=0, stop=len(out), step=1).astype(str)
    if "season" not in out.columns:
        out["season"] = season
    if "event_type" not in out.columns and "events" in out.columns:
        out["event_type"] = out["events"]
    for col in PA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def _normalize_parks_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.copy()
    if "park_id" not in out.columns and "venue_id" in out.columns:
        out["park_id"] = out["venue_id"]
    if "venue_id" not in out.columns and "park_id" in out.columns:
        out["venue_id"] = out["park_id"]
    if "season" not in out.columns:
        out["season"] = season
    for col in PARK_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out




def _normalize_games_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.copy()

    if "season" not in out.columns:
        out["season"] = season

    # Raw games files may not yet have enriched identity columns.
    if "park_id" not in out.columns:
        out["park_id"] = pd.NA
    if "venue_id" not in out.columns:
        out["venue_id"] = out["park_id"]
    if "canonical_park_key" not in out.columns:
        out["canonical_park_key"] = pd.NA
    if "park_name" not in out.columns:
        out["park_name"] = pd.NA
    if "home_sp_id" not in out.columns:
        out["home_sp_id"] = pd.NA
    if "away_sp_id" not in out.columns:
        out["away_sp_id"] = pd.NA

    for col in GAMES_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    return out

def load_or_placeholder(raw_path: Path, columns: list[str], label: str, season: int) -> pd.DataFrame:
    if raw_path.exists():
        df = read_parquet(raw_path)
        if label == "games":
            df = _normalize_games_df(df, season)
            print_rowcount("games_normalized", df)
        if label == "plate_appearances":
            df = _normalize_pa_df(df, season)
            print_rowcount("plate_appearances_normalized", df)
        if label == "parks":
            df = _normalize_parks_df(df, season)
            print_rowcount("parks_normalized", df)
        require_columns(df, columns, label)
        print_rowcount(label, df)
        return df
    logging.warning("Missing raw input for %s season %s. Creating empty placeholder: %s", label, season, raw_path)
    df = _empty_df(columns)
    print_rowcount(label, df)
    return df


def _apply_overrides(games_df: pd.DataFrame, overrides_df: pd.DataFrame, season: int) -> pd.DataFrame:
    if overrides_df.empty:
        return games_df
    out = games_df.copy()
    for _, rule in overrides_df.iterrows():
        s0 = int(rule.get("season_start", season))
        s1 = int(rule.get("season_end", season))
        if not (s0 <= season <= s1):
            continue
        team = canonical_team_abbr(rule.get("team"), season)
        name_contains = str(rule.get("park_name_contains", "")).lower()
        mask = out["home_team"].astype(str).map(lambda x: canonical_team_abbr(x, season) == team)
        if name_contains and "park_name" in out.columns:
            mask &= out["park_name"].astype(str).str.lower().str.contains(name_contains, na=False)
        if pd.notna(rule.get("venue_id")):
            out.loc[mask & out["venue_id"].isna(), "venue_id"] = rule.get("venue_id")
        if pd.notna(rule.get("park_id_override")):
            out.loc[mask & out["park_id"].isna(), "park_id"] = rule.get("park_id_override")
    return out


def _enrich_games_with_park_identity(games_df: pd.DataFrame, parks_df: pd.DataFrame, reference_dir: Path, season: int) -> pd.DataFrame:
    out = games_df.copy()
    for c in ["park_id", "venue_id", "park_name"]:
        if c not in out.columns:
            out[c] = pd.NA

    overrides_df = load_park_overrides(reference_dir / "park_overrides.csv")
    out = _apply_overrides(out, overrides_df, season)

    parks_map = parks_df.copy()
    if "venue_id" not in parks_map.columns and "park_id" in parks_map.columns:
        parks_map["venue_id"] = parks_map["park_id"]

    park_rows = []
    for _, row in out.iterrows():
        resolved = resolve_park_for_game(row, parks_map)
        park_rows.append(resolved)
    park_resolved = pd.DataFrame(park_rows, index=out.index)

    for c in ["park_id", "venue_id", "park_name", "canonical_park_key"]:
        if c in park_resolved.columns:
            out.loc[out[c].isna() if c in out.columns else slice(None), c] = park_resolved[c]
            if c not in out.columns:
                out[c] = park_resolved[c]

    if "canonical_park_key" not in out.columns:
        out["canonical_park_key"] = park_resolved.get("canonical_park_key", pd.NA)

    return out


def build_spine_for_season(season: int, dirs: dict[str, Path], force: bool = False) -> dict[str, Path]:
    processed_by_season = dirs["processed_dir"] / "by_season"
    raw_by_season = dirs["raw_dir"] / "by_season"
    processed_by_season.mkdir(parents=True, exist_ok=True)

    raw_games = raw_by_season / f"games_{season}.parquet"
    raw_pa = raw_by_season / f"pa_{season}.parquet"
    raw_weather = raw_by_season / f"weather_game_{season}.parquet"
    raw_parks = raw_by_season / f"parks_{season}.parquet"

    out_games = processed_by_season / f"games_{season}.parquet"
    out_pa = processed_by_season / f"pa_{season}.parquet"
    out_weather = processed_by_season / f"weather_game_{season}.parquet"
    out_parks = processed_by_season / f"parks_{season}.parquet"

    if out_games.exists() and not force:
        logging.info("Season outputs already exist and force=False; reloading processed tables for season %s", season)
        games_df = read_parquet(out_games)
        pa_df = read_parquet(out_pa)
        weather_df = read_parquet(out_weather)
        parks_df = read_parquet(out_parks)
    else:
        games_df = load_or_placeholder(raw_games, GAMES_COLUMNS, "games", season)
        pa_df = load_or_placeholder(raw_pa, PA_COLUMNS, "plate_appearances", season)
        weather_df = load_or_placeholder(raw_weather, WEATHER_COLUMNS, "weather", season)
        parks_df = load_or_placeholder(raw_parks, PARK_COLUMNS, "parks", season)

        for df in [games_df, pa_df, weather_df, parks_df]:
            if "season" in df.columns:
                df["season"] = df["season"].fillna(season)

        games_df["home_team"] = games_df.get("home_team", pd.Series(index=games_df.index, dtype="object")).map(
            lambda x: canonical_team_abbr(x, season)
        )
        games_df["away_team"] = games_df.get("away_team", pd.Series(index=games_df.index, dtype="object")).map(
            lambda x: canonical_team_abbr(x, season)
        )

        games_df = _enrich_games_with_park_identity(games_df, parks_df, dirs["reference_dir"], season)
        pa_df = _normalize_pa_df(pa_df, season)
        print_rowcount("plate_appearances_processed", pa_df)

        print(f"Writing to: {out_games.resolve()}")
        write_parquet(games_df, out_games)
        print(f"Writing to: {out_pa.resolve()}")
        write_parquet(pa_df, out_pa)
        print(f"Writing to: {out_weather.resolve()}")
        write_parquet(weather_df, out_weather)
        print(f"Writing to: {out_parks.resolve()}")
        write_parquet(parks_df, out_parks)

    return {"games": out_games, "pa": out_pa, "weather": out_weather, "parks": out_parks}


def _pick_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _populate_starter_ids_from_events(model_spine: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    events_path = processed_dir / "events_pa.parquet"
    if not events_path.exists():
        logging.warning("events_pa.parquet not found at %s; starter IDs will remain as-is", events_path)
        return model_spine

    events = read_parquet(events_path)
    pitcher_col = _pick_optional_column(events, PITCHER_ID_CANDIDATES)
    pitching_team_col = _pick_optional_column(events, PITCHING_TEAM_CANDIDATES)
    inning_col = _pick_optional_column(events, INNING_CANDIDATES)
    inning_half_col = _pick_optional_column(events, INNING_HALF_CANDIDATES)

    if pitcher_col is None or pitching_team_col is None or inning_col is None or "game_pk" not in events.columns:
        logging.warning(
            "Cannot populate starters from events_pa due to missing columns. pitcher=%s pitching_team=%s inning=%s has_game_pk=%s",
            pitcher_col,
            pitching_team_col,
            inning_col,
            "game_pk" in events.columns,
        )
        return model_spine

    starter_events = events.copy()
    starter_events[pitcher_col] = pd.to_numeric(starter_events[pitcher_col], errors="coerce")
    starter_events[inning_col] = pd.to_numeric(starter_events[inning_col], errors="coerce")
    starter_events = starter_events[starter_events[inning_col] == 1]
    starter_events = starter_events.dropna(subset=[pitcher_col, pitching_team_col, "game_pk"])

    seq_col = _pick_optional_column(starter_events, PITCH_SEQUENCE_CANDIDATES)
    sort_cols = ["game_pk", pitching_team_col]
    if seq_col is not None:
        sort_cols.append(seq_col)
    elif inning_half_col is not None:
        sort_cols.append(inning_half_col)

    starter_events = starter_events.sort_values(sort_cols)
    starters = (
        starter_events.groupby(["game_pk", pitching_team_col], dropna=False)[pitcher_col]
        .first()
        .reset_index()
        .rename(columns={pitching_team_col: "pitching_team", pitcher_col: "starter_pitcher_id"})
    )

    out = model_spine.copy()
    if "home_sp_id" not in out.columns:
        out["home_sp_id"] = pd.NA
    if "away_sp_id" not in out.columns:
        out["away_sp_id"] = pd.NA

    home_map = out[["game_pk", "home_team"]].merge(
        starters, left_on=["game_pk", "home_team"], right_on=["game_pk", "pitching_team"], how="left"
    )
    away_map = out[["game_pk", "away_team"]].merge(
        starters, left_on=["game_pk", "away_team"], right_on=["game_pk", "pitching_team"], how="left"
    )

    home_vals = pd.to_numeric(home_map["starter_pitcher_id"], errors="coerce")
    away_vals = pd.to_numeric(away_map["starter_pitcher_id"], errors="coerce")

    home_missing = out["home_sp_id"].isna()
    away_missing = out["away_sp_id"].isna()
    out.loc[home_missing, "home_sp_id"] = home_vals[home_missing].values
    out.loc[away_missing, "away_sp_id"] = away_vals[away_missing].values

    home_null_pct = float(out["home_sp_id"].isna().mean() * 100) if len(out) else 0.0
    away_null_pct = float(out["away_sp_id"].isna().mean() * 100) if len(out) else 0.0
    logging.info("home_sp_id null %% after events starter mapping: %.2f%%", home_null_pct)
    logging.info("away_sp_id null %% after events starter mapping: %.2f%%", away_null_pct)

    return out


def _apply_park_venue_mapping(model_spine: pd.DataFrame, processed_by_season: Path, seasons: list[int]) -> pd.DataFrame:
    if "park_id" not in model_spine.columns:
        model_spine["park_id"] = pd.NA
    if "venue_id" not in model_spine.columns:
        model_spine["venue_id"] = pd.NA

    parks_frames: list[pd.DataFrame] = []
    for season in seasons:
        parks_path = processed_by_season / f"parks_{season}.parquet"
        if parks_path.exists():
            parks_df = _normalize_parks_df(read_parquet(parks_path), season)
            parks_frames.append(parks_df)
    if not parks_frames:
        return model_spine

    parks_all = pd.concat(parks_frames, ignore_index=True).drop_duplicates()
    venue_to_park = dict(parks_all.dropna(subset=["venue_id", "park_id"])[["venue_id", "park_id"]].values)
    park_to_venue = dict(parks_all.dropna(subset=["park_id", "venue_id"])[["park_id", "venue_id"]].values)

    park_missing = model_spine["park_id"].isna() & model_spine["venue_id"].notna()
    model_spine.loc[park_missing, "park_id"] = model_spine.loc[park_missing, "venue_id"].map(venue_to_park)

    venue_missing = model_spine["venue_id"].isna() & model_spine["park_id"].notna()
    model_spine.loc[venue_missing, "venue_id"] = model_spine.loc[venue_missing, "park_id"].map(park_to_venue)
    return model_spine


def build_model_spine(dirs: dict[str, Path], seasons: list[int]) -> Path:
    processed_by_season = dirs["processed_dir"] / "by_season"
    model_spine_path = dirs["processed_dir"] / "model_spine_game.parquet"

    frames: list[pd.DataFrame] = []
    for season in seasons:
        season_path = processed_by_season / f"games_{season}.parquet"
        if season_path.exists():
            df = read_parquet(season_path)
            if "season" not in df.columns:
                df["season"] = season
            frames.append(df)

    model_spine = pd.concat(frames, ignore_index=True) if frames else _empty_df(GAMES_COLUMNS)
    model_spine = _apply_park_venue_mapping(model_spine, processed_by_season, seasons)
    model_spine = _populate_starter_ids_from_events(model_spine, dirs["processed_dir"])

    keep_cols = [
        "game_pk",
        "game_date",
        "home_team",
        "away_team",
        "home_sp_id",
        "away_sp_id",
        "park_id",
        "venue_id",
        "park_name",
        "canonical_park_key",
        "season",
    ]
    for col in keep_cols:
        if col not in model_spine.columns:
            model_spine[col] = pd.NA
    model_spine = model_spine[keep_cols]

    missing = model_spine[model_spine["park_id"].isna()].head(5)
    if not missing.empty:
        print("WARNING: sample games with missing park mapping:")
        print(missing[["game_pk", "game_date", "home_team", "away_team", "park_name"]].to_string(index=False))

    print_rowcount("model_spine_game", model_spine)
    print(f"Writing to: {model_spine_path.resolve()}")
    write_parquet(model_spine, model_spine_path)
    return model_spine_path
