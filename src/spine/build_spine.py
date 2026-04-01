from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

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

_TEAM_ALIAS_TO_ABBR = {
    "ARI": "ARI",
    "ARIZONA": "ARI",
    "ARIZONA DIAMONDBACKS": "ARI",
    "ATL": "ATL",
    "ATLANTA": "ATL",
    "ATLANTA BRAVES": "ATL",
    "BAL": "BAL",
    "BALTIMORE": "BAL",
    "BALTIMORE ORIOLES": "BAL",
    "BOS": "BOS",
    "BOSTON": "BOS",
    "BOSTON RED SOX": "BOS",
    "CHC": "CHC",
    "CHICAGO CUBS": "CHC",
    "CIN": "CIN",
    "CINCINNATI": "CIN",
    "CINCINNATI REDS": "CIN",
    "CLE": "CLE",
    "CLEVELAND": "CLE",
    "COLORADO": "COL",
    "COLORADO ROCKIES": "COL",
    "COL": "COL",
    "CWS": "CWS",
    "CHW": "CWS",
    "CHICAGO WHITE SOX": "CWS",
    "DET": "DET",
    "DETROIT": "DET",
    "DETROIT TIGERS": "DET",
    "HOU": "HOU",
    "HOUSTON": "HOU",
    "HOUSTON ASTROS": "HOU",
    "KC": "KC",
    "KCR": "KC",
    "KANSAS CITY": "KC",
    "KANSAS CITY ROYALS": "KC",
    "LAA": "LAA",
    "ANA": "LAA",
    "LOS ANGELES ANGELS": "LAA",
    "LAD": "LAD",
    "LOS ANGELES DODGERS": "LAD",
    "MIA": "MIA",
    "FLA": "MIA",
    "MIAMI": "MIA",
    "MIAMI MARLINS": "MIA",
    "MIL": "MIL",
    "MILWAUKEE": "MIL",
    "MILWAUKEE BREWERS": "MIL",
    "MIN": "MIN",
    "MINNESOTA": "MIN",
    "MINNESOTA TWINS": "MIN",
    "NYM": "NYM",
    "NEW YORK METS": "NYM",
    "NYY": "NYY",
    "NEW YORK YANKEES": "NYY",
    "OAK": "OAK",
    "ATHLETICS": "OAK",
    "OAKLAND": "OAK",
    "OAKLAND ATHLETICS": "OAK",
    "PHI": "PHI",
    "PHILADELPHIA": "PHI",
    "PHILADELPHIA PHILLIES": "PHI",
    "PIT": "PIT",
    "PITTSBURGH": "PIT",
    "PITTSBURGH PIRATES": "PIT",
    "SD": "SD",
    "SDP": "SD",
    "SAN DIEGO": "SD",
    "SAN DIEGO PADRES": "SD",
    "SEA": "SEA",
    "SEATTLE": "SEA",
    "SEATTLE MARINERS": "SEA",
    "SF": "SF",
    "SFG": "SF",
    "SAN FRANCISCO": "SF",
    "SAN FRANCISCO GIANTS": "SF",
    "STL": "STL",
    "STL CARDINALS": "STL",
    "ST LOUIS": "STL",
    "ST. LOUIS": "STL",
    "ST LOUIS CARDINALS": "STL",
    "ST. LOUIS CARDINALS": "STL",
    "TB": "TB",
    "TBR": "TB",
    "TAMPA BAY": "TB",
    "TAMPA BAY RAYS": "TB",
    "TEX": "TEX",
    "TEXAS": "TEX",
    "TEXAS RANGERS": "TEX",
    "TOR": "TOR",
    "TORONTO": "TOR",
    "TORONTO BLUE JAYS": "TOR",
    "WSH": "WSH",
    "WAS": "WSH",
    "WASHINGTON": "WSH",
    "WASHINGTON NATIONALS": "WSH",
}


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


def _normalize_weather_df(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.copy()
    if "season" not in out.columns:
        out["season"] = season
    for col in WEATHER_COLUMNS:
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
        if label == "weather":
            df = _normalize_weather_df(df, season)
            if df.empty:
                logging.info("weather_by_season empty source normalized with required schema path=%s", raw_path)
            print_rowcount("weather_normalized", df)
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

    before_park_fill = float(out["park_id"].notna().mean() * 100.0) if len(out) else 0.0
    before_venue_fill = float(out["venue_id"].notna().mean() * 100.0) if len(out) else 0.0

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

    # Fallback map by home team for seasons where games often miss park metadata.
    team_map_path = reference_dir / "parks" / "team_home_park_map_from_2024.parquet"
    if team_map_path.exists():
        team_map = read_parquet(team_map_path)
        map_cols = [c for c in ["home_team", "venue_id", "park_id", "park_name"] if c in team_map.columns]
        if "home_team" in map_cols:
            team_map = team_map[map_cols].copy().drop_duplicates(subset=["home_team"], keep="first")
            out = out.merge(team_map, on="home_team", how="left", suffixes=("", "_map"))

            def _missing_id(series: pd.Series) -> pd.Series:
                numeric = pd.to_numeric(series, errors="coerce")
                return series.isna() | numeric.eq(0)

            for c in ["venue_id", "park_id", "park_name"]:
                map_col = f"{c}_map"
                if map_col in out.columns:
                    if c in ["venue_id", "park_id"]:
                        missing_mask = _missing_id(out[c])
                    else:
                        missing_mask = out[c].isna() | out[c].astype(str).str.strip().eq("")
                    out.loc[missing_mask, c] = out.loc[missing_mask, map_col]
                    out = out.drop(columns=[map_col])
        else:
            logging.warning("team_home_park_map_from_2024.parquet missing home_team; fallback skipped")
    else:
        logging.info("team home park fallback map not found at %s", team_map_path.resolve())

    if "canonical_park_key" not in out.columns:
        out["canonical_park_key"] = park_resolved.get("canonical_park_key", pd.NA)

    after_park_fill = float(out["park_id"].notna().mean() * 100.0) if len(out) else 0.0
    after_venue_fill = float(out["venue_id"].notna().mean() * 100.0) if len(out) else 0.0
    logging.info(
        "park mapping fill rates season=%s park_id %.2f%%->%.2f%% venue_id %.2f%%->%.2f%%",
        season,
        before_park_fill,
        after_park_fill,
        before_venue_fill,
        after_venue_fill,
    )

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




def _log_spine_quality_by_season(df: pd.DataFrame) -> None:
    if df.empty or "season" not in df.columns:
        return
    for season_val, grp in df.groupby("season", dropna=False):
        season_label = "unknown" if pd.isna(season_val) else str(int(season_val) if isinstance(season_val, (int, float, np.integer, np.floating)) else season_val)
        park_fill = float(grp["park_id"].notna().mean() * 100.0) if "park_id" in grp.columns else 0.0
        venue_fill = float(grp["venue_id"].notna().mean() * 100.0) if "venue_id" in grp.columns else 0.0
        home_null = float(grp["home_sp_id"].isna().mean() * 100.0) if "home_sp_id" in grp.columns else 0.0
        away_null = float(grp["away_sp_id"].isna().mean() * 100.0) if "away_sp_id" in grp.columns else 0.0
        logging.info(
            "spine season=%s park_id_filled=%.2f%% venue_id_filled=%.2f%% home_sp_null=%.2f%% away_sp_null=%.2f%%",
            season_label,
            park_fill,
            venue_fill,
            home_null,
            away_null,
        )

def _pick_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_team_for_match(val: object) -> str:
    raw = str(val).strip().upper() if pd.notna(val) else ""
    if not raw:
        return "UNK"
    canon = canonical_team_abbr(raw, None)
    if canon != "UNK":
        return canon
    return _TEAM_ALIAS_TO_ABBR.get(raw, raw)


def _populate_starter_ids_from_events(model_spine: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    def _pick_first_pitcher(df: pd.DataFrame) -> pd.Series:
        sort_cols = [c for c in ["inning", "at_bat_number", "pitch_number"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort")
        return df.iloc[0]

    out = model_spine.copy()
    if "game_pk" not in out.columns:
        return out

    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    if "home_sp_id" not in out.columns:
        out["home_sp_id"] = pd.NA
    if "away_sp_id" not in out.columns:
        out["away_sp_id"] = pd.NA

    seasons: list[int] = []
    if "season" in out.columns:
        seasons = sorted(set(pd.to_numeric(out["season"], errors="coerce").dropna().astype(int).tolist()))

    def _resolve_pa_path(season: int) -> Path | None:
        p1 = processed_dir / "events_pa.parquet"
        if p1.exists():
            return p1
        p2 = processed_dir / "by_season" / f"pa_{season}.parquet"
        if p2.exists():
            return p2
        p3 = processed_dir.parent / "raw" / "by_season" / f"pa_{season}.parquet"
        if p3.exists():
            return p3
        return None

    if not seasons:
        logging.warning("starter mapping skipped: no seasons found in model_spine")
        return out

    for season in seasons:
        pa_path = _resolve_pa_path(int(season))
        if pa_path is None:
            logging.warning("starter mapping no PA source for season=%s; tried processed/events_pa.parquet, processed/by_season/pa_%s.parquet, raw/by_season/pa_%s.parquet", season, season, season)
            continue

        dataset = ds.dataset(pa_path, format="parquet")
        avail = set(dataset.schema.names)
        keep = [c for c in ["game_pk", "inning_topbot", "inning", "pitcher_id", "pitcher", "at_bat_number", "pitch_number"] if c in avail]
        if "game_pk" not in keep or "inning_topbot" not in keep:
            logging.warning("starter mapping PA missing required columns game_pk/inning_topbot source=%s", pa_path.resolve())
            continue

        pa = dataset.to_table(columns=keep).to_pandas()
        pa["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")

        if "pitcher_id" in pa.columns:
            pa["pitcher_id"] = pd.to_numeric(pa["pitcher_id"], errors="coerce").astype("Int64")
        elif "pitcher" in pa.columns:
            pa["pitcher_id"] = pd.to_numeric(pa["pitcher"], errors="coerce").astype("Int64")
        else:
            logging.warning("starter mapping PA missing pitcher identifier source=%s", pa_path.resolve())
            continue

        season_games = out.loc[pd.to_numeric(out["season"], errors="coerce") == int(season), "game_pk"].dropna().unique() if "season" in out.columns else out["game_pk"].dropna().unique()
        if len(season_games):
            pa = pa[pa["game_pk"].isin(season_games)]

        if "inning" in pa.columns:
            pa["inning"] = pd.to_numeric(pa["inning"], errors="coerce").astype("Int64")
            pa1 = pa[pa["inning"] == 1].copy()
            if len(pa1):
                pa = pa1

        if pa.empty:
            continue

        half = pa["inning_topbot"].astype(str).str.lower().str.strip()
        home_df = pa[half.str.startswith("top")].copy()  # away batting -> home pitching
        away_df = pa[half.str.startswith("bot")].copy()  # home batting -> away pitching

        if not home_df.empty:
            home_first = (
                home_df.groupby("game_pk", as_index=False)
                .apply(_pick_first_pitcher, include_groups=False)
                .reset_index(drop=True)[["game_pk", "pitcher_id"]]
                .rename(columns={"pitcher_id": "home_sp_id_s"})
            )
        else:
            home_first = pd.DataFrame(columns=["game_pk", "home_sp_id_s"])

        if not away_df.empty:
            away_first = (
                away_df.groupby("game_pk", as_index=False)
                .apply(_pick_first_pitcher, include_groups=False)
                .reset_index(drop=True)[["game_pk", "pitcher_id"]]
                .rename(columns={"pitcher_id": "away_sp_id_s"})
            )
        else:
            away_first = pd.DataFrame(columns=["game_pk", "away_sp_id_s"])

        starters = home_first.merge(away_first, on="game_pk", how="outer")
        if starters.empty:
            continue
        starters["game_pk"] = pd.to_numeric(starters["game_pk"], errors="coerce").astype("Int64")
        starters["home_sp_id_s"] = pd.to_numeric(starters.get("home_sp_id_s"), errors="coerce").astype("Int64")
        starters["away_sp_id_s"] = pd.to_numeric(starters.get("away_sp_id_s"), errors="coerce").astype("Int64")

        out = out.merge(starters, on="game_pk", how="left")
        out["home_sp_id"] = pd.to_numeric(out["home_sp_id"], errors="coerce").fillna(pd.to_numeric(out.get("home_sp_id_s"), errors="coerce")).astype("Int64")
        out["away_sp_id"] = pd.to_numeric(out["away_sp_id"], errors="coerce").fillna(pd.to_numeric(out.get("away_sp_id_s"), errors="coerce")).astype("Int64")
        out = out.drop(columns=["home_sp_id_s", "away_sp_id_s"], errors="ignore")

        logging.info("starter mapping source=%s season=%s starters_rows=%s", pa_path.resolve(), season, len(starters))

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
    _log_spine_quality_by_season(model_spine)

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

    if len(seasons) == 1:
        season = seasons[0]
        season_spine_path = processed_by_season / f"model_spine_game_{season}.parquet"
        print(f"Writing to: {season_spine_path.resolve()}")
        write_parquet(model_spine, season_spine_path)

    return model_spine_path
