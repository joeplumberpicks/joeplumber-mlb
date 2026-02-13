from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.paths import processed_dir, reference_dir
from src.weather.stadiums import load_park_overrides, load_stadium_reference, normalize_team_abbr, resolve_park_for_game

TEAM_NAME_TO_ABBR = {
    "arizona diamondbacks": "ARI",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CHW",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KCR",
    "los angeles angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "athletics": "ATH",
    "oakland athletics": "ATH",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SDP",
    "san francisco giants": "SFG",
    "seattle mariners": "SEA",
    "st louis cardinals": "STL",
    "tampa bay rays": "TBR",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSN",
}



def _canonicalize_team_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    upper = text.upper()
    if len(upper) <= 4:
        return normalize_team_abbr(upper)
    mapped = TEAM_NAME_TO_ABBR.get(text.casefold())
    if mapped:
        return normalize_team_abbr(mapped)
    return normalize_team_abbr(upper)



def _rename_first_match(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    out = df
    col = next((c for c in candidates if c in out.columns), None)
    if col is not None and col != target:
        out = out.rename(columns={col: target})
    return out



def _prepare_spine(spine: pd.DataFrame) -> pd.DataFrame:
    out = spine.copy()
    out = _rename_first_match(out, "game_date", ["game_date", "date", "game_dt"])
    out = _rename_first_match(out, "game_pk", ["game_pk", "game_id", "mlb_game_id"])
    out = _rename_first_match(
        out,
        "home_team",
        ["home_team", "home_team_abbr", "home_abbr", "home", "home_name", "home_team_name"],
    )
    out = _rename_first_match(
        out,
        "away_team",
        ["away_team", "away_team_abbr", "away_abbr", "away", "away_name", "away_team_name"],
    )
    if "game_pk" not in out.columns:
        return pd.DataFrame(columns=["game_pk", "game_date", "home_team", "away_team"])
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.normalize()
    else:
        out["game_date"] = pd.NaT
    if "home_team" in out.columns:
        out["home_team"] = out["home_team"].map(_canonicalize_team_value)
    else:
        out["home_team"] = pd.NA
    if "away_team" in out.columns:
        out["away_team"] = out["away_team"].map(_canonicalize_team_value)
    else:
        out["away_team"] = pd.NA
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first")
    return out[["game_pk", "game_date", "home_team", "away_team"]].reset_index(drop=True)



def _resolve_games(games: pd.DataFrame, season: int, spine: pd.DataFrame | None = None) -> pd.DataFrame:
    out = games.copy()
    out = _rename_first_match(out, "game_date", ["game_date", "date", "game_dt"])
    out = _rename_first_match(out, "game_pk", ["game_pk", "game_id", "mlb_game_id"])
    out = _rename_first_match(
        out,
        "home_team",
        ["home_team", "home_team_abbr", "home_abbr", "home", "home_name", "home_team_name"],
    )
    out = _rename_first_match(
        out,
        "away_team",
        ["away_team", "away_team_abbr", "away_abbr", "away", "away_name", "away_team_name"],
    )

    if "game_pk" not in out.columns:
        raise ValueError("games parquet missing required column: game_pk")

    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce").dt.normalize()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    if "home_team" in out.columns:
        out["home_team"] = out["home_team"].map(_canonicalize_team_value)
    if "away_team" in out.columns:
        out["away_team"] = out["away_team"].map(_canonicalize_team_value)

    if spine is not None:
        sp = _prepare_spine(spine)
        if "home_team" not in out.columns:
            out["home_team"] = pd.NA
        if "away_team" not in out.columns:
            out["away_team"] = pd.NA
        out = out.merge(sp, on=["game_pk"], how="left", suffixes=("", "_sp"))
        if "home_team_sp" in out.columns:
            out["home_team"] = out["home_team"].fillna(out["home_team_sp"])
        if "away_team_sp" in out.columns:
            out["away_team"] = out["away_team"].fillna(out["away_team_sp"])

    required = {"game_date", "game_pk", "home_team", "away_team"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            "games parquet missing required canonical columns after variant mapping and spine fill: "
            f"{sorted(missing)}"
        )

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



def run_smoke_test(
    season: int,
    games_path: Path,
    spine_path: Path,
    logger: logging.Logger | None = None,
) -> None:
    log = logger or logging.getLogger(__name__)
    games = pd.read_parquet(games_path, engine="pyarrow")
    spine = pd.read_parquet(spine_path, engine="pyarrow")
    subset = games.copy()
    for col in ["home_team", "away_team", "home_team_abbr", "away_team_abbr", "home_abbr", "away_abbr"]:
        if col in subset.columns:
            subset = subset.drop(columns=[col])
    resolved = _resolve_games(subset, season=season, spine=spine)
    if resolved.empty:
        raise AssertionError("Smoke test produced empty resolved games frame")
    if resolved[["home_team", "away_team"]].isna().any().any():
        raise AssertionError("Smoke test failed to fill home/away team from spine")
    log.info("park_game_smoke_ok season=%s rows=%d", season, len(resolved))
