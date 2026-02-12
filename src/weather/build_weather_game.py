from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.weather.features import add_weather_transforms
from src.weather.providers import NWSClient, RetryConfig, VisualCrossingClient
from src.weather.stadiums import load_park_overrides, load_stadium_reference, resolve_park_for_game


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _pick_provider(name: str, cfg: dict, logger: logging.Logger) -> object:
    retry_cfg = RetryConfig(
        max_retries=int(cfg.get("query", {}).get("max_retries", 4)),
        base_backoff_seconds=float(cfg.get("query", {}).get("base_backoff_seconds", 1.0)),
        backoff_cap_seconds=float(cfg.get("query", {}).get("backoff_cap_seconds", 30.0)),
        timeout_seconds=int(cfg.get("query", {}).get("timeout_seconds", 25)),
    )
    if name == "visualcrossing":
        env_name = str(cfg.get("visualcrossing_api_key_env", "VISUAL_CROSSING_API_KEY"))
        return VisualCrossingClient.from_env(api_key_env=env_name, retry=retry_cfg, logger=logger)
    if name == "nws":
        return NWSClient(logger=logger)
    raise ValueError(f"Unsupported weather provider: {name}")


def _nearest_row(hourly: pd.DataFrame, target: pd.Timestamp) -> pd.Series | None:
    if hourly.empty or "obs_time" not in hourly.columns:
        return None
    temp = hourly.copy()
    temp["delta_s"] = (temp["obs_time"] - target).abs().dt.total_seconds()
    temp = temp.sort_values(["delta_s", "obs_time"], kind="mergesort")
    return temp.iloc[0]


def _resolve_games(games: pd.DataFrame, spine: pd.DataFrame | None, season: int) -> pd.DataFrame:
    if "game_pk" not in games.columns:
        raise ValueError("games parquet missing required column: game_pk")

    g = games.copy()
    g["game_date"] = pd.to_datetime(g.get("game_date"), errors="coerce").dt.normalize()
    g["game_pk"] = pd.to_numeric(g["game_pk"], errors="coerce").astype("Int64")

    if "home_team" not in g.columns or "away_team" not in g.columns:
        if spine is not None and {"game_pk", "home_team", "away_team"}.issubset(spine.columns):
            g = g.merge(spine[["game_pk", "home_team", "away_team"]], on="game_pk", how="left", suffixes=("", "_sp"))
            g["home_team"] = g["home_team"].fillna(g.get("home_team_sp"))
            g["away_team"] = g["away_team"].fillna(g.get("away_team_sp"))

    required = {"game_date", "game_pk", "home_team", "away_team"}
    miss = required - set(g.columns)
    if miss:
        raise ValueError(f"games/spine missing required columns: {sorted(miss)}")

    g = g.dropna(subset=["game_date", "game_pk", "home_team"]).copy()
    g = g.loc[g["game_date"].dt.year == int(season)].copy()
    g = g.sort_values(["game_date", "game_pk"], kind="mergesort")
    g = g.drop_duplicates(subset=["game_pk"], keep="first")
    return g.reset_index(drop=True)


def build_weather_game(
    season: int,
    start: str | None,
    end: str | None,
    games_path: Path,
    spine_path: Path | None,
    out_path: Path,
    provider: str = "visualcrossing",
    *,
    max_games: int | None = None,
    allow_partial: bool = False,
    config_path: Path = Path("config/weather.yaml"),
    overrides_path: Path = Path("data/reference/park_overrides.csv"),
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build one weather row per game using nearest hourly conditions to game start."""
    log = logger or logging.getLogger(__name__)
    if not games_path.exists():
        raise FileNotFoundError(f"Missing games parquet: {games_path}")

    games = pd.read_parquet(games_path, engine="pyarrow")
    spine = pd.read_parquet(spine_path, engine="pyarrow") if spine_path and spine_path.exists() else None
    cfg = _load_yaml(config_path)
    provider_name = provider or str(cfg.get("provider", "visualcrossing"))
    client = _pick_provider(provider_name, cfg, log)

    stadiums = load_stadium_reference(logger=log)
    overrides = load_park_overrides(overrides_path)
    bearings_cfg = cfg.get("park_bearing_to_cf_deg", {})
    tz_default = str(cfg.get("timezone_default", "America/New_York"))

    base = _resolve_games(games, spine, season=season)
    if start:
        base = base.loc[base["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        base = base.loc[base["game_date"] <= pd.to_datetime(end).normalize()].copy()
    if max_games is not None:
        base = base.head(int(max_games)).copy()

    park_ids: list[str | None] = []
    sources: list[str] = []
    for _, row in base.iterrows():
        park_id, source = resolve_park_for_game(row, season, stadiums, overrides, logger=log)
        park_ids.append(park_id)
        sources.append(source)
    base["park_id"] = park_ids
    base["park_resolution_source"] = sources

    counters = base["park_resolution_source"].value_counts(dropna=False).to_dict()
    unresolved_mask = base["park_id"].isna()
    unresolved_count = int(unresolved_mask.sum())
    total_games = len(base)
    unresolved_rate = unresolved_count / total_games if total_games else 0.0

    log.info(
        "park_resolution total=%d override=%d explicit_venue=%d team_season=%d unresolved=%d",
        total_games,
        int(counters.get("override", 0)),
        int(counters.get("explicit_venue", 0)),
        int(counters.get("team_season", 0)),
        unresolved_count,
    )

    if unresolved_count > 0:
        sample = base.loc[unresolved_mask, ["game_pk", "home_team", "game_date"]].head(10)
        log.warning("park_unresolved_samples:\n%s", sample.to_string(index=False))

    if unresolved_rate > 0.05 and not allow_partial:
        raise SystemExit(
            f"Unresolved park mapping rate {unresolved_rate:.2%} exceeds 5%. "
            "Populate data/reference/park_overrides.csv or stadium mappings."
        )

    base = base.merge(stadiums, on=["park_id"], how="left", suffixes=("", "_stad"))
    missing_coords = base["lat"].isna() | base["lon"].isna()
    if missing_coords.any():
        missing_parks = sorted(base.loc[missing_coords, "park_id"].dropna().unique().tolist())
        raise ValueError(f"Missing stadium coordinates for resolved park_ids: {missing_parks}")

    rows: list[dict[str, object]] = []
    for rec in base.to_dict(orient="records"):
        game_date = pd.to_datetime(rec["game_date"]).normalize()
        start_col = rec.get("game_datetime") or rec.get("first_pitch") or rec.get("start_time")
        if pd.isna(start_col) or start_col is None:
            local_dt = pd.Timestamp.combine(game_date.date(), time(19, 0))
            log.warning("game_pk=%s missing start time; defaulting to 7pm local", rec["game_pk"])
        else:
            local_dt = pd.to_datetime(start_col, errors="coerce")
            if pd.isna(local_dt):
                local_dt = pd.Timestamp.combine(game_date.date(), time(19, 0))

        tz_name = rec.get("timezone") or tz_default
        local_dt = pd.Timestamp(local_dt)
        if local_dt.tzinfo is None:
            local_dt = local_dt.tz_localize(tz_name)
        target_utc = local_dt.tz_convert("UTC")

        hourly = client.fetch_hourly(
            float(rec["lat"]),
            float(rec["lon"]),
            (target_utc - timedelta(hours=2)).to_pydatetime(),
            (target_utc + timedelta(hours=2)).to_pydatetime(),
        )
        nearest = _nearest_row(hourly, pd.Timestamp(target_utc))
        if nearest is None:
            continue

        park_id = str(rec.get("park_id"))
        bearing = bearings_cfg.get(park_id, rec.get("cf_bearing_deg"))
        if pd.isna(bearing):
            log.warning("Missing cf bearing for park_id=%s; using 0.0", park_id)
            bearing = 0.0

        rows.append(
            {
                "game_date": game_date,
                "game_pk": int(rec["game_pk"]),
                "home_team": rec["home_team"],
                "away_team": rec["away_team"],
                "park_id": park_id,
                "stadium_name": rec.get("stadium_name"),
                "city": rec.get("city"),
                "state": rec.get("state"),
                "lat": rec.get("lat"),
                "lon": rec.get("lon"),
                "timezone": rec.get("timezone") or tz_name,
                "roof_type": rec.get("roof_type"),
                "cf_bearing_deg": bearing,
                "park_resolution_source": rec.get("park_resolution_source"),
                "provider": nearest.get("provider"),
                "obs_time": nearest.get("obs_time"),
                "temp_f": nearest.get("temp_f"),
                "humidity": nearest.get("humidity"),
                "dewpoint_f": nearest.get("dewpoint_f"),
                "pressure_mb": nearest.get("pressure_mb"),
                "wind_speed_mph": nearest.get("wind_speed_mph"),
                "wind_dir_deg": nearest.get("wind_dir_deg"),
                "precip_in": nearest.get("precip_in"),
                "conditions": nearest.get("conditions"),
                "source_timezone": tz_name,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No weather rows were fetched. Check provider settings and API key.")

    out = out.rename(columns={"cf_bearing_deg": "park_cf_bearing_deg"})
    out = add_weather_transforms(out, park_cf_bearing_default=0.0)
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    expected = len(base)
    coverage = len(out) / expected if expected else 0.0
    log.info("weather_games_expected=%d weather_games_built=%d coverage=%.4f", expected, len(out), coverage)
    for col in ["temp_f", "humidity", "wind_speed_mph", "wind_out_to_cf_mph"]:
        if col in out.columns:
            log.info("null_rate_%s=%.4f", col, float(out[col].isna().mean()))

    if coverage < 0.90 and not allow_partial:
        raise SystemExit(2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("wrote_weather_game rows=%d path=%s", len(out), out_path)
    return out
