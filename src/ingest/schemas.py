"""
Schema contracts for Joe Plumber MLB Engine Layer 1 ingest tables.

Purpose
-------
Define stable schema metadata and validation helpers for normalized ingest
outputs.

This module is Layer 1 only:
- schema definitions
- required columns
- primary keys
- light validation helpers

No modeling logic.
No feature engineering.
No target creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TableSchema:
    """Schema contract for a normalized ingest table."""

    name: str
    columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    dtypes: dict[str, str]


SCHEMAS: dict[str, TableSchema] = {
    "games_schedule": TableSchema(
        name="games_schedule",
        columns=(
            "game_pk",
            "game_date",
            "season",
            "game_type",
            "status",
            "status_detailed",
            "coded_game_state",
            "scheduled_start_time_utc",
            "scheduled_start_time_et",
            "away_team",
            "home_team",
            "away_team_name",
            "home_team_name",
            "away_team_id",
            "home_team_id",
            "venue_id",
            "venue_name",
            "doubleheader_flag",
            "game_number",
            "series_description",
            "day_night",
            "scheduled_innings",
            "if_necessary",
            "if_necessary_description",
            "resume_date",
            "resume_game_date",
            "public_facing",
        ),
        required_columns=(
            "game_pk",
            "game_date",
            "season",
            "away_team",
            "home_team",
            "venue_id",
        ),
        primary_key=("game_pk",),
        dtypes={
            "game_pk": "Int64",
            "game_date": "string",
            "season": "Int64",
            "away_team": "string",
            "home_team": "string",
            "venue_id": "Int64",
        },
    ),
    "games": TableSchema(
        name="games",
        columns=(
            "game_pk",
            "game_date",
            "season",
            "game_type",
            "status",
            "status_detailed",
            "coded_game_state",
            "away_team",
            "home_team",
            "away_team_name",
            "home_team_name",
            "away_team_id",
            "home_team_id",
            "venue_id",
            "venue_name",
            "away_score",
            "home_score",
            "winning_team",
            "losing_team",
            "home_win",
            "start_time_utc",
            "start_time_et",
            "resume_time_utc",
            "resume_time_et",
            "day_night",
            "doubleheader_flag",
            "game_number",
            "series_description",
            "scheduled_innings",
            "innings",
            "current_inning",
            "current_inning_ordinal",
            "inning_state",
            "inning_half",
            "is_tied",
            "balls",
            "strikes",
            "outs",
            "scheduled_start_time_utc",
            "scheduled_start_time_et",
            "if_necessary",
            "if_necessary_description",
            "resume_date",
            "resume_game_date",
            "public_facing",
            "series_game_number",
            "games_in_series",
            "abstract_game_code",
        ),
        required_columns=(
            "game_pk",
            "game_date",
            "season",
            "away_team",
            "home_team",
            "venue_id",
            "status",
        ),
        primary_key=("game_pk",),
        dtypes={
            "game_pk": "Int64",
            "game_date": "string",
            "season": "Int64",
            "away_team": "string",
            "home_team": "string",
            "venue_id": "Int64",
            "status": "string",
        },
    ),
    "weather_game": TableSchema(
        name="weather_game",
        columns=(
            "game_pk",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "venue_id",
            "venue_name",
            "station_id",
            "observed_at_utc",
            "temperature_f",
            "temperature_c",
            "dewpoint_f",
            "dewpoint_c",
            "wind_mph",
            "wind_kt",
            "wind_gust_mph",
            "wind_gust_kt",
            "wind_direction_deg",
            "wind_dir_text",
            "pressure_hpa",
            "visibility_miles",
            "ceiling_ft",
            "weather_condition",
            "flight_category",
            "humidity",
            "precipitation_mm",
            "roof_status",
            "weather_wind_out",
            "weather_wind_in",
            "weather_crosswind",
            "raw_metar",
            "weather_source",
            "weather_pull_ts",
        ),
        required_columns=(
            "game_pk",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "venue_id",
        ),
        primary_key=("game_pk",),
        dtypes={
            "game_pk": "Int64",
            "game_date": "string",
            "season": "Int64",
            "home_team": "string",
            "away_team": "string",
            "venue_id": "Int64",
            "station_id": "string",
        },
    ),
    "projected_lineups": TableSchema(
        name="projected_lineups",
        columns=(
            "game_pk",
            "game_date",
            "season",
            "team",
            "opponent",
            "is_home",
            "lineup_status",
            "source",
            "source_pull_ts",
            "batting_order",
            "player_id",
            "player_name",
            "handedness_bat",
            "handedness_throw",
            "position",
            "is_starting_lineup",
        ),
        required_columns=(
            "game_pk",
            "game_date",
            "season",
            "team",
            "lineup_status",
            "batting_order",
            "player_name",
        ),
        primary_key=("game_pk", "team", "lineup_status", "batting_order"),
        dtypes={
            "game_pk": "Int64",
            "game_date": "string",
            "season": "Int64",
            "team": "string",
            "lineup_status": "string",
            "batting_order": "Int64",
            "player_name": "string",
        },
    ),
    "confirmed_lineups": TableSchema(
        name="confirmed_lineups",
        columns=(
            "game_pk",
            "game_date",
            "season",
            "team",
            "opponent",
            "is_home",
            "lineup_status",
            "source",
            "source_pull_ts",
            "batting_order",
            "player_id",
            "player_name",
            "handedness_bat",
            "handedness_throw",
            "position",
            "is_starting_lineup",
        ),
        required_columns=(
            "game_pk",
            "game_date",
            "season",
            "team",
            "lineup_status",
            "batting_order",
            "player_name",
        ),
        primary_key=("game_pk", "team", "lineup_status", "batting_order"),
        dtypes={
            "game_pk": "Int64",
            "game_date": "string",
            "season": "Int64",
            "team": "string",
            "lineup_status": "string",
            "batting_order": "Int64",
            "player_name": "string",
        },
    ),
    "starting_pitchers": TableSchema(
        name="starting_pitchers",
        columns=(
            "game_pk",
            "game_date",
            "season",
            "team",
            "opponent",
            "is_home",
            "pitcher_id",
            "pitcher_name",
            "throws",
            "starter_status",
            "source",
            "source_pull_ts",
        ),
        required_columns=(
            "game_pk",
            "game_date",
            "season",
            "team",
            "starter_status",
            "pitcher_name",
        ),
        primary_key=("game_pk", "team", "starter_status"),
        dtypes={
            "game_pk": "Int64",
            "game_date": "string",
            "season": "Int64",
            "team": "string",
            "starter_status": "string",
            "pitcher_name": "string",
        },
    ),
    "plate_appearances": TableSchema(
        name="plate_appearances",
        columns=(
            "game_pk",
            "game_date",
            "season",
            "inning",
            "inning_topbot",
            "batting_team",
            "fielding_team",
            "batter_id",
            "batter_name",
            "pitcher_id",
            "pitcher_name",
            "pa_index",
            "pitch_number_start",
            "pitch_number_end",
            "outs_before_pa",
            "outs_after_pa",
            "base_state_before",
            "base_state_after",
            "runs_scored_on_pa",
            "rbi",
            "event_type",
            "event_text",
            "is_pa",
            "is_ab",
            "is_hit",
            "is_1b",
            "is_2b",
            "is_3b",
            "is_hr",
            "is_bb",
            "is_hbp",
            "is_so",
            "is_rbi",
            "is_sac_fly",
            "is_reached_on_error",
            "source",
            "source_pull_ts",
        ),
        required_columns=(
            "game_pk",
            "game_date",
            "pa_index",
            "event_type",
        ),
        primary_key=("game_pk", "pa_index"),
        dtypes={
            "game_pk": "Int64",
            "game_date": "string",
            "pa_index": "Int64",
            "event_type": "string",
            "batter_name": "string",
            "pitcher_name": "string",
        },
    ),
    "parks": TableSchema(
        name="parks",
        columns=(
            "venue_id",
            "venue_name",
            "team",
            "team_name",
            "city",
            "state",
            "country",
            "surface_type",
            "roof_type",
            "is_dome",
            "is_retractable",
            "altitude_ft",
            "latitude",
            "longitude",
            "time_zone",
            "weather_station_id",
            "park_factor_runs",
            "park_factor_hr",
            "park_factor_hits",
            "park_factor_2b",
            "park_factor_3b",
            "park_factor_bb",
            "park_source",
            "park_source_season",
            "source",
            "source_pull_ts",
        ),
        required_columns=(
            "venue_id",
            "venue_name",
        ),
        primary_key=("venue_id",),
        dtypes={
            "venue_id": "Int64",
            "venue_name": "string",
            "team": "string",
            "weather_station_id": "string",
        },
    ),
}


def get_schema(name: str) -> TableSchema:
    """Return a schema contract by table name."""
    if name not in SCHEMAS:
        available = ", ".join(sorted(SCHEMAS))
        raise KeyError(f"Unknown schema '{name}'. Available schemas: {available}")
    return SCHEMAS[name]


def list_schemas() -> list[str]:
    """Return available schema names."""
    return sorted(SCHEMAS.keys())


def get_required_columns(name: str) -> tuple[str, ...]:
    """Return required columns for a schema."""
    return get_schema(name).required_columns


def get_primary_key(name: str) -> tuple[str, ...]:
    """Return primary key columns for a schema."""
    return get_schema(name).primary_key


def get_dtype_hints(name: str) -> dict[str, str]:
    """Return dtype hints for a schema."""
    return dict(get_schema(name).dtypes)


def ensure_columns(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """
    Add any missing schema columns as NA columns and reorder known columns first.

    This is useful for standardizing outputs before writing.
    """
    schema = get_schema(schema_name)
    out = df.copy()

    for col in schema.columns:
        if col not in out.columns:
            out[col] = pd.NA

    ordered = list(schema.columns) + [c for c in out.columns if c not in schema.columns]
    return out.loc[:, ordered]


def apply_dtype_hints(df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
    """
    Apply schema dtype hints where possible.

    This is best-effort and will not raise for individual conversion failures.
    """
    schema = get_schema(schema_name)
    out = df.copy()

    for col, dtype in schema.dtypes.items():
        if col not in out.columns:
            continue
        try:
            if dtype == "Int64":
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
            elif dtype == "boolean":
                out[col] = out[col].astype("boolean")
            elif dtype == "string":
                out[col] = out[col].astype("string")
            else:
                out[col] = out[col].astype(dtype)
        except Exception:
            continue

    return out


def validate_required_columns(df: pd.DataFrame, schema_name: str) -> list[str]:
    """Return missing required columns for a schema."""
    schema = get_schema(schema_name)
    return sorted(set(schema.required_columns).difference(df.columns))


def validate_primary_key_nulls(df: pd.DataFrame, schema_name: str) -> dict[str, int]:
    """Return null counts for primary key columns."""
    schema = get_schema(schema_name)
    results: dict[str, int] = {}

    for col in schema.primary_key:
        if col not in df.columns:
            results[col] = len(df)
        else:
            results[col] = int(df[col].isna().sum())

    return results


def validate_primary_key_duplicates(df: pd.DataFrame, schema_name: str) -> int:
    """Return duplicate row count on primary key."""
    schema = get_schema(schema_name)
    if not set(schema.primary_key).issubset(df.columns):
        return len(df)
    return int(df.duplicated(subset=list(schema.primary_key)).sum())


def validate_schema(
    df: pd.DataFrame,
    schema_name: str,
    *,
    require_all_columns: bool = False,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """
    Validate a DataFrame against a schema contract.

    Parameters
    ----------
    df:
        DataFrame to validate.
    schema_name:
        Schema name from SCHEMAS.
    require_all_columns:
        If True, validates that every declared schema column is present.
        If False, only required columns are enforced.
    raise_on_error:
        If True, raises ValueError when validation fails.

    Returns
    -------
    dict
        Validation report.
    """
    schema = get_schema(schema_name)

    missing_required = validate_required_columns(df, schema_name)
    missing_declared = sorted(set(schema.columns).difference(df.columns)) if require_all_columns else []
    pk_nulls = validate_primary_key_nulls(df, schema_name)
    pk_duplicates = validate_primary_key_duplicates(df, schema_name)

    errors: list[str] = []

    if missing_required:
        errors.append(f"missing required columns: {missing_required}")

    if missing_declared:
        errors.append(f"missing declared columns: {missing_declared}")

    bad_pk_nulls = {k: v for k, v in pk_nulls.items() if v > 0}
    if bad_pk_nulls:
        errors.append(f"primary key null counts: {bad_pk_nulls}")

    if pk_duplicates > 0:
        errors.append(f"primary key duplicate count: {pk_duplicates}")

    report = {
        "schema_name": schema.name,
        "row_count": int(len(df)),
        "missing_required_columns": missing_required,
        "missing_declared_columns": missing_declared,
        "primary_key": list(schema.primary_key),
        "primary_key_nulls": pk_nulls,
        "primary_key_duplicate_count": pk_duplicates,
        "valid": len(errors) == 0,
        "errors": errors,
    }

    if raise_on_error and errors:
        raise ValueError(f"{schema_name} schema validation failed: " + " | ".join(errors))

    return report


__all__ = [
    "TableSchema",
    "SCHEMAS",
    "get_schema",
    "list_schemas",
    "get_required_columns",
    "get_primary_key",
    "get_dtype_hints",
    "ensure_columns",
    "apply_dtype_hints",
    "validate_required_columns",
    "validate_primary_key_nulls",
    "validate_primary_key_duplicates",
    "validate_schema",
]
