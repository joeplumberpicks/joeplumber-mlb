"""
Joe Plumber MLB Engine — Layer 1: Ingest

This package contains all raw data ingestion logic.

Responsibilities:
- Pull raw data from external providers (MLB, Statcast, Fangraphs, Rotowire, etc.)
- Normalize into consistent schemas
- Write clean raw + processed tables

STRICT RULES:
- No modeling logic
- No feature engineering
- No targets
- No rolling stats
- No leakage-prone transformations

Outputs from this layer feed directly into the Spine and Feature layers.
"""

# Core ingest modules
from .schedule import build_schedule_games
from .games import build_games_metadata
from .weather import build_weather_games
from .lineups import build_projected_lineups, build_confirmed_lineups, build_starting_pitchers
from .plate_appearances import build_plate_appearances
from .parks import build_parks_reference

# Utilities
from .schemas import validate_schema, get_schema
from .io import write_parquet, read_parquet
from .normalize import normalize_columns

# Versioning (optional but useful for debugging)
INGEST_VERSION = "v1.0"

__all__ = [
    # Core builders
    "build_schedule_games",
    "build_games_metadata",
    "build_weather_games",
    "build_projected_lineups",
    "build_confirmed_lineups",
    "build_starting_pitchers",
    "build_plate_appearances",
    "build_parks_reference",

    # Utilities
    "validate_schema",
    "get_schema",
    "write_parquet",
    "read_parquet",
    "normalize_columns",

    # Metadata
    "INGEST_VERSION",
]
