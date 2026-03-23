from __future__ import annotations
import logging
from pathlib import Path
from scripts.ingest.ingest_weather_game import write_weather_for_season

def write_weather_stub_for_games(dirs: dict[str, Path], season: int) -> Path:
    logging.warning("write_weather_stub_for_games is deprecated. Using write_weather_for_season instead.")
    return write_weather_for_season(dirs=dirs, season=season)
