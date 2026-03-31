from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

SOURCE_NAME = "fangraphs_rosterresource"
SOURCE_URL = "https://www.fangraphs.com/roster-resource/lineup-tracker"

POS_SET = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH", "OF"}

TEAM_NICKNAME_TO_CANONICAL = {
    "Angels": "LOS ANGELES ANGELS",
    "Astros": "HOUSTON ASTROS",
    "Athletics": "ATHLETICS",
    "Blue Jays": "TORONTO BLUE J
