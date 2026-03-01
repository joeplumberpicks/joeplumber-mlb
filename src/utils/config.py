from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}
