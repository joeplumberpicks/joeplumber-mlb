from __future__ import annotations

from typing import Any

import pandas as pd
import statsapi


def get_schedule(season: int, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch MLB schedule via statsapi and return as DataFrame."""
    rows = statsapi.schedule(season=season, start_date=start, end_date=end)
    return pd.DataFrame(rows)


def get_game_feed(game_pk: int) -> dict[str, Any]:
    """Fetch raw game feed payload for a gamePk."""
    return statsapi.get("game", {"gamePk": int(game_pk)})


def safe_json_get(d: Any, path_list: list[Any], default: Any = None) -> Any:
    """Safely get nested key/index from dict/list payloads."""
    cur = d
    for key in path_list:
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        elif isinstance(cur, list) and isinstance(key, int):
            if 0 <= key < len(cur):
                cur = cur[key]
            else:
                return default
        else:
            return default

        if cur is None:
            return default
    return cur
