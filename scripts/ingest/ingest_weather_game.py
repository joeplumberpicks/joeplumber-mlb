from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.checks import print_rowcount, require_files
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header

GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"


def _to_int(value: object) -> int | None:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    return int(num)


def _to_float(value: object) -> float | None:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    return float(num)


def _parse_wind(raw_wind: object) -> tuple[float | None, str | None]:
    if raw_wind is None:
        return None, None
    text = str(raw_wind).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None, None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    speed = float(match.group(1)) if match else None
    return speed, text


def _wind_flags(direction: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    d = direction.fillna("").astype(str).str.lower()
    out = d.str.contains(r"\bout\b|out to|to (left|right|center|cf|lf|rf)")
    inn = d.str.contains(r"\bin\b|in from|from (left|right|center|cf|lf|rf)")
    cross = d.str.contains(r"cross|left to right|right to left")
    return out.astype(float), inn.astype(float), cross.astype(float)


def _safe_get(d: dict, *keys: str) -> object:
    cur: object = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _metadata_row_map(games_df: pd.DataFrame) -> dict[int, dict[str, object]]:
    if games_df.empty or "game_pk" not in games_df.columns:
        return {}
    out: dict[int, dict[str, object]] = {}
    for _, row in games_df.iterrows():
        game_pk = _to_int(row.get("game_pk"))
        if game_pk is None:
            continue
        out[game_pk] = {
            "game_date": pd.to_datetime(row.get("game_date"), errors="coerce"),
            "season": _to_int(row.get("season")),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "venue_id": _to_int(row.get("venue_id")),
        }
    return out


def fetch_weather_for_games(games_df: pd.DataFrame, timeout: int = 30) -> pd.DataFrame:
    meta = _metadata_row_map(games_df)
    game_pks = sorted(meta)
    session = requests.Session()
    rows: list[dict[str, object]] = []

    for game_pk in game_pks:
        meta_row = meta.get(game_pk, {})
        row: dict[str, object] = {
            "game_pk": game_pk,
            "game_date": meta_row.get("game_date"),
            "season": meta_row.get("season"),
            "home_team": meta_row.get("home_team"),
            "away_team": meta_row.get("away_team"),
            "venue_id": meta_row.get("venue_id"),
            "temperature": None,
            "wind_speed": None,
            "wind_direction": None,
            "temperature_f": None,
            "wind_mph": None,
            "wind_dir": None,
            "weather_wind_out": None,
            "weather_wind_in": None,
            "weather_crosswind": None,
        }
        try:
            resp = session.get(GAME_FEED_URL.format(game_pk=game_pk), timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            gd = payload.get("gameData", {})
            weather = gd.get("weather", {}) if isinstance(gd.get("weather"), dict) else {}

            temp = _to_float(weather.get("temp"))
            wind_speed, wind_direction = _parse_wind(weather.get("wind"))
            if temp is None:
                bx_info = _safe_get(payload, "liveData", "boxscore", "info")
                if isinstance(bx_info, list):
                    for item in bx_info:
                        label = str(item.get("label", "")).strip().lower() if isinstance(item, dict) else ""
                        value = item.get("value") if isinstance(item, dict) else None
                        if label == "weather" and temp is None:
                            temp = _to_float(value)
                        if label == "wind" and wind_direction is None:
                            wind_speed, wind_direction = _parse_wind(value)

            official_date = pd.to_datetime(_safe_get(gd, "datetime", "officialDate"), errors="coerce")
            if pd.notna(official_date):
                row["game_date"] = official_date

            season_val = _to_int(_safe_get(gd, "game", "season"))
            if season_val is not None:
                row["season"] = season_val

            home_team = _safe_get(gd, "teams", "home", "abbreviation") or _safe_get(gd, "teams", "home", "name")
            away_team = _safe_get(gd, "teams", "away", "abbreviation") or _safe_get(gd, "teams", "away", "name")
            if home_team is not None:
                row["home_team"] = home_team
            if away_team is not None:
                row["away_team"] = away_team

            venue_id = _to_int(_safe_get(gd, "venue", "id"))
            if venue_id is not None:
                row["venue_id"] = venue_id

            row["temperature"] = temp
            row["wind_speed"] = wind_speed
            row["wind_direction"] = wind_direction
            row["temperature_f"] = temp
            row["wind_mph"] = wind_speed
            row["wind_dir"] = wind_direction
        except Exception as exc:  # noqa: BLE001
            logging.warning("weather fetch failed game_pk=%s error=%s", game_pk, exc)

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce")
    out["season"] = pd.to_numeric(out.get("season"), errors="coerce").astype("Int64")
    out["venue_id"] = pd.to_numeric(out.get("venue_id"), errors="coerce").astype("Int64")
    out["temperature"] = pd.to_numeric(out.get("temperature"), errors="coerce")
    out["wind_speed"] = pd.to_numeric(out.get("wind_speed"), errors="coerce")
    out["temperature_f"] = pd.to_numeric(out.get("temperature_f"), errors="coerce")
    out["wind_mph"] = pd.to_numeric(out.get("wind_mph"), errors="coerce")
    out["wind_direction"] = out.get("wind_direction").astype("string")
    out["wind_dir"] = out.get("wind_dir").astype("string")

    wind_out, wind_in, wind_cross = _wind_flags(out["wind_direction"])
    out["weather_wind_out"] = wind_out
    out["weather_wind_in"] = wind_in
    out["weather_crosswind"] = wind_cross

    out["weather_non_null_fields"] = (
        out[["temperature", "wind_speed", "wind_direction"]].notna().sum(axis=1)
    )
    out = out.sort_values(["weather_non_null_fields", "game_pk"], ascending=[False, True], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first")
    out = out.drop(columns=["weather_non_null_fields"], errors="ignore")
    return out


def _read_historical_games(dirs: dict[str, Path], season: int) -> pd.DataFrame:
    games_path = dirs["raw_dir"] / "by_season" / f"games_{season}.parquet"
    require_files([games_path], f"weather_games_{season}")
    return read_parquet(games_path)


def _read_live_games(dirs: dict[str, Path], season: int, date: str) -> pd.DataFrame:
    date_scoped = dirs["raw_dir"] / "live" / f"games_schedule_{season}_{date}.parquet"
    cumulative = dirs["raw_dir"] / "live" / f"games_schedule_{season}.parquet"
    in_path = date_scoped if date_scoped.exists() else cumulative
    require_files([in_path], f"weather_live_games_{season}_{date}")
    games = read_parquet(in_path)
    games["game_date"] = pd.to_datetime(games.get("game_date"), errors="coerce")
    day = pd.to_datetime(date, errors="raise")
    return games[games["game_date"].dt.date == day.date()].copy()


def write_weather_for_season(dirs: dict[str, Path], season: int) -> Path:
    games_df = _read_historical_games(dirs, season)
    weather_df = fetch_weather_for_games(games_df)
    out_path = dirs["raw_dir"] / "by_season" / f"weather_game_{season}.parquet"
    print_rowcount(f"weather_game_{season}", weather_df)
    print(f"Writing to: {out_path.resolve()}")
    write_parquet(weather_df, out_path)
    logging.info("weather ingest season=%s rows=%s out=%s", season, len(weather_df), out_path)
    return out_path


def write_weather_for_live_date(dirs: dict[str, Path], season: int, date: str) -> Path:
    games_df = _read_live_games(dirs, season, date)
    weather_df = fetch_weather_for_games(games_df)
    out_path = dirs["raw_dir"] / "live" / f"weather_game_{season}_{date}.parquet"
    print_rowcount(f"weather_game_{season}_{date}", weather_df)
    print(f"Writing to: {out_path.resolve()}")
    write_parquet(weather_df, out_path)
    logging.info("weather ingest live season=%s date=%s rows=%s out=%s", season, date, len(weather_df), out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest game-level weather from MLB game feeds.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", default=None, help="YYYY-MM-DD. If set, writes live date-scoped weather file.")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "ingest_weather_game.log")
    log_header("scripts/ingest/ingest_weather_game.py", repo_root, config_path, dirs)

    if args.date:
        write_weather_for_live_date(dirs=dirs, season=args.season, date=args.date)
    else:
        write_weather_for_season(dirs=dirs, season=args.season)


if __name__ == "__main__":
    main()
