from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import write_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build normalized game-level weather table.")
    p.add_argument("--season-start", type=int, default=2019)
    p.add_argument("--season-end", type=int, default=2026)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], cands: list[str]) -> str | None:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "build_weather_game.log")
    log_header("scripts/reference/build_weather_game.py", repo_root, config_path, dirs)

    frames: list[pd.DataFrame] = []
    for season in range(args.season_start, args.season_end + 1):
        spine_candidates = [
            dirs["processed_dir"] / f"model_spine_game_{season}.parquet",
            dirs["processed_dir"] / "live" / f"model_spine_game_{season}.parquet",
        ]
        path = next((p for p in spine_candidates if p.exists()), None)
        if path is None:
            continue
        df = pd.read_parquet(path).copy()
        if "game_pk" not in df.columns:
            continue
        temp_col = _pick(list(df.columns), ["temperature", "temp_f", "game_temp", "weather_temp"])
        wind_col = _pick(list(df.columns), ["weather_wind", "wind_speed", "wind_mph", "wind"])
        game_date_col = _pick(list(df.columns), ["game_date", "date"])
        home_col = _pick(list(df.columns), ["home_team", "home_team_abbr"])
        away_col = _pick(list(df.columns), ["away_team", "away_team_abbr"])

        w = pd.DataFrame()
        w["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
        w["game_date"] = pd.to_datetime(df[game_date_col], errors="coerce") if game_date_col else pd.NaT
        w["home_team"] = df[home_col] if home_col else pd.NA
        w["away_team"] = df[away_col] if away_col else pd.NA
        w["temperature"] = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else pd.NA
        w["weather_wind"] = pd.to_numeric(df[wind_col], errors="coerce") if wind_col else pd.NA
        frames.append(w)

    if not frames:
        raise FileNotFoundError("No spine files found to build weather_game.parquet")

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=["game_pk"], keep="last")
    out_path = dirs["processed_dir"] / "weather_game.parquet"
    write_parquet(out, out_path)
    logging.info("weather_game rows=%s path=%s", len(out), out_path)
    print(f"weather_out={out_path}")


if __name__ == "__main__":
    main()
