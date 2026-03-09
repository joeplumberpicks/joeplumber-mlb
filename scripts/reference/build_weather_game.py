from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
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


def _parse_wind_direction(raw: pd.Series) -> pd.Series:
    s = raw.astype(str).str.lower()
    out = pd.Series(pd.NA, index=raw.index, dtype="object")
    out = out.mask(s.str.contains("out"), "out")
    out = out.mask(s.str.contains("in"), "in")
    out = out.mask(s.str.contains("cross|left to right|right to left"), "cross")
    return out


def _parse_wind_speed(raw: pd.Series) -> pd.Series:
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        return num
    txt = raw.astype(str).str.lower()
    extracted = txt.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


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
        candidates = [
            dirs["processed_dir"] / f"model_spine_game_{season}.parquet",
            dirs["processed_dir"] / "live" / f"model_spine_game_{season}.parquet",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        df = pd.read_parquet(path)
        if "game_pk" not in df.columns:
            continue

        gd_col = _pick(list(df.columns), ["game_date", "date"])
        ht_col = _pick(list(df.columns), ["home_team", "home_team_abbr"])
        at_col = _pick(list(df.columns), ["away_team", "away_team_abbr"])
        temp_col = _pick(list(df.columns), ["temperature", "temp_f", "game_temp", "weather_temp"])
        wind_spd_col = _pick(list(df.columns), ["wind_speed", "wind_mph", "weather_wind", "wind"])
        wind_dir_col = _pick(list(df.columns), ["wind_direction", "wind_dir", "weather_wind_direction"])

        out = pd.DataFrame(index=df.index)
        out["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
        out["game_date"] = pd.to_datetime(df[gd_col], errors="coerce") if gd_col else pd.NaT
        out["home_team"] = df[ht_col] if ht_col else pd.NA
        out["away_team"] = df[at_col] if at_col else pd.NA
        out["temperature"] = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else np.nan

        wind_raw = df[wind_spd_col] if wind_spd_col else pd.Series(pd.NA, index=df.index)
        out["wind_speed"] = _parse_wind_speed(wind_raw)

        if wind_dir_col:
            wind_dir = df[wind_dir_col].astype(str).str.lower()
        else:
            wind_dir = _parse_wind_direction(wind_raw).astype("string")
        out["wind_direction"] = wind_dir
        out["weather_wind_out"] = wind_dir.str.contains("out").astype(float)
        out["weather_wind_in"] = wind_dir.str.contains("in").astype(float)
        out["weather_crosswind"] = wind_dir.str.contains("cross|left to right|right to left").astype(float)
        frames.append(out)

    if not frames:
        raise FileNotFoundError("No spine files found to build weather_game.parquet")

    full = pd.concat(frames, ignore_index=True, sort=False)
    full = full.drop_duplicates(subset=["game_pk"], keep="last")

    out_path = dirs["processed_dir"] / "weather_game.parquet"
    write_parquet(full, out_path)
    logging.info("weather_game rows=%s path=%s", len(full), out_path)
    print(f"weather_out={out_path}")


if __name__ == "__main__":
    main()
