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
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header

NORMALIZED_COLUMNS = [
    "game_pk",
    "game_date",
    "season",
    "home_team",
    "away_team",
    "venue_id",
    "temperature",
    "wind_speed",
    "wind_direction",
    "weather_wind_out",
    "weather_wind_in",
    "weather_crosswind",
]


LIVE_WEATHER_RE = re.compile(r"weather_game_(\d{4})_(\d{4}-\d{2}-\d{2})\.parquet$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build normalized game-level weather table.")
    p.add_argument("--season-start", type=int, default=2019)
    p.add_argument("--season-end", type=int, default=2026)
    p.add_argument("--season", type=int, default=None, help="Optional single season shortcut.")
    p.add_argument("--date", type=str, default=None, help="Optional YYYY-MM-DD to focus live weather normalization.")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], cands: list[str]) -> str | None:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def _parse_wind_direction(raw: pd.Series) -> pd.Series:
    s = raw.astype("string").fillna("").str.lower().str.strip()
    out = pd.Series(pd.NA, index=raw.index, dtype="string")
    out = out.mask(s.str.contains(r"\bout\b|out to|to (left|right|center|cf|lf|rf)"), "out")
    out = out.mask(s.str.contains(r"\bin\b|in from|from (left|right|center|cf|lf|rf)"), "in")
    out = out.mask(s.str.contains(r"cross|left to right|right to left"), "cross")
    return out


def _parse_wind_speed(raw: pd.Series) -> pd.Series:
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().any():
        return num
    txt = raw.astype("string").fillna("").str.lower()
    extracted = txt.str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _derive_wind_flags(wind_direction: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    s = wind_direction.astype("string").fillna("").str.lower()
    wind_out = s.str.contains(r"\bout\b|out to|to (left|right|center|cf|lf|rf)|^out$").astype(float)
    wind_in = s.str.contains(r"\bin\b|in from|from (left|right|center|cf|lf|rf)|^in$").astype(float)
    cross = s.str.contains(r"cross|left to right|right to left|^cross$").astype(float)
    return wind_out, wind_in, cross


def normalize_weather_frame(df: pd.DataFrame, fallback_season: int | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)

    gd_col = _pick(list(df.columns), ["game_date", "date"])
    ht_col = _pick(list(df.columns), ["home_team", "home_team_abbr"])
    at_col = _pick(list(df.columns), ["away_team", "away_team_abbr"])
    venue_col = _pick(list(df.columns), ["venue_id", "park_id"])
    season_col = _pick(list(df.columns), ["season"])

    temp_col = _pick(list(df.columns), ["temperature", "temp_f", "temperature_f", "game_temp", "weather_temp"])
    wind_spd_col = _pick(list(df.columns), ["wind_speed", "wind_mph", "weather_wind", "wind"])
    wind_dir_col = _pick(list(df.columns), ["wind_direction", "wind_dir", "weather_wind_direction"])

    out = pd.DataFrame(index=df.index)
    out["game_pk"] = pd.to_numeric(df.get("game_pk"), errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(df[gd_col], errors="coerce") if gd_col else pd.NaT
    if season_col:
        out["season"] = pd.to_numeric(df[season_col], errors="coerce").astype("Int64")
    else:
        out["season"] = pd.to_numeric(out["game_date"].dt.year, errors="coerce").astype("Int64")
        if fallback_season is not None:
            out["season"] = out["season"].fillna(int(fallback_season)).astype("Int64")

    out["home_team"] = df[ht_col].astype("string") if ht_col else pd.Series(pd.NA, index=df.index, dtype="string")
    out["away_team"] = df[at_col].astype("string") if at_col else pd.Series(pd.NA, index=df.index, dtype="string")
    out["venue_id"] = pd.to_numeric(df[venue_col], errors="coerce").astype("Int64") if venue_col else pd.array([pd.NA] * len(df), dtype="Int64")

    out["temperature"] = pd.to_numeric(df[temp_col], errors="coerce") if temp_col else np.nan
    wind_raw = df[wind_spd_col] if wind_spd_col else pd.Series(pd.NA, index=df.index)
    out["wind_speed"] = _parse_wind_speed(wind_raw)

    if wind_dir_col:
        out["wind_direction"] = df[wind_dir_col].astype("string")
    else:
        out["wind_direction"] = _parse_wind_direction(wind_raw)

    out["wind_direction"] = out["wind_direction"].astype("string").str.strip()

    wind_out_col = _pick(list(df.columns), ["weather_wind_out"])
    wind_in_col = _pick(list(df.columns), ["weather_wind_in"])
    wind_cross_col = _pick(list(df.columns), ["weather_crosswind"])

    parsed_out, parsed_in, parsed_cross = _derive_wind_flags(out["wind_direction"])
    out["weather_wind_out"] = pd.to_numeric(df[wind_out_col], errors="coerce") if wind_out_col else np.nan
    out["weather_wind_in"] = pd.to_numeric(df[wind_in_col], errors="coerce") if wind_in_col else np.nan
    out["weather_crosswind"] = pd.to_numeric(df[wind_cross_col], errors="coerce") if wind_cross_col else np.nan
    out["weather_wind_out"] = out["weather_wind_out"].fillna(parsed_out)
    out["weather_wind_in"] = out["weather_wind_in"].fillna(parsed_in)
    out["weather_crosswind"] = out["weather_crosswind"].fillna(parsed_cross)

    out = out.dropna(subset=["game_pk"]).copy()
    return out[NORMALIZED_COLUMNS]


def _dedupe_weather(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    score_cols = ["temperature", "wind_speed", "wind_direction", "weather_wind_out", "weather_wind_in", "weather_crosswind"]
    dedupe = df.copy()
    dedupe["_quality"] = dedupe[score_cols].notna().sum(axis=1)
    dedupe = dedupe.sort_values(["_quality", "game_pk", "game_date"], ascending=[False, True, False], kind="mergesort")
    dedupe = dedupe.drop_duplicates(subset=["game_pk"], keep="first")
    dedupe = dedupe.drop(columns=["_quality"], errors="ignore")
    return dedupe


def _collect_weather_sources(dirs: dict[str, Path], season_start: int, season_end: int, date: str | None) -> list[tuple[Path, int | None, str]]:
    sources: list[tuple[Path, int | None, str]] = []
    for season in range(season_start, season_end + 1):
        p = dirs["raw_dir"] / "by_season" / f"weather_game_{season}.parquet"
        if p.exists():
            sources.append((p, season, "historical"))

    live_dir = dirs["raw_dir"] / "live"
    if live_dir.exists():
        for p in sorted(live_dir.glob("weather_game_*.parquet")):
            m = LIVE_WEATHER_RE.search(p.name)
            if m:
                season = int(m.group(1))
                file_date = m.group(2)
                if season_start <= season <= season_end and (date is None or file_date == date):
                    sources.append((p, season, "live"))
    return sources


def build_weather_game_table(
    dirs: dict[str, Path],
    season_start: int,
    season_end: int,
    date: str | None = None,
) -> tuple[Path, dict[str, Path]]:
    frames: list[pd.DataFrame] = []
    live_outputs: dict[str, Path] = {}
    sources = _collect_weather_sources(dirs, season_start, season_end, date)
    if not sources:
        raise FileNotFoundError(
            f"No weather source files found under {dirs['raw_dir'] / 'by_season'} and {dirs['raw_dir'] / 'live'}"
        )

    for path, fallback_season, source_type in sources:
        raw_df = read_parquet(path)
        norm = normalize_weather_frame(raw_df, fallback_season=fallback_season)
        if norm.empty:
            logging.warning("weather source normalized to empty rows path=%s", path)
            continue
        norm["_source_type"] = source_type
        norm["_source_path"] = str(path)
        frames.append(norm)

        m = LIVE_WEATHER_RE.search(path.name)
        if source_type == "live" and m:
            season = int(m.group(1))
            slate_date = m.group(2)
            live_out = dirs["processed_dir"] / "live" / f"weather_game_{season}_{slate_date}.parquet"
            deduped = _dedupe_weather(norm.drop(columns=["_source_type", "_source_path"], errors="ignore"))
            write_parquet(deduped, live_out)
            live_outputs[f"{season}_{slate_date}"] = live_out
            logging.info("processed live weather rows=%s out=%s", len(deduped), live_out)

    if not frames:
        raise RuntimeError("Weather source files were found but all normalized frames were empty.")

    full = pd.concat(frames, ignore_index=True, sort=False)
    full = _dedupe_weather(full)
    full = full[NORMALIZED_COLUMNS]

    out_path = dirs["processed_dir"] / "weather_game.parquet"
    write_parquet(full, out_path)
    logging.info("weather_game rows=%s path=%s", len(full), out_path)
    print(f"Row count [weather_game]: {len(full):,}")
    print(f"Writing to: {out_path.resolve()}")
    return out_path, live_outputs


def main() -> None:
    args = parse_args()
    season_start = args.season if args.season is not None else args.season_start
    season_end = args.season if args.season is not None else args.season_end

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "build_weather_game.log")
    log_header("scripts/reference/build_weather_game.py", repo_root, config_path, dirs)

    out_path, live_outputs = build_weather_game_table(
        dirs=dirs,
        season_start=season_start,
        season_end=season_end,
        date=args.date,
    )
    for slate_key, slate_path in sorted(live_outputs.items()):
        print(f"live_weather_out[{slate_key}]={slate_path}")
    print(f"weather_out={out_path}")


if __name__ == "__main__":
    main()
