#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = "configs/project.yaml"


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_data_root(config: dict) -> Path:
    drive_root = config.get("drive_data_root", "joeplumber-mlb/data")
    return Path("/content/drive/MyDrive") / drive_root


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = False) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {list(candidates)}")
    return None


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_team_abbr(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    mapping = {
        "ATH": "OAK",
        "ATHLETICS": "OAK",
        "A'S": "OAK",
        "WSH": "WSN",
        "WAS": "WSN",
        "AZ": "ARI",
        "D-BACKS": "ARI",
        "CHW": "CWS",
    }
    return s.replace(mapping)


def last_non_null(x: pd.Series):
    x = x.dropna()
    if x.empty:
        return np.nan
    return x.iloc[-1]


def safe_dt_string(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.strftime("%Y-%m-%d")


def fetch_statcast_for_range(start_date: str, end_date: str) -> pd.DataFrame:
    from pybaseball import statcast

    print(f"Fetching Statcast: {start_date} -> {end_date}")
    sc = statcast(start_dt=start_date, end_dt=end_date)

    if sc is None or sc.empty:
        return pd.DataFrame()

    sc = sc.copy()

    keep_cols = [
        "game_pk",
        "game_date",
        "batter",
        "pitcher",
        "at_bat_number",
        "pitch_number",
        "events",
        "description",
        "launch_speed",
        "launch_angle",
        "hit_distance_sc",
        "bb_type",
        "hc_x",
        "hc_y",
    ]
    keep_cols = [c for c in keep_cols if c in sc.columns]
    sc = sc[keep_cols].copy()

    if "game_pk" in sc.columns:
        sc["game_pk"] = to_num(sc["game_pk"]).astype("Int64")
    if "batter" in sc.columns:
        sc["batter"] = to_num(sc["batter"]).astype("Int64")
    if "pitcher" in sc.columns:
        sc["pitcher"] = to_num(sc["pitcher"]).astype("Int64")
    if "at_bat_number" in sc.columns:
        sc["at_bat_number"] = to_num(sc["at_bat_number"]).astype("Int64")
    if "pitch_number" in sc.columns:
        sc["pitch_number"] = to_num(sc["pitch_number"]).astype("Int64")

    if "game_date" in sc.columns:
        sc["game_date"] = safe_dt_string(sc["game_date"])

    for c in ["launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y"]:
        if c in sc.columns:
            sc[c] = to_num(sc[c])

    for c in ["events", "description", "bb_type"]:
        if c in sc.columns:
            sc[c] = sc[c].astype("string")

    return sc


def build_statcast_pa_table(sc: pd.DataFrame) -> pd.DataFrame:
    if sc.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "game_date",
                "batter_id",
                "pitcher_id",
                "pa_index",
                "launch_speed",
                "launch_angle",
                "hit_distance_sc",
                "bb_type",
                "hc_x",
                "hc_y",
                "sc_events",
                "sc_description",
            ]
        )

    required = ["game_pk", "game_date", "batter", "pitcher", "at_bat_number"]
    missing = [c for c in required if c not in sc.columns]
    if missing:
        raise ValueError(f"Statcast payload missing required columns: {missing}")

    agg = (
        sc.groupby(["game_pk", "game_date", "batter", "pitcher", "at_bat_number"], dropna=False)
        .agg(
            launch_speed=("launch_speed", last_non_null),
            launch_angle=("launch_angle", last_non_null),
            hit_distance_sc=("hit_distance_sc", last_non_null),
            bb_type=("bb_type", last_non_null),
            hc_x=("hc_x", last_non_null),
            hc_y=("hc_y", last_non_null),
            sc_events=("events", last_non_null),
            sc_description=("description", last_non_null),
        )
        .reset_index()
    )

    agg = agg.rename(
        columns={
            "batter": "batter_id",
            "pitcher": "pitcher_id",
            "at_bat_number": "pa_index",
        }
    )

    agg["game_pk"] = to_num(agg["game_pk"]).astype("Int64")
    agg["batter_id"] = to_num(agg["batter_id"]).astype("Int64")
    agg["pitcher_id"] = to_num(agg["pitcher_id"]).astype("Int64")
    agg["pa_index"] = to_num(agg["pa_index"]).astype("Int64")
    agg["game_date"] = agg["game_date"].astype("string")

    return agg


def enrich_pa_with_statcast(pa: pd.DataFrame, sc_pa: pd.DataFrame) -> pd.DataFrame:
    if pa.empty:
        return pa.copy()
    if sc_pa.empty:
        out = pa.copy()
        for c in [
            "launch_speed",
            "launch_angle",
            "hit_distance_sc",
            "bb_type",
            "hc_x",
            "hc_y",
        ]:
            if c not in out.columns:
                out[c] = np.nan
        return out

    out = pa.copy()

    if "game_date" in out.columns:
        out["game_date"] = safe_dt_string(out["game_date"])
    if "game_pk" in out.columns:
        out["game_pk"] = to_num(out["game_pk"]).astype("Int64")
    if "batter_id" in out.columns:
        out["batter_id"] = to_num(out["batter_id"]).astype("Int64")
    if "pitcher_id" in out.columns:
        out["pitcher_id"] = to_num(out["pitcher_id"]).astype("Int64")
    if "pa_index" in out.columns:
        out["pa_index"] = to_num(out["pa_index"]).astype("Int64")

    merge_keys = ["game_pk", "game_date", "batter_id", "pitcher_id", "pa_index"]
    out = out.merge(
        sc_pa,
        on=merge_keys,
        how="left",
        suffixes=("", "_sc"),
    )

    for c in [
        "launch_speed",
        "launch_angle",
        "hit_distance_sc",
        "bb_type",
        "hc_x",
        "hc_y",
    ]:
        if c not in out.columns:
            out[c] = np.nan

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill Statcast EV/LA fields into season PA parquet.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--config", type=str, default=CONFIG_PATH)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--write-debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    data_root = resolve_data_root(config)
    processed_dir = data_root / "processed"
    by_season_dir = processed_dir / "by_season"
    ensure_dir(by_season_dir)

    pa_path = by_season_dir / f"pa_{args.season}.parquet"
    if not pa_path.exists():
        raise FileNotFoundError(f"Missing PA file: {pa_path}")

    pa = pd.read_parquet(pa_path).copy()
    if pa.empty:
        raise ValueError(f"PA file is empty: {pa_path}")

    if "game_date" not in pa.columns:
        raise ValueError("PA file missing game_date")
    if "game_pk" not in pa.columns:
        raise ValueError("PA file missing game_pk")
    if "batter_id" not in pa.columns:
        raise ValueError("PA file missing batter_id")
    if "pitcher_id" not in pa.columns:
        raise ValueError("PA file missing pitcher_id")
    if "pa_index" not in pa.columns:
        raise ValueError("PA file missing pa_index")

    pa["game_date"] = safe_dt_string(pa["game_date"])

    start_date = args.start_date or str(pa["game_date"].dropna().min())
    end_date = args.end_date or str(pa["game_date"].dropna().max())

    sc = fetch_statcast_for_range(start_date=start_date, end_date=end_date)
    print(f"Raw Statcast rows: {len(sc):,}")

    sc_pa = build_statcast_pa_table(sc)
    print(f"Statcast PA rows: {len(sc_pa):,}")

    enriched = enrich_pa_with_statcast(pa, sc_pa)

    for c in ["launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y"]:
        if c in enriched.columns:
            print(f"{c} non-null %: {enriched[c].notna().mean() * 100:.2f}")

    enriched.to_parquet(pa_path, index=False)
    print(f"✅ Enriched PA written: {pa_path}")

    if args.write_debug:
        debug_dir = processed_dir / "debug"
        ensure_dir(debug_dir)
        sc_raw_path = debug_dir / f"statcast_raw_{args.season}_{start_date}_{end_date}.parquet"
        sc_pa_path = debug_dir / f"statcast_pa_{args.season}_{start_date}_{end_date}.parquet"
        sc.to_parquet(sc_raw_path, index=False)
        sc_pa.to_parquet(sc_pa_path, index=False)
        print(f"debug_statcast_raw={sc_raw_path}")
        print(f"debug_statcast_pa={sc_pa_path}")


if __name__ == "__main__":
    main()