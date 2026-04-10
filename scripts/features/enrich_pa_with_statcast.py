#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

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


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def rate(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = np.where((den.notna()) & (den != 0), num / den, np.nan)
    return pd.Series(out, index=num.index, dtype=float)


def fetch_statcast_daily(start_date: str, end_date: str) -> pd.DataFrame:
    from pybaseball import statcast

    print(f"Fetching Statcast from {start_date} to {end_date}")
    sc = statcast(start_dt=start_date, end_dt=end_date)

    if sc is None or sc.empty:
        print("⚠️ Statcast returned empty")
        return pd.DataFrame()

    sc = sc.copy()

    keep_cols = [
        "game_date",
        "game_pk",
        "batter",
        "player_name",
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

    if "game_date" in sc.columns:
        sc["game_date"] = safe_date_series(sc["game_date"])
    if "game_pk" in sc.columns:
        sc["game_pk"] = to_num(sc["game_pk"]).astype("Int64")
    if "batter" in sc.columns:
        sc["batter"] = to_num(sc["batter"]).astype("Int64")

    for c in ["launch_speed", "launch_angle", "hit_distance_sc", "hc_x", "hc_y"]:
        if c in sc.columns:
            sc[c] = to_num(sc[c])

    for c in ["events", "description", "bb_type", "player_name"]:
        if c in sc.columns:
            sc[c] = sc[c].astype("string")

    return sc


def build_statcast_batter_daily(sc: pd.DataFrame) -> pd.DataFrame:
    if sc.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "batter_id",
                "launch_speed",
                "launch_angle",
                "hit_distance_sc",
                "bb_type",
                "hc_x",
                "hc_y",
                "barrels",
                "hard_hit",
                "bbe",
                "barrel_rate_sc",
                "hard_hit_rate_sc",
            ]
        )

    required = ["game_date", "batter"]
    missing = [c for c in required if c not in sc.columns]
    if missing:
        raise ValueError(f"Statcast payload missing required columns: {missing}")

    sc = sc.copy()

    ls = sc["launch_speed"] if "launch_speed" in sc.columns else pd.Series(np.nan, index=sc.index)
    la = sc["launch_angle"] if "launch_angle" in sc.columns else pd.Series(np.nan, index=sc.index)

    sc["is_bbe"] = ls.notna().astype(int)
    sc["is_hard_hit"] = (ls >= 95).fillna(False).astype(int)
    sc["is_barrel"] = ((ls >= 98) & la.between(26, 30, inclusive="both")).fillna(False).astype(int)

    grouped = (
        sc.groupby(["game_date", "batter"], dropna=False)
        .agg(
            launch_speed=("launch_speed", "mean"),
            launch_angle=("launch_angle", "mean"),
            hit_distance_sc=("hit_distance_sc", "mean"),
            hc_x=("hc_x", "mean"),
            hc_y=("hc_y", "mean"),
            bb_type=("bb_type", lambda x: x.dropna().iloc[0] if x.dropna().shape[0] else pd.NA),
            barrels=("is_barrel", "sum"),
            hard_hit=("is_hard_hit", "sum"),
            bbe=("is_bbe", "sum"),
        )
        .reset_index()
    )

    grouped = grouped.rename(columns={"batter": "batter_id"})
    grouped["batter_id"] = to_num(grouped["batter_id"]).astype("Int64")

    grouped["barrel_rate_sc"] = rate(grouped["barrels"], grouped["bbe"])
    grouped["hard_hit_rate_sc"] = rate(grouped["hard_hit"], grouped["bbe"])

    return grouped


def enrich_pa_with_statcast(pa: pd.DataFrame, sc_daily: pd.DataFrame) -> pd.DataFrame:
    out = pa.copy()

    if out.empty:
        return out

    out["game_date"] = safe_date_series(out["game_date"])
    out["batter_id"] = to_num(out["batter_id"]).astype("Int64")

    if sc_daily.empty:
        print("⚠️ No Statcast daily table available; returning PA unchanged with empty Statcast columns.")
        for c in [
            "launch_speed",
            "launch_angle",
            "hit_distance_sc",
            "bb_type",
            "hc_x",
            "hc_y",
            "barrels_sc",
            "hard_hit_sc",
            "bbe_sc",
            "barrel_rate_sc",
            "hard_hit_rate_sc",
        ]:
            if c not in out.columns:
                out[c] = np.nan
        return out

    merged = out.merge(
        sc_daily.rename(
            columns={
                "barrels": "barrels_sc",
                "hard_hit": "hard_hit_sc",
                "bbe": "bbe_sc",
            }
        ),
        how="left",
        on=["game_date", "batter_id"],
        suffixes=("", "_scdup"),
    )

    # Prefer real Statcast values for these fields
    for c in ["launch_speed", "launch_angle", "hit_distance_sc", "bb_type", "hc_x", "hc_y"]:
        if c not in merged.columns:
            merged[c] = np.nan

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich season PA parquet with Statcast daily batter data.")
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

    required_cols = ["game_date", "batter_id"]
    missing = [c for c in required_cols if c not in pa.columns]
    if missing:
        raise ValueError(f"PA file missing required columns: {missing}")

    pa["game_date"] = safe_date_series(pa["game_date"])

    start_date = args.start_date or str(pd.Series(pa["game_date"]).dropna().min())
    end_date = args.end_date or str(pd.Series(pa["game_date"]).dropna().max())

    sc_raw = fetch_statcast_daily(start_date, end_date)
    print(f"Raw Statcast rows: {len(sc_raw):,}")

    sc_daily = build_statcast_batter_daily(sc_raw)
    print(f"Daily Statcast batter rows: {len(sc_daily):,}")

    enriched = enrich_pa_with_statcast(pa, sc_daily)

    for c in [
        "launch_speed",
        "launch_angle",
        "hit_distance_sc",
        "bb_type",
        "hc_x",
        "hc_y",
        "barrel_rate_sc",
        "hard_hit_rate_sc",
    ]:
        if c in enriched.columns:
            if str(enriched[c].dtype) == "string" or enriched[c].dtype == object:
                pct = enriched[c].notna().mean() * 100.0
            else:
                pct = pd.to_numeric(enriched[c], errors="coerce").notna().mean() * 100.0
            print(f"{c} non-null %: {pct:.2f}")

    enriched.to_parquet(pa_path, index=False)
    print(f"✅ Enriched PA written: {pa_path}")

    if args.write_debug:
        debug_dir = processed_dir / "debug"
        ensure_dir(debug_dir)

        raw_path = debug_dir / f"statcast_raw_{args.season}_{start_date}_{end_date}.parquet"
        daily_path = debug_dir / f"statcast_batter_daily_{args.season}_{start_date}_{end_date}.parquet"
        enriched_path = debug_dir / f"pa_enriched_preview_{args.season}_{start_date}_{end_date}.parquet"

        sc_raw.to_parquet(raw_path, index=False)
        sc_daily.to_parquet(daily_path, index=False)
        enriched.head(50000).to_parquet(enriched_path, index=False)

        print(f"debug_statcast_raw={raw_path}")
        print(f"debug_statcast_batter_daily={daily_path}")
        print(f"debug_pa_enriched_preview={enriched_path}")


if __name__ == "__main__":
    main() 