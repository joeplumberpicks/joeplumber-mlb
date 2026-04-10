#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml


CONFIG_PATH = Path("configs/project.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_data_root(config: dict) -> Path:
    drive_data_root = config.get("drive_data_root", "joeplumber-mlb/data")
    return Path("/content/drive/MyDrive") / drive_data_root


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def first_existing_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    num = pd.to_numeric(num, errors="coerce")
    return (num / den).replace([np.inf, -np.inf], np.nan)


def rolling_sum(shifted: pd.Series, group_key: pd.Series, window: int) -> pd.Series:
    return (
        shifted.groupby(group_key)
        .rolling(window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )


def rolling_mean(shifted: pd.Series, group_key: pd.Series, window: int) -> pd.Series:
    return (
        shifted.groupby(group_key)
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )


def add_missing_numeric_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_batter_rollings(df: pd.DataFrame) -> pd.DataFrame:
    batter_id_col = first_existing_col(df, ["batter_id", "batter"])
    date_col = first_existing_col(df, ["game_date"])

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([batter_id_col, date_col, "game_pk"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Backward-compatible normalization of expected input columns
    # ------------------------------------------------------------------
    rename_map = {}
    if "avg_exit_velocity" in df.columns and "avg_ev" not in df.columns:
        rename_map["avg_exit_velocity"] = "avg_ev"
    if "avg_launch_angle" in df.columns and "avg_la" not in df.columns:
        rename_map["avg_launch_angle"] = "avg_la"
    if "hardhit_rate" in df.columns and "hard_hit_rate" not in df.columns:
        rename_map["hardhit_rate"] = "hard_hit_rate"
    if rename_map:
        df = df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # Ensure required columns exist
    # ------------------------------------------------------------------
    required_numeric = [
        "pa",
        "hits",
        "hr",
        "rbi",
        "tb",
        "bb",
        "so",
        "barrels",
        "barrel_rate",
        "hardhit",
        "hard_hit_rate",
        "avg_ev",
        "avg_la",
        "iso",
        "hr_per_pa",
        "tb_per_pa",
        "bb_rate",
        "k_rate",
        "fb_rate",
        "pulled_air_rate",
    ]
    df = add_missing_numeric_cols(df, required_numeric)

    # Optional old-style components if available
    optional_old = ["ab", "1b", "2b", "3b", "bip", "fb", "pull_air", "hard_hit"]
    df = add_missing_numeric_cols(df, optional_old)

    # ------------------------------------------------------------------
    # Fill in derivable fields if absent
    # ------------------------------------------------------------------
    if df["hard_hit_rate"].isna().all():
        if "hard_hit" in df.columns and "bip" in df.columns and df["hard_hit"].notna().any():
            df["hard_hit_rate"] = safe_rate(df["hard_hit"], df["bip"])
        elif "hardhit" in df.columns:
            df["hard_hit_rate"] = safe_rate(df["hardhit"], df["pa"])

    if df["barrel_rate"].isna().all() and "barrels" in df.columns:
        df["barrel_rate"] = safe_rate(df["barrels"], df["pa"])

    if df["hr_per_pa"].isna().all():
        df["hr_per_pa"] = safe_rate(df["hr"], df["pa"])

    if df["tb_per_pa"].isna().all():
        df["tb_per_pa"] = safe_rate(df["tb"], df["pa"])

    if df["bb_rate"].isna().all():
        df["bb_rate"] = safe_rate(df["bb"], df["pa"])

    if df["k_rate"].isna().all():
        df["k_rate"] = safe_rate(df["so"], df["pa"])

    if df["iso"].isna().all():
        if "ab" in df.columns and df["ab"].notna().any():
            singles = df["1b"] if "1b" in df.columns else 0
            doubles = df["2b"] if "2b" in df.columns else 0
            triples = df["3b"] if "3b" in df.columns else 0
            extra_bases = pd.to_numeric(doubles, errors="coerce").fillna(0)
            extra_bases += 2 * pd.to_numeric(triples, errors="coerce").fillna(0)
            extra_bases += 3 * pd.to_numeric(df["hr"], errors="coerce").fillna(0)
            df["iso"] = safe_rate(extra_bases, df["ab"])
        else:
            df["iso"] = safe_rate(df["tb"] - df["hits"], df["pa"])

    if df["fb_rate"].isna().all() and "fb" in df.columns and "bip" in df.columns and df["fb"].notna().any():
        df["fb_rate"] = safe_rate(df["fb"], df["bip"])

    if df["pulled_air_rate"].isna().all() and "pull_air" in df.columns and "bip" in df.columns and df["pull_air"].notna().any():
        df["pulled_air_rate"] = safe_rate(df["pull_air"], df["bip"])

    # If avg_ev / avg_la are still absent, try old columns directly
    if df["avg_ev"].isna().all() and "ev_mean" in df.columns:
        df["avg_ev"] = pd.to_numeric(df["ev_mean"], errors="coerce")
    if df["avg_la"].isna().all() and "la_mean" in df.columns:
        df["avg_la"] = pd.to_numeric(df["la_mean"], errors="coerce")

    grouped = df.groupby(batter_id_col, sort=False)
    windows = [3, 7, 15, 30]

    # ------------------------------------------------------------------
    # Shifted event totals
    # ------------------------------------------------------------------
    pa_s = grouped["pa"].shift(1)
    hits_s = grouped["hits"].shift(1)
    hr_s = grouped["hr"].shift(1)
    rbi_s = grouped["rbi"].shift(1)
    tb_s = grouped["tb"].shift(1)
    bb_s = grouped["bb"].shift(1)
    so_s = grouped["so"].shift(1)

    # ------------------------------------------------------------------
    # Shifted game-level rates/means
    # ------------------------------------------------------------------
    barrel_rate_s = grouped["barrel_rate"].shift(1)
    hard_hit_rate_s = grouped["hard_hit_rate"].shift(1)
    avg_ev_s = grouped["avg_ev"].shift(1)
    avg_la_s = grouped["avg_la"].shift(1)
    iso_s = grouped["iso"].shift(1)
    hr_per_pa_s = grouped["hr_per_pa"].shift(1)
    tb_per_pa_s = grouped["tb_per_pa"].shift(1)
    bb_rate_s = grouped["bb_rate"].shift(1)
    k_rate_s = grouped["k_rate"].shift(1)
    fb_rate_s = grouped["fb_rate"].shift(1)
    pulled_air_rate_s = grouped["pulled_air_rate"].shift(1)

    for w in windows:
        # Volume rollups
        hits_roll = rolling_sum(hits_s, df[batter_id_col], w)
        hr_roll = rolling_sum(hr_s, df[batter_id_col], w)
        rbi_roll = rolling_sum(rbi_s, df[batter_id_col], w)
        tb_roll = rolling_sum(tb_s, df[batter_id_col], w)
        bb_roll = rolling_sum(bb_s, df[batter_id_col], w)
        so_roll = rolling_sum(so_s, df[batter_id_col], w)

        # Rate/mean rollups
        barrel_rate_roll = rolling_mean(barrel_rate_s, df[batter_id_col], w)
        hard_hit_rate_roll = rolling_mean(hard_hit_rate_s, df[batter_id_col], w)
        avg_ev_roll = rolling_mean(avg_ev_s, df[batter_id_col], w)
        avg_la_roll = rolling_mean(avg_la_s, df[batter_id_col], w)
        iso_roll = rolling_mean(iso_s, df[batter_id_col], w)
        hr_per_pa_roll = rolling_mean(hr_per_pa_s, df[batter_id_col], w)
        tb_per_pa_roll = rolling_mean(tb_per_pa_s, df[batter_id_col], w)
        bb_rate_roll = rolling_mean(bb_rate_s, df[batter_id_col], w)
        k_rate_roll = rolling_mean(k_rate_s, df[batter_id_col], w)
        fb_rate_roll = rolling_mean(fb_rate_s, df[batter_id_col], w)
        pulled_air_rate_roll = rolling_mean(pulled_air_rate_s, df[batter_id_col], w)

        # Legacy batting average if AB exists, otherwise use hit/pa proxy
        if "ab" in df.columns and df["ab"].notna().any():
            ab_s = grouped["ab"].shift(1)
            ab_roll = rolling_sum(ab_s, df[batter_id_col], w)
            ba_roll = safe_rate(hits_roll, ab_roll)
        else:
            pa_roll = rolling_sum(pa_s, df[batter_id_col], w)
            ba_roll = safe_rate(hits_roll, pa_roll)

        df[f"bat_hits_roll{w}"] = hits_roll
        df[f"bat_hr_roll{w}"] = hr_roll
        df[f"bat_rbi_roll{w}"] = rbi_roll
        df[f"bat_tb_roll{w}"] = tb_roll
        df[f"bat_bb_roll{w}"] = bb_roll
        df[f"bat_so_roll{w}"] = so_roll

        df[f"bat_ba_roll{w}"] = ba_roll
        df[f"bat_hr_per_pa_roll{w}"] = hr_per_pa_roll
        df[f"bat_tb_per_pa_roll{w}"] = tb_per_pa_roll
        df[f"bat_bb_rate_roll{w}"] = bb_rate_roll
        df[f"bat_k_rate_roll{w}"] = k_rate_roll

        df[f"bat_barrel_rate_roll{w}"] = barrel_rate_roll
        df[f"bat_hard_hit_rate_roll{w}"] = hard_hit_rate_roll
        df[f"bat_fb_rate_roll{w}"] = fb_rate_roll
        df[f"bat_pulled_air_rate_roll{w}"] = pulled_air_rate_roll

        df[f"bat_avg_ev_roll{w}"] = avg_ev_roll
        df[f"bat_avg_la_roll{w}"] = avg_la_roll
        df[f"bat_iso_roll{w}"] = iso_roll

    return df


def add_pitcher_rollings(df: pd.DataFrame) -> pd.DataFrame:
    pitcher_id_col = first_existing_col(df, ["pitcher_id", "pitcher"])
    date_col = first_existing_col(df, ["game_date"])

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([pitcher_id_col, date_col, "game_pk"]).reset_index(drop=True)

    required_numeric = [
        "batters_faced",
        "bb_allowed",
        "so",
        "hr_allowed",
        "barrels_allowed",
        "hardhit_allowed",
        "barrel_rate",
        "hard_hit_rate",
        "bb_rate",
        "k_rate",
        "hr_per_bf",
        "hr9",
    ]
    df = add_missing_numeric_cols(df, required_numeric)

    if df["bb_rate"].isna().all():
        df["bb_rate"] = safe_rate(df["bb_allowed"], df["batters_faced"])
    if df["k_rate"].isna().all():
        df["k_rate"] = safe_rate(df["so"], df["batters_faced"])
    if df["barrel_rate"].isna().all():
        df["barrel_rate"] = safe_rate(df["barrels_allowed"], df["batters_faced"])
    if df["hard_hit_rate"].isna().all():
        df["hard_hit_rate"] = safe_rate(df["hardhit_allowed"], df["batters_faced"])
    if df["hr_per_bf"].isna().all():
        df["hr_per_bf"] = safe_rate(df["hr_allowed"], df["batters_faced"])
    if df["hr9"].isna().all():
        df["hr9"] = df["hr_per_bf"] * 27.0

    grouped = df.groupby(pitcher_id_col, sort=False)
    windows = [3, 7, 15, 30]

    hr_allowed_s = grouped["hr_allowed"].shift(1)
    bb_allowed_s = grouped["bb_allowed"].shift(1)
    so_s = grouped["so"].shift(1)

    hr9_s = grouped["hr9"].shift(1)
    bb_rate_s = grouped["bb_rate"].shift(1)
    k_rate_s = grouped["k_rate"].shift(1)
    barrel_rate_s = grouped["barrel_rate"].shift(1)
    hard_hit_rate_s = grouped["hard_hit_rate"].shift(1)

    for w in windows:
        df[f"pit_hr_allowed_roll{w}"] = rolling_sum(hr_allowed_s, df[pitcher_id_col], w)
        df[f"pit_bb_allowed_roll{w}"] = rolling_sum(bb_allowed_s, df[pitcher_id_col], w)
        df[f"pit_so_roll{w}"] = rolling_sum(so_s, df[pitcher_id_col], w)

        df[f"pit_hr9_roll{w}"] = rolling_mean(hr9_s, df[pitcher_id_col], w)
        df[f"pit_bb_rate_roll{w}"] = rolling_mean(bb_rate_s, df[pitcher_id_col], w)
        df[f"pit_k_rate_roll{w}"] = rolling_mean(k_rate_s, df[pitcher_id_col], w)
        df[f"pit_barrel_rate_roll{w}"] = rolling_mean(barrel_rate_s, df[pitcher_id_col], w)
        df[f"pit_hard_hit_rate_roll{w}"] = rolling_mean(hard_hit_rate_s, df[pitcher_id_col], w)

    return df


def main() -> None:
    config = load_config()
    data_root = resolve_data_root(config)
    processed_dir = data_root / "processed"

    ensure_dir(processed_dir)

    batter_in = processed_dir / "batter_game_statcast.parquet"
    pitcher_in = processed_dir / "pitcher_game_statcast.parquet"

    if not batter_in.exists():
        raise FileNotFoundError(f"Missing batter game table: {batter_in}")
    if not pitcher_in.exists():
        raise FileNotFoundError(f"Missing pitcher game table: {pitcher_in}")

    bat = pd.read_parquet(batter_in)
    pit = pd.read_parquet(pitcher_in)

    if bat.empty:
        raise ValueError(f"Batter game table is empty: {batter_in}")
    if pit.empty:
        raise ValueError(f"Pitcher game table is empty: {pitcher_in}")

    bat = add_batter_rollings(bat)
    pit = add_pitcher_rollings(pit)

    bat_out = processed_dir / "batter_statcast_rolling.parquet"
    pit_out = processed_dir / "pitcher_statcast_rolling.parquet"

    ensure_dir(bat_out.parent)
    ensure_dir(pit_out.parent)

    bat.to_parquet(bat_out, index=False)
    pit.to_parquet(pit_out, index=False)

    print("✅ cross-season statcast rolling built")
    print(f"batter_rows={len(bat):,}")
    print(f"pitcher_rows={len(pit):,}")
    print(f"bat_out={bat_out}")
    print(f"pit_out={pit_out}")


if __name__ == "__main__":
    main()