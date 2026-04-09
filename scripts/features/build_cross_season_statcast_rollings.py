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


def add_batter_rollings(df: pd.DataFrame) -> pd.DataFrame:
    batter_id_col = first_existing_col(df, ["batter_id", "batter"])
    date_col = first_existing_col(df, ["game_date"])
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Core numeric coercions
    numeric_cols = [
        "pa", "ab", "hits", "1b", "2b", "3b", "hr", "bb", "so",
        "tb", "barrels", "hard_hit", "bip", "fb", "pull_air",
        "avg_exit_velocity", "avg_launch_angle"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values([batter_id_col, date_col]).reset_index(drop=True)

    # Derived columns
    if "tb" not in df.columns:
        df["tb"] = (
            df.get("1b", 0).fillna(0)
            + 2 * df.get("2b", 0).fillna(0)
            + 3 * df.get("3b", 0).fillna(0)
            + 4 * df.get("hr", 0).fillna(0)
        )

    if "hits" not in df.columns:
        df["hits"] = (
            df.get("1b", 0).fillna(0)
            + df.get("2b", 0).fillna(0)
            + df.get("3b", 0).fillna(0)
            + df.get("hr", 0).fillna(0)
        )

    windows = [3, 7, 15, 30]

    grouped = df.groupby(batter_id_col, sort=False)

    # Shift(1) to avoid leakage
    pa_s = grouped["pa"].shift(1) if "pa" in df.columns else pd.Series(np.nan, index=df.index)
    ab_s = grouped["ab"].shift(1) if "ab" in df.columns else pd.Series(np.nan, index=df.index)
    hits_s = grouped["hits"].shift(1)
    hr_s = grouped["hr"].shift(1) if "hr" in df.columns else pd.Series(np.nan, index=df.index)
    tb_s = grouped["tb"].shift(1)
    bb_s = grouped["bb"].shift(1) if "bb" in df.columns else pd.Series(np.nan, index=df.index)
    so_s = grouped["so"].shift(1) if "so" in df.columns else pd.Series(np.nan, index=df.index)
    barrels_s = grouped["barrels"].shift(1) if "barrels" in df.columns else pd.Series(np.nan, index=df.index)
    hard_hit_s = grouped["hard_hit"].shift(1) if "hard_hit" in df.columns else pd.Series(np.nan, index=df.index)
    bip_s = grouped["bip"].shift(1) if "bip" in df.columns else pd.Series(np.nan, index=df.index)
    fb_s = grouped["fb"].shift(1) if "fb" in df.columns else pd.Series(np.nan, index=df.index)
    pull_air_s = grouped["pull_air"].shift(1) if "pull_air" in df.columns else pd.Series(np.nan, index=df.index)
    ev_s = grouped["avg_exit_velocity"].shift(1) if "avg_exit_velocity" in df.columns else pd.Series(np.nan, index=df.index)
    la_s = grouped["avg_launch_angle"].shift(1) if "avg_launch_angle" in df.columns else pd.Series(np.nan, index=df.index)

    for w in windows:
        pa_roll = pa_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        ab_roll = ab_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        hits_roll = hits_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        hr_roll = hr_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        tb_roll = tb_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        bb_roll = bb_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        so_roll = so_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        barrels_roll = barrels_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        hard_hit_roll = hard_hit_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        bip_roll = bip_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        fb_roll = fb_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        pull_air_roll = pull_air_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)

        ev_roll = ev_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        la_roll = la_s.groupby(df[batter_id_col]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)

        df[f"bat_hits_roll{w}"] = hits_roll
        df[f"bat_hr_roll{w}"] = hr_roll
        df[f"bat_tb_roll{w}"] = tb_roll
        df[f"bat_bb_roll{w}"] = bb_roll
        df[f"bat_so_roll{w}"] = so_roll

        df[f"bat_ba_roll{w}"] = safe_rate(hits_roll, ab_roll)
        df[f"bat_hr_per_pa_roll{w}"] = safe_rate(hr_roll, pa_roll)
        df[f"bat_tb_per_pa_roll{w}"] = safe_rate(tb_roll, pa_roll)
        df[f"bat_bb_rate_roll{w}"] = safe_rate(bb_roll, pa_roll)
        df[f"bat_k_rate_roll{w}"] = safe_rate(so_roll, pa_roll)

        df[f"bat_barrel_rate_roll{w}"] = safe_rate(barrels_roll, bip_roll)
        df[f"bat_hard_hit_rate_roll{w}"] = safe_rate(hard_hit_roll, bip_roll)
        df[f"bat_fb_rate_roll{w}"] = safe_rate(fb_roll, bip_roll)
        df[f"bat_pulled_air_rate_roll{w}"] = safe_rate(pull_air_roll, bip_roll)

        df[f"bat_avg_ev_roll{w}"] = ev_roll
        df[f"bat_avg_la_roll{w}"] = la_roll

        # ISO proxy
        singles_roll = (
            grouped["1b"].shift(1).groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
            if "1b" in df.columns else 0
        )
        doubles_roll = (
            grouped["2b"].shift(1).groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
            if "2b" in df.columns else 0
        )
        triples_roll = (
            grouped["3b"].shift(1).groupby(df[batter_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
            if "3b" in df.columns else 0
        )
        extra_bases = doubles_roll + 2 * triples_roll + 3 * hr_roll
        df[f"bat_iso_roll{w}"] = safe_rate(extra_bases, ab_roll)

    return df


def add_pitcher_rollings(df: pd.DataFrame) -> pd.DataFrame:
    pitcher_id_col = first_existing_col(df, ["pitcher_id", "pitcher"])
    date_col = first_existing_col(df, ["game_date"])
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    numeric_cols = [
        "batters_faced", "ip_outs", "hits_allowed", "hr_allowed", "bb_allowed", "so",
        "barrels_allowed", "hard_hit_allowed", "bip_allowed"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values([pitcher_id_col, date_col]).reset_index(drop=True)
    grouped = df.groupby(pitcher_id_col, sort=False)

    bf_s = grouped["batters_faced"].shift(1) if "batters_faced" in df.columns else pd.Series(np.nan, index=df.index)
    hr_s = grouped["hr_allowed"].shift(1) if "hr_allowed" in df.columns else pd.Series(np.nan, index=df.index)
    bb_s = grouped["bb_allowed"].shift(1) if "bb_allowed" in df.columns else pd.Series(np.nan, index=df.index)
    so_s = grouped["so"].shift(1) if "so" in df.columns else pd.Series(np.nan, index=df.index)
    barrels_s = grouped["barrels_allowed"].shift(1) if "barrels_allowed" in df.columns else pd.Series(np.nan, index=df.index)
    hard_hit_s = grouped["hard_hit_allowed"].shift(1) if "hard_hit_allowed" in df.columns else pd.Series(np.nan, index=df.index)
    bip_s = grouped["bip_allowed"].shift(1) if "bip_allowed" in df.columns else pd.Series(np.nan, index=df.index)

    windows = [3, 7, 15, 30]
    for w in windows:
        bf_roll = bf_s.groupby(df[pitcher_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        hr_roll = hr_s.groupby(df[pitcher_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        bb_roll = bb_s.groupby(df[pitcher_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        so_roll = so_s.groupby(df[pitcher_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        barrels_roll = barrels_s.groupby(df[pitcher_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        hard_hit_roll = hard_hit_s.groupby(df[pitcher_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        bip_roll = bip_s.groupby(df[pitcher_id_col]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)

        df[f"pit_hr_allowed_roll{w}"] = hr_roll
        df[f"pit_bb_allowed_roll{w}"] = bb_roll
        df[f"pit_so_roll{w}"] = so_roll
        df[f"pit_hr9_roll{w}"] = safe_rate(hr_roll * 27.0, bf_roll)
        df[f"pit_bb_rate_roll{w}"] = safe_rate(bb_roll, bf_roll)
        df[f"pit_k_rate_roll{w}"] = safe_rate(so_roll, bf_roll)
        df[f"pit_barrel_rate_roll{w}"] = safe_rate(barrels_roll, bip_roll)
        df[f"pit_hard_hit_rate_roll{w}"] = safe_rate(hard_hit_roll, bip_roll)

    return df


def main() -> None:
    config = load_config()
    data_root = resolve_data_root(config)
    processed_dir = data_root / "processed"

    # Make absolutely sure destination exists
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

    # Safety again, in case of odd runtime state
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