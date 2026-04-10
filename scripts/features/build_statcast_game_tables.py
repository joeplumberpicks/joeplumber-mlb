#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


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


def safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    out = np.where((denom.notna()) & (denom != 0), numer / denom, np.nan)
    return pd.Series(out, index=numer.index, dtype=float)


def normalize_team_abbr(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    s = series.astype(str).str.upper().str.strip()
    mapping = {
        "ATHLETICS": "OAK",
        "A'S": "OAK",
        "WSH": "WSN",
        "WAS": "WSN",
        "D-BACKS": "ARI",
        "AZ": "ARI",
    }
    return s.replace(mapping)


def build_batter_game_table(pa_df: pd.DataFrame) -> pd.DataFrame:
    game_pk_col = pick_col(pa_df, ["game_pk"], required=True)
    game_date_col = pick_col(pa_df, ["game_date"], required=True)
    batter_id_col = pick_col(pa_df, ["batter_id", "batter"], required=True)
    batter_name_col = pick_col(pa_df, ["batter_name", "player_name"], required=False)
    team_col = pick_col(pa_df, ["batting_team", "team", "batter_team", "offense_team"], required=False)

    # IMPORTANT:
    # Do NOT use launch_speed_angle as EV. It is not exit velocity.
    launch_speed_col = pick_col(pa_df, ["launch_speed", "hit_speed"], required=False)
    launch_angle_col = pick_col(pa_df, ["launch_angle", "hit_angle"], required=False)

    is_hit_col = pick_col(pa_df, ["is_hit"], required=False)
    is_1b_col = pick_col(pa_df, ["is_1b"], required=False)
    is_2b_col = pick_col(pa_df, ["is_2b"], required=False)
    is_3b_col = pick_col(pa_df, ["is_3b"], required=False)
    is_hr_col = pick_col(pa_df, ["is_hr"], required=False)
    is_bb_col = pick_col(pa_df, ["is_bb"], required=False)
    is_so_col = pick_col(pa_df, ["is_so"], required=False)
    is_rbi_col = pick_col(pa_df, ["is_rbi"], required=False)

    df = pa_df.copy()

    for c in [is_hit_col, is_1b_col, is_2b_col, is_3b_col, is_hr_col, is_bb_col, is_so_col, is_rbi_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if is_hit_col is None:
        hit_inputs = []
        for c in [is_1b_col, is_2b_col, is_3b_col, is_hr_col]:
            if c and c in df.columns:
                hit_inputs.append(pd.to_numeric(df[c], errors="coerce").fillna(0))
        if hit_inputs:
            df["_is_hit_derived"] = sum(hit_inputs).clip(upper=1).astype(int)
            is_hit_col = "_is_hit_derived"
        else:
            df["_is_hit_derived"] = 0
            is_hit_col = "_is_hit_derived"

    if is_1b_col is None:
        df["_is_1b_derived"] = 0
        is_1b_col = "_is_1b_derived"
    if is_2b_col is None:
        df["_is_2b_derived"] = 0
        is_2b_col = "_is_2b_derived"
    if is_3b_col is None:
        df["_is_3b_derived"] = 0
        is_3b_col = "_is_3b_derived"
    if is_hr_col is None:
        df["_is_hr_derived"] = 0
        is_hr_col = "_is_hr_derived"
    if is_bb_col is None:
        df["_is_bb_derived"] = 0
        is_bb_col = "_is_bb_derived"
    if is_so_col is None:
        df["_is_so_derived"] = 0
        is_so_col = "_is_so_derived"
    if is_rbi_col is None:
        df["_is_rbi_derived"] = 0
        is_rbi_col = "_is_rbi_derived"

    if launch_speed_col and launch_speed_col in df.columns:
        df[launch_speed_col] = pd.to_numeric(df[launch_speed_col], errors="coerce")
    else:
        df["_launch_speed_null"] = np.nan
        launch_speed_col = "_launch_speed_null"

    if launch_angle_col and launch_angle_col in df.columns:
        df[launch_angle_col] = pd.to_numeric(df[launch_angle_col], errors="coerce")
    else:
        df["_launch_angle_null"] = np.nan
        launch_angle_col = "_launch_angle_null"

    # Tracked batted-ball contact
    df["_tracked_contact"] = (
        df[launch_speed_col].notna() &
        df[launch_angle_col].notna()
    ).astype(int)

    # Barrel proxy
    df["_barrel_event"] = (
        df["_tracked_contact"].eq(1) &
        df[launch_speed_col].ge(98) &
        df[launch_angle_col].between(26, 30, inclusive="both")
    ).astype(int)

    # Hard-hit proxy
    df["_hard_hit_event"] = (
        df["_tracked_contact"].eq(1) &
        df[launch_speed_col].ge(95)
    ).astype(int)

    # Fly-ball proxy
    df["_flyball_event"] = (
        df["_tracked_contact"].eq(1) &
        df[launch_angle_col].between(10, 50, inclusive="both")
    ).astype(int)

    # Pulled-air placeholder
    # Without spray-angle / direction data, we cannot do true pulled_air here.
    df["_pulled_air_event"] = np.nan

    df["_total_bases"] = (
        pd.to_numeric(df[is_1b_col], errors="coerce").fillna(0) * 1
        + pd.to_numeric(df[is_2b_col], errors="coerce").fillna(0) * 2
        + pd.to_numeric(df[is_3b_col], errors="coerce").fillna(0) * 3
        + pd.to_numeric(df[is_hr_col], errors="coerce").fillna(0) * 4
    )

    group_cols = [game_pk_col, game_date_col, batter_id_col]
    if batter_name_col:
        group_cols.append(batter_name_col)
    if team_col:
        group_cols.append(team_col)

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            pa=(batter_id_col, "size"),
            hits=(is_hit_col, "sum"),
            hr=(is_hr_col, "sum"),
            rbi=(is_rbi_col, "sum"),
            tb=("_total_bases", "sum"),
            tracked_bbe=("_tracked_contact", "sum"),
            ev_mean=(launch_speed_col, "mean"),
            ev_max=(launch_speed_col, "max"),
            la_mean=(launch_angle_col, "mean"),
            barrels=("_barrel_event", "sum"),
            hardhit=("_hard_hit_event", "sum"),
            flyballs=("_flyball_event", "sum"),
            bb=(is_bb_col, "sum"),
            so=(is_so_col, "sum"),
        )
        .reset_index()
    )

    rename_map = {
        game_pk_col: "game_pk",
        game_date_col: "game_date",
        batter_id_col: "batter_id",
    }
    if batter_name_col:
        rename_map[batter_name_col] = "batter_name"
    if team_col:
        rename_map[team_col] = "team"

    grouped = grouped.rename(columns=rename_map)

    if "batter_name" not in grouped.columns:
        grouped["batter_name"] = ""
    if "team" not in grouped.columns:
        grouped["team"] = ""

    grouped["team"] = normalize_team_abbr(grouped["team"])

    # Core rates
    grouped["barrel_rate"] = safe_div(grouped["barrels"], grouped["tracked_bbe"])
    grouped["hardhit_rate"] = safe_div(grouped["hardhit"], grouped["tracked_bbe"])
    grouped["k_rate"] = safe_div(grouped["so"], grouped["pa"])
    grouped["bb_rate"] = safe_div(grouped["bb"], grouped["pa"])

    # Normalized columns required downstream
    grouped["avg_ev"] = grouped["ev_mean"]
    grouped["avg_la"] = grouped["la_mean"]
    grouped["hard_hit_rate"] = grouped["hardhit_rate"]
    grouped["iso"] = safe_div(grouped["tb"] - grouped["hits"], grouped["pa"])
    grouped["hr_per_pa"] = safe_div(grouped["hr"], grouped["pa"])
    grouped["tb_per_pa"] = safe_div(grouped["tb"], grouped["pa"])
    grouped["fb_rate"] = safe_div(grouped["flyballs"], grouped["tracked_bbe"])
    grouped["pulled_air_rate"] = np.nan

    ordered_cols = [
        "game_pk",
        "game_date",
        "batter_id",
        "batter_name",
        "team",
        "pa",
        "hits",
        "hr",
        "rbi",
        "tb",
        "tracked_bbe",
        "ev_mean",
        "ev_max",
        "la_mean",
        "barrels",
        "hardhit",
        "flyballs",
        "bb",
        "so",
        "barrel_rate",
        "hardhit_rate",
        "k_rate",
        "bb_rate",
        "avg_ev",
        "avg_la",
        "hard_hit_rate",
        "iso",
        "hr_per_pa",
        "tb_per_pa",
        "fb_rate",
        "pulled_air_rate",
    ]
    existing_cols = [c for c in ordered_cols if c in grouped.columns]
    other_cols = [c for c in grouped.columns if c not in existing_cols]
    grouped = grouped[existing_cols + other_cols].copy()

    grouped["game_date"] = pd.to_datetime(grouped["game_date"], errors="coerce")
    grouped = grouped.sort_values(["batter_id", "game_date", "game_pk"]).reset_index(drop=True)

    return grouped


def build_pitcher_game_table(pa_df: pd.DataFrame) -> pd.DataFrame:
    game_pk_col = pick_col(pa_df, ["game_pk"], required=True)
    game_date_col = pick_col(pa_df, ["game_date"], required=True)
    pitcher_id_col = pick_col(pa_df, ["pitcher_id", "pitcher"], required=True)
    pitcher_name_col = pick_col(pa_df, ["pitcher_name"], required=False)
    team_col = pick_col(pa_df, ["pitching_team", "defense_team", "team_allowed"], required=False)

    is_bb_col = pick_col(pa_df, ["is_bb"], required=False)
    is_so_col = pick_col(pa_df, ["is_so"], required=False)
    is_hr_col = pick_col(pa_df, ["is_hr"], required=False)

    launch_speed_col = pick_col(pa_df, ["launch_speed", "hit_speed"], required=False)
    launch_angle_col = pick_col(pa_df, ["launch_angle", "hit_angle"], required=False)

    df = pa_df.copy()

    for c in [is_bb_col, is_so_col, is_hr_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if is_bb_col is None:
        df["_is_bb_derived"] = 0
        is_bb_col = "_is_bb_derived"
    if is_so_col is None:
        df["_is_so_derived"] = 0
        is_so_col = "_is_so_derived"
    if is_hr_col is None:
        df["_is_hr_derived"] = 0
        is_hr_col = "_is_hr_derived"

    if launch_speed_col and launch_speed_col in df.columns:
        df[launch_speed_col] = pd.to_numeric(df[launch_speed_col], errors="coerce")
    else:
        df["_launch_speed_null"] = np.nan
        launch_speed_col = "_launch_speed_null"

    if launch_angle_col and launch_angle_col in df.columns:
        df[launch_angle_col] = pd.to_numeric(df[launch_angle_col], errors="coerce")
    else:
        df["_launch_angle_null"] = np.nan
        launch_angle_col = "_launch_angle_null"

    df["_tracked_contact"] = (
        df[launch_speed_col].notna() &
        df[launch_angle_col].notna()
    ).astype(int)

    df["_barrel_event"] = (
        df["_tracked_contact"].eq(1) &
        df[launch_speed_col].ge(98) &
        df[launch_angle_col].between(26, 30, inclusive="both")
    ).astype(int)

    df["_hard_hit_event"] = (
        df["_tracked_contact"].eq(1) &
        df[launch_speed_col].ge(95)
    ).astype(int)

    group_cols = [game_pk_col, game_date_col, pitcher_id_col]
    if pitcher_name_col:
        group_cols.append(pitcher_name_col)
    if team_col:
        group_cols.append(team_col)

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            batters_faced=(pitcher_id_col, "size"),
            tracked_bbe=("_tracked_contact", "sum"),
            bb_allowed=(is_bb_col, "sum"),
            so=(is_so_col, "sum"),
            hr_allowed=(is_hr_col, "sum"),
            barrels_allowed=("_barrel_event", "sum"),
            hardhit_allowed=("_hard_hit_event", "sum"),
        )
        .reset_index()
    )

    rename_map = {
        game_pk_col: "game_pk",
        game_date_col: "game_date",
        pitcher_id_col: "pitcher_id",
    }
    if pitcher_name_col:
        rename_map[pitcher_name_col] = "pitcher_name"
    if team_col:
        rename_map[team_col] = "team"

    grouped = grouped.rename(columns=rename_map)

    if "pitcher_name" not in grouped.columns:
        grouped["pitcher_name"] = ""
    if "team" not in grouped.columns:
        grouped["team"] = ""

    grouped["team"] = normalize_team_abbr(grouped["team"])

    grouped["bb_rate"] = safe_div(grouped["bb_allowed"], grouped["batters_faced"])
    grouped["k_rate"] = safe_div(grouped["so"], grouped["batters_faced"])
    grouped["barrel_rate"] = safe_div(grouped["barrels_allowed"], grouped["tracked_bbe"])
    grouped["hard_hit_rate"] = safe_div(grouped["hardhit_allowed"], grouped["tracked_bbe"])
    grouped["hr_per_bf"] = safe_div(grouped["hr_allowed"], grouped["batters_faced"])
    grouped["hr9"] = grouped["hr_per_bf"] * 27.0

    grouped["game_date"] = pd.to_datetime(grouped["game_date"], errors="coerce")
    grouped = grouped.sort_values(["pitcher_id", "game_date", "game_pk"]).reset_index(drop=True)

    return grouped


def load_pa_by_seasons(data_root: Path, seasons: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in seasons:
        path = data_root / "processed" / "by_season" / f"pa_{season}.parquet"
        if not path.exists():
            print(f"Missing or empty: {path}")
            continue
        try:
            df = pd.read_parquet(path)
        except Exception:
            print(f"Could not read: {path}")
            continue
        if df.empty:
            print(f"Missing or empty: {path}")
            continue
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No usable pa_*.parquet files found for requested seasons.")

    out = pd.concat(frames, ignore_index=True)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out[out["game_date"].notna()].copy()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build game-level statcast tables from PA data.")
    parser.add_argument("--seasons", nargs="+", required=True, type=int)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    data_root = resolve_data_root(config)
    processed_dir = data_root / "processed"
    ensure_dir(processed_dir)

    pa_df = load_pa_by_seasons(data_root, args.seasons)

    batter_game = build_batter_game_table(pa_df)
    pitcher_game = build_pitcher_game_table(pa_df)

    batter_out = processed_dir / "batter_game_statcast.parquet"
    pitcher_out = processed_dir / "pitcher_game_statcast.parquet"

    batter_game.to_parquet(batter_out, index=False)
    pitcher_game.to_parquet(pitcher_out, index=False)

    print("✅ statcast game tables built")
    print(f"batter_rows={len(batter_game):,}")
    print(f"pitcher_rows={len(pitcher_game):,}")
    print(f"batter_out={batter_out}")
    print(f"pitcher_out={pitcher_out}")


if __name__ == "__main__":
    main()