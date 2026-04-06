#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.ingest.io import log_kv, log_section, read_dataset, write_parquet
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


B_WINDOWS = (3, 7, 15, 30)
P_WINDOWS = (3, 7, 15, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cross-season batter/pitcher rolling features.")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def _safe_rate(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce")
    num = pd.to_numeric(num, errors="coerce")
    out = num / den.where(den.ne(0))
    return out


def _prep_pa(pa: pd.DataFrame) -> pd.DataFrame:
    out = pa.copy()

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["batter_id"] = pd.to_numeric(out["batter_id"], errors="coerce").astype("Int64")
    out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")

    for col in [
        "is_pa", "is_ab", "is_hit", "is_1b", "is_2b", "is_3b", "is_hr",
        "is_bb", "is_hbp", "is_so", "is_rbi", "is_sac_fly", "is_reached_on_error"
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype("boolean")
        else:
            out[col] = False

    if "runs_scored_on_pa" not in out.columns:
        out["runs_scored_on_pa"] = 0
    if "rbi" not in out.columns:
        out["rbi"] = 0

    out["runs_scored_on_pa"] = pd.to_numeric(out["runs_scored_on_pa"], errors="coerce").fillna(0.0)
    out["rbi"] = pd.to_numeric(out["rbi"], errors="coerce").fillna(0.0)

    out["tb"] = (
        out["is_1b"].astype(int)
        + 2 * out["is_2b"].astype(int)
        + 3 * out["is_3b"].astype(int)
        + 4 * out["is_hr"].astype(int)
    )

    return out


def _build_batter_game(pa: pd.DataFrame) -> pd.DataFrame:
    grp = (
        pa.groupby(["season", "game_date", "game_pk", "batter_id", "batter_name"], dropna=False)
        .agg(
            pa=("is_pa", "sum"),
            ab=("is_ab", "sum"),
            hits=("is_hit", "sum"),
            singles=("is_1b", "sum"),
            doubles=("is_2b", "sum"),
            triples=("is_3b", "sum"),
            hr=("is_hr", "sum"),
            bb=("is_bb", "sum"),
            hbp=("is_hbp", "sum"),
            so=("is_so", "sum"),
            rbi=("rbi", "sum"),
            tb=("tb", "sum"),
            runs_scored=("runs_scored_on_pa", "sum"),
            sac_fly=("is_sac_fly", "sum"),
            roe=("is_reached_on_error", "sum"),
            pitcher_id_nunique=("pitcher_id", "nunique"),
        )
        .reset_index()
    )

    grp = grp.sort_values(["batter_id", "game_date", "game_pk"], kind="stable").reset_index(drop=True)

    grp["hit_1_plus"] = grp["hits"].gt(0).astype(int)
    grp["tb_2_plus"] = grp["tb"].ge(2).astype(int)
    grp["hr_1_plus"] = grp["hr"].gt(0).astype(int)
    grp["rbi_1_plus"] = grp["rbi"].gt(0).astype(int)

    return grp


def _build_pitcher_game(pa: pd.DataFrame) -> pd.DataFrame:
    grp = (
        pa.groupby(["season", "game_date", "game_pk", "pitcher_id", "pitcher_name"], dropna=False)
        .agg(
            batters_faced=("is_pa", "sum"),
            ab_allowed=("is_ab", "sum"),
            hits_allowed=("is_hit", "sum"),
            singles_allowed=("is_1b", "sum"),
            doubles_allowed=("is_2b", "sum"),
            triples_allowed=("is_3b", "sum"),
            hr_allowed=("is_hr", "sum"),
            bb_allowed=("is_bb", "sum"),
            hbp_allowed=("is_hbp", "sum"),
            so=("is_so", "sum"),
            rbi_allowed=("rbi", "sum"),
            tb_allowed=("tb", "sum"),
            runs_allowed=("runs_scored_on_pa", "sum"),
            batter_id_nunique=("batter_id", "nunique"),
        )
        .reset_index()
    )

    grp = grp.sort_values(["pitcher_id", "game_date", "game_pk"], kind="stable").reset_index(drop=True)

    return grp


def _add_batter_rolls(df: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["batter_id", "game_date", "game_pk"], kind="stable").reset_index(drop=True)

    base_cols = ["pa", "ab", "hits", "hr", "rbi", "tb", "bb", "so"]

    for col in base_cols:
        shifted = out.groupby("batter_id", dropna=False)[col].shift(1)
        for w in windows:
            out[f"{col}_roll{w}"] = (
                shifted.groupby(out["batter_id"]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
            )

    for w in windows:
        out[f"hit_rate_roll{w}"] = _safe_rate(out[f"hits_roll{w}"], out[f"ab_roll{w}"])
        out[f"hr_rate_roll{w}"] = _safe_rate(out[f"hr_roll{w}"], out[f"ab_roll{w}"])
        out[f"rbi_pa_rate_roll{w}"] = _safe_rate(out[f"rbi_roll{w}"], out[f"pa_roll{w}"])
        out[f"tb_pa_rate_roll{w}"] = _safe_rate(out[f"tb_roll{w}"], out[f"pa_roll{w}"])
        out[f"bb_rate_roll{w}"] = _safe_rate(out[f"bb_roll{w}"], out[f"pa_roll{w}"])
        out[f"so_rate_roll{w}"] = _safe_rate(out[f"so_roll{w}"], out[f"pa_roll{w}"])

    return out


def _add_pitcher_rolls(df: pd.DataFrame, windows: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["pitcher_id", "game_date", "game_pk"], kind="stable").reset_index(drop=True)

    base_cols = ["batters_faced", "hits_allowed", "hr_allowed", "bb_allowed", "so", "runs_allowed", "tb_allowed"]

    for col in base_cols:
        shifted = out.groupby("pitcher_id", dropna=False)[col].shift(1)
        for w in windows:
            out[f"{col}_roll{w}"] = (
                shifted.groupby(out["pitcher_id"]).rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
            )

    for w in windows:
        out[f"k_rate_roll{w}"] = _safe_rate(out[f"so_roll{w}"], out[f"batters_faced_roll{w}"])
        out[f"bb_rate_roll{w}"] = _safe_rate(out[f"bb_allowed_roll{w}"], out[f"batters_faced_roll{w}"])
        out[f"hr_rate_roll{w}"] = _safe_rate(out[f"hr_allowed_roll{w}"], out[f"batters_faced_roll{w}"])
        out[f"hit_rate_roll{w}"] = _safe_rate(out[f"hits_allowed_roll{w}"], out[f"batters_faced_roll{w}"])
        out[f"runs_rate_roll{w}"] = _safe_rate(out[f"runs_allowed_roll{w}"], out[f"batters_faced_roll{w}"])

    return out


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/features/build_cross_season_rollings.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    pa_path = processed_dir / "pa.parquet"

    if not pa_path.exists():
        raise FileNotFoundError(
            f"Missing historical PA parquet: {pa_path}\n"
            "Run scripts/features/build_historical_bridge.py first."
        )

    pa = read_dataset(pa_path)
    pa = _prep_pa(pa)

    batter_game = _build_batter_game(pa)
    pitcher_game = _build_pitcher_game(pa)

    batter_roll = _add_batter_rolls(batter_game, B_WINDOWS)
    pitcher_roll = _add_pitcher_rolls(pitcher_game, P_WINDOWS)

    print(f"Row count [batter_game_rolling]: {len(batter_roll):,}")
    print(f"Distinct batter_id: {batter_roll['batter_id'].nunique(dropna=True):,}")
    if not batter_roll.empty:
        print(f"Min game_date [batter]: {batter_roll['game_date'].min()}")
        print(f"Max game_date [batter]: {batter_roll['game_date'].max()}")

    print(f"Row count [pitcher_game_rolling]: {len(pitcher_roll):,}")
    print(f"Distinct pitcher_id: {pitcher_roll['pitcher_id'].nunique(dropna=True):,}")
    if not pitcher_roll.empty:
        print(f"Min game_date [pitcher]: {pitcher_roll['game_date'].min()}")
        print(f"Max game_date [pitcher]: {pitcher_roll['game_date'].max()}")

    write_parquet(batter_roll, processed_dir / "batter_game_rolling.parquet")
    write_parquet(pitcher_roll, processed_dir / "pitcher_game_rolling.parquet")

    print("")
    print("rolling_out:")
    print(processed_dir / "batter_game_rolling.parquet")
    print(processed_dir / "pitcher_game_rolling.parquet")


if __name__ == "__main__":
    main()
