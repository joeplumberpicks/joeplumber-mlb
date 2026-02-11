#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.utils.drive import resolve_data_dirs
from src.utils.io import load_config, read_parquet, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rolling pregame features.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--windows", type=str, default=None, help="Comma-separated windows, e.g. 3,7,15,30")
    parser.add_argument("--min_history_days", type=int, default=None)
    return parser.parse_args()


def parse_windows(arg_windows: str | None, config: dict) -> list[int]:
    if arg_windows:
        return [int(w.strip()) for w in arg_windows.split(",") if w.strip()]
    return [int(w) for w in config["features"]["rolling_windows_days"]]


def rolling_sum_prior_days(df: pd.DataFrame, entity_col: str, value_col: str, days: int) -> pd.Series:
    out = pd.Series(index=df.index, dtype="float64")
    for _, idx in df.groupby(entity_col, sort=False).groups.items():
        g = df.loc[idx].sort_values("game_date", kind="mergesort")
        s = g.set_index("game_date")[value_col].astype(float)
        prior = s.shift(1)
        roll = prior.rolling(f"{days}D", min_periods=1).sum().fillna(0.0)
        out.loc[g.index] = roll.values
    return out.fillna(0.0)


def add_rolling_features(
    df: pd.DataFrame,
    entity_col: str,
    metric_cols: list[str],
    denom_col: str,
    windows: list[int],
    min_history_days: int,
) -> pd.DataFrame:
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    out = out.sort_values([entity_col, "game_date", "game_pk"], kind="mergesort").reset_index(drop=True)

    for w in windows:
        for m in metric_cols:
            out[f"rolling_{m}_{w}"] = rolling_sum_prior_days(out, entity_col, m, w)

        out[f"rolling_{denom_col}_{w}"] = rolling_sum_prior_days(out, entity_col, denom_col, w)

        denom = out[f"rolling_{denom_col}_{w}"].replace(0, pd.NA)
        if "hr" in metric_cols:
            out[f"hr_rate_{w}"] = (out.get(f"rolling_hr_{w}", 0) / denom).fillna(0.0)
        if "hits" in metric_cols:
            out[f"hit_rate_{w}"] = (out.get(f"rolling_hits_{w}", 0) / denom).fillna(0.0)
        if "bb" in metric_cols:
            out[f"bb_rate_{w}"] = (out.get(f"rolling_bb_{w}", 0) / denom).fillna(0.0)
        if "so" in metric_cols:
            out[f"so_rate_{w}"] = (out.get(f"rolling_so_{w}", 0) / denom).fillna(0.0)
        if "tb" in metric_cols:
            out[f"tb_per_pa_{w}"] = (out.get(f"rolling_tb_{w}", 0) / denom).fillna(0.0)
        if "outs_recorded" in metric_cols:
            out[f"outs_per_bf_{w}"] = (out.get(f"rolling_outs_recorded_{w}", 0) / denom).fillna(0.0)
        if "runs_allowed" in metric_cols:
            out[f"runs_per_bf_{w}"] = (out.get(f"rolling_runs_allowed_{w}", 0) / denom).fillna(0.0)

    min_start = out["game_date"] - pd.to_timedelta(min_history_days, unit="D")
    hist_ok = []
    for _, idx in out.groupby(entity_col, sort=False).groups.items():
        g = out.loc[idx].sort_values("game_date", kind="mergesort")
        first_date = g["game_date"].min()
        hist_ok.extend((g["game_date"] >= (first_date + pd.to_timedelta(min_history_days, unit="D"))).tolist())
    out["has_min_history"] = hist_ok
    _ = min_start  # keep min_history explicit for readability
    return out


def validate_pk(df: pd.DataFrame, keys: list[str], name: str) -> None:
    if df[keys].isna().any().any():
        raise ValueError(f"{name} has null PK values")
    if df.duplicated(subset=keys).any():
        raise ValueError(f"{name} has duplicate PK values")


def main() -> None:
    args = parse_args()
    config = load_config()
    windows = parse_windows(args.windows, config)
    min_history_days = args.min_history_days or int(config["features"]["min_history_days"])

    dirs = resolve_data_dirs(config)
    processed = dirs["processed"]

    games = read_parquet(processed / "games.parquet")
    games = games.loc[games["season"] == args.season, ["game_pk", "game_date"]].copy()
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")

    batter = read_parquet(processed / "batter_game.parquet").merge(games, on="game_pk", how="inner")
    pitcher = read_parquet(processed / "pitcher_game.parquet").merge(games, on="game_pk", how="inner")
    team = read_parquet(processed / "team_game.parquet").merge(games, on="game_pk", how="inner")

    batter_roll = add_rolling_features(
        batter,
        entity_col="batter_id",
        metric_cols=["hits", "hr", "2b", "bb", "so", "tb"],
        denom_col="pa",
        windows=windows,
        min_history_days=min_history_days,
    )
    for w in windows:
        batter_roll[f"hr_per_pa_{w}"] = batter_roll[f"hr_rate_{w}"]

    pitcher_roll = add_rolling_features(
        pitcher,
        entity_col="pitcher_id",
        metric_cols=["so", "bb", "hr_allowed", "outs_recorded", "runs_allowed"],
        denom_col="batters_faced",
        windows=windows,
        min_history_days=min_history_days,
    )
    for w in windows:
        pitcher_roll[f"k_rate_{w}"] = pitcher_roll[f"so_rate_{w}"]
        pitcher_roll[f"bb_rate_{w}"] = pitcher_roll[f"bb_rate_{w}"]
        pitcher_roll[f"hr_rate_{w}"] = pitcher_roll[f"hr_rate_{w}"]

    team_roll = add_rolling_features(
        team,
        entity_col="team_id",
        metric_cols=["runs_scored", "hits", "hr", "bb", "so", "tb"],
        denom_col="pa",
        windows=windows,
        min_history_days=min_history_days,
    )
    for w in windows:
        team_roll[f"run_rate_{w}"] = (team_roll[f"rolling_runs_scored_{w}"] / team_roll[f"rolling_pa_{w}"].replace(0, pd.NA)).fillna(0.0)

    batter_keep = ["game_pk", "batter_id"]
    pitcher_keep = ["game_pk", "pitcher_id"]
    team_keep = ["game_pk", "team_id"]

    for w in windows:
        batter_keep += [
            f"rolling_pa_{w}", f"rolling_hits_{w}", f"rolling_hr_{w}", f"rolling_2b_{w}",
            f"rolling_bb_{w}", f"rolling_so_{w}", f"rolling_tb_{w}",
            f"hr_per_pa_{w}", f"hit_rate_{w}", f"bb_rate_{w}", f"so_rate_{w}", f"tb_per_pa_{w}",
        ]
        pitcher_keep += [
            f"rolling_batters_faced_{w}", f"rolling_so_{w}", f"rolling_bb_{w}",
            f"rolling_hr_allowed_{w}", f"rolling_outs_recorded_{w}", f"rolling_runs_allowed_{w}",
            f"k_rate_{w}", f"bb_rate_{w}", f"hr_rate_{w}", f"outs_per_bf_{w}", f"runs_per_bf_{w}",
        ]
        team_keep += [
            f"rolling_runs_scored_{w}", f"rolling_hits_{w}", f"rolling_hr_{w}", f"rolling_bb_{w}",
            f"rolling_so_{w}", f"rolling_tb_{w}",
            f"run_rate_{w}", f"hr_rate_{w}", f"bb_rate_{w}", f"so_rate_{w}", f"tb_per_pa_{w}",
        ]

    batter_out = batter_roll[batter_keep].rename(columns={f"rolling_pa_{w}": f"rolling_pa_{w}" for w in windows})
    pitcher_out = pitcher_roll[pitcher_keep].rename(columns={f"rolling_batters_faced_{w}": f"rolling_bf_{w}" for w in windows})
    team_out = team_roll[team_keep].rename(columns={f"rolling_runs_scored_{w}": f"rolling_runs_{w}" for w in windows})

    batter_out = batter_out.sort_values(["game_pk", "batter_id"], kind="mergesort").drop_duplicates(["game_pk", "batter_id"], keep="first")
    pitcher_out = pitcher_out.sort_values(["game_pk", "pitcher_id"], kind="mergesort").drop_duplicates(["game_pk", "pitcher_id"], keep="first")
    team_out = team_out.sort_values(["game_pk", "team_id"], kind="mergesort").drop_duplicates(["game_pk", "team_id"], keep="first")

    validate_pk(batter_out, ["game_pk", "batter_id"], "batter_rolling")
    validate_pk(pitcher_out, ["game_pk", "pitcher_id"], "pitcher_rolling")
    validate_pk(team_out, ["game_pk", "team_id"], "team_rolling")

    print(f"batter_rolling rows: {len(batter_out)}")
    print(f"pitcher_rolling rows: {len(pitcher_out)}")
    print(f"team_rolling rows: {len(team_out)}")

    write_parquet(batter_out, processed / "batter_rolling.parquet")
    write_parquet(pitcher_out, processed / "pitcher_rolling.parquet")
    write_parquet(team_out, processed / "team_rolling.parquet")


if __name__ == "__main__":
    main()
