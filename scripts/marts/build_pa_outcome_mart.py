#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.pa_outcome_model import build_pa_target
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PA outcome mart.")
    parser.add_argument("--season", type=int, default=None, help="Optional single season build.")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required parquet: {path}")
    return pd.read_parquet(path)


def _prep_pa(pa: pd.DataFrame) -> pd.DataFrame:
    out = pa.copy()

    if "game_date" not in out.columns:
        raise ValueError("PA table missing game_date")

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if "season" not in out.columns:
        out["season"] = out["game_date"].dt.year

    for col in ["game_pk", "batter_id", "pitcher_id", "season"]:
        if col not in out.columns:
            raise ValueError(f"PA table missing required column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["inning", "outs_before_pa", "outs_after_pa", "pa_index"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["event_type", "inning_topbot", "base_state_before", "batting_team", "fielding_team"]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    flag_cols = ["is_pa", "is_ab", "is_1b", "is_2b", "is_3b", "is_hr", "is_bb", "is_hbp", "is_so"]
    for col in flag_cols:
        if col in out.columns:
            out[col] = out[col].fillna(False).astype(bool)
        else:
            out[col] = False

    out = out[out["game_date"].notna()].copy()
    out = out[out["season"].notna()].copy()
    out = out[out["game_pk"].notna()].copy()
    out = out[out["batter_id"].notna()].copy()
    out = out[out["pitcher_id"].notna()].copy()

    if "is_pa" in out.columns:
        out = out[out["is_pa"] == True].copy()

    out["pa_outcome_target"] = build_pa_target(out)

    return out


def _prep_batter_roll(br: pd.DataFrame) -> pd.DataFrame:
    out = br.copy()

    if "batter" in out.columns and "batter_id" not in out.columns:
        out = out.rename(columns={"batter": "batter_id"})

    for col in ["game_pk", "batter_id", "season"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    return out


def _prep_pitcher_roll(pr: pd.DataFrame) -> pd.DataFrame:
    out = pr.copy()

    if "pitcher" in out.columns and "pitcher_id" not in out.columns:
        out = out.rename(columns={"pitcher": "pitcher_id"})

    for col in ["game_pk", "pitcher_id", "season"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    return out


def _select_batter_roll_cols(br: pd.DataFrame) -> list[str]:
    keep = ["game_pk", "batter_id"]
    keep += [c for c in br.columns if c.startswith("bat_")]
    return [c for c in keep if c in br.columns]


def _select_pitcher_roll_cols(pr: pd.DataFrame) -> list[str]:
    keep = ["game_pk", "pitcher_id"]
    keep += [c for c in pr.columns if c.startswith("pit_")]
    return [c for c in keep if c in pr.columns]


def _join_rollings(pa: pd.DataFrame, br: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    out = pa.copy()

    br_cols = _select_batter_roll_cols(br)
    pr_cols = _select_pitcher_roll_cols(pr)

    br_small = br[br_cols].copy().drop_duplicates(subset=["game_pk", "batter_id"], keep="last")
    pr_small = pr[pr_cols].copy().drop_duplicates(subset=["game_pk", "pitcher_id"], keep="last")

    out = out.merge(br_small, on=["game_pk", "batter_id"], how="left")
    out = out.merge(pr_small, on=["game_pk", "pitcher_id"], how="left")

    return out


def _build_base_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "base_state_before" in out.columns:
        bsb = out["base_state_before"].astype("string").fillna("")
        out["on_1b"] = bsb.str.contains("1", regex=False).astype(int)
        out["on_2b"] = bsb.str.contains("2", regex=False).astype(int)
        out["on_3b"] = bsb.str.contains("3", regex=False).astype(int)
        out["base_runner_count"] = out[["on_1b", "on_2b", "on_3b"]].sum(axis=1)
        out["risp_flag"] = ((out["on_2b"] == 1) | (out["on_3b"] == 1)).astype(int)
        out["bases_empty_flag"] = (out["base_runner_count"] == 0).astype(int)
    else:
        out["on_1b"] = 0
        out["on_2b"] = 0
        out["on_3b"] = 0
        out["base_runner_count"] = 0
        out["risp_flag"] = 0
        out["bases_empty_flag"] = 1

    if "inning_topbot" in out.columns:
        itb = out["inning_topbot"].astype("string").str.upper()
        out["is_top_inning"] = itb.eq("TOP").astype(int)
        out["is_bot_inning"] = itb.eq("BOT").astype(int)
    else:
        out["is_top_inning"] = np.nan
        out["is_bot_inning"] = np.nan

    if "inning" in out.columns:
        out["inning"] = pd.to_numeric(out["inning"], errors="coerce")
        out["inning_bucket"] = pd.cut(
            out["inning"],
            bins=[0, 3, 6, 99],
            labels=["early", "mid", "late"],
            right=True,
        ).astype("string")
    else:
        out["inning_bucket"] = pd.Series(pd.NA, index=out.index, dtype="string")

    if "outs_before_pa" in out.columns:
        out["outs_before_pa"] = pd.to_numeric(out["outs_before_pa"], errors="coerce")
        out["two_out_flag"] = out["outs_before_pa"].eq(2).astype(int)
    else:
        out["two_out_flag"] = np.nan

    # placeholder until real lineup slot source is joined
    out["lineup_slot"] = np.nan

    return out


def _engineer_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    pairs_diff = [
        ("matchup_hit_rate_diff", "bat_hit_rate_roll30", "pit_hit_rate_roll30"),
        ("matchup_hr_rate_diff", "bat_hr_rate_roll30", "pit_hr_rate_roll30"),
        ("matchup_bb_rate_diff", "bat_bb_rate_roll30", "pit_bb_rate_roll30"),
        ("matchup_k_pressure_diff", "bat_so_rate_roll30", "pit_k_rate_roll30"),
        ("matchup_power_diff", "bat_tb_per_pa_roll30", "pit_tb_allowed_per_bf_roll30"),
    ]

    for new_col, a, b in pairs_diff:
        if a in out.columns and b in out.columns:
            out[new_col] = pd.to_numeric(out[a], errors="coerce") - pd.to_numeric(out[b], errors="coerce")

    pairs_x = [
        ("matchup_hr_pressure_x", "bat_hr_rate_roll30", "pit_hr_rate_roll30"),
        ("matchup_hit_pressure_x", "bat_hit_rate_roll30", "pit_hit_rate_roll30"),
        ("matchup_walk_pressure_x", "bat_bb_rate_roll30", "pit_bb_rate_roll30"),
        ("matchup_k_pressure_x", "bat_so_rate_roll30", "pit_k_rate_roll30"),
    ]

    for new_col, a, b in pairs_x:
        if a in out.columns and b in out.columns:
            out[new_col] = pd.to_numeric(out[a], errors="coerce") * pd.to_numeric(out[b], errors="coerce")

    return out


def _postprocess_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    string_cols = [
        c for c in [
            "pa_outcome_target",
            "event_type",
            "inning_topbot",
            "base_state_before",
            "batting_team",
            "fielding_team",
            "inning_bucket",
        ] if c in out.columns
    ]
    for col in string_cols:
        out[col] = out[col].astype("string")

    return out


def _sort_cols_for_preview(df: pd.DataFrame) -> list[str]:
    front = [
        c for c in [
            "game_date",
            "season",
            "game_pk",
            "pa_index",
            "batter_id",
            "pitcher_id",
            "inning",
            "outs_before_pa",
            "base_state_before",
            "pa_outcome_target",
            "event_type",
        ] if c in df.columns
    ]
    rest = [c for c in df.columns if c not in front]
    return front + rest


def build_pa_outcome_mart(
    pa: pd.DataFrame,
    batter_roll: pd.DataFrame,
    pitcher_roll: pd.DataFrame,
) -> pd.DataFrame:
    pa = _prep_pa(pa)
    batter_roll = _prep_batter_roll(batter_roll)
    pitcher_roll = _prep_pitcher_roll(pitcher_roll)

    df = _join_rollings(pa, batter_roll, pitcher_roll)
    df = _build_base_context_features(df)
    df = _engineer_matchup_features(df)
    df = _postprocess_types(df)

    return df


def main() -> None:
    args = parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    marts_dir = Path(dirs["marts_dir"])
    by_season_out_dir = marts_dir / "by_season"
    pa_outcome_out_dir = marts_dir / "pa_outcome"

    by_season_out_dir.mkdir(parents=True, exist_ok=True)
    pa_outcome_out_dir.mkdir(parents=True, exist_ok=True)

    pa = _load_parquet(processed_dir / "pa.parquet")
    batter_roll = _load_parquet(processed_dir / "batter_game_rolling.parquet")
    pitcher_roll = _load_parquet(processed_dir / "pitcher_game_rolling.parquet")

    print("========================================")
    print("JOE PLUMBER PA OUTCOME MART BUILD")
    print("========================================")
    print(f"pa_rows={len(pa):,}")
    print(f"batter_roll_rows={len(batter_roll):,}")
    print(f"pitcher_roll_rows={len(pitcher_roll):,}")

    mart = build_pa_outcome_mart(pa=pa, batter_roll=batter_roll, pitcher_roll=pitcher_roll)

    bat_cols = [c for c in mart.columns if c.startswith("bat_")]
    pit_cols = [c for c in mart.columns if c.startswith("pit_")]
    matchup_cols = [c for c in mart.columns if c.startswith("matchup_")]

    print("")
    print(f"mart_rows={len(mart):,}")
    print(f"mart_cols={len(mart.columns):,}")
    print(f"bat_cols={len(bat_cols)}")
    print(f"pit_cols={len(pit_cols)}")
    print(f"matchup_cols={len(matchup_cols)}")

    if args.season is not None:
        mart = mart[mart["season"] == args.season].copy()
        if mart.empty:
            raise ValueError(f"No rows found for season={args.season}")

        season_out = by_season_out_dir / f"pa_outcome_features_{args.season}.parquet"
        if season_out.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists, use --overwrite: {season_out}")

        mart = mart[_sort_cols_for_preview(mart)]
        mart.to_parquet(season_out, index=False)

        print("")
        print("=== COMPLETE ===")
        print(f"season={args.season}")
        print(f"rows={len(mart):,}")
        print(f"cols={len(mart.columns):,}")
        print(f"out={season_out}")
        return

    global_out = pa_outcome_out_dir / "pa_outcome_features.parquet"
    if global_out.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists, use --overwrite: {global_out}")

    mart = mart[_sort_cols_for_preview(mart)]
    mart.to_parquet(global_out, index=False)

    print("")
    print("=== GLOBAL MART COMPLETE ===")
    print(f"rows={len(mart):,}")
    print(f"cols={len(mart.columns):,}")
    print(f"out={global_out}")

    seasons = sorted(pd.Series(mart["season"].dropna().unique()).astype(int).tolist())
    for season in seasons:
        season_df = mart[mart["season"] == season].copy()
        season_out = by_season_out_dir / f"pa_outcome_features_{season}.parquet"
        season_df.to_parquet(season_out, index=False)
        print(f"season_out[{season}]={season_out} rows={len(season_df):,}")


if __name__ == "__main__":
    main()
