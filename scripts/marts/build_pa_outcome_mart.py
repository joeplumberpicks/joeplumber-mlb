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


def _pick_existing(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def _safe_to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


def _clip_rate(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").clip(lower=0.0, upper=1.0)


def _prep_pa(pa: pd.DataFrame) -> pd.DataFrame:
    out = pa.copy()

    if "game_date" not in out.columns:
        raise ValueError("PA table missing game_date")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if "season" not in out.columns:
        out["season"] = out["game_date"].dt.year

    required = ["game_pk", "batter_id", "pitcher_id"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"PA table missing required column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")

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

    batter_id_col = "batter_id" if "batter_id" in out.columns else ("batter" if "batter" in out.columns else None)
    if batter_id_col is None:
        raise ValueError("batter_game_rolling.parquet missing batter_id/batter column")

    if batter_id_col != "batter_id":
        out = out.rename(columns={batter_id_col: "batter_id"})

    out["batter_id"] = pd.to_numeric(out["batter_id"], errors="coerce").astype("Int64")
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    return out


def _prep_pitcher_roll(pr: pd.DataFrame) -> pd.DataFrame:
    out = pr.copy()

    pitcher_id_col = "pitcher_id" if "pitcher_id" in out.columns else ("pitcher" if "pitcher" in out.columns else None)
    if pitcher_id_col is None:
        raise ValueError("pitcher_game_rolling.parquet missing pitcher_id/pitcher column")

    if pitcher_id_col != "pitcher_id":
        out = out.rename(columns={pitcher_id_col: "pitcher_id"})

    out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

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

    if "inning_topbot" in out.columns:
        itb = out["inning_topbot"].astype("string").str.upper()
        out["is_top_inning"] = itb.eq("TOP").astype(int)
        out["is_bot_inning"] = itb.eq("BOT").astype(int)

    if "inning" in out.columns:
        out["inning"] = pd.to_numeric(out["inning"], errors="coerce")
        out["inning_bucket"] = pd.cut(
            out["inning"],
            bins=[0, 3, 6, 20],
            labels=["early", "mid", "late"],
            right=True,
        ).astype("string")

    if "outs_before_pa" in out.columns:
        out["outs_before_pa"] = pd.to_numeric(out["outs_before_pa"], errors="coerce")
        out["two_out_flag"] = out["outs_before_pa"].eq(2).astype(int)

    return out


def _select_batter_roll_cols(br: pd.DataFrame) -> list[str]:
    candidates = [
        "game_pk", "batter_id",
        "pa_roll3", "pa_roll7", "pa_roll15", "pa_roll30",
        "ab_roll3", "ab_roll7", "ab_roll15", "ab_roll30",
        "hits_roll3", "hits_roll7", "hits_roll15", "hits_roll30",
        "hr_roll3", "hr_roll7", "hr_roll15", "hr_roll30",
        "tb_roll3", "tb_roll7", "tb_roll15", "tb_roll30",
        "bb_roll3", "bb_roll7", "bb_roll15", "bb_roll30",
        "so_roll3", "so_roll7", "so_roll15", "so_roll30",
        "hit_rate_roll3", "hit_rate_roll7", "hit_rate_roll15", "hit_rate_roll30",
        "hr_rate_roll3", "hr_rate_roll7", "hr_rate_roll15", "hr_rate_roll30",
        "tb_pa_rate_roll3", "tb_pa_rate_roll7", "tb_pa_rate_roll15", "tb_pa_rate_roll30",
        "bb_rate_roll3", "bb_rate_roll7", "bb_rate_roll15", "bb_rate_roll30",
        "so_rate_roll3", "so_rate_roll7", "so_rate_roll15", "so_rate_roll30",
        "contact_rate_roll3", "contact_rate_roll7", "contact_rate_roll15", "contact_rate_roll30",
        "whiff_rate_roll3", "whiff_rate_roll7", "whiff_rate_roll15", "whiff_rate_roll30",
        "hard_hit_rate_roll3", "hard_hit_rate_roll7", "hard_hit_rate_roll15", "hard_hit_rate_roll30",
        "barrel_rate_roll3", "barrel_rate_roll7", "barrel_rate_roll15", "barrel_rate_roll30",
        "avg_ev_roll3", "avg_ev_roll7", "avg_ev_roll15", "avg_ev_roll30",
        "avg_la_roll3", "avg_la_roll7", "avg_la_roll15", "avg_la_roll30",
        "gb_rate_roll3", "gb_rate_roll7", "gb_rate_roll15", "gb_rate_roll30",
        "fb_rate_roll3", "fb_rate_roll7", "fb_rate_roll15", "fb_rate_roll30",
        "pull_rate_roll3", "pull_rate_roll7", "pull_rate_roll15", "pull_rate_roll30",
        "iso_roll3", "iso_roll7", "iso_roll15", "iso_roll30",
    ]
    return _pick_existing(br, candidates)


def _select_pitcher_roll_cols(pr: pd.DataFrame) -> list[str]:
    candidates = [
        "game_pk", "pitcher_id",
        "batters_faced_roll3", "batters_faced_roll7", "batters_faced_roll15", "batters_faced_roll30",
        "hits_allowed_roll3", "hits_allowed_roll7", "hits_allowed_roll15", "hits_allowed_roll30",
        "hr_allowed_roll3", "hr_allowed_roll7", "hr_allowed_roll15", "hr_allowed_roll30",
        "bb_allowed_roll3", "bb_allowed_roll7", "bb_allowed_roll15", "bb_allowed_roll30",
        "so_roll3", "so_roll7", "so_roll15", "so_roll30",
        "runs_allowed_roll3", "runs_allowed_roll7", "runs_allowed_roll15", "runs_allowed_roll30",
        "tb_allowed_roll3", "tb_allowed_roll7", "tb_allowed_roll15", "tb_allowed_roll30",
        "k_rate_roll3", "k_rate_roll7", "k_rate_roll15", "k_rate_roll30",
        "bb_rate_roll3", "bb_rate_roll7", "bb_rate_roll15", "bb_rate_roll30",
        "hr_rate_roll3", "hr_rate_roll7", "hr_rate_roll15", "hr_rate_roll30",
        "hit_rate_roll3", "hit_rate_roll7", "hit_rate_roll15", "hit_rate_roll30",
        "runs_rate_roll3", "runs_rate_roll7", "runs_rate_roll15", "runs_rate_roll30",
        "contact_rate_roll3", "contact_rate_roll7", "contact_rate_roll15", "contact_rate_roll30",
        "whiff_rate_roll3", "whiff_rate_roll7", "whiff_rate_roll15", "whiff_rate_roll30",
        "hard_hit_rate_roll3", "hard_hit_rate_roll7", "hard_hit_rate_roll15", "hard_hit_rate_roll30",
        "barrel_rate_roll3", "barrel_rate_roll7", "barrel_rate_roll15", "barrel_rate_roll30",
        "avg_ev_roll3", "avg_ev_roll7", "avg_ev_roll15", "avg_ev_roll30",
        "avg_la_roll3", "avg_la_roll7", "avg_la_roll15", "avg_la_roll30",
        "gb_rate_roll3", "gb_rate_roll7", "gb_rate_roll15", "gb_rate_roll30",
        "fb_rate_roll3", "fb_rate_roll7", "fb_rate_roll15", "fb_rate_roll30",
    ]
    return _pick_existing(pr, candidates)


def _join_rollings(pa: pd.DataFrame, br: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:
    out = pa.copy()

    br_cols = _select_batter_roll_cols(br)
    pr_cols = _select_pitcher_roll_cols(pr)

    br_small = br[br_cols].copy().rename(
        columns={c: f"bat_{c}" for c in br_cols if c not in {"game_pk", "batter_id"}}
    )
    pr_small = pr[pr_cols].copy().rename(
        columns={c: f"pit_{c}" for c in pr_cols if c not in {"game_pk", "pitcher_id"}}
    )

    out = out.merge(br_small, on=["game_pk", "batter_id"], how="left")
    out = out.merge(pr_small, on=["game_pk", "pitcher_id"], how="left")

    return out


def _engineer_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "bat_hit_rate_roll30", "bat_hr_rate_roll30", "bat_bb_rate_roll30", "bat_so_rate_roll30",
        "bat_contact_rate_roll30", "bat_whiff_rate_roll30", "bat_hard_hit_rate_roll30", "bat_barrel_rate_roll30",
        "bat_iso_roll30", "bat_tb_pa_rate_roll30",
        "pit_hit_rate_roll30", "pit_hr_rate_roll30", "pit_bb_rate_roll30", "pit_k_rate_roll30",
        "pit_contact_rate_roll30", "pit_whiff_rate_roll30", "pit_hard_hit_rate_roll30", "pit_barrel_rate_roll30",
        "pit_runs_rate_roll30",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    diff_pairs = [
        ("matchup_hit_rate_diff", "bat_hit_rate_roll30", "pit_hit_rate_roll30"),
        ("matchup_hr_rate_diff", "bat_hr_rate_roll30", "pit_hr_rate_roll30"),
        ("matchup_bb_rate_diff", "bat_bb_rate_roll30", "pit_bb_rate_roll30"),
        ("matchup_k_pressure_diff", "bat_so_rate_roll30", "pit_k_rate_roll30"),
        ("matchup_contact_diff", "bat_contact_rate_roll30", "pit_contact_rate_roll30"),
        ("matchup_whiff_diff", "bat_whiff_rate_roll30", "pit_whiff_rate_roll30"),
        ("matchup_hard_hit_diff", "bat_hard_hit_rate_roll30", "pit_hard_hit_rate_roll30"),
        ("matchup_barrel_diff", "bat_barrel_rate_roll30", "pit_barrel_rate_roll30"),
        ("matchup_power_diff", "bat_iso_roll30", "pit_hr_rate_roll30"),
    ]

    for new_col, a, b in diff_pairs:
        if a in out.columns and b in out.columns:
            out[new_col] = pd.to_numeric(out[a], errors="coerce") - pd.to_numeric(out[b], errors="coerce")

    interaction_pairs = [
        ("matchup_hr_pressure_x", "bat_hr_rate_roll30", "pit_hr_rate_roll30"),
        ("matchup_hit_pressure_x", "bat_hit_rate_roll30", "pit_hit_rate_roll30"),
        ("matchup_walk_pressure_x", "bat_bb_rate_roll30", "pit_bb_rate_roll30"),
        ("matchup_k_pressure_x", "bat_so_rate_roll30", "pit_k_rate_roll30"),
        ("matchup_contact_x", "bat_contact_rate_roll30", "pit_contact_rate_roll30"),
        ("matchup_hard_hit_x", "bat_hard_hit_rate_roll30", "pit_hard_hit_rate_roll30"),
        ("matchup_barrel_x", "bat_barrel_rate_roll30", "pit_barrel_rate_roll30"),
    ]

    for new_col, a, b in interaction_pairs:
        if a in out.columns and b in out.columns:
            out[new_col] = pd.to_numeric(out[a], errors="coerce") * pd.to_numeric(out[b], errors="coerce")

    return out


def _engineer_form_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    windows = [3, 7, 15, 30]

    for w in windows:
        bat_pa = f"bat_pa_roll{w}"
        bat_hits = f"bat_hits_roll{w}"
        bat_hr = f"bat_hr_roll{w}"
        bat_bb = f"bat_bb_roll{w}"
        bat_so = f"bat_so_roll{w}"
        bat_tb = f"bat_tb_roll{w}"

        if bat_pa in out.columns and bat_hits in out.columns:
            out[f"bat_hit_rate_calc_roll{w}"] = pd.to_numeric(out[bat_hits], errors="coerce") / pd.to_numeric(out[bat_pa], errors="coerce")
        if bat_pa in out.columns and bat_hr in out.columns:
            out[f"bat_hr_rate_calc_roll{w}"] = pd.to_numeric(out[bat_hr], errors="coerce") / pd.to_numeric(out[bat_pa], errors="coerce")
        if bat_pa in out.columns and bat_bb in out.columns:
            out[f"bat_bb_rate_calc_roll{w}"] = pd.to_numeric(out[bat_bb], errors="coerce") / pd.to_numeric(out[bat_pa], errors="coerce")
        if bat_pa in out.columns and bat_so in out.columns:
            out[f"bat_so_rate_calc_roll{w}"] = pd.to_numeric(out[bat_so], errors="coerce") / pd.to_numeric(out[bat_pa], errors="coerce")
        if bat_pa in out.columns and bat_tb in out.columns:
            out[f"bat_tb_pa_rate_calc_roll{w}"] = pd.to_numeric(out[bat_tb], errors="coerce") / pd.to_numeric(out[bat_pa], errors="coerce")

        pit_bf = f"pit_batters_faced_roll{w}"
        pit_hits = f"pit_hits_allowed_roll{w}"
        pit_hr = f"pit_hr_allowed_roll{w}"
        pit_bb = f"pit_bb_allowed_roll{w}"
        pit_runs = f"pit_runs_allowed_roll{w}"

        if pit_bf in out.columns and pit_hits in out.columns:
            out[f"pit_hit_rate_calc_roll{w}"] = pd.to_numeric(out[pit_hits], errors="coerce") / pd.to_numeric(out[pit_bf], errors="coerce")
        if pit_bf in out.columns and pit_hr in out.columns:
            out[f"pit_hr_rate_calc_roll{w}"] = pd.to_numeric(out[pit_hr], errors="coerce") / pd.to_numeric(out[pit_bf], errors="coerce")
        if pit_bf in out.columns and pit_bb in out.columns:
            out[f"pit_bb_rate_calc_roll{w}"] = pd.to_numeric(out[pit_bb], errors="coerce") / pd.to_numeric(out[pit_bf], errors="coerce")
        if pit_bf in out.columns and pit_runs in out.columns:
            out[f"pit_runs_rate_calc_roll{w}"] = pd.to_numeric(out[pit_runs], errors="coerce") / pd.to_numeric(out[pit_bf], errors="coerce")

    return out


def _engineer_stability_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def delta(col_a: str, col_b: str, new_col: str) -> None:
        if col_a in out.columns and col_b in out.columns:
            out[new_col] = pd.to_numeric(out[col_a], errors="coerce") - pd.to_numeric(out[col_b], errors="coerce")

    delta("bat_hit_rate_roll7", "bat_hit_rate_roll30", "bat_hit_rate_trend_7v30")
    delta("bat_hr_rate_roll7", "bat_hr_rate_roll30", "bat_hr_rate_trend_7v30")
    delta("bat_bb_rate_roll7", "bat_bb_rate_roll30", "bat_bb_rate_trend_7v30")
    delta("bat_so_rate_roll7", "bat_so_rate_roll30", "bat_so_rate_trend_7v30")
    delta("bat_contact_rate_roll7", "bat_contact_rate_roll30", "bat_contact_trend_7v30")
    delta("bat_hard_hit_rate_roll7", "bat_hard_hit_rate_roll30", "bat_hard_hit_trend_7v30")
    delta("bat_barrel_rate_roll7", "bat_barrel_rate_roll30", "bat_barrel_trend_7v30")

    delta("pit_hit_rate_roll7", "pit_hit_rate_roll30", "pit_hit_rate_trend_7v30")
    delta("pit_hr_rate_roll7", "pit_hr_rate_roll30", "pit_hr_rate_trend_7v30")
    delta("pit_bb_rate_roll7", "pit_bb_rate_roll30", "pit_bb_rate_trend_7v30")
    delta("pit_k_rate_roll7", "pit_k_rate_roll30", "pit_k_rate_trend_7v30")
    delta("pit_hard_hit_rate_roll7", "pit_hard_hit_rate_roll30", "pit_hard_hit_trend_7v30")
    delta("pit_barrel_rate_roll7", "pit_barrel_rate_roll30", "pit_barrel_trend_7v30")

    return out


def _postprocess_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    string_cols = _pick_existing(
        out,
        [
            "pa_outcome_target",
            "event_type",
            "inning_topbot",
            "base_state_before",
            "batting_team",
            "fielding_team",
            "inning_bucket",
        ],
    )
    for col in string_cols:
        out[col] = out[col].astype("string")

    num_cols = [c for c in out.columns if c not in string_cols and c not in {"game_date"}]
    for col in num_cols:
        if out[col].dtype == "object":
            maybe_num = pd.to_numeric(out[col], errors="coerce")
            if maybe_num.notna().sum() > 0:
                out[col] = maybe_num

    return out


def _sort_cols_for_preview(df: pd.DataFrame) -> list[str]:
    front = _pick_existing(
        df,
        [
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
        ],
    )
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
    df = _engineer_form_features(df)
    df = _engineer_stability_features(df)
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

    pa_path = processed_dir / "pa.parquet"
    batter_roll_path = processed_dir / "batter_game_rolling.parquet"
    pitcher_roll_path = processed_dir / "pitcher_game_rolling.parquet"

    pa = _load_parquet(pa_path)
    batter_roll = _load_parquet(batter_roll_path)
    pitcher_roll = _load_parquet(pitcher_roll_path)

    print("========================================")
    print("JOE PLUMBER PA OUTCOME MART BUILD")
    print("========================================")
    print(f"pa_path={pa_path}")
    print(f"batter_roll_path={batter_roll_path}")
    print(f"pitcher_roll_path={pitcher_roll_path}")

    mart = build_pa_outcome_mart(pa=pa, batter_roll=batter_roll, pitcher_roll=pitcher_roll)

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
