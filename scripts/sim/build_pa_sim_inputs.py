#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


LEAKAGE_DROP_COLS = {
    "pa_outcome_target",
    "event_type",
    "game_date",
    "game_pk",
    "pa_index",
    "outs_after_pa",
    "rbi",
    "runs_scored_on_pa",
    "is_pa",
    "is_ab",
    "is_1b",
    "is_2b",
    "is_3b",
    "is_hr",
    "is_bb",
    "is_hbp",
    "is_so",
    "pitch_number_start",
    "pitch_number_end",
    "launch_speed",
    "launch_angle",
    "hit_distance_sc",
    "hc_x",
    "hc_y",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sim inputs from PA mart using explicit player IDs.")
    parser.add_argument("--game-date", type=str, required=True)
    parser.add_argument("--away-team", type=str, required=True)
    parser.add_argument("--home-team", type=str, required=True)
    parser.add_argument("--away-batters", type=str, required=True, help="Comma-separated batter_ids in lineup order")
    parser.add_argument("--home-batters", type=str, required=True, help="Comma-separated batter_ids in lineup order")
    parser.add_argument("--away-pitcher-id", type=int, required=True)
    parser.add_argument("--home-pitcher-id", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def _parse_ids(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _strip_to_sim_safe(df: pd.DataFrame, keep_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in LEAKAGE_DROP_COLS or c in keep_cols]
    out = df[cols].copy()

    if "lineup_slot" in out.columns:
        out["lineup_slot"] = pd.to_numeric(out["lineup_slot"], errors="coerce")

    return out


def _latest_rows_for_batters(
    mart: pd.DataFrame,
    batter_ids: list[int],
    as_of_date: pd.Timestamp,
    team_code: str,
) -> pd.DataFrame:
    df = mart.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["batter_id"] = pd.to_numeric(df["batter_id"], errors="coerce").astype("Int64")
    df = df[df["game_date"] < as_of_date].copy()

    out_rows = []
    for slot, batter_id in enumerate(batter_ids, start=1):
        sub = df[df["batter_id"] == batter_id].sort_values("game_date")
        if sub.empty:
            raise ValueError(f"No historical mart row found for batter_id={batter_id}")
        row = sub.iloc[-1].copy()
        row["batter_id"] = batter_id
        row["lineup_slot"] = slot
        row["batting_team"] = team_code
        out_rows.append(row)

    return pd.DataFrame(out_rows).reset_index(drop=True)


def _latest_row_for_pitcher(
    mart: pd.DataFrame,
    pitcher_id: int,
    as_of_date: pd.Timestamp,
    team_code: str,
) -> pd.DataFrame:
    df = mart.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce").astype("Int64")
    df = df[df["game_date"] < as_of_date].copy()

    sub = df[df["pitcher_id"] == pitcher_id].sort_values("game_date")
    if sub.empty:
        raise ValueError(f"No historical mart row found for pitcher_id={pitcher_id}")

    row = sub.iloc[-1].copy()
    row["pitcher_id"] = pitcher_id
    row["fielding_team"] = team_code
    return pd.DataFrame([row])


def main() -> None:
    args = parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    mart_path = Path(dirs["marts_dir"]) / "pa_outcome" / "pa_outcome_features.parquet"
    out_dir = Path(dirs["outputs_dir"]) / "pa_sim_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    mart = pd.read_parquet(mart_path)
    as_of_date = pd.Timestamp(args.game_date)

    away_batters = _parse_ids(args.away_batters)
    home_batters = _parse_ids(args.home_batters)

    lineup_away = _latest_rows_for_batters(
        mart=mart,
        batter_ids=away_batters,
        as_of_date=as_of_date,
        team_code=args.away_team,
    )
    lineup_home = _latest_rows_for_batters(
        mart=mart,
        batter_ids=home_batters,
        as_of_date=as_of_date,
        team_code=args.home_team,
    )
    pitcher_away = _latest_row_for_pitcher(
        mart=mart,
        pitcher_id=args.away_pitcher_id,
        as_of_date=as_of_date,
        team_code=args.away_team,
    )
    pitcher_home = _latest_row_for_pitcher(
        mart=mart,
        pitcher_id=args.home_pitcher_id,
        as_of_date=as_of_date,
        team_code=args.home_team,
    )

    lineup_away = _strip_to_sim_safe(lineup_away, keep_cols=["batter_id", "lineup_slot", "batting_team"])
    lineup_home = _strip_to_sim_safe(lineup_home, keep_cols=["batter_id", "lineup_slot", "batting_team"])
    pitcher_away = _strip_to_sim_safe(pitcher_away, keep_cols=["pitcher_id", "fielding_team"])
    pitcher_home = _strip_to_sim_safe(pitcher_home, keep_cols=["pitcher_id", "fielding_team"])

    away_lineup_out = out_dir / f"{args.away_team}_{args.game_date}_lineup.csv"
    home_lineup_out = out_dir / f"{args.home_team}_{args.game_date}_lineup.csv"
    away_pitcher_out = out_dir / f"{args.away_team}_{args.game_date}_pitcher.csv"
    home_pitcher_out = out_dir / f"{args.home_team}_{args.game_date}_pitcher.csv"

    lineup_away.to_csv(away_lineup_out, index=False)
    lineup_home.to_csv(home_lineup_out, index=False)
    pitcher_away.to_csv(away_pitcher_out, index=False)
    pitcher_home.to_csv(home_pitcher_out, index=False)

    print("========================================")
    print("JOE PLUMBER PA SIM INPUT BUILD")
    print("========================================")
    print(f"mart_path={mart_path}")
    print(f"away_lineup_out={away_lineup_out}")
    print(f"home_lineup_out={home_lineup_out}")
    print(f"away_pitcher_out={away_pitcher_out}")
    print(f"home_pitcher_out={home_pitcher_out}")


if __name__ == "__main__":
    main()
