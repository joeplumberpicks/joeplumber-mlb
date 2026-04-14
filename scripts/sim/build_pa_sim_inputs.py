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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean PA sim inputs from PA outcome mart.")
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


def _coerce_numeric_except_strings(df: pd.DataFrame, string_cols: set[str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in string_cols:
            out[col] = out[col].astype("string")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
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
        keep_cols = [c for c in row.index if c.startswith("bat_")]

        clean = row[keep_cols].to_dict()
        clean["batter_id"] = batter_id
        clean["lineup_slot"] = slot
        clean["batting_team"] = team_code
        out_rows.append(clean)

    out = pd.DataFrame(out_rows).reset_index(drop=True)
    out = _coerce_numeric_except_strings(out, {"batting_team"})
    return out


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
    keep_cols = [c for c in row.index if c.startswith("pit_")]

    clean = row[keep_cols].to_dict()
    clean["pitcher_id"] = pitcher_id
    clean["fielding_team"] = team_code

    out = pd.DataFrame([clean])
    out = _coerce_numeric_except_strings(out, {"fielding_team"})
    return out


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
    print(f"away_lineup_out={away_lineup_out} rows={len(lineup_away):,} cols={len(lineup_away.columns):,}")
    print(f"home_lineup_out={home_lineup_out} rows={len(lineup_home):,} cols={len(lineup_home.columns):,}")
    print(f"away_pitcher_out={away_pitcher_out} rows={len(pitcher_away):,} cols={len(pitcher_away.columns):,}")
    print(f"home_pitcher_out={home_pitcher_out} rows={len(pitcher_home):,} cols={len(pitcher_home.columns):,}")


if __name__ == "__main__":
    main()
