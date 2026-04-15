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
    parser = argparse.ArgumentParser(
        description="Build live PA sim inputs from rolling tables with mart fallback."
    )
    parser.add_argument("--game-date", type=str, required=True)
    parser.add_argument("--away-team", type=str, required=True)
    parser.add_argument("--home-team", type=str, required=True)
    parser.add_argument(
        "--away-batters",
        type=str,
        required=True,
        help="Comma-separated batter_ids in lineup order",
    )
    parser.add_argument(
        "--home-batters",
        type=str,
        required=True,
        help="Comma-separated batter_ids in lineup order",
    )
    parser.add_argument("--away-pitcher-id", type=int, required=True)
    parser.add_argument("--home-pitcher-id", type=int, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    return parser.parse_args()


def _parse_ids(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _load_batter_rollings(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    if "batter" in df.columns and "batter_id" not in df.columns:
        df = df.rename(columns={"batter": "batter_id"})

    required = ["game_date", "game_pk", "batter_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"batter rolling missing columns: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce")
    df["batter_id"] = pd.to_numeric(df["batter_id"], errors="coerce")

    return df


def _load_pitcher_rollings(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()

    if "pitcher" in df.columns and "pitcher_id" not in df.columns:
        df = df.rename(columns={"pitcher": "pitcher_id"})

    required = ["game_date", "game_pk", "pitcher_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"pitcher rolling missing columns: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce")
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce")

    return df


def _load_pa_mart(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path).copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    if "batter_id" in df.columns:
        df["batter_id"] = pd.to_numeric(df["batter_id"], errors="coerce")
    if "pitcher_id" in df.columns:
        df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce")
    if "game_pk" in df.columns:
        df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce")
    return df


def _coerce_output_types(df: pd.DataFrame, string_cols: set[str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in string_cols:
            out[col] = out[col].astype("string")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _latest_batter_from_rollings(
    batter_roll: pd.DataFrame,
    batter_id: int,
    as_of_date: pd.Timestamp,
) -> dict | None:
    sub = batter_roll[
        (batter_roll["batter_id"] == batter_id) &
        (batter_roll["game_date"] < as_of_date)
    ].sort_values(["game_date", "game_pk"], kind="stable")

    if sub.empty:
        return None

    row = sub.iloc[-1].copy()
    keep_cols = [c for c in row.index if c in {"batter_id", "batter_name"} or c.startswith("bat_")]
    return row[keep_cols].to_dict()


def _latest_batter_from_mart(
    pa_mart: pd.DataFrame,
    batter_id: int,
    as_of_date: pd.Timestamp,
) -> dict | None:
    sub = pa_mart[
        (pa_mart["batter_id"] == batter_id) &
        (pa_mart["game_date"] < as_of_date)
    ].sort_values(["game_date", "game_pk"], kind="stable")

    if sub.empty:
        return None

    row = sub.iloc[-1].copy()
    keep_cols = [c for c in row.index if c in {"batter_id", "batter_name"} or c.startswith("bat_")]
    return row[keep_cols].to_dict()


def _latest_pitcher_from_rollings(
    pitcher_roll: pd.DataFrame,
    pitcher_id: int,
    as_of_date: pd.Timestamp,
) -> dict | None:
    sub = pitcher_roll[
        (pitcher_roll["pitcher_id"] == pitcher_id) &
        (pitcher_roll["game_date"] < as_of_date)
    ].sort_values(["game_date", "game_pk"], kind="stable")

    if sub.empty:
        return None

    row = sub.iloc[-1].copy()
    keep_cols = [c for c in row.index if c in {"pitcher_id", "pitcher_name"} or c.startswith("pit_")]
    return row[keep_cols].to_dict()


def _latest_pitcher_from_mart(
    pa_mart: pd.DataFrame,
    pitcher_id: int,
    as_of_date: pd.Timestamp,
) -> dict | None:
    sub = pa_mart[
        (pa_mart["pitcher_id"] == pitcher_id) &
        (pa_mart["game_date"] < as_of_date)
    ].sort_values(["game_date", "game_pk"], kind="stable")

    if sub.empty:
        return None

    row = sub.iloc[-1].copy()
    keep_cols = [c for c in row.index if c in {"pitcher_id", "pitcher_name"} or c.startswith("pit_")]
    return row[keep_cols].to_dict()


def _latest_rows_for_batters(
    batter_roll: pd.DataFrame,
    pa_mart: pd.DataFrame,
    batter_ids: list[int],
    as_of_date: pd.Timestamp,
    team_code: str,
) -> pd.DataFrame:
    out_rows: list[dict] = []

    for slot, batter_id in enumerate(batter_ids, start=1):
        clean = _latest_batter_from_rollings(batter_roll, batter_id, as_of_date)

        source = "rolling"
        if clean is None:
            clean = _latest_batter_from_mart(pa_mart, batter_id, as_of_date)
            source = "mart_fallback"

        if clean is None:
            raise ValueError(
                f"No batter row found before {as_of_date.date()} for batter_id={batter_id} "
                f"in rollings or mart"
            )

        clean["batter_id"] = int(batter_id)
        clean["lineup_slot"] = slot
        clean["batting_team"] = team_code
        clean["_source"] = source
        out_rows.append(clean)

    out = pd.DataFrame(out_rows).reset_index(drop=True)
    return out


def _latest_row_for_pitcher(
    pitcher_roll: pd.DataFrame,
    pa_mart: pd.DataFrame,
    pitcher_id: int,
    as_of_date: pd.Timestamp,
    team_code: str,
) -> pd.DataFrame:
    clean = _latest_pitcher_from_rollings(pitcher_roll, pitcher_id, as_of_date)
    source = "rolling"

    if clean is None:
        clean = _latest_pitcher_from_mart(pa_mart, pitcher_id, as_of_date)
        source = "mart_fallback"

    if clean is None:
        raise ValueError(
            f"No pitcher row found before {as_of_date.date()} for pitcher_id={pitcher_id} "
            f"in rollings or mart"
        )

    clean["pitcher_id"] = int(pitcher_id)
    clean["fielding_team"] = team_code
    clean["_source"] = source

    out = pd.DataFrame([clean]).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed_dir = Path(dirs["processed_dir"])
    marts_dir = Path(dirs["marts_dir"])
    out_dir = Path(dirs["outputs_dir"]) / "pa_sim_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    batter_roll_path = processed_dir / "batter_game_rolling.parquet"
    pitcher_roll_path = processed_dir / "pitcher_game_rolling.parquet"
    mart_path = marts_dir / "pa_outcome" / "pa_outcome_features.parquet"

    batter_roll = _load_batter_rollings(batter_roll_path)
    pitcher_roll = _load_pitcher_rollings(pitcher_roll_path)
    pa_mart = _load_pa_mart(mart_path)

    as_of_date = pd.Timestamp(args.game_date)

    away_batters = _parse_ids(args.away_batters)
    home_batters = _parse_ids(args.home_batters)

    lineup_away = _latest_rows_for_batters(
        batter_roll=batter_roll,
        pa_mart=pa_mart,
        batter_ids=away_batters,
        as_of_date=as_of_date,
        team_code=args.away_team,
    )
    lineup_home = _latest_rows_for_batters(
        batter_roll=batter_roll,
        pa_mart=pa_mart,
        batter_ids=home_batters,
        as_of_date=as_of_date,
        team_code=args.home_team,
    )

    pitcher_away = _latest_row_for_pitcher(
        pitcher_roll=pitcher_roll,
        pa_mart=pa_mart,
        pitcher_id=args.away_pitcher_id,
        as_of_date=as_of_date,
        team_code=args.away_team,
    )
    pitcher_home = _latest_row_for_pitcher(
        pitcher_roll=pitcher_roll,
        pa_mart=pa_mart,
        pitcher_id=args.home_pitcher_id,
        as_of_date=as_of_date,
        team_code=args.home_team,
    )

    lineup_away = _coerce_output_types(lineup_away, {"batter_name", "batting_team", "_source"})
    lineup_home = _coerce_output_types(lineup_home, {"batter_name", "batting_team", "_source"})
    pitcher_away = _coerce_output_types(pitcher_away, {"pitcher_name", "fielding_team", "_source"})
    pitcher_home = _coerce_output_types(pitcher_home, {"pitcher_name", "fielding_team", "_source"})

    away_lineup_out = out_dir / f"{args.away_team}_{args.game_date}_lineup.csv"
    home_lineup_out = out_dir / f"{args.home_team}_{args.game_date}_lineup.csv"
    away_pitcher_out = out_dir / f"{args.away_team}_{args.game_date}_pitcher.csv"
    home_pitcher_out = out_dir / f"{args.home_team}_{args.game_date}_pitcher.csv"

    lineup_away.to_csv(away_lineup_out, index=False)
    lineup_home.to_csv(home_lineup_out, index=False)
    pitcher_away.to_csv(away_pitcher_out, index=False)
    pitcher_home.to_csv(home_pitcher_out, index=False)

    print("========================================")
    print("JOE PLUMBER HYBRID PA SIM INPUT BUILD")
    print("========================================")
    print(f"batter_roll_path={batter_roll_path}")
    print(f"pitcher_roll_path={pitcher_roll_path}")
    print(f"mart_path={mart_path}")
    print(f"away_lineup_out={away_lineup_out} rows={len(lineup_away):,} cols={len(lineup_away.columns):,}")
    print(f"home_lineup_out={home_lineup_out} rows={len(lineup_home):,} cols={len(lineup_home.columns):,}")
    print(f"away_pitcher_out={away_pitcher_out} rows={len(pitcher_away):,} cols={len(pitcher_away.columns):,}")
    print(f"home_pitcher_out={home_pitcher_out} rows={len(pitcher_home):,} cols={len(pitcher_home.columns):,}")

    print("")
    print("away_lineup_source_counts:")
    print(lineup_away["_source"].value_counts(dropna=False).to_string())
    print("")
    print("home_lineup_source_counts:")
    print(lineup_home["_source"].value_counts(dropna=False).to_string())
    print("")
    print("away_pitcher_source_counts:")
    print(pitcher_away["_source"].value_counts(dropna=False).to_string())
    print("")
    print("home_pitcher_source_counts:")
    print(pitcher_home["_source"].value_counts(dropna=False).to_string())

    away_non_null = lineup_away.drop(columns=["_source"], errors="ignore").notna().mean().mean()
    home_non_null = lineup_home.drop(columns=["_source"], errors="ignore").notna().mean().mean()
    away_pitcher_non_null = pitcher_away.drop(columns=["_source"], errors="ignore").notna().mean().mean()
    home_pitcher_non_null = pitcher_home.drop(columns=["_source"], errors="ignore").notna().mean().mean()

    print("")
    print(f"away_lineup_non_null_rate={away_non_null:.3f}")
    print(f"home_lineup_non_null_rate={home_non_null:.3f}")
    print(f"away_pitcher_non_null_rate={away_pitcher_non_null:.3f}")
    print(f"home_pitcher_non_null_rate={home_pitcher_non_null:.3f}")


if __name__ == "__main__":
    main()
