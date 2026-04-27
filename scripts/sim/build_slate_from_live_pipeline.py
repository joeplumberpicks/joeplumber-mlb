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


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_parquet(path)


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find any of:\n" + "\n".join(str(p) for p in paths))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    processed = Path(dirs["processed_dir"]) / "live"
    raw = Path(dirs["raw_dir"]) / "live"
    outputs = Path(dirs["outputs_dir"]) / "slate_inputs"
    outputs.mkdir(parents=True, exist_ok=True)

    spine = read_parquet(processed / f"model_spine_game_{args.season}_{args.date}.parquet")

    projected = read_parquet(
        first_existing([
            raw / f"projected_lineups_{args.season}_{args.date}.parquet",
            processed / f"projected_lineups_{args.season}_{args.date}.parquet",
        ])
    )

    confirmed = None
    for path in [
        raw / f"confirmed_lineups_{args.season}_{args.date}.parquet",
        processed / f"confirmed_lineups_{args.season}_{args.date}.parquet",
    ]:
        if path.exists():
            confirmed = pd.read_parquet(path)
            break

    starters = read_parquet(
        first_existing([
            raw / f"starting_pitchers_{args.season}_{args.date}.parquet",
            processed / f"starting_pitchers_{args.season}_{args.date}.parquet",
        ])
    )

    rows = []

    for _, game in spine.iterrows():
        game_pk = game["game_pk"]
        away = game["away_team"]
        home = game["home_team"]

        away_lu = None
        home_lu = None
        away_source = "none"
        home_source = "none"

        if confirmed is not None:
            away_conf = confirmed[(confirmed.game_pk == game_pk) & (confirmed.team == away)]
            home_conf = confirmed[(confirmed.game_pk == game_pk) & (confirmed.team == home)]

            if len(away_conf.dropna(subset=["player_id"])) >= 9:
                away_lu = away_conf
                away_source = "confirmed"

            if len(home_conf.dropna(subset=["player_id"])) >= 9:
                home_lu = home_conf
                home_source = "confirmed"

        if away_lu is None:
            away_proj = projected[(projected.game_pk == game_pk) & (projected.team == away)]
            if len(away_proj.dropna(subset=["player_id"])) >= 9:
                away_lu = away_proj
                away_source = "projected"

        if home_lu is None:
            home_proj = projected[(projected.game_pk == game_pk) & (projected.team == home)]
            if len(home_proj.dropna(subset=["player_id"])) >= 9:
                home_lu = home_proj
                home_source = "projected"

        if away_lu is None or home_lu is None:
            print(f"[SKIP] Missing lineup IDs: {away} @ {home}")
            continue

        away_ids = (
            away_lu.sort_values("batting_order")["player_id"]
            .dropna()
            .astype(int)
            .astype(str)
            .tolist()
        )

        home_ids = (
            home_lu.sort_values("batting_order")["player_id"]
            .dropna()
            .astype(int)
            .astype(str)
            .tolist()
        )

        if len(away_ids) < 9 or len(home_ids) < 9:
            print(f"[SKIP] Incomplete lineup IDs: {away} @ {home}")
            continue

        away_sp = starters[(starters.game_pk == game_pk) & (starters.team == away)]
        home_sp = starters[(starters.game_pk == game_pk) & (starters.team == home)]

        if away_sp.empty or home_sp.empty:
            print(f"[SKIP] Missing starter: {away} @ {home}")
            continue

        away_pid = away_sp.iloc[0].get("pitcher_id", away_sp.iloc[0].get("player_id"))
        home_pid = home_sp.iloc[0].get("pitcher_id", home_sp.iloc[0].get("player_id"))

        if pd.isna(away_pid) or pd.isna(home_pid):
            print(f"[SKIP] Unresolved starter ID: {away} @ {home}")
            continue

        rows.append({
            "game_pk": game_pk,
            "game_date": args.date,
            "away_team": away,
            "home_team": home,
            "away_batters": ",".join(away_ids[:9]),
            "home_batters": ",".join(home_ids[:9]),
            "away_pitcher_id": int(away_pid),
            "home_pitcher_id": int(home_pid),
            "away_lineup_source": away_source,
            "home_lineup_source": home_source,
        })

    out = pd.DataFrame(rows)
    out_path = outputs / f"slate_input_{args.date}.csv"
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Games ready: {len(out)}")
    if len(out):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
