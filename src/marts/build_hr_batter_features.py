from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount, require_files
from src.utils.io import read_parquet, write_parquet

BATTER_TEAM_CANDIDATES = ["bat_team", "team", "batting_team", "batter_team", "offense_team"]
BATTER_ID_CANDIDATES = ["batter", "batter_id", "mlbam_batter_id", "player_id"]
PITCHER_ID_CANDIDATES = ["pitcher", "pitcher_id", "mlbam_pitcher_id", "player_id"]


def _pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {label} column. Candidates: {candidates}. Available: {list(df.columns)}")


def build_hr_batter_features(
    dirs: dict[str, Path],
    season: int,
    start: str | None = None,
    end: str | None = None,
) -> Path:
    batter_game_path = dirs["processed_dir"] / "by_season" / f"batter_game_{season}.parquet"
    spine_path = dirs["processed_dir"] / "model_spine_game.parquet"
    batter_roll_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    pitcher_roll_path = dirs["processed_dir"] / "pitcher_game_rolling.parquet"
    require_files([batter_game_path, spine_path, batter_roll_path, pitcher_roll_path], f"hr_batter_features_{season}")

    batter_game = read_parquet(batter_game_path)
    spine = read_parquet(spine_path)
    batter_roll = read_parquet(batter_roll_path)
    pitcher_roll = read_parquet(pitcher_roll_path)

    if "bat_hr" not in batter_game.columns:
        raise ValueError(f"Expected bat_hr column in batter_game_{season}. Available: {list(batter_game.columns)}")

    batter_team_col = _pick_column(batter_game, BATTER_TEAM_CANDIDATES, "batter_team")
    batter_id_col = _pick_column(batter_game, BATTER_ID_CANDIDATES, "batter_id")

    batter_game = batter_game.copy()
    batter_game["batter_id"] = pd.to_numeric(batter_game[batter_id_col], errors="coerce").astype("Int64")
    batter_game["batter_team"] = batter_game[batter_team_col]
    batter_game["target_hr"] = (pd.to_numeric(batter_game["bat_hr"], errors="coerce").fillna(0) > 0).astype("Int64")
    batter_game["game_date"] = pd.to_datetime(batter_game.get("game_date"), errors="coerce")
    batter_game["season"] = pd.to_numeric(batter_game.get("season"), errors="coerce").fillna(season).astype("Int64")

    if start:
        batter_game = batter_game[batter_game["game_date"] >= pd.to_datetime(start)]
    if end:
        batter_game = batter_game[batter_game["game_date"] <= pd.to_datetime(end)]

    spine = spine.copy()
    spine["game_date"] = pd.to_datetime(spine.get("game_date"), errors="coerce")
    if "season" in spine.columns:
        spine = spine[pd.to_numeric(spine["season"], errors="coerce") == season]
    if start:
        spine = spine[spine["game_date"] >= pd.to_datetime(start)]
    if end:
        spine = spine[spine["game_date"] <= pd.to_datetime(end)]

    spine_cols = [
        c
        for c in ["game_pk", "game_date", "home_team", "away_team", "home_sp_id", "away_sp_id", "park_id", "season"]
        if c in spine.columns
    ]
    hr = batter_game.merge(spine[spine_cols].drop_duplicates(subset=["game_pk"]), on="game_pk", how="left", suffixes=("", "_sp"))

    hr["opp_sp_id"] = pd.NA
    home_mask = hr["batter_team"].astype(str) == hr["home_team"].astype(str)
    away_mask = hr["batter_team"].astype(str) == hr["away_team"].astype(str)
    if "away_sp_id" in hr.columns:
        hr.loc[home_mask, "opp_sp_id"] = hr.loc[home_mask, "away_sp_id"]
    if "home_sp_id" in hr.columns:
        hr.loc[away_mask, "opp_sp_id"] = hr.loc[away_mask, "home_sp_id"]
    hr["opp_sp_id"] = pd.to_numeric(hr["opp_sp_id"], errors="coerce").astype("Int64")

    unknown_opp = int(hr["opp_sp_id"].isna().sum())
    logging.info("hr_batter_features: opp_sp_id null rows=%s (%.2f%%)", unknown_opp, 100.0 * unknown_opp / max(len(hr), 1))

    batter_roll = batter_roll.copy()
    pitch_roll = pitcher_roll.copy()
    br_id = _pick_column(batter_roll, BATTER_ID_CANDIDATES, "batter rolling id")
    pr_id = _pick_column(pitch_roll, PITCHER_ID_CANDIDATES, "pitcher rolling id")
    batter_roll["batter"] = pd.to_numeric(batter_roll[br_id], errors="coerce").astype("Int64")
    pitch_roll["pitcher"] = pd.to_numeric(pitch_roll[pr_id], errors="coerce").astype("Int64")

    br_cols = [c for c in batter_roll.columns if c not in {"game_pk", br_id, "batter"}]
    pr_cols = [c for c in pitch_roll.columns if c not in {"game_pk", pr_id, "pitcher"}]

    hr = hr.merge(
        batter_roll[["game_pk", "batter", *br_cols]].rename(columns={c: f"bat_{c}" for c in br_cols}),
        left_on=["game_pk", "batter_id"],
        right_on=["game_pk", "batter"],
        how="left",
    )
    hr = hr.drop(columns=["batter"], errors="ignore")

    hr = hr.merge(
        pitch_roll[["game_pk", "pitcher", *pr_cols]].rename(columns={c: f"opp_{c}" for c in pr_cols}),
        left_on=["game_pk", "opp_sp_id"],
        right_on=["game_pk", "pitcher"],
        how="left",
    )
    hr = hr.drop(columns=["pitcher"], errors="ignore")

    stable_cols = [
        "game_pk",
        "game_date",
        "season",
        "park_id",
        "home_team",
        "away_team",
        "batter_id",
        "batter_team",
        "opp_sp_id",
        "target_hr",
    ]
    for col in stable_cols:
        if col not in hr.columns:
            hr[col] = pd.NA

    front = stable_cols
    tail = [c for c in hr.columns if c not in front]
    hr = hr[front + tail]

    out_path = dirs["marts_dir"] / "hr_batter_features.parquet"
    print_rowcount("hr_batter_features", hr)
    print(f"opp_sp_id null rate: {hr['opp_sp_id'].isna().mean():.2%}")
    print(f"Writing to: {out_path.resolve()}")
    write_parquet(hr, out_path)
    return out_path
