from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.targets.paths import target_input_candidates
from src.utils.checks import print_rowcount, require_files
from src.utils.io import read_parquet, write_parquet

BATTER_TEAM_CANDIDATES = ["bat_team", "team", "batting_team", "batter_team", "offense_team"]
BATTER_ID_CANDIDATES = ["batter_id", "mlbam_batter_id", "batter", "player_id"]
PITCHER_ID_CANDIDATES = ["pitcher", "pitcher_id", "mlbam_pitcher_id", "player_id"]
HR_CANDIDATES = ["bat_hr", "hr"]
EVENTS_TEAM_CANDIDATES = ["batting_team", "bat_team", "team", "offense_team"]
INNING_HALF_CANDIDATES = ["inning_topbot", "topbot", "inning_half", "inning_top_bot"]
HOME_TEAM_CANDIDATES = ["home_team"]
AWAY_TEAM_CANDIDATES = ["away_team"]


def _pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing {label} column. Candidates: {candidates}. Available: {list(df.columns)}")


def _pick_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _mode(series: pd.Series) -> object:
    m = series.mode(dropna=True)
    return m.iloc[0] if not m.empty else pd.NA


def _infer_batter_team_from_events(
    batter_game: pd.DataFrame,
    processed_dir: Path,
    batter_id_col: str,
    season: int,
) -> pd.DataFrame:
    season_pa_path = processed_dir / "by_season" / f"pa_{season}.parquet"
    events_dir = processed_dir / "events_pa"
    events_file = processed_dir / "events_pa.parquet"

    pa_cols = [
        "game_pk",
        "batter_id",
        "mlbam_batter_id",
        "batter",
        "batting_team",
        "bat_team",
        "team",
        "offense_team",
        "inning_topbot",
        "topbot",
        "inning_half",
        "inning_top_bot",
        "home_team",
        "away_team",
    ]

    if season_pa_path.exists():
        logging.info("hr_batter_features loading PA for batter_team from by_season file: %s", season_pa_path.resolve())
        events = read_parquet(season_pa_path, columns=pa_cols)
    else:
        fallback_path = events_dir if events_dir.exists() else events_file
        if not fallback_path.exists():
            logging.warning("events_pa dataset not found; cannot infer batter_team fallback")
            batter_game["batter_team"] = pd.NA
            return batter_game
        logging.info("hr_batter_features loading PA for batter_team from fallback path: %s", fallback_path.resolve())
        events = read_parquet(fallback_path, columns=pa_cols, filters=[("season", "=", season)])
    events_batter_col = _pick_optional_column(events, BATTER_ID_CANDIDATES)
    events_team_col = _pick_optional_column(events, EVENTS_TEAM_CANDIDATES)

    derived_batting_team_col = None
    if events_team_col is None:
        inning_col = _pick_optional_column(events, INNING_HALF_CANDIDATES)
        home_col = _pick_optional_column(events, HOME_TEAM_CANDIDATES)
        away_col = _pick_optional_column(events, AWAY_TEAM_CANDIDATES)
        if inning_col and home_col and away_col:
            half = events[inning_col].astype(str).str.lower()
            derived_batting_team_col = "_derived_batting_team"
            events[derived_batting_team_col] = np.where(
                half.str.startswith("top"),
                events[away_col],
                np.where(half.str.startswith("bot"), events[home_col], pd.NA),
            )
            events_team_col = derived_batting_team_col

    if events_batter_col is None or events_team_col is None or "game_pk" not in events.columns:
        logging.warning(
            "events_pa missing required columns for batter_team inference. batter_col=%s team_col=%s has_game_pk=%s",
            events_batter_col,
            events_team_col,
            "game_pk" in events.columns,
        )
        batter_game["batter_team"] = pd.NA
        return batter_game

    events_map = events[["game_pk", events_batter_col, events_team_col]].copy()
    events_map["batter_id"] = pd.to_numeric(events_map[events_batter_col], errors="coerce").astype("Int64")
    events_map = events_map.dropna(subset=["batter_id"])
    events_mode = (
        events_map.groupby(["game_pk", "batter_id"], dropna=False)[events_team_col]
        .agg(_mode)
        .reset_index()
        .rename(columns={events_team_col: "batter_team"})
    )

    out = batter_game.copy()
    out = out.merge(events_mode, left_on=["game_pk", batter_id_col], right_on=["game_pk", "batter_id"], how="left")
    return out


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

    hr_col = _pick_optional_column(batter_game, HR_CANDIDATES)
    if hr_col is None:
        raise ValueError(
            f"Expected one of {HR_CANDIDATES} in batter_game_{season}. Available: {list(batter_game.columns)}"
        )

    batter_id_col = _pick_column(batter_game, BATTER_ID_CANDIDATES, "batter_id")
    batter_team_col = _pick_optional_column(batter_game, BATTER_TEAM_CANDIDATES)

    batter_game = batter_game.copy()
    batter_game["batter_id"] = pd.to_numeric(batter_game[batter_id_col], errors="coerce").astype("Int64")
    if batter_team_col is not None:
        batter_game["batter_team"] = batter_game[batter_team_col]
    else:
        batter_game = _infer_batter_team_from_events(batter_game, dirs["processed_dir"], "batter_id", season)

    batter_game["target_hr"] = (pd.to_numeric(batter_game[hr_col], errors="coerce").fillna(0) > 0).astype("Int64")
    batter_game["game_date"] = pd.to_datetime(batter_game.get("game_date"), errors="coerce")
    if "season" in batter_game.columns:
        batter_game["season"] = pd.to_numeric(batter_game["season"], errors="coerce").fillna(season).astype("Int64")
    else:
        batter_game["season"] = pd.Series([season] * len(batter_game), index=batter_game.index, dtype="Int64")

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

    null_team = int(hr["batter_team"].isna().sum()) if "batter_team" in hr.columns else len(hr)
    if null_team:
        logging.info("hr_batter_features: batter_team null rows=%s", null_team)

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


    if season is not None:
        target_paths = target_input_candidates(dirs["processed_dir"], "hr_batter", season)
        tpath = next((tp for tp in target_paths if tp.exists()), None)
        if tpath is None:
            logging.warning("Missing HR batter targets. Run: python scripts/build_targets_hr_batter.py --season %s --force", season)
        else:
            tgt = read_parquet(tpath)
            needed = {"game_pk", "batter_id", "target_hr"}
            if needed.issubset(tgt.columns):
                tgt = tgt[["game_pk", "batter_id", "target_hr"]].copy()
                tgt["game_pk"] = pd.to_numeric(tgt["game_pk"], errors="coerce").astype("Int64")
                tgt["batter_id"] = pd.to_numeric(tgt["batter_id"], errors="coerce").astype("Int64")
                hr = hr.drop(columns=["target_hr"], errors="ignore").merge(
                    tgt.drop_duplicates(subset=["game_pk", "batter_id"]), on=["game_pk", "batter_id"], how="left"
                )
            else:
                logging.warning("HR batter target file malformed: %s", tpath.resolve())

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

    marts_by_season_dir = dirs["marts_dir"] / "by_season"
    marts_by_season_dir.mkdir(parents=True, exist_ok=True)
    out_path = marts_by_season_dir / f"hr_batter_features_{season}.parquet"
    print_rowcount("hr_batter_features", hr)
    print(f"opp_sp_id null rate: {hr['opp_sp_id'].isna().mean():.2%}")
    print(f"Writing to: {out_path.resolve()}")
    write_parquet(hr, out_path)
    return out_path
