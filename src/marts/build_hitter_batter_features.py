from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import read_parquet, write_parquet

_BATTER_ID_CANDIDATES = ["batter_id", "batter", "mlbam_batter_id", "player_id"]
_HIT_EVENTS = {"single", "double", "triple", "home_run"}



_LEAKY_SAME_GAME_COLS = {
    "pitches",
    "swings",
    "contacts",
    "whiffs",
    "in_zone_pitches",
    "chases",
    "k",
    "bb",
    "hbp",
    "hr",
    "h",
    "launch_speed_mean",
    "launch_speed_max",
    "launch_angle_mean",
    "launch_angle_max",
}
_TARGET_COLS = ["target_hit_1p", "target_tb_2p", "target_rbi_1p", "target_bb_1p"]
_IDENTIFIER_COLS = {
    "game_pk",
    "batter_id",
    "game_date",
    "season",
    "home_team",
    "away_team",
    "park_id",
    "park_name",
    "canonical_park_key",
    "batting_team",
}


def _prune_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols: list[str] = []
    for col in df.columns:
        keep = False
        if col in _IDENTIFIER_COLS or col in _TARGET_COLS:
            keep = True
        elif "_roll" in col:
            keep = True
        elif ("_rate_roll" in col) or col.endswith(("_rate", "_pct")):
            # allow pregame rates derived from rolling (e.g. chase_rate_roll7)
            keep = "_roll" in col

        if keep:
            keep_cols.append(col)

    dropped_same_game = [c for c in df.columns if c in _LEAKY_SAME_GAME_COLS]
    # Hard-exclude known leaky columns even if caught by generic rules
    keep_cols = [c for c in keep_cols if c not in _LEAKY_SAME_GAME_COLS]

    dropped_cols = [c for c in df.columns if c not in keep_cols]
    pruned = df[keep_cols].copy()
    logging.info(
        "hitter_batter_features dropped_cols=%s dropped_same_game_cols=%s",
        len(dropped_cols),
        len(dropped_same_game),
    )
    logging.info("hitter_batter_features leaky columns absent check: h=%s bb=%s", "h" in pruned.columns, "bb" in pruned.columns)
    return pruned


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")

def build_hitter_batter_features(dirs: dict[str, Path], season: int) -> Path:
    rolling_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    events_path = dirs["processed_dir"] / "events_pa.parquet"

    if not rolling_path.exists():
        raise FileNotFoundError(f"Missing batter rolling features: {rolling_path.resolve()}")
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events_pa for targets: {events_path.resolve()}")

    roll = read_parquet(rolling_path)
    if "game_date" in roll.columns:
        roll["game_date"] = pd.to_datetime(roll["game_date"], errors="coerce")
        roll = roll[roll["game_date"].dt.year == season].copy()
    if "season" in roll.columns:
        roll["season"] = pd.to_numeric(roll["season"], errors="coerce")
        roll = roll[(roll["season"].isna()) | (roll["season"] == season)].copy()

    batter_col = _pick_col(roll, _BATTER_ID_CANDIDATES)
    if batter_col is None:
        raise ValueError(f"No batter id column in batter rolling. Available: {sorted(roll.columns.tolist())}")
    if batter_col != "batter_id":
        roll = roll.rename(columns={batter_col: "batter_id"})
    roll["batter_id"] = pd.to_numeric(roll["batter_id"], errors="coerce").astype("Int64")
    roll["game_pk"] = pd.to_numeric(roll["game_pk"], errors="coerce").astype("Int64")

    events = read_parquet(events_path)
    ev_batter_col = _pick_col(events, _BATTER_ID_CANDIDATES)
    if ev_batter_col is None:
        raise ValueError(f"No batter id column in events_pa. Available: {sorted(events.columns.tolist())}")

    needed = [
        c
        for c in [
            "game_pk",
            "game_date",
            "inning_topbot",
            "events",
            "event_type",
            "home_team",
            "away_team",
            ev_batter_col,
            "bat_score",
            "post_bat_score",
            "home_score",
            "post_home_score",
            "away_score",
            "post_away_score",
        ]
        if c in events.columns
    ]
    events = events[needed].copy()
    events["game_pk"] = pd.to_numeric(events["game_pk"], errors="coerce").astype("Int64")
    events["game_date"] = pd.to_datetime(events.get("game_date"), errors="coerce")
    events = events[events["game_date"].dt.year == season].copy()
    events = events.rename(columns={ev_batter_col: "batter_id"})
    events["batter_id"] = pd.to_numeric(events["batter_id"], errors="coerce").astype("Int64")

    half = events.get("inning_topbot", pd.Series(index=events.index, dtype="object")).astype(str).str.lower().str.strip()
    events["batting_team"] = pd.NA
    events.loc[half.str.startswith("top"), "batting_team"] = events.get("away_team")
    events.loc[half.str.startswith("bot"), "batting_team"] = events.get("home_team")

    ev = events.get("events", events.get("event_type", pd.Series(index=events.index, dtype="object"))).astype(str).str.lower()
    events["is_hit"] = ev.isin(_HIT_EVENTS).astype(int)
    events["tb"] = (
        (ev == "single").astype(int)
        + 2 * (ev == "double").astype(int)
        + 3 * (ev == "triple").astype(int)
        + 4 * (ev == "home_run").astype(int)
    )
    events["is_bb"] = ev.isin({"walk", "intent_walk"}).astype(int)

    bat_score = _numeric_series(events, "bat_score")
    post_bat_score = _numeric_series(events, "post_bat_score")
    events["rbi_pa"] = (post_bat_score - bat_score).fillna(0).clip(lower=0, upper=4)

    # Optional cross-check using offense-side team score delta from inning half.
    home_delta = (
        _numeric_series(events, "post_home_score")
        - _numeric_series(events, "home_score")
    ).fillna(0)
    away_delta = (
        _numeric_series(events, "post_away_score")
        - _numeric_series(events, "away_score")
    ).fillna(0)
    offense_delta = pd.Series(0, index=events.index, dtype="float64")
    offense_delta.loc[half.str.startswith("bot")] = home_delta.loc[half.str.startswith("bot")]
    offense_delta.loc[half.str.startswith("top")] = away_delta.loc[half.str.startswith("top")]
    offense_delta = offense_delta.fillna(0).clip(lower=0, upper=4)
    mismatch = (offense_delta - events["rbi_pa"]).abs() > 0
    logging.info(
        "hitter_batter_features rbi_pa score-delta mismatch_rate=%.4f mismatch_n=%s",
        float(mismatch.mean()) if len(events) else 0.0,
        int(mismatch.sum()),
    )

    agg_map: dict[str, tuple[str, str]] = {
        "hits": ("is_hit", "sum"),
        "tb": ("tb", "sum"),
        "bb": ("is_bb", "sum"),
        "rbi_game": ("rbi_pa", "sum"),
        "pa": ("events", "size") if "events" in events.columns else ("is_hit", "size"),
    }

    targets = events.groupby(["game_pk", "batter_id"], dropna=False).agg(**agg_map).reset_index()
    targets["target_hit_1p"] = (targets["hits"] >= 1).astype("Int64")
    targets["target_tb_2p"] = (targets["tb"] >= 2).astype("Int64")
    targets["target_bb_1p"] = (targets["bb"] >= 1).astype("Int64")
    targets["target_rbi_1p"] = (pd.to_numeric(targets["rbi_game"], errors="coerce").fillna(0) >= 1).astype("Int64")

    out = roll.merge(
        targets[["game_pk", "batter_id", "target_hit_1p", "target_tb_2p", "target_rbi_1p", "target_bb_1p"]],
        on=["game_pk", "batter_id"],
        how="left",
    )

    if "season" not in out.columns:
        out["season"] = season

    out = _prune_leaky_columns(out)

    for col in ["target_hit_1p", "target_tb_2p", "target_bb_1p", "target_rbi_1p"]:
        null_rate = float(out[col].isna().mean()) if len(out) else 0.0
        pos_rate = float(pd.to_numeric(out[col], errors="coerce").fillna(0).mean()) if len(out) else 0.0
        logging.info("hitter_batter_features %s null_rate=%.4f pos_rate=%.4f", col, null_rate, pos_rate)

    out_path = dirs["marts_dir"] / "hitter_batter_features.parquet"
    write_parquet(out, out_path)
    logging.info("hitter_batter_features rows=%s path=%s", len(out), out_path.resolve())
    return out_path
