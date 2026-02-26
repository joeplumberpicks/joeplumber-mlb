from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import read_parquet, write_parquet

_BATTER_ID_CANDIDATES = ["batter_id", "batter", "mlbam_batter_id", "player_id"]
_HIT_EVENTS = {"single", "double", "triple", "home_run"}


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


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

    needed = [c for c in ["game_pk", "game_date", "inning_topbot", "events", "home_team", "away_team", ev_batter_col, "rbi"] if c in events.columns]
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

    ev = events.get("events", pd.Series(index=events.index, dtype="object")).astype(str).str.lower()
    events["is_hit"] = ev.isin(_HIT_EVENTS).astype(int)
    events["tb"] = (
        (ev == "single").astype(int)
        + 2 * (ev == "double").astype(int)
        + 3 * (ev == "triple").astype(int)
        + 4 * (ev == "home_run").astype(int)
    )
    events["is_bb"] = ev.isin({"walk", "intent_walk"}).astype(int)

    agg_map: dict[str, tuple[str, str]] = {
        "hits": ("is_hit", "sum"),
        "tb": ("tb", "sum"),
        "bb": ("is_bb", "sum"),
        "pa": ("events", "size") if "events" in events.columns else ("is_hit", "size"),
    }
    has_rbi = "rbi" in events.columns
    if has_rbi:
        events["rbi"] = pd.to_numeric(events["rbi"], errors="coerce").fillna(0)
        agg_map["rbi"] = ("rbi", "sum")
    else:
        logging.warning("events_pa missing rbi column; target_rbi_1p will be NA")

    targets = events.groupby(["game_pk", "batter_id"], dropna=False).agg(**agg_map).reset_index()
    targets["target_hit_1p"] = (targets["hits"] >= 1).astype("Int64")
    targets["target_tb_2p"] = (targets["tb"] >= 2).astype("Int64")
    targets["target_bb_1p"] = (targets["bb"] >= 1).astype("Int64")
    if has_rbi:
        targets["target_rbi_1p"] = (targets["rbi"] >= 1).astype("Int64")
    else:
        targets["target_rbi_1p"] = pd.Series(pd.NA, index=targets.index, dtype="Int64")

    out = roll.merge(
        targets[["game_pk", "batter_id", "target_hit_1p", "target_tb_2p", "target_rbi_1p", "target_bb_1p"]],
        on=["game_pk", "batter_id"],
        how="left",
    )

    if "season" not in out.columns:
        out["season"] = season

    for col in ["target_hit_1p", "target_tb_2p", "target_bb_1p", "target_rbi_1p"]:
        null_rate = float(out[col].isna().mean()) if len(out) else 0.0
        pos_rate = float(pd.to_numeric(out[col], errors="coerce").fillna(0).mean()) if len(out) else 0.0
        logging.info("hitter_batter_features %s null_rate=%.4f pos_rate=%.4f", col, null_rate, pos_rate)

    out_path = dirs["marts_dir"] / "hitter_batter_features.parquet"
    write_parquet(out, out_path)
    logging.info("hitter_batter_features rows=%s path=%s", len(out), out_path.resolve())
    return out_path
