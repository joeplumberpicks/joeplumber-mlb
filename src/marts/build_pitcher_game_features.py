from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import read_parquet, write_parquet

_PITCHER_ID_CANDIDATES = ["pitcher_id", "pitcher", "mlbam_pitcher_id", "player_id"]
_K_CANDIDATES = ["so", "k", "strikeouts", "pitcher_k"]
_OUTS_CANDIDATES = ["outs", "outs_recorded", "pitcher_outs"]


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_pitcher_game_features(dirs: dict[str, Path], season: int) -> Path:
    rolling_path = dirs["processed_dir"] / "pitcher_game_rolling.parquet"
    pg_path = dirs["processed_dir"] / "by_season" / f"pitcher_game_{season}.parquet"
    events_path = dirs["processed_dir"] / "events_pa.parquet"

    if not rolling_path.exists():
        raise FileNotFoundError(f"Missing pitcher rolling features: {rolling_path.resolve()}")

    roll = read_parquet(rolling_path)
    if "game_date" in roll.columns:
        roll["game_date"] = pd.to_datetime(roll["game_date"], errors="coerce")
        roll = roll[roll["game_date"].dt.year == season].copy()
    if "season" in roll.columns:
        roll["season"] = pd.to_numeric(roll["season"], errors="coerce")
        roll = roll[(roll["season"].isna()) | (roll["season"] == season)].copy()

    roll_pitcher_col = _pick_col(roll, _PITCHER_ID_CANDIDATES)
    if roll_pitcher_col is None:
        raise ValueError(f"No pitcher id column in pitcher rolling. Available: {sorted(roll.columns.tolist())}")
    if roll_pitcher_col != "pitcher_id":
        roll = roll.rename(columns={roll_pitcher_col: "pitcher_id"})
    roll["pitcher_id"] = pd.to_numeric(roll["pitcher_id"], errors="coerce").astype("Int64")
    roll["game_pk"] = pd.to_numeric(roll["game_pk"], errors="coerce").astype("Int64")

    target_df = pd.DataFrame(columns=["game_pk", "pitcher_id", "target_k", "target_outs"])

    if pg_path.exists():
        pg = read_parquet(pg_path)
        pg_pitcher_col = _pick_col(pg, _PITCHER_ID_CANDIDATES)
        if pg_pitcher_col is not None and "game_pk" in pg.columns:
            k_col = _pick_col(pg, _K_CANDIDATES)
            outs_col = _pick_col(pg, _OUTS_CANDIDATES)
            logging.info("pitcher_game target source cols: k=%s outs=%s", k_col, outs_col)
            target_df = pg[["game_pk", pg_pitcher_col] + [c for c in [k_col, outs_col] if c is not None]].copy()
            target_df = target_df.rename(columns={pg_pitcher_col: "pitcher_id"})
            target_df["game_pk"] = pd.to_numeric(target_df["game_pk"], errors="coerce").astype("Int64")
            target_df["pitcher_id"] = pd.to_numeric(target_df["pitcher_id"], errors="coerce").astype("Int64")
            target_df["target_k"] = pd.to_numeric(target_df[k_col], errors="coerce") if k_col else pd.NA
            target_df["target_outs"] = pd.to_numeric(target_df[outs_col], errors="coerce") if outs_col else pd.NA
            target_df = target_df[["game_pk", "pitcher_id", "target_k", "target_outs"]]

    if (target_df.empty or target_df["target_k"].isna().all()) and events_path.exists():
        ev = read_parquet(events_path)
        ev_pitcher_col = _pick_col(ev, _PITCHER_ID_CANDIDATES)
        if ev_pitcher_col and "game_pk" in ev.columns:
            ev = ev[["game_pk", ev_pitcher_col] + [c for c in ["events", "event_type", "game_date"] if c in ev.columns]].copy()
            ev["game_pk"] = pd.to_numeric(ev["game_pk"], errors="coerce").astype("Int64")
            ev["pitcher_id"] = pd.to_numeric(ev[ev_pitcher_col], errors="coerce").astype("Int64")
            if "game_date" in ev.columns:
                ev["game_date"] = pd.to_datetime(ev["game_date"], errors="coerce")
                ev = ev[ev["game_date"].dt.year == season].copy()
            event_series = ev.get("events", ev.get("event_type", pd.Series(index=ev.index, dtype="object"))).astype(str).str.lower()
            ev["k_ev"] = (event_series == "strikeout").astype(int)
            k_fallback = ev.groupby(["game_pk", "pitcher_id"], dropna=False)["k_ev"].sum().reset_index()
            k_fallback = k_fallback.rename(columns={"k_ev": "target_k"})
            if target_df.empty:
                target_df = k_fallback
                target_df["target_outs"] = pd.NA
            else:
                target_df = target_df.merge(k_fallback, on=["game_pk", "pitcher_id"], how="left", suffixes=("", "_fb"))
                target_df["target_k"] = pd.to_numeric(target_df["target_k"], errors="coerce").fillna(
                    pd.to_numeric(target_df.get("target_k_fb"), errors="coerce")
                )
                if "target_k_fb" in target_df.columns:
                    target_df = target_df.drop(columns=["target_k_fb"])

    if "target_outs" in target_df.columns and target_df["target_outs"].isna().all():
        logging.warning("pitcher_game targets missing outs source; target_outs will be NA")

    out = roll.merge(target_df, on=["game_pk", "pitcher_id"], how="left")
    if "season" not in out.columns:
        out["season"] = season

    k_null = float(out["target_k"].isna().mean()) if len(out) else 0.0
    k_mean = float(pd.to_numeric(out["target_k"], errors="coerce").mean()) if len(out) else 0.0
    outs_null = float(out["target_outs"].isna().mean()) if len(out) else 0.0
    outs_mean = float(pd.to_numeric(out["target_outs"], errors="coerce").mean()) if len(out) else 0.0
    logging.info("pitcher_game_features target_k mean=%.4f null_pct=%.4f", k_mean, k_null)
    logging.info("pitcher_game_features target_outs mean=%.4f null_pct=%.4f", outs_mean, outs_null)

    out_path = dirs["marts_dir"] / "pitcher_game_features.parquet"
    write_parquet(out, out_path)
    logging.info("pitcher_game_features rows=%s path=%s", len(out), out_path.resolve())
    return out_path
