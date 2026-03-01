from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import read_parquet, write_parquet

_PITCHER_ID_CANDIDATES = ["pitcher_id", "pitcher", "mlbam_pitcher_id", "player_id"]
_K_CANDIDATES = ["so", "k", "strikeouts", "pitcher_k"]
_OUTS_CANDIDATES = ["outs", "outs_recorded", "pitcher_outs"]


_ONE_OUT_EVENTS = {
    "strikeout",
    "field_out",
    "force_out",
    "ground_out",
    "fly_out",
    "line_out",
    "pop_out",
    "sac_fly",
    "sac_bunt",
    "fielders_choice_out",
    "sac_fly_error",
}
_TWO_OUT_EVENTS = {
    "double_play",
    "grounded_into_double_play",
    "strikeout_double_play",
    "sac_fly_double_play",
}
_THREE_OUT_EVENTS = {"triple_play"}

_STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}


def _outs_from_events(events_series: pd.Series) -> pd.Series:
    lower = events_series.astype(str).str.lower()
    outs = pd.Series(0, index=events_series.index, dtype="int16")
    outs[lower.isin(_ONE_OUT_EVENTS)] = 1
    outs[lower.isin(_TWO_OUT_EVENTS)] = 2
    outs[lower.isin(_THREE_OUT_EVENTS)] = 3
    return outs


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_pitcher_game_features(dirs: dict[str, Path], season: int) -> Path:
    rolling_path = dirs["processed_dir"] / "pitcher_game_rolling.parquet"
    pg_path = dirs["processed_dir"] / "by_season" / f"pitcher_game_{season}.parquet"
    season_pa_path = dirs["processed_dir"] / "by_season" / f"pa_{season}.parquet"
    events_dir = dirs["processed_dir"] / "events_pa"
    events_file = dirs["processed_dir"] / "events_pa.parquet"

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

    k_agg = pd.DataFrame(columns=["game_pk", "pitcher_id", "target_k_events"])
    outs_agg = pd.DataFrame(columns=["game_pk", "pitcher_id", "target_outs_events"])

    ev: pd.DataFrame | None = None
    if season_pa_path.exists():
        pa_cols = ["game_pk", "pitcher_id", "pitcher", "events", "event_type", "outs_when_up"]
        logging.info("pitcher_game targets loading PA from by_season file: %s", season_pa_path.resolve())
        ev = read_parquet(season_pa_path, columns=pa_cols)
    else:
        fallback_path = events_dir if events_dir.exists() else events_file
        if fallback_path.exists():
            pa_cols = ["game_pk", "pitcher_id", "pitcher", "events", "event_type", "game_date", "outs_when_up"]
            logging.info("pitcher_game targets loading PA from fallback path: %s", fallback_path.resolve())
            ev = read_parquet(fallback_path, columns=pa_cols, filters=[("season", "=", season)])

    if ev is not None and not ev.empty:
        ev_pitcher_col = "pitcher_id" if "pitcher_id" in ev.columns else ("pitcher" if "pitcher" in ev.columns else None)
        event_col = "events" if "events" in ev.columns else ("event_type" if "event_type" in ev.columns else None)
        if ev_pitcher_col is not None and "game_pk" in ev.columns and event_col is not None:
            cols = ["game_pk", ev_pitcher_col, event_col] + (["game_date"] if "game_date" in ev.columns else [])
            ev = ev[cols].copy()
            ev["game_pk"] = pd.to_numeric(ev["game_pk"], errors="coerce").astype("Int64")
            ev["pitcher_id"] = pd.to_numeric(ev[ev_pitcher_col], errors="coerce").astype("Int64")
            if "game_date" in ev.columns:
                ev["game_date"] = pd.to_datetime(ev["game_date"], errors="coerce")
                ev = ev[ev["game_date"].dt.year == season].copy()

            lower_events = ev[event_col].astype(str).str.lower()
            k_pa = lower_events.isin(_STRIKEOUT_EVENTS).astype("int16")
            k_agg = (
                ev.assign(_k=k_pa)
                .groupby(["game_pk", "pitcher_id"], as_index=False)["_k"]
                .sum()
                .rename(columns={"_k": "target_k_events"})
            )

            outs_pa = _outs_from_events(lower_events)
            outs_agg = (
                ev.assign(_outs=outs_pa)
                .groupby(["game_pk", "pitcher_id"], as_index=False)["_outs"]
                .sum()
                .rename(columns={"_outs": "target_outs_events"})
            )

    if "target_outs" in target_df.columns and target_df["target_outs"].isna().all() and outs_agg.empty:
        logging.warning("pitcher_game targets missing outs source; target_outs will be NA")

    out = roll.merge(target_df, on=["game_pk", "pitcher_id"], how="left")

    if not k_agg.empty:
        out = out.merge(k_agg, on=["game_pk", "pitcher_id"], how="left")
        out["target_k"] = pd.to_numeric(out["target_k_events"], errors="coerce")
        out = out.drop(columns=["target_k_events"])
    out["target_k"] = pd.to_numeric(out.get("target_k"), errors="coerce")

    if not outs_agg.empty:
        out = out.merge(outs_agg, on=["game_pk", "pitcher_id"], how="left")
        out["target_outs"] = pd.to_numeric(out.get("target_outs"), errors="coerce").fillna(
            pd.to_numeric(out["target_outs_events"], errors="coerce")
        )
        out = out.drop(columns=["target_outs_events"])
    out["target_outs"] = pd.to_numeric(out["target_outs"], errors="coerce")

    if "season" not in out.columns:
        out["season"] = season

    k_series = pd.to_numeric(out["target_k"], errors="coerce")
    k_null = float(k_series.isna().mean()) if len(out) else 0.0
    k_mean = float(k_series.mean()) if len(out) else 0.0
    k_min = float(k_series.min()) if len(out) else 0.0
    k_max = float(k_series.max()) if len(out) else 0.0
    outs_series = pd.to_numeric(out["target_outs"], errors="coerce")
    outs_null = float(outs_series.isna().mean()) if len(out) else 0.0
    outs_non_null = 1.0 - outs_null
    outs_mean = float(outs_series.mean()) if len(out) else 0.0
    outs_min = float(outs_series.min()) if len(out) else 0.0
    outs_max = float(outs_series.max()) if len(out) else 0.0
    logging.info(
        "pitcher_game_features target_k mean=%.4f min=%.4f max=%.4f null_pct=%.4f",
        k_mean,
        k_min,
        k_max,
        k_null,
    )
    logging.info("pitcher_game_features target_outs non_null_pct=%.4f", outs_non_null)
    logging.info("pitcher_game_features target_outs mean=%.4f min=%.4f max=%.4f", outs_mean, outs_min, outs_max)

    marts_by_season_dir = dirs["marts_dir"] / "by_season"
    marts_by_season_dir.mkdir(parents=True, exist_ok=True)
    out_path = marts_by_season_dir / f"pitcher_game_features_{season}.parquet"
    write_parquet(out, out_path)
    logging.info("pitcher_game_features rows=%s path=%s", len(out), out_path.resolve())
    return out_path
