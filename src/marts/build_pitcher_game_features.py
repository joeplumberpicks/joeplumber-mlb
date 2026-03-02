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


def _load_pitcher_prop_targets(processed_dir: Path, season: int) -> pd.DataFrame:
    """Authoritative pitcher-game targets built by scripts/build_targets_pitcher_props.py."""
    t_path = processed_dir / "targets" / "pitcher_props" / f"targets_pitcher_props_{season}.parquet"
    t = read_parquet(
        t_path,
        columns=["game_pk", "pitcher_id", "target_k", "target_outs", "target_er", "target_bb"],
    ).copy()
    t["game_pk"] = pd.to_numeric(t["game_pk"], errors="coerce").astype("Int64")
    t["pitcher_id"] = pd.to_numeric(t["pitcher_id"], errors="coerce").astype("Int64")
    t = t.dropna(subset=["game_pk", "pitcher_id"]).drop_duplicates(subset=["game_pk", "pitcher_id"])
    return t


def build_pitcher_game_features(dirs: dict[str, Path], season: int) -> Path:
    rolling_path = dirs["processed_dir"] / "pitcher_game_rolling.parquet"
    pg_path = dirs["processed_dir"] / "by_season" / f"pitcher_game_{season}.parquet"

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

    out = roll.copy()

    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")
    out["season"] = int(season)

    targets = _load_pitcher_prop_targets(Path(dirs["processed_dir"]), season)
    before = len(out)
    out = out.merge(targets, on=["game_pk", "pitcher_id"], how="left")
    labeled = int(out["target_k"].notna().sum()) if "target_k" in out.columns else 0
    logging.info(
        "pitcher_game_features target merge: rows_before=%s rows_after=%s labeled=%s labeled_pct=%.4f target_k_mean=%.4f",
        before,
        len(out),
        labeled,
        (labeled / len(out)) if len(out) else 0.0,
        float(pd.to_numeric(out["target_k"], errors="coerce").mean()) if "target_k" in out.columns and len(out) else 0.0,
    )
    out = out[out["target_k"].notna()].copy()
    out = out.dropna(subset=["game_pk", "pitcher_id"]).copy()

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
