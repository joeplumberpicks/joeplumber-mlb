from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils.io import read_parquet, write_parquet

_BATTER_ID_CANDIDATES = ["batter_id", "mlbam_batter_id", "batter", "player_id"]
_PITCHER_ID_CANDIDATES = ["pitcher_id", "pitcher", "mlbam_pitcher_id", "player_id"]
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
_TARGET_COLS = [
    "target_hit_1p",
    "target_tb_2p",
    "target_rbi_1p",
    "target_bb_1p",
    "target_hit1p",
    "target_tb2p",
    "target_rbi1p",
    "target_bb1p",
]
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


def _load_hitter_prop_targets(processed_dir: Path, season: int) -> pd.DataFrame:
    """Authoritative batter-game targets built by scripts/build_targets_hitter_props.py."""
    t_path = processed_dir / "targets" / "hitter_props" / f"targets_hitter_props_{season}.parquet"
    t = read_parquet(
        t_path,
        columns=["game_pk", "batter_id", "target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"],
    ).copy()
    t["game_pk"] = pd.to_numeric(t["game_pk"], errors="coerce").astype("Int64")
    t["batter_id"] = pd.to_numeric(t["batter_id"], errors="coerce").astype("Int64")
    t = t.dropna(subset=["game_pk", "batter_id"]).drop_duplicates(subset=["game_pk", "batter_id"])
    return t

def build_hitter_batter_features(dirs: dict[str, Path], season: int) -> Path:
    rolling_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    pa_path = dirs["processed_dir"] / "by_season" / f"pa_{season}.parquet"

    if not rolling_path.exists():
        raise FileNotFoundError(f"Missing batter rolling features: {rolling_path.resolve()}")

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

    out = roll.copy()

    # --- ensure keys + season ---
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["batter_id"] = pd.to_numeric(out["batter_id"], errors="coerce").astype("Int64")
    out["season"] = int(season)

    # --- merge hitter targets (authoritative) ---
    targets = _load_hitter_prop_targets(Path(dirs["processed_dir"]), season)
    before = len(out)
    out = out.merge(targets, on=["game_pk", "batter_id"], how="left")
    labeled = int(out["target_hit1p"].notna().sum()) if "target_hit1p" in out.columns else 0
    logging.info(
        "hitter_batter_features target merge: rows_before=%s rows_after=%s labeled=%s labeled_pct=%.4f",
        before,
        len(out),
        labeled,
        (labeled / len(out)) if len(out) else 0.0,
    )
    out = out[out["target_hit1p"].notna()].copy()

    # IMPORTANT: do NOT drop rows due to roll-window NaNs; only enforce keys.
    out = out.dropna(subset=["game_pk", "batter_id"]).copy()

    out = _prune_leaky_columns(out)

    for col in ["target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]:
        if col in out.columns:
            null_rate = float(out[col].isna().mean()) if len(out) else 0.0
            pos_rate = float(pd.to_numeric(out[col], errors="coerce").fillna(0).mean()) if len(out) else 0.0
            logging.info("hitter_batter_features %s null_rate=%.4f pos_rate=%.4f", col, null_rate, pos_rate)

    marts_by_season_dir = dirs["marts_dir"] / "by_season"
    marts_by_season_dir.mkdir(parents=True, exist_ok=True)
    out_path = marts_by_season_dir / f"hitter_batter_features_{season}.parquet"
    write_parquet(out, out_path)
    logging.info("hitter_batter_features rows=%s path=%s", len(out), out_path.resolve())
    return out_path
