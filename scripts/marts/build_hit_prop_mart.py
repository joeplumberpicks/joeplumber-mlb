from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.paths import target_input_candidates
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import write_parquet
from src.utils.logging import configure_logging, log_header

LINEUP_PA_MAP = {1: 4.65, 2: 4.55, 3: 4.45, 4: 4.35, 5: 4.25, 6: 4.10, 7: 3.95, 8: 3.80, 9: 3.70}


SAFE_ID_CONTEXT_COLS = {
    "game_pk", "game_date", "batter_id", "opp_pitcher_id", "season", "home_away", "park_id", "batter_team", "opponent_team"
}
SAFE_ENGINEERED_COLS = {
    "lineup_slot", "expected_batting_order_pa", "lineup_confidence", "expected_ab_proxy", "park_factor_hits", "temperature", "weather_wind"
}
ROLL_SUFFIXES = ("_roll3", "_roll7", "_roll15", "_roll30")
RAW_LEAKAGE_BASES = {
    "h", "hr", "bb", "k", "hbp", "pitches", "swings", "whiffs", "contacts", "chases", "in_zone_pitches",
    "release_speed_mean", "release_speed_max", "release_spin_rate_mean", "release_spin_rate_max",
    "launch_speed_mean", "launch_speed_max", "launch_angle_mean", "launch_angle_max",
    "chase_rate", "whiff_rate", "contact_rate", "swing_rate", "zone_swing_rate",
}


def _base_feature_name(col: str) -> str:
    out = col
    for prefix in ("bat_", "pit_", "diff_off_", "home_", "away_"):
        if out.startswith(prefix):
            out = out[len(prefix):]
    return out


def _is_rolling_feature(col: str) -> bool:
    return col.endswith(ROLL_SUFFIXES)


def _is_leakage_prone_raw(col: str) -> bool:
    if _is_rolling_feature(col):
        return False
    return _base_feature_name(col) in RAW_LEAKAGE_BASES


def _safe_hit_mart_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    keep: list[str] = []
    dropped_raw: list[str] = []
    dropped_other: list[str] = []
    for c in df.columns:
        if c in SAFE_ID_CONTEXT_COLS or c in SAFE_ENGINEERED_COLS or c == "target_hit_1_plus":
            keep.append(c)
            continue
        if _is_rolling_feature(c) or (c.startswith("pit_") and c.endswith("_roll30")):
            keep.append(c)
            continue
        if _is_leakage_prone_raw(c):
            dropped_raw.append(c)
        else:
            dropped_other.append(c)
    return keep, dropped_raw, dropped_other


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build 1+ hit prop mart at batter-game grain.")
    p.add_argument("--season-start", type=int, required=True)
    p.add_argument("--season-end", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], cands: list[str]) -> str | None:
    colset = set(cols)
    for c in cands:
        if c in colset:
            return c
    return None


def _read_targets(processed_dir: Path, season: int) -> pd.DataFrame:
    for path in target_input_candidates(processed_dir, "hitter_props", season):
        if not path.exists():
            continue
        t = pd.read_parquet(path).copy()
        if "target_hit_1_plus" not in t.columns:
            if "target_hit1p" in t.columns:
                t["target_hit_1_plus"] = pd.to_numeric(t["target_hit1p"], errors="coerce").astype("Int64")
            elif "target_hit_1p" in t.columns:
                t["target_hit_1_plus"] = pd.to_numeric(t["target_hit_1p"], errors="coerce").astype("Int64")
        if "target_hit_1_plus" not in t.columns:
            continue
        batter_col = _pick(list(t.columns), ["batter_id", "batter", "player_id", "mlbam_batter_id"])
        if batter_col is None:
            continue
        t["game_pk"] = pd.to_numeric(t.get("game_pk"), errors="coerce").astype("Int64")
        t["batter_id"] = pd.to_numeric(t[batter_col], errors="coerce").astype("Int64")
        return t[["game_pk", "batter_id", "target_hit_1_plus"]].drop_duplicates()
    return pd.DataFrame(columns=["game_pk", "batter_id", "target_hit_1_plus"])


def _lineup_conf(slot: pd.Series) -> pd.Series:
    s = pd.to_numeric(slot, errors="coerce")
    out = pd.Series(0.75, index=slot.index, dtype="float64")
    out = out.where(~s.between(1, 5), 1.00)
    out = out.where(~s.between(6, 7), 0.92)
    out = out.where(~s.between(8, 9), 0.85)
    return out


def _ensure_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _ensure_season(df: pd.DataFrame) -> pd.Series:
    if "season" in df.columns:
        s = pd.to_numeric(df["season"], errors="coerce")
        if s.notna().any():
            return s.astype("Int64")
    if "game_date" in df.columns:
        gd = pd.to_datetime(df["game_date"], errors="coerce")
        return gd.dt.year.astype("Int64")
    return pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)


def _derive_ab_pa_per_game(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "bat_ab_per_game_roll15" not in out.columns:
        ab_pg = _pick(list(out.columns), ["bat_ab_per_game_roll15", "ab_per_game_roll15", "ab_pg_roll15", "bat_ab_per_g_roll15", "bat_ab_pg_roll15"])
        ab_roll = _pick(list(out.columns), ["bat_ab_roll15", "ab_roll15", "at_bats_roll15", "bat_ab_roll_15"])
        g_roll = _pick(list(out.columns), ["bat_g_roll15", "games_roll15", "g_roll15", "bat_games_roll15"])
        if ab_pg and ab_pg in out.columns:
            out["bat_ab_per_game_roll15"] = pd.to_numeric(out[ab_pg], errors="coerce")
        elif ab_roll and g_roll and ab_roll in out.columns and g_roll in out.columns:
            out["bat_ab_per_game_roll15"] = pd.to_numeric(out[ab_roll], errors="coerce") / pd.to_numeric(out[g_roll], errors="coerce").replace(0, np.nan)
        else:
            out["bat_ab_per_game_roll15"] = np.nan

    if "bat_pa_per_game_roll15" not in out.columns:
        pa_pg = _pick(list(out.columns), ["bat_pa_per_game_roll15", "pa_per_game_roll15", "pa_pg_roll15", "bat_pa_per_g_roll15", "bat_pa_pg_roll15"])
        pa_roll = _pick(list(out.columns), ["bat_pa_roll15", "pa_roll15", "plate_appearances_roll15", "bat_pa_roll_15"])
        g_roll = _pick(list(out.columns), ["bat_g_roll15", "games_roll15", "g_roll15", "bat_games_roll15"])
        if pa_pg and pa_pg in out.columns:
            out["bat_pa_per_game_roll15"] = pd.to_numeric(out[pa_pg], errors="coerce")
        elif pa_roll and g_roll and pa_roll in out.columns and g_roll in out.columns:
            out["bat_pa_per_game_roll15"] = pd.to_numeric(out[pa_roll], errors="coerce") / pd.to_numeric(out[g_roll], errors="coerce").replace(0, np.nan)
        else:
            out["bat_pa_per_game_roll15"] = np.nan
    return out




def _coalesce_feature(df: pd.DataFrame, out_col: str, candidates: list[str]) -> pd.DataFrame:
    if out_col in df.columns and pd.to_numeric(df[out_col], errors="coerce").notna().any():
        return df
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in candidates:
        if c in df.columns:
            out = out.fillna(pd.to_numeric(df[c], errors="coerce"))
    df[out_col] = out
    return df


def _add_required_batter_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    feature_map = {
        "h_roll3": ["h_roll3", "bat_h_roll3"],
        "h_roll7": ["h_roll7", "bat_h_roll7"],
        "h_roll15": ["h_roll15", "bat_h_roll15"],
        "h_roll30": ["h_roll30", "bat_h_roll30"],
        "launch_speed_mean_roll15": ["launch_speed_mean_roll15", "bat_launch_speed_mean_roll15"],
        "launch_angle_mean_roll15": ["launch_angle_mean_roll15", "bat_launch_angle_mean_roll15"],
        "whiff_rate_roll30": ["whiff_rate_roll30", "bat_whiff_rate_roll30"],
        "chase_rate_roll30": ["chase_rate_roll30", "bat_chase_rate_roll30"],
    }
    for k, cands in feature_map.items():
        out = _coalesce_feature(out, k, cands)
    return out

def _context_from_spine(spine: pd.DataFrame) -> pd.DataFrame:
    s = spine.copy()
    s["game_pk"] = pd.to_numeric(s.get("game_pk"), errors="coerce").astype("Int64")
    batter_id_col = _pick(list(s.columns), ["batter_id", "batter", "player_id"])
    lineup_col = _pick(list(s.columns), ["lineup_slot", "bat_order", "batting_order", "lineup_position", "order"])
    park_col = _pick(list(s.columns), ["park_factor", "park_factor_run", "venue_factor", "park_run_factor"])
    wind_col = _pick(list(s.columns), ["weather_wind", "wind_speed", "wind_mph", "wind"])
    temp_col = _pick(list(s.columns), ["temperature", "temp_f", "game_temp", "weather_temp"])

    logging.info(
        "hit_prop_mart spine source cols lineup=%s park=%s wind=%s temp=%s batter_id=%s",
        lineup_col,
        park_col,
        wind_col,
        temp_col,
        batter_id_col,
    )

    by_game = pd.DataFrame({
        "game_pk": s["game_pk"],
        "park_factor": _ensure_series(s, park_col),
        "weather_wind": _ensure_series(s, wind_col),
        "temperature": _ensure_series(s, temp_col),
    })
    by_game = by_game.groupby("game_pk", as_index=False).last()

    by_player = pd.DataFrame(columns=["game_pk", "batter_id", "lineup_slot"])
    if batter_id_col and lineup_col:
        by_player = s[["game_pk", batter_id_col, lineup_col]].copy()
        by_player = by_player.rename(columns={batter_id_col: "batter_id", lineup_col: "lineup_slot"})
        by_player["batter_id"] = pd.to_numeric(by_player["batter_id"], errors="coerce").astype("Int64")
        by_player["lineup_slot"] = pd.to_numeric(by_player["lineup_slot"], errors="coerce")
        by_player = by_player.dropna(subset=["batter_id"]).drop_duplicates(subset=["game_pk", "batter_id"], keep="last")

    return by_game, by_player




def _join_optional_park_weather(batter: pd.DataFrame, dirs: dict[str, Path], spine: pd.DataFrame) -> pd.DataFrame:
    out = batter.copy()

    parks_path = dirs["reference_dir"] / "parks.parquet"
    if parks_path.exists():
        try:
            parks = pd.read_parquet(parks_path).copy()
            parks_key = _pick(list(parks.columns), ["canonical_park_key", "venue_id", "park_id"])
            if parks_key and "park_factor_hits" in parks.columns:
                out_key_col = _pick(list(out.columns), ["canonical_park_key", "venue_id", "park_id"])
                if out_key_col is None and not spine.empty:
                    spine_k = _pick(list(spine.columns), ["canonical_park_key", "venue_id", "park_id"])
                    if spine_k:
                        sp = spine[["game_pk", spine_k]].copy().rename(columns={spine_k: "_park_join_key"})
                        sp["game_pk"] = pd.to_numeric(sp["game_pk"], errors="coerce").astype("Int64")
                        out = out.merge(sp.drop_duplicates(subset=["game_pk"], keep="last"), on="game_pk", how="left")
                        out_key_col = "_park_join_key"
                if out_key_col:
                    left = out.copy()
                    left["_park_join_key"] = left[out_key_col].astype(str)
                    p2 = parks[[parks_key, "park_factor_hits"]].copy()
                    p2["_park_join_key"] = p2[parks_key].astype(str)
                    out = left.merge(p2[["_park_join_key", "park_factor_hits"]].drop_duplicates(subset=["_park_join_key"]), on="_park_join_key", how="left")
                    out = out.drop(columns=["_park_join_key"], errors="ignore")
            else:
                logging.warning("parks.parquet found but missing expected keys/park_factor_hits")
        except Exception:
            logging.exception("Failed optional parks join; continuing without park_factor_hits")
    else:
        logging.warning("Optional parks reference missing: %s", parks_path)

    weather_path = dirs["processed_dir"] / "weather_game.parquet"
    if weather_path.exists():
        try:
            wg = pd.read_parquet(weather_path).copy()
            if "game_pk" in wg.columns:
                wg["game_pk"] = pd.to_numeric(wg["game_pk"], errors="coerce").astype("Int64")
                temp_col = _pick(list(wg.columns), ["temperature", "temp_f", "game_temp", "weather_temp"])
                wind_col = _pick(list(wg.columns), ["weather_wind", "wind_speed", "wind_mph", "wind"])
                keep = ["game_pk"]
                if temp_col:
                    keep.append(temp_col)
                if wind_col:
                    keep.append(wind_col)
                tmp = wg[keep].drop_duplicates(subset=["game_pk"], keep="last")
                out = out.merge(tmp, on="game_pk", how="left", suffixes=("", "_wg"))
                if temp_col:
                    out["temperature"] = pd.to_numeric(out.get("temperature"), errors="coerce").fillna(pd.to_numeric(out.get(f"{temp_col}_wg"), errors="coerce"))
                    out = out.drop(columns=[f"{temp_col}_wg"], errors="ignore")
                if wind_col:
                    out["weather_wind"] = pd.to_numeric(out.get("weather_wind"), errors="coerce").fillna(pd.to_numeric(out.get(f"{wind_col}_wg"), errors="coerce"))
                    out = out.drop(columns=[f"{wind_col}_wg"], errors="ignore")
        except Exception:
            logging.exception("Failed optional weather_game join; continuing")
    else:
        logging.warning("Optional weather table missing: %s", weather_path)

    if "park_factor_hits" not in out.columns:
        out["park_factor_hits"] = np.nan
    return out

def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_hit_prop_mart.log")
    log_header("scripts/marts/build_hit_prop_mart.py", repo_root, config_path, dirs)

    bat_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    pit_path = dirs["processed_dir"] / "pitcher_game_rolling.parquet"
    spine_path = dirs["processed_dir"] / "model_spine_game.parquet"

    batter = pd.read_parquet(bat_path).copy()
    pitcher = pd.read_parquet(pit_path).copy() if pit_path.exists() else pd.DataFrame()
    spine = pd.read_parquet(spine_path).copy() if spine_path.exists() else pd.DataFrame()

    logging.info("hit_prop_mart source_cols batter_sample=%s", sorted(list(batter.columns))[:80])
    logging.info("hit_prop_mart source_cols pitcher_sample=%s", sorted(list(pitcher.columns))[:80] if not pitcher.empty else [])
    logging.info("hit_prop_mart source_cols spine_sample=%s", sorted(list(spine.columns))[:80] if not spine.empty else [])

    batter["game_pk"] = pd.to_numeric(batter.get("game_pk"), errors="coerce").astype("Int64")
    batter["game_date"] = pd.to_datetime(batter.get("game_date"), errors="coerce")
    batter_id_col = _pick(list(batter.columns), ["batter_id", "batter", "player_id", "mlbam_batter_id"])
    if batter_id_col is None:
        raise ValueError("batter rolling missing batter identifier")
    batter["batter_id"] = pd.to_numeric(batter[batter_id_col], errors="coerce").astype("Int64")

    team_col = _pick(list(batter.columns), ["batter_team", "bat_team", "batting_team", "team", "team_abbrev", "team_name"])
    home_col = _pick(list(batter.columns), ["home_team", "home_team_abbr"])
    away_col = _pick(list(batter.columns), ["away_team", "away_team_abbr"])
    lineup_col = _pick(list(batter.columns), ["lineup_slot", "bat_order", "batting_order", "lineup_position", "order"])
    park_col = _pick(list(batter.columns), ["park_factor", "park_factor_run", "venue_factor", "park_run_factor"])
    wind_col = _pick(list(batter.columns), ["weather_wind", "wind_speed", "wind_mph", "wind"])
    temp_col = _pick(list(batter.columns), ["temperature", "temp_f", "game_temp", "weather_temp"])

    logging.info(
        "hit_prop_mart batter source cols lineup=%s park=%s wind=%s temp=%s team=%s",
        lineup_col,
        park_col,
        wind_col,
        temp_col,
        team_col,
    )

    batter["batter_team"] = batter[team_col] if team_col else pd.NA
    if team_col and home_col and away_col:
        batter["home_away"] = np.where(batter[team_col].astype(str) == batter[home_col].astype(str), 1.0, 0.0)
        batter["opponent_team"] = np.where(batter["home_away"] == 1.0, batter[away_col], batter[home_col])
    else:
        batter["home_away"] = np.nan
        batter["opponent_team"] = pd.NA

    batter["lineup_slot"] = _ensure_series(batter, lineup_col)
    batter["park_factor"] = _ensure_series(batter, park_col)
    batter["weather_wind"] = _ensure_series(batter, wind_col)
    batter["temperature"] = _ensure_series(batter, temp_col)

    if not spine.empty:
        spine_game, spine_player = _context_from_spine(spine)
        batter = batter.merge(spine_game, on="game_pk", how="left", suffixes=("", "_spine"))
        for c in ["park_factor", "weather_wind", "temperature"]:
            batter[c] = pd.to_numeric(batter[c], errors="coerce").fillna(pd.to_numeric(batter.get(f"{c}_spine"), errors="coerce"))
            batter = batter.drop(columns=[f"{c}_spine"], errors="ignore")

        if not spine_player.empty:
            batter = batter.merge(spine_player, on=["game_pk", "batter_id"], how="left", suffixes=("", "_spine"))
            batter["lineup_slot"] = pd.to_numeric(batter["lineup_slot"], errors="coerce").fillna(pd.to_numeric(batter.get("lineup_slot_spine"), errors="coerce"))
            batter = batter.drop(columns=["lineup_slot_spine"], errors="ignore")

    batter = _join_optional_park_weather(batter, dirs, spine)

    # derive opp_pitcher_id from live/model spine when available
    if not spine.empty:
        sp_cols = [c for c in ["game_pk", "home_sp_id", "away_sp_id", "venue_id", "park_id"] if c in spine.columns]
        if sp_cols:
            sp = spine[sp_cols].copy()
            sp["game_pk"] = pd.to_numeric(sp["game_pk"], errors="coerce").astype("Int64")
            if "home_sp_id" in sp.columns:
                sp["home_sp_id"] = pd.to_numeric(sp["home_sp_id"], errors="coerce").astype("Int64")
            if "away_sp_id" in sp.columns:
                sp["away_sp_id"] = pd.to_numeric(sp["away_sp_id"], errors="coerce").astype("Int64")
            batter = batter.merge(sp.drop_duplicates(subset=["game_pk"], keep="last"), on="game_pk", how="left", suffixes=("", "_sp"))
            if "opp_pitcher_id" not in batter.columns:
                batter["opp_pitcher_id"] = pd.Series(pd.NA, index=batter.index, dtype="Int64")
            away_mask = pd.to_numeric(batter.get("home_away"), errors="coerce") == 0.0
            home_mask = pd.to_numeric(batter.get("home_away"), errors="coerce") == 1.0
            if "home_sp_id" in batter.columns:
                batter.loc[away_mask, "opp_pitcher_id"] = pd.to_numeric(batter.loc[away_mask, "home_sp_id"], errors="coerce").astype("Int64")
            if "away_sp_id" in batter.columns:
                batter.loc[home_mask, "opp_pitcher_id"] = pd.to_numeric(batter.loc[home_mask, "away_sp_id"], errors="coerce").astype("Int64")
            if "park_id" not in batter.columns:
                if "park_id_sp" in batter.columns:
                    batter["park_id"] = batter["park_id_sp"]
                elif "venue_id" in batter.columns:
                    batter["park_id"] = batter["venue_id"]

    # explicit batter rolling feature coverage from required family
    batter = _add_required_batter_rolling_features(batter)

    if not pitcher.empty:
        pitcher["game_pk"] = pd.to_numeric(pitcher.get("game_pk"), errors="coerce").astype("Int64")
        pid_col = _pick(list(pitcher.columns), ["pitcher_id", "pitcher", "player_id", "mlb_id"])
        if pid_col:
            pitcher["pitcher_id"] = pd.to_numeric(pitcher[pid_col], errors="coerce").astype("Int64")
            pit_needed = {
                "pit_whiff_rate_roll30": ["whiff_rate_roll30", "pit_whiff_rate_roll30"],
                "pit_contact_rate_roll30": ["contact_rate_roll30", "pit_contact_rate_roll30"],
                "pit_k_roll30": ["k_roll30", "pit_k_roll30"],
                "pit_bb_roll30": ["bb_roll30", "pit_bb_roll30"],
            }
            pit_join = pitcher[["game_pk", "pitcher_id"]].copy()
            for out_col, cands in pit_needed.items():
                ser = pd.Series(np.nan, index=pitcher.index, dtype="float64")
                for c in cands:
                    if c in pitcher.columns:
                        ser = ser.fillna(pd.to_numeric(pitcher[c], errors="coerce"))
                pit_join[out_col] = ser
            pit_join = pit_join.drop_duplicates(subset=["game_pk", "pitcher_id"], keep="last")
            batter["opp_pitcher_id"] = pd.to_numeric(batter.get("opp_pitcher_id"), errors="coerce").astype("Int64")
            batter = batter.merge(pit_join, left_on=["game_pk", "opp_pitcher_id"], right_on=["game_pk", "pitcher_id"], how="left")
            batter = batter.drop(columns=["pitcher_id"], errors="ignore")

    batter = _derive_ab_pa_per_game(batter)
    batter["expected_batting_order_pa"] = pd.to_numeric(batter["lineup_slot"], errors="coerce").map(LINEUP_PA_MAP)
    batter["lineup_confidence"] = _lineup_conf(batter["lineup_slot"])
    batter["expected_ab_proxy"] = batter["lineup_confidence"] * (
        0.65 * pd.to_numeric(batter.get("bat_ab_per_game_roll15"), errors="coerce")
        + 0.35 * pd.to_numeric(batter.get("expected_batting_order_pa"), errors="coerce")
    )
    batter["expected_ab_proxy"] = batter["expected_ab_proxy"].fillna(
        batter["lineup_confidence"] * pd.to_numeric(batter.get("expected_batting_order_pa"), errors="coerce")
    )

    if "diff_off_contact_rate_roll30" not in batter.columns:
        batter["diff_off_contact_rate_roll30"] = pd.to_numeric(batter.get("bat_contact_rate_roll30"), errors="coerce") - pd.to_numeric(batter.get("pit_contact_rate_roll30"), errors="coerce")
    if "diff_off_whiff_rate_roll30" not in batter.columns:
        batter["diff_off_whiff_rate_roll30"] = pd.to_numeric(batter.get("bat_whiff_rate_roll30"), errors="coerce") - pd.to_numeric(batter.get("pit_whiff_rate_roll30"), errors="coerce")
    if "diff_off_launch_speed_roll30" not in batter.columns:
        batter["diff_off_launch_speed_roll30"] = pd.to_numeric(batter.get("bat_launch_speed_mean_roll15"), errors="coerce")

    batter["season"] = _ensure_season(batter)

    all_frames: list[pd.DataFrame] = []
    by_season_dir = dirs["marts_dir"] / "by_season"
    by_season_dir.mkdir(parents=True, exist_ok=True)

    context_cols = [
        "lineup_slot",
        "expected_batting_order_pa",
        "lineup_confidence",
        "expected_ab_proxy",
        "bat_ab_per_game_roll15",
        "bat_pa_per_game_roll15",
        "park_factor_hits",
        "temperature",
        "weather_wind",
    ]

    for season in range(args.season_start, args.season_end + 1):
        s_df = batter[pd.to_numeric(batter["season"], errors="coerce") == season].copy()
        targets = _read_targets(dirs["processed_dir"], season)
        if not targets.empty:
            s_df = s_df.merge(targets, on=["game_pk", "batter_id"], how="left")
        s_df = s_df.drop_duplicates(subset=["game_pk", "batter_id"], keep="last")

        keep_cols, dropped_raw, dropped_other = _safe_hit_mart_columns(s_df)
        s_df = s_df[[c for c in keep_cols if c in s_df.columns]].copy()

        context_rates = {c: float(pd.to_numeric(s_df.get(c), errors="coerce").notna().mean()) if c in s_df.columns else 0.0 for c in context_cols}
        numeric_cols = [c for c in s_df.columns if pd.api.types.is_numeric_dtype(s_df[c]) and c not in {"game_pk", "batter_id", "opp_pitcher_id", "season"}]
        nn = {c: float(pd.to_numeric(s_df[c], errors="coerce").notna().mean()) for c in numeric_cols}
        top_nn = sorted(nn.items(), key=lambda kv: kv[1], reverse=True)[:20]
        logging.info("hit_prop_mart season=%s rows=%s feature_count=%s approved_rolling_features=%s dropped_leakage_raw=%s context_non_null_rates=%s", season, len(s_df), len(s_df.columns), sum(1 for c in s_df.columns if _is_rolling_feature(c)), len(dropped_raw), context_rates)
        logging.info("hit_prop_mart season=%s top20_feature_non_null_pct=%s", season, top_nn)
        sample_cols = [c for c in ["lineup_slot","expected_batting_order_pa","lineup_confidence","bat_ab_per_game_roll15","bat_pa_per_game_roll15","expected_ab_proxy"] if c in s_df.columns]
        if sample_cols:
            logging.info("hit_prop_mart season=%s sample_opportunity_rows=%s", season, s_df[sample_cols].head(5).to_dict(orient="records"))

        out_season = by_season_dir / f"hit_prop_features_{season}.parquet"
        write_parquet(s_df, out_season)
        all_frames.append(s_df)

    full = pd.concat(all_frames, ignore_index=True, sort=False) if all_frames else pd.DataFrame()
    full_sort_cols = [c for c in ["game_date", "game_pk", "batter_id"] if c in full.columns]
    if full_sort_cols:
        full = full.sort_values(full_sort_cols, kind="mergesort")

    full_context_rates = {c: float(pd.to_numeric(full.get(c), errors="coerce").notna().mean()) if c in full.columns else 0.0 for c in context_cols}
    full_numeric = [c for c in full.columns if pd.api.types.is_numeric_dtype(full[c]) and c not in {"game_pk", "batter_id", "opp_pitcher_id", "season"}]
    full_nn = {c: float(pd.to_numeric(full[c], errors="coerce").notna().mean()) for c in full_numeric}
    top_full_nn = sorted(full_nn.items(), key=lambda kv: kv[1], reverse=True)[:30]
    logging.info("hit_prop_mart full_rows=%s feature_count=%s context_non_null_rates=%s", len(full), len(full.columns), full_context_rates)
    logging.info("hit_prop_mart top30_feature_non_null_pct=%s", top_full_nn)

    if len(full.columns) <= 100:
        logging.warning("hit_prop mart feature_count=%s (<=100)", len(full.columns))

    out_full = dirs["marts_dir"] / "hit_prop_features.parquet"
    write_parquet(full, out_full)
    logging.info("hit_prop_mart complete rows=%s path=%s", len(full), out_full)
    print(f"hit_prop_mart={out_full}")


if __name__ == "__main__":
    main()
