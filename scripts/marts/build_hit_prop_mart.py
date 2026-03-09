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


def _derive_ab_pa_per_game(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "bat_ab_per_game_roll15" not in out.columns:
        ab_col = _pick(list(out.columns), ["bat_ab_per_game_roll15", "ab_per_game_roll15", "bat_ab_roll15", "ab_roll15"])
        g_col = _pick(list(out.columns), ["bat_g_roll15", "games_roll15", "g_roll15", "bat_games_roll15"])
        if ab_col and g_col and ab_col != "bat_ab_per_game_roll15":
            out["bat_ab_per_game_roll15"] = pd.to_numeric(out[ab_col], errors="coerce") / pd.to_numeric(out[g_col], errors="coerce").replace(0, np.nan)
    if "bat_pa_per_game_roll15" not in out.columns:
        pa_col = _pick(list(out.columns), ["bat_pa_per_game_roll15", "pa_per_game_roll15", "bat_pa_roll15", "pa_roll15"])
        g_col = _pick(list(out.columns), ["bat_g_roll15", "games_roll15", "g_roll15", "bat_games_roll15"])
        if pa_col and g_col and pa_col != "bat_pa_per_game_roll15":
            out["bat_pa_per_game_roll15"] = pd.to_numeric(out[pa_col], errors="coerce") / pd.to_numeric(out[g_col], errors="coerce").replace(0, np.nan)
    return out


def _context_from_spine(spine: pd.DataFrame) -> pd.DataFrame:
    s = spine.copy()
    s["game_pk"] = pd.to_numeric(s.get("game_pk"), errors="coerce").astype("Int64")

    park_col = _pick(list(s.columns), ["park_factor", "park_factor_run", "venue_factor", "park_run_factor"])
    wind_col = _pick(list(s.columns), ["weather_wind", "wind_speed", "wind_mph", "wind"])
    temp_col = _pick(list(s.columns), ["temperature", "temp_f", "game_temp", "weather_temp"])

    out = s[["game_pk"]].drop_duplicates().copy()
    out["park_factor"] = pd.to_numeric(s[park_col], errors="coerce") if park_col else np.nan
    out["weather_wind"] = pd.to_numeric(s[wind_col], errors="coerce") if wind_col else np.nan
    out["temperature"] = pd.to_numeric(s[temp_col], errors="coerce") if temp_col else np.nan

    if "lineup_slot" in s.columns:
        lineup = pd.to_numeric(s["lineup_slot"], errors="coerce")
        if lineup.notna().any() and "batter_id" in s.columns:
            tmp = s[["game_pk", "batter_id", "lineup_slot"]].copy()
            tmp["game_pk"] = pd.to_numeric(tmp["game_pk"], errors="coerce").astype("Int64")
            tmp["batter_id"] = pd.to_numeric(tmp["batter_id"], errors="coerce").astype("Int64")
            tmp["lineup_slot"] = pd.to_numeric(tmp["lineup_slot"], errors="coerce")
            out = out.merge(tmp.drop_duplicates(subset=["game_pk", "batter_id"]), on="game_pk", how="left")
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

    batter["game_pk"] = pd.to_numeric(batter.get("game_pk"), errors="coerce").astype("Int64")
    batter["game_date"] = pd.to_datetime(batter.get("game_date"), errors="coerce")
    batter_id_col = _pick(list(batter.columns), ["batter_id", "batter", "player_id", "mlbam_batter_id"])
    if batter_id_col is None:
        raise ValueError("batter rolling missing batter identifier")
    batter["batter_id"] = pd.to_numeric(batter[batter_id_col], errors="coerce").astype("Int64")

    team_col = _pick(list(batter.columns), ["batter_team", "bat_team", "batting_team", "team", "team_abbrev", "team_name"])
    home_col = _pick(list(batter.columns), ["home_team", "home_team_abbr"])
    away_col = _pick(list(batter.columns), ["away_team", "away_team_abbr"])
    lineup_col = _pick(list(batter.columns), ["lineup_slot", "bat_order", "batting_order", "lineup_position"])

    batter["batter_team"] = batter[team_col] if team_col else pd.NA
    if team_col and home_col and away_col:
        batter["home_away"] = np.where(batter[team_col].astype(str) == batter[home_col].astype(str), 1.0, 0.0)
        batter["opponent_team"] = np.where(batter["home_away"] == 1.0, batter[away_col], batter[home_col])
    else:
        batter["home_away"] = np.nan
        batter["opponent_team"] = pd.NA

    batter["lineup_slot"] = pd.to_numeric(batter[lineup_col], errors="coerce") if lineup_col else np.nan

    if not spine.empty:
        ctx = _context_from_spine(spine)
        merge_cols = ["game_pk"]
        if "batter_id" in ctx.columns and ctx["batter_id"].notna().any():
            merge_cols = ["game_pk", "batter_id"]
        batter = batter.merge(ctx.drop_duplicates(subset=merge_cols), on=merge_cols, how="left", suffixes=("", "_spine"))
        for c in ["lineup_slot", "park_factor", "weather_wind", "temperature"]:
            spine_c = f"{c}_spine"
            if spine_c in batter.columns:
                batter[c] = pd.to_numeric(batter[c], errors="coerce").fillna(pd.to_numeric(batter[spine_c], errors="coerce"))
                batter = batter.drop(columns=[spine_c], errors="ignore")

    if "park_factor" not in batter.columns:
        park_col = _pick(list(batter.columns), ["park_factor", "park_factor_run", "venue_factor", "park_run_factor"])
        batter["park_factor"] = pd.to_numeric(batter[park_col], errors="coerce") if park_col else np.nan
    if "weather_wind" not in batter.columns:
        wind_col = _pick(list(batter.columns), ["weather_wind", "wind_speed", "wind_mph", "wind"])
        batter["weather_wind"] = pd.to_numeric(batter[wind_col], errors="coerce") if wind_col else np.nan
    if "temperature" not in batter.columns:
        temp_col = _pick(list(batter.columns), ["temperature", "temp_f", "game_temp", "weather_temp"])
        batter["temperature"] = pd.to_numeric(batter[temp_col], errors="coerce") if temp_col else np.nan

    if not pitcher.empty:
        pitcher["game_pk"] = pd.to_numeric(pitcher.get("game_pk"), errors="coerce").astype("Int64")
        pid_col = _pick(list(pitcher.columns), ["pitcher_id", "pitcher", "player_id", "mlb_id"])
        pteam_col = _pick(list(pitcher.columns), ["pitcher_team", "pit_team", "team", "team_abbrev", "team_name"])
        if pid_col and pteam_col:
            pitcher["pitcher_id"] = pd.to_numeric(pitcher[pid_col], errors="coerce").astype("Int64")
            pnum = [c for c in pitcher.select_dtypes(include=[np.number]).columns if c not in {"game_pk", "pitcher_id"}]
            pit_game_team = pitcher.groupby(["game_pk", pteam_col], as_index=False)[["pitcher_id"] + pnum].mean(numeric_only=True)
            pit_game_team = pit_game_team.rename(columns={pteam_col: "opponent_team", "pitcher_id": "opp_pitcher_id"})
            pit_game_team = pit_game_team.rename(columns={c: f"pit_{c}" for c in pnum})
            batter = batter.merge(pit_game_team, on=["game_pk", "opponent_team"], how="left")

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

    season_series = pd.to_numeric(batter.get("season"), errors="coerce")
    batter["season"] = season_series.astype("Int64") if season_series.notna().any() else batter["game_date"].dt.year.astype("Int64")

    all_frames: list[pd.DataFrame] = []
    by_season_dir = dirs["marts_dir"] / "by_season"
    by_season_dir.mkdir(parents=True, exist_ok=True)

    for season in range(args.season_start, args.season_end + 1):
        s_df = batter[pd.to_numeric(batter["season"], errors="coerce") == season].copy()
        targets = _read_targets(dirs["processed_dir"], season)
        if not targets.empty:
            s_df = s_df.merge(targets, on=["game_pk", "batter_id"], how="left")
        s_df = s_df.drop_duplicates(subset=["game_pk", "batter_id"], keep="last")

        context_cols = ["lineup_slot", "park_factor", "weather_wind", "temperature"]
        context_rates = {c: float(pd.to_numeric(s_df.get(c), errors="coerce").notna().mean()) if c in s_df.columns else 0.0 for c in context_cols}
        logging.info("hit_prop_mart season=%s rows=%s context_non_null_rates=%s", season, len(s_df), context_rates)

        out_season = by_season_dir / f"hit_prop_features_{season}.parquet"
        write_parquet(s_df, out_season)
        all_frames.append(s_df)

    full = pd.concat(all_frames, ignore_index=True, sort=False) if all_frames else pd.DataFrame()
    full_sort_cols = [c for c in ["game_date", "game_pk", "batter_id"] if c in full.columns]
    if full_sort_cols:
        full = full.sort_values(full_sort_cols, kind="mergesort")

    full_context_rates = {
        c: float(pd.to_numeric(full.get(c), errors="coerce").notna().mean()) if c in full.columns else 0.0
        for c in ["lineup_slot", "park_factor", "weather_wind", "temperature"]
    }
    logging.info("hit_prop_mart full_rows=%s context_non_null_rates=%s", len(full), full_context_rates)

    out_full = dirs["marts_dir"] / "hit_prop_features.parquet"
    write_parquet(full, out_full)
    logging.info("hit_prop_mart complete rows=%s path=%s", len(full), out_full)
    print(f"hit_prop_mart={out_full}")


if __name__ == "__main__":
    main()
