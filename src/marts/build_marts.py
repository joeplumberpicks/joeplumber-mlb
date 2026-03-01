from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.marts.build_hitter_batter_features import build_hitter_batter_features
from src.marts.build_pitcher_game_features import build_pitcher_game_features
from src.targets.paths import target_input_candidates
from src.utils.checks import print_rowcount, require_files
from src.utils.io import read_parquet, write_parquet

MART_SCHEMAS = {
    "hr_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_hr"],
    "nrfi_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_nrfi", "target_yrfi"],
    "moneyline_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_home_win"],
    "hitter_props_features.parquet": ["game_pk", "batter_id", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_hitter_prop", "target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"],
    "pitcher_props_features.parquet": ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key", "target_pitcher_prop"],
}


def _marts_by_season_dir(marts_dir: Path) -> Path:
    path = marts_dir / "by_season"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _mart_out_path(marts_dir: Path, mart_name: str, season: int | None) -> Path:
    if season is None:
        return marts_dir / mart_name
    stem = mart_name.replace(".parquet", "")
    return _marts_by_season_dir(marts_dir) / f"{stem}_{season}.parquet"


def _filter_to_season(df: pd.DataFrame, season: int | None) -> pd.DataFrame:
    if season is None or df.empty:
        return df
    out = df.copy()
    if "game_date" in out.columns:
        gd = pd.to_datetime(out["game_date"], errors="coerce")
        out = out[gd.dt.year == season].copy()
    elif "season" in out.columns:
        out = out[pd.to_numeric(out["season"], errors="coerce") == season].copy()
    elif "game_year" in out.columns:
        out = out[pd.to_numeric(out["game_year"], errors="coerce") == season].copy()
    return out


def _apply_date_range(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if df.empty or "game_date" not in df.columns:
        return df
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    if start:
        out = out[out["game_date"] >= pd.to_datetime(start)]
    if end:
        out = out[out["game_date"] <= pd.to_datetime(end)]
    return out


def _log_mart_stats(name: str, df: pd.DataFrame) -> None:
    n_games = int(df["game_pk"].nunique()) if "game_pk" in df.columns else 0
    if "game_date" in df.columns and len(df):
        gd = pd.to_datetime(df["game_date"], errors="coerce")
        min_d = gd.min()
        max_d = gd.max()
    else:
        min_d = pd.NaT
        max_d = pd.NaT
    msg = f"{name} rows={len(df)} unique_games={n_games} game_date_min={min_d} game_date_max={max_d}"
    if "target_home_win" in df.columns:
        msg += f" target_home_win_null_rate={float(df['target_home_win'].isna().mean()) if len(df) else 0.0:.4f}"
    logging.info(msg)


def _game_level_rollups(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty or "game_pk" not in df.columns:
        return pd.DataFrame(columns=["game_pk"])
    numeric = [c for c in df.select_dtypes(include=["number"]).columns if c != "game_pk"]
    if not numeric:
        return df[["game_pk"]].drop_duplicates().assign(**{f"{prefix}_feature_count": 0})
    agg = df.groupby("game_pk", dropna=False)[numeric].mean().reset_index()
    return agg.rename(columns={c: f"{prefix}_{c}" for c in numeric})


def _base_mart(spine: pd.DataFrame, schema: list[str]) -> pd.DataFrame:
    base_cols = [
        c
        for c in ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key"]
        if c in spine.columns
    ]
    df = spine[base_cols].copy() if not spine.empty else pd.DataFrame(columns=base_cols)
    for col in schema:
        if col not in df.columns:
            df[col] = pd.NA
    return df[schema]




def _read_target_file(processed_dir: Path, market: str, season: int | None, required_cols: list[str]) -> pd.DataFrame:
    if season is None:
        return pd.DataFrame()
    for p in target_input_candidates(processed_dir, market, season):
        if p.exists():
            df = read_parquet(p)
            miss = [c for c in required_cols if c not in df.columns]
            if miss:
                logging.warning("Target file %s missing cols=%s", p.resolve(), miss)
                return pd.DataFrame()
            return df
    logging.warning("Missing target file for market=%s season=%s. Run: python scripts/build_targets_%s.py --season %s --force", market, season, market, season)
    return pd.DataFrame()

def _merge_moneyline_targets(mart_df: pd.DataFrame, processed_dir: Path, season: int | None) -> pd.DataFrame:
    out = mart_df.copy()
    if season is None:
        return out

    targets = _read_target_file(processed_dir, "moneyline", season, ["game_pk", "target_home_win"])
    if targets.empty:
        return out

    slim = targets[["game_pk", "target_home_win"]].copy()
    slim["game_pk"] = pd.to_numeric(slim["game_pk"], errors="coerce").astype("Int64")
    slim["target_home_win"] = pd.to_numeric(slim["target_home_win"], errors="coerce").astype("Int64")

    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out = out.merge(slim.drop_duplicates(subset=["game_pk"], keep="last"), on="game_pk", how="left", suffixes=("", "_t"))
    if "target_home_win_t" in out.columns:
        out["target_home_win"] = pd.to_numeric(out.get("target_home_win"), errors="coerce").fillna(
            pd.to_numeric(out["target_home_win_t"], errors="coerce")
        ).astype("Int64")
        out = out.drop(columns=["target_home_win_t"])

    null_rate = float(out["target_home_win"].isna().mean()) if len(out) else 0.0
    pos_rate = float(pd.to_numeric(out["target_home_win"], errors="coerce").fillna(0).mean()) if len(out) else 0.0
    logging.info("moneyline target_home_win null_rate=%.6f pos_rate=%.6f", null_rate, pos_rate)
    return out


def _load_hitter_targets(processed_dir: Path, season: int | None) -> pd.DataFrame:
    if season is None:
        return pd.DataFrame()
    df = _read_target_file(processed_dir, "hitter_props", season, ["game_pk", "batter_id"])
    if df.empty:
        return pd.DataFrame()
    need = ["game_pk", "batter_id", "target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]
    if not set(["game_pk", "batter_id"]).issubset(df.columns):
        return pd.DataFrame()
    for c in need:
        if c not in df.columns:
            df[c] = pd.NA
    return df[need].copy()


def _moneyline_side_offense_features(batter_roll: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    if batter_roll.empty or spine.empty:
        return spine

    team_col = next((c for c in ["batter_team", "batting_team", "bat_team", "team", "offense_team"] if c in batter_roll.columns), None)
    if team_col is None:
        logging.warning("moneyline side offense features skipped: no batter team column found in batter rolling")
        return spine

    exclude_cols = {
        "game_pk", "batter", "batter_id", "bat_batter_id", "pitcher_id", "season", "park_id", "home_team", "away_team", "game_date"
    }
    feat_cols = [c for c in batter_roll.select_dtypes(include=["number"]).columns if c not in exclude_cols]
    if not feat_cols:
        return spine

    bat_team_agg = batter_roll.groupby(["game_pk", team_col])[feat_cols].mean().reset_index()
    home_agg = bat_team_agg.rename(columns={c: f"home_off_{c}" for c in feat_cols})
    out = spine.merge(home_agg, left_on=["game_pk", "home_team"], right_on=["game_pk", team_col], how="left")
    out = out.drop(columns=[team_col], errors="ignore")

    away_agg = bat_team_agg.rename(columns={c: f"away_off_{c}" for c in feat_cols})
    out = out.merge(away_agg, left_on=["game_pk", "away_team"], right_on=["game_pk", team_col], how="left")
    out = out.drop(columns=[team_col], errors="ignore")

    home_cols = [c for c in out.columns if c.startswith("home_off_")]
    diff_map: dict[str, pd.Series] = {}
    for home_col in home_cols:
        suffix = home_col[len("home_off_"):]
        away_col = f"away_off_{suffix}"
        if away_col in out.columns:
            diff_map[f"diff_off_{suffix}"] = out[home_col] - out[away_col]
    if diff_map:
        out = pd.concat([out, pd.DataFrame(diff_map)], axis=1)
    return out


def build_marts(
    dirs: dict[str, Path],
    season: int | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Path]:
    spine_path = dirs["processed_dir"] / "model_spine_game.parquet"
    require_files([spine_path], "mart_build_model_spine")

    spine = read_parquet(spine_path)
    spine["game_pk"] = pd.to_numeric(spine.get("game_pk"), errors="coerce").astype("Int64")
    spine = _filter_to_season(spine, season)
    spine = _apply_date_range(spine, start, end)

    batter_roll_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    pitcher_roll_path = dirs["processed_dir"] / "pitcher_game_rolling.parquet"
    batter_roll = read_parquet(batter_roll_path) if batter_roll_path.exists() else pd.DataFrame()
    pitcher_roll = read_parquet(pitcher_roll_path) if pitcher_roll_path.exists() else pd.DataFrame()
    batter_roll = _filter_to_season(batter_roll, season)
    pitcher_roll = _filter_to_season(pitcher_roll, season)
    batter_roll = _apply_date_range(batter_roll, start, end)
    pitcher_roll = _apply_date_range(pitcher_roll, start, end)

    batter_game_rollup = _game_level_rollups(batter_roll, "bat")
    pitcher_game_rollup = _game_level_rollups(pitcher_roll, "pit")

    outputs: dict[str, Path] = {}
    for filename, schema in MART_SCHEMAS.items():
        mart_df = _base_mart(spine, schema)
        if not batter_game_rollup.empty:
            mart_df = mart_df.merge(batter_game_rollup, on="game_pk", how="left")
        if not pitcher_game_rollup.empty:
            mart_df = mart_df.merge(pitcher_game_rollup, on="game_pk", how="left")

        if filename == "moneyline_features.parquet":
            mart_df = _moneyline_side_offense_features(batter_roll, mart_df)
            mart_df = _merge_moneyline_targets(mart_df, dirs["processed_dir"], season)
            if "target_home_win" in mart_df.columns:
                mart_df = mart_df[mart_df["target_home_win"].notna()].copy()
                logging.info(
                    "moneyline labeled rows=%s unique_games=%s null_rate=%.4f",
                    len(mart_df),
                    int(mart_df["game_pk"].nunique()) if "game_pk" in mart_df.columns else 0,
                    float(mart_df["target_home_win"].isna().mean()) if len(mart_df) else 0.0,
                )
        elif filename == "nrfi_features.parquet":
            nrfi_t = _read_target_file(dirs["processed_dir"], "nrfi", season, ["game_pk", "target_nrfi", "target_yrfi"])
            if not nrfi_t.empty:
                mart_df = mart_df.merge(nrfi_t[["game_pk", "target_nrfi", "target_yrfi"]].drop_duplicates(subset=["game_pk"]), on="game_pk", how="left", suffixes=("", "_t"))
                for c in ["target_nrfi", "target_yrfi"]:
                    tc = f"{c}_t"
                    if tc in mart_df.columns:
                        mart_df[c] = pd.to_numeric(mart_df.get(c), errors="coerce").fillna(pd.to_numeric(mart_df[tc], errors="coerce")).astype("Int64")
                        mart_df = mart_df.drop(columns=[tc])
        elif filename == "hitter_props_features.parquet":
            if "batter_id" in batter_roll.columns:
                batter_ids = batter_roll[["game_pk", "batter_id"]].copy()
            elif "batter" in batter_roll.columns:
                batter_ids = batter_roll[["game_pk", "batter"]].rename(columns={"batter": "batter_id"})
            else:
                batter_ids = pd.DataFrame(columns=["game_pk", "batter_id"])
            if not batter_ids.empty:
                mart_df = mart_df.merge(batter_ids.drop_duplicates(), on="game_pk", how="left")
            targets = _load_hitter_targets(dirs["processed_dir"], season)
            if not targets.empty and {"game_pk", "batter_id"}.issubset(mart_df.columns):
                mart_df = mart_df.merge(targets, on=["game_pk", "batter_id"], how="left", suffixes=("", "_t"))
                for c in ["target_hit1p", "target_tb2p", "target_bb1p", "target_rbi1p"]:
                    tc = f"{c}_t"
                    if tc in mart_df.columns:
                        mart_df[c] = pd.to_numeric(mart_df.get(c), errors="coerce").fillna(pd.to_numeric(mart_df[tc], errors="coerce")).astype("Int64")
                        mart_df = mart_df.drop(columns=[tc])

        elif filename == "pitcher_props_features.parquet":
            pt = _read_target_file(dirs["processed_dir"], "pitcher_props", season, ["game_pk", "pitcher_id", "target_k", "target_outs", "target_er", "target_bb"])
            if not pt.empty:
                if "pitcher_id" in pitcher_roll.columns:
                    pid = pitcher_roll[["game_pk", "pitcher_id"]].drop_duplicates()
                elif "pitcher" in pitcher_roll.columns:
                    pid = pitcher_roll[["game_pk", "pitcher"]].rename(columns={"pitcher": "pitcher_id"}).drop_duplicates()
                else:
                    pid = pd.DataFrame(columns=["game_pk", "pitcher_id"])
                if not pid.empty:
                    mart_df = mart_df.merge(pid, on="game_pk", how="left")
                    mart_df = mart_df.merge(pt, on=["game_pk", "pitcher_id"], how="left", suffixes=("", "_t"))
                    for c in ["target_k", "target_outs", "target_er", "target_bb"]:
                        tc=f"{c}_t"
                        if tc in mart_df.columns:
                            mart_df[c]=pd.to_numeric(mart_df.get(c), errors="coerce").fillna(pd.to_numeric(mart_df[tc], errors="coerce"))
                            mart_df = mart_df.drop(columns=[tc])

        mart_df = _filter_to_season(mart_df, season)
        mart_df = _apply_date_range(mart_df, start, end)

        print_rowcount(filename.replace(".parquet", ""), mart_df)
        _log_mart_stats(filename.replace(".parquet", ""), mart_df)
        out_path = _mart_out_path(dirs["marts_dir"], filename, season)
        print(f"Writing to: {out_path.resolve()}")
        write_parquet(mart_df, out_path)
        outputs[filename] = out_path

    if season is not None:
        outputs["hitter_batter_features.parquet"] = build_hitter_batter_features(dirs, season)
        outputs["pitcher_game_features.parquet"] = build_pitcher_game_features(dirs, season)

    return outputs
