from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.drive import resolve_data_dirs
from src.utils.team_ids import normalize_team_abbr


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    cset = set(cols)
    for c in candidates:
        if c in cset:
            return c
    return None


def _run_cmd(cmd: list[str], repo_root: Path) -> None:
    logging.info("live preflight running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def resolve_live_paths(*, config: dict, season: int, date_str: str) -> dict[str, Path]:
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    return {
        "live_spine_path": dirs["processed_dir"] / "live" / f"model_spine_game_{season}_{date_str}.parquet",
        "projected_lineups_path": dirs["processed_dir"] / "live" / f"projected_lineups_{season}_{date_str}.parquet",
        "live_weather_path": dirs["processed_dir"] / "live" / f"weather_game_{season}_{date_str}.parquet",
        "live_schedule_path": dirs["raw_dir"] / "live" / f"games_schedule_{season}_{date_str}.parquet",
        "live_schedule_cumulative_path": dirs["raw_dir"] / "live" / f"games_schedule_{season}.parquet",
    }


def run_live_preflight(
    *,
    repo_root: Path,
    config_path: Path,
    season: int,
    date_str: str,
    auto_build: bool = True,
    force_spine: bool = True,
    build_lineups: bool = True,
    build_weather: bool = True,
    permissive_live_context: bool = False,
) -> dict:
    from src.utils.config import load_config

    config = load_config(config_path)
    paths = resolve_live_paths(config=config, season=season, date_str=date_str)

    if auto_build:
        _run_cmd(
            [
                sys.executable,
                "scripts/ingest/ingest_schedule_games.py",
                "--date",
                date_str,
                "--season",
                str(season),
                "--game-types",
                "S,R",
                "--config",
                str(config_path),
            ],
            repo_root,
        )

        spine_cmd = [
            sys.executable,
            "scripts/live/build_spine_from_schedule.py",
            "--season",
            str(season),
            "--date",
            date_str,
            "--config",
            str(config_path),
        ]
        if force_spine:
            spine_cmd.append("--force")
        _run_cmd(spine_cmd, repo_root)

        if build_lineups:
            try:
                _run_cmd(
                    [
                        sys.executable,
                        "scripts/live/build_projected_lineups.py",
                        "--season",
                        str(season),
                        "--date",
                        date_str,
                        "--config",
                        str(config_path),
                    ],
                    repo_root,
                )
            except Exception:
                if not permissive_live_context:
                    raise
                logging.exception("Projected lineups build failed but permissive mode is enabled.")

        if build_weather:
            try:
                _run_cmd(
                    [
                        sys.executable,
                        "scripts/live/build_live_weather.py",
                        "--season",
                        str(season),
                        "--date",
                        date_str,
                        "--config",
                        str(config_path),
                    ],
                    repo_root,
                )
            except Exception:
                if not permissive_live_context:
                    raise
                logging.exception("Live weather build failed but permissive mode is enabled.")

    out = dict(paths)
    for key, p in paths.items():
        out[f"{key}_exists"] = p.exists()
    return out


def load_live_spine(live_spine_path: Path) -> pd.DataFrame:
    if not live_spine_path.exists():
        raise FileNotFoundError(f"Live spine not found: {live_spine_path}")
    df = pd.read_parquet(live_spine_path).copy()
    required = ["game_pk", "game_date", "season", "away_team", "home_team"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Live spine missing required columns: {missing}")
    df["game_pk"] = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    keep = required + [c for c in ["away_sp_id", "home_sp_id", "away_sp_name", "home_sp_name", "venue_id"] if c in df.columns]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    out = df[keep].drop_duplicates(subset=["game_pk"], keep="last").copy()
    return out


def load_live_weather(live_weather_path: Path) -> pd.DataFrame:
    if not live_weather_path.exists():
        return pd.DataFrame(columns=["game_pk", "weather_data_available"])
    wx = pd.read_parquet(live_weather_path).copy()
    if wx.empty:
        return pd.DataFrame(columns=["game_pk", "weather_data_available"])

    if "game_pk" in wx.columns:
        wx["game_pk"] = pd.to_numeric(wx["game_pk"], errors="coerce").astype("Int64")
    if "game_date" in wx.columns:
        wx["game_date"] = pd.to_datetime(wx["game_date"], errors="coerce")

    team_candidates = ["away_team", "home_team", "away_team_abbr", "home_team_abbr", "away", "home"]
    for c in team_candidates:
        if c in wx.columns:
            wx[c] = wx[c].map(normalize_team_abbr)

    numeric = [c for c in wx.select_dtypes(include=[np.number]).columns if c != "game_pk"]
    keep = [c for c in ["game_pk", "game_date", "away_team", "home_team"] if c in wx.columns]
    cols = keep + numeric
    wx = wx[cols].copy()

    key_cols = [c for c in ["game_pk", "game_date", "away_team", "home_team"] if c in wx.columns]
    wx = wx.sort_values(key_cols, kind="mergesort") if key_cols else wx
    wx = wx.drop_duplicates(subset=key_cols if key_cols else [wx.columns[0]], keep="last")
    wx["weather_data_available"] = True
    return wx


def load_projected_lineups(projected_lineups_path: Path) -> pd.DataFrame:
    if not projected_lineups_path.exists():
        return pd.DataFrame()
    lu = pd.read_parquet(projected_lineups_path).copy()
    if lu.empty:
        return lu
    team_col = _pick(list(lu.columns), ["batter_team", "team", "team_abbrev", "batting_team"])
    if team_col:
        lu["batter_team"] = lu[team_col].map(normalize_team_abbr)
    if "game_pk" in lu.columns:
        lu["game_pk"] = pd.to_numeric(lu["game_pk"], errors="coerce").astype("Int64")
    if "lineup_slot" in lu.columns:
        lu["lineup_slot"] = pd.to_numeric(lu["lineup_slot"], errors="coerce")
    return lu


def _pick_proxy_columns(batter_roll: pd.DataFrame) -> dict[str, str | None]:
    cols = [c for c in batter_roll.columns if pd.api.types.is_numeric_dtype(batter_roll[c])]
    low = {c.lower(): c for c in cols}

    def pick(tokens: list[str]) -> str | None:
        for c in cols:
            lc = c.lower()
            if any(t in lc for t in tokens):
                return c
        return None

    chosen = {
        "onbase": pick(["obp", "bb_rate", "walk", "on_base", "woba"]),
        "contact": pick(["contact", "hit_rate", "whiff_inv", "k_inv", "k_rate", "avg"]),
        "power": pick(["iso", "slug", "xbh", "hard_hit", "barrel", "hr", "slg"]),
    }
    logging.info("Lineup proxy columns selected: %s", chosen)
    return chosen


def build_game_level_lineup_features(
    projected_lineups: pd.DataFrame,
    batter_rolling: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    if projected_lineups.empty:
        return pd.DataFrame(columns=["game_pk"])

    lu = projected_lineups.copy()
    if "game_pk" not in lu.columns:
        return pd.DataFrame(columns=["game_pk"])
    lu["game_pk"] = pd.to_numeric(lu["game_pk"], errors="coerce").astype("Int64")

    slot_col = _pick(list(lu.columns), ["lineup_slot", "bat_order", "batting_order", "order"])
    if slot_col is None:
        lu["lineup_slot"] = np.nan
        slot_col = "lineup_slot"
    lu["lineup_slot"] = pd.to_numeric(lu[slot_col], errors="coerce")

    bid_col = _pick(list(lu.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(lu.columns), ["player_name", "batter_name", "name"])
    team_col = _pick(list(lu.columns), ["batter_team", "team", "team_abbrev", "batting_team"])

    if team_col is None:
        return pd.DataFrame(columns=["game_pk"])
    lu["batter_team"] = lu[team_col].map(normalize_team_abbr)

    br = batter_rolling.copy()
    if br.empty:
        br = pd.DataFrame(columns=["batter_team"])
    br["game_date"] = pd.to_datetime(br.get("game_date"), errors="coerce")
    br = br[br["game_date"].notna() & (br["game_date"] < target_date)].copy()
    metric_map = _pick_proxy_columns(br) if not br.empty else {"onbase": None, "contact": None, "power": None}

    roll = pd.DataFrame()
    if not br.empty:
        br_team_col = _pick(list(br.columns), ["batter_team", "team", "team_abbrev", "batting_team"])
        if br_team_col:
            br["batter_team"] = br[br_team_col].map(normalize_team_abbr)
        if bid_col and bid_col in br.columns:
            br["join_batter_id"] = pd.to_numeric(br[bid_col], errors="coerce").astype("Int64")
        else:
            br_bid_col = _pick(list(br.columns), ["batter_id", "batter", "player_id"])
            br["join_batter_id"] = pd.to_numeric(br.get(br_bid_col), errors="coerce").astype("Int64")
        br_name_col = _pick(list(br.columns), ["player_name", "batter_name", "name"])
        br["join_name"] = br[br_name_col].astype(str).str.lower().str.replace(r"[^a-z0-9 ]+", "", regex=True).str.strip() if br_name_col else ""

        group_cols = ["join_batter_id"] if br["join_batter_id"].notna().any() else ["batter_team", "join_name"]
        keep = group_cols + [c for c in metric_map.values() if c is not None] + ["game_date"]
        roll = br[keep].sort_values("game_date").groupby(group_cols, as_index=False).tail(1)

    lu["join_batter_id"] = pd.to_numeric(lu[bid_col], errors="coerce").astype("Int64") if bid_col and bid_col in lu.columns else pd.Series(pd.NA, index=lu.index, dtype="Int64")
    lu["join_name"] = lu[name_col].astype(str).str.lower().str.replace(r"[^a-z0-9 ]+", "", regex=True).str.strip() if name_col and name_col in lu.columns else ""

    merged = lu.copy()
    if not roll.empty:
        if merged["join_batter_id"].notna().any() and "join_batter_id" in roll.columns:
            merged = merged.merge(roll.drop(columns=["game_date"], errors="ignore"), on=["join_batter_id"], how="left", suffixes=("", "_roll"))
        elif {"batter_team", "join_name"}.issubset(roll.columns):
            merged = merged.merge(roll.drop(columns=["game_date"], errors="ignore"), on=["batter_team", "join_name"], how="left", suffixes=("", "_roll"))

    def _side_agg(side_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for game_pk, g in side_df.groupby("game_pk", dropna=False):
            g = g.sort_values("lineup_slot", kind="mergesort")
            rec = {"game_pk": game_pk, "lineup_proj_hitters_count": int(g["lineup_slot"].notna().sum())}
            for tag, n in [("top3", 3), ("top5", 5)]:
                sub = g[g["lineup_slot"] <= n] if g["lineup_slot"].notna().any() else g.head(n)
                rec[f"lineup_{tag}_count"] = int(len(sub))
                for mtag, col in metric_map.items():
                    rec[f"lineup_{tag}_{mtag}_mean"] = pd.to_numeric(sub.get(col), errors="coerce").mean() if col in sub.columns else np.nan
            for mtag, col in metric_map.items():
                rec[f"lineup_full_{mtag}_mean"] = pd.to_numeric(g.get(col), errors="coerce").mean() if col in g.columns else np.nan
            full_means = [rec.get("lineup_full_onbase_mean"), rec.get("lineup_full_contact_mean"), rec.get("lineup_full_power_mean")]
            rec["lineup_quality_score"] = np.nanmean(full_means)
            rec["lineup_completeness_score"] = min(rec["lineup_proj_hitters_count"] / 9.0, 1.0)
            records.append(rec)
        return pd.DataFrame(records)

    team_game_cols = [c for c in ["game_pk", "away_team", "home_team"] if c in merged.columns]
    games = merged[team_game_cols].drop_duplicates(subset=["game_pk"], keep="last")

    away = pd.DataFrame(columns=["game_pk"])
    home = pd.DataFrame(columns=["game_pk"])
    if {"away_team", "batter_team"}.issubset(games.columns):
        away_rows = merged.merge(games[["game_pk", "away_team"]], on="game_pk", how="left")
        away_rows = away_rows[away_rows["batter_team"] == away_rows["away_team"].map(normalize_team_abbr)]
        away = _side_agg(away_rows)
        away = away.rename(columns={c: f"away_{c}" for c in away.columns if c != "game_pk"})
    if {"home_team", "batter_team"}.issubset(games.columns):
        home_rows = merged.merge(games[["game_pk", "home_team"]], on="game_pk", how="left")
        home_rows = home_rows[home_rows["batter_team"] == home_rows["home_team"].map(normalize_team_abbr)]
        home = _side_agg(home_rows)
        home = home.rename(columns={c: f"home_{c}" for c in home.columns if c != "game_pk"})

    out = games[["game_pk"]].merge(away, on="game_pk", how="left").merge(home, on="game_pk", how="left")
    return out.drop_duplicates(subset=["game_pk"], keep="last")


def merge_live_context(spine: pd.DataFrame, weather: pd.DataFrame, lineup_features: pd.DataFrame) -> pd.DataFrame:
    out = spine.drop_duplicates(subset=["game_pk"], keep="last").copy()

    if not weather.empty:
        wx = weather.copy()
        wx_key = "game_pk" if "game_pk" in wx.columns and wx["game_pk"].notna().any() else None
        if wx_key:
            out = out.merge(wx, on="game_pk", how="left", suffixes=("", "_wx"))
            out["weather_source"] = "live_weather_game_pk"
        else:
            join_cols = [c for c in ["game_date", "away_team", "home_team"] if c in out.columns and c in wx.columns]
            out = out.merge(wx, on=join_cols, how="left", suffixes=("", "_wx"))
            out["weather_source"] = "live_weather_team_date"
    else:
        out["weather_source"] = pd.NA

    if not lineup_features.empty:
        out = out.merge(lineup_features, on="game_pk", how="left")
        out["lineup_source"] = "projected_lineups"
    else:
        out["lineup_source"] = pd.NA

    out["has_weather"] = out.get("weather_data_available", False).fillna(False).astype(bool)
    away_comp = pd.to_numeric(out.get("away_lineup_completeness_score"), errors="coerce")
    home_comp = pd.to_numeric(out.get("home_lineup_completeness_score"), errors="coerce")
    out["has_projected_lineups"] = (away_comp.notna() | home_comp.notna())
    out = out.drop_duplicates(subset=["game_pk"], keep="last")
    return out


def summarize_live_context(ctx: pd.DataFrame) -> dict[str, float | int]:
    n = len(ctx)
    if n == 0:
        return {
            "games": 0,
            "pct_with_starters": 0.0,
            "pct_with_weather": 0.0,
            "pct_with_projected_lineups": 0.0,
            "away_lineup_found": 0,
            "home_lineup_found": 0,
        }
    starters = pd.to_numeric(ctx.get("away_sp_id"), errors="coerce").notna() & pd.to_numeric(ctx.get("home_sp_id"), errors="coerce").notna()
    away_found = int(pd.to_numeric(ctx.get("away_lineup_completeness_score"), errors="coerce").notna().sum())
    home_found = int(pd.to_numeric(ctx.get("home_lineup_completeness_score"), errors="coerce").notna().sum())
    return {
        "games": n,
        "pct_with_starters": float(starters.mean() * 100.0),
        "pct_with_weather": float(pd.Series(ctx.get("has_weather", False)).astype(bool).mean() * 100.0),
        "pct_with_projected_lineups": float(pd.Series(ctx.get("has_projected_lineups", False)).astype(bool).mean() * 100.0),
        "away_lineup_found": away_found,
        "home_lineup_found": home_found,
    }
