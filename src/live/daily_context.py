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


def _run_cmd(cmd: list[str], repo_root: Path, *, raise_on_error: bool = True) -> bool:
    logging.info("live preflight running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    if proc.returncode != 0:
        msg = f"Command failed (exit={proc.returncode}): {' '.join(cmd)}"
        if raise_on_error:
            raise RuntimeError(msg)
        logging.warning(msg)
        return False
    return True


def _lineup_output_stats(projected_lineups_path: Path) -> tuple[bool, int, int]:
    if not projected_lineups_path.exists():
        return False, 0, 0
    try:
        lu = pd.read_parquet(projected_lineups_path)
    except Exception:
        logging.exception("Failed reading projected lineup parquet: %s", projected_lineups_path)
        return False, 0, 0
    rows = int(len(lu))
    if rows == 0:
        return False, 0, 0
    team_col = _pick(list(lu.columns), ["batter_team", "team", "team_abbrev", "batting_team"])
    if team_col is None:
        return True, rows, 0
    teams = lu[team_col].dropna().astype(str).str.upper().str.strip()
    return True, rows, int(teams.nunique())


def _run_lineup_build_with_fallback(
    *,
    repo_root: Path,
    config_path: Path,
    season: int,
    date_str: str,
    projected_lineups_path: Path,
    permissive_live_context: bool,
) -> dict[str, object]:
    attempts = [
        ("scripts/live/build_projected_lineups.py", "primary"),
        ("scripts/live/build_projected_lineups_rotogrinders.py", "fallback"),
    ]
    errors: list[str] = []
    last_rows = 0
    last_teams = 0

    for idx, (script_path, stage) in enumerate(attempts):
        cmd = [
            sys.executable,
            script_path,
            "--season",
            str(season),
            "--date",
            date_str,
            "--config",
            str(config_path),
        ]
        if stage == "primary":
            logging.info("projected lineup build: starting primary builder=%s", script_path)
        else:
            logging.info("projected lineup build: starting fallback builder=%s", script_path)

        try:
            _run_cmd(cmd, repo_root, raise_on_error=True)
            exists, rows, teams = _lineup_output_stats(projected_lineups_path)
            last_rows, last_teams = rows, teams
            if exists:
                logging.info(
                    "projected lineup build: %s builder succeeded script=%s rows=%s unique_teams=%s",
                    stage,
                    script_path,
                    rows,
                    teams,
                )
                return {
                    "lineup_builder_used": script_path,
                    "lineup_rows": rows,
                    "lineup_unique_teams": teams,
                    "lineup_build_succeeded": True,
                }
            raise RuntimeError(
                f"Projected lineup parquet missing or empty after {script_path}: {projected_lineups_path}"
            )
        except Exception as exc:
            msg = f"{script_path} failed: {exc}"
            errors.append(msg)
            if idx + 1 < len(attempts):
                logging.warning("projected lineup build: %s; trying fallback builder next", msg)
            else:
                logging.error("projected lineup build: fallback failed: %s", msg)

    logging.error(
        "projected lineup build failed for all builders permissive_live_context=%s path=%s errors=%s",
        permissive_live_context,
        projected_lineups_path,
        errors,
    )
    if not permissive_live_context:
        raise RuntimeError("Projected lineup build failed for all builders: " + " | ".join(errors))
    logging.warning("permissive_live_context=True so continuing after projected lineup build failure")
    return {
        "lineup_builder_used": None,
        "lineup_rows": last_rows,
        "lineup_unique_teams": last_teams,
        "lineup_build_succeeded": False,
    }


def resolve_live_paths(*, config: dict, season: int, date_str: str) -> dict[str, Path]:
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    return {
        "live_spine_path": dirs["processed_dir"] / "live" / f"model_spine_game_{season}_{date_str}.parquet",
        "projected_lineups_path": dirs["processed_dir"] / "live" / f"projected_lineups_{season}_{date_str}.parquet",
        "confirmed_lineups_path": dirs["processed_dir"] / "live" / f"confirmed_lineups_{season}_{date_str}.parquet",
        "live_lineups_path": dirs["processed_dir"] / "live" / f"lineups_{season}_{date_str}.parquet",
        "live_weather_path": dirs["processed_dir"] / "live" / f"weather_game_{season}_{date_str}.parquet",
        "live_schedule_path": dirs["raw_dir"] / "live" / f"games_schedule_{season}_{date_str}.parquet",
        "live_schedule_cumulative_path": dirs["raw_dir"] / "live" / f"games_schedule_{season}.parquet",
    }


def _coverage_pct(numer: int, denom: int) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom * 100.0)


def validate_live_context(*, live_spine_path: Path) -> dict[str, float | int]:
    spine = load_live_spine(live_spine_path)
    n = int(len(spine))
    if n == 0:
        return {
            "slate_game_count": 0,
            "home_starter_coverage_pct": 0.0,
            "away_starter_coverage_pct": 0.0,
            "projected_lineup_coverage_pct": 0.0,
            "weather_coverage_pct": 0.0,
            "final_game_spine_row_count": 0,
        }

    home_sp = pd.to_numeric(spine.get("home_sp_id"), errors="coerce")
    away_sp = pd.to_numeric(spine.get("away_sp_id"), errors="coerce")
    has_lineups = _to_series(spine.get("has_projected_lineups", False), spine.index).fillna(False).astype(bool)
    has_weather = _to_series(spine.get("has_weather", False), spine.index).fillna(False).astype(bool)

    summary = {
        "slate_game_count": n,
        "home_starter_coverage_pct": _coverage_pct(int(home_sp.notna().sum()), n),
        "away_starter_coverage_pct": _coverage_pct(int(away_sp.notna().sum()), n),
        "projected_lineup_coverage_pct": _coverage_pct(int(has_lineups.sum()), n),
        "weather_coverage_pct": _coverage_pct(int(has_weather.sum()), n),
        "final_game_spine_row_count": n,
    }
    return summary


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
    emit_summary_line: bool = True,
) -> dict:
    from src.utils.config import load_config

    config = load_config(config_path)
    paths = resolve_live_paths(config=config, season=season, date_str=date_str)

    def _ensure_spine_exists_before_lineups() -> None:
        logging.info("live preflight: ensuring game spine exists before lineup build")
        if paths["live_spine_path"].exists() and not force_spine:
            logging.info("live preflight: spine confirmed path=%s", paths["live_spine_path"])
            return
        if not auto_build:
            raise FileNotFoundError(
                "Live spine required before lineup build but auto_build=False and spine is missing: "
                f"{paths['live_spine_path']}"
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
        if not paths["live_spine_path"].exists():
            raise FileNotFoundError(
                "Live spine still missing after build_spine_from_schedule.py ran: "
                f"{paths['live_spine_path']}"
            )
        logging.info("live preflight: spine confirmed path=%s", paths["live_spine_path"])

    if auto_build and not paths["live_schedule_path"].exists():
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

    spine_ensured_before_lineups = False
    if build_lineups:
        _ensure_spine_exists_before_lineups()
        spine_ensured_before_lineups = True

    lineup_meta = {
        "lineup_builder_used": None,
        "lineup_rows": 0,
        "lineup_unique_teams": 0,
        "lineup_build_succeeded": paths["projected_lineups_path"].exists(),
    }
    if auto_build and build_lineups and not paths["projected_lineups_path"].exists():
        if not paths["live_spine_path"].exists():
            _ensure_spine_exists_before_lineups()
            if not paths["live_spine_path"].exists():
                raise FileNotFoundError(
                    "Live spine is required before projected lineup build and is still missing: "
                    f"{paths['live_spine_path']}"
                )
        lineup_meta = _run_lineup_build_with_fallback(
            repo_root=repo_root,
            config_path=config_path,
            season=season,
            date_str=date_str,
            projected_lineups_path=paths["projected_lineups_path"],
            permissive_live_context=permissive_live_context,
        )
    if auto_build and build_weather and not paths["live_weather_path"].exists():
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
            raise_on_error=not permissive_live_context,
        )

    if auto_build and (force_spine or not paths["live_spine_path"].exists()) and not spine_ensured_before_lineups:
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
        if paths["live_spine_path"].exists():
            logging.info("live preflight: spine confirmed path=%s", paths["live_spine_path"])

    out = dict(paths)
    out.update(lineup_meta)
    for key, p in paths.items():
        out[f"{key}_exists"] = p.exists()
    if paths["live_spine_path"].exists():
        validation = validate_live_context(live_spine_path=paths["live_spine_path"])
        out.update(validation)
        logging.info(
            "Live context validation summary | slate_game_count=%s home_starter_coverage_pct=%.2f away_starter_coverage_pct=%.2f projected_lineup_coverage_pct=%.2f weather_coverage_pct=%.2f final_game_spine_row_count=%s",
            validation["slate_game_count"],
            validation["home_starter_coverage_pct"],
            validation["away_starter_coverage_pct"],
            validation["projected_lineup_coverage_pct"],
            validation["weather_coverage_pct"],
            validation["final_game_spine_row_count"],
        )
        if emit_summary_line:
            print(
                "live_preflight_validation "
                f"slate_games={validation['slate_game_count']} "
                f"home_starter_coverage_pct={validation['home_starter_coverage_pct']:.2f} "
                f"away_starter_coverage_pct={validation['away_starter_coverage_pct']:.2f} "
                f"projected_lineup_coverage_pct={validation['projected_lineup_coverage_pct']:.2f} "
                f"weather_coverage_pct={validation['weather_coverage_pct']:.2f} "
                f"final_spine_rows={validation['final_game_spine_row_count']}"
            )
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

    keep = required + [
        c
        for c in [
            "away_sp_id",
            "home_sp_id",
            "away_sp_name",
            "home_sp_name",
            "venue_id",
            "has_projected_lineups",
            "has_weather",
            "projected_lineup_rows",
        ]
        if c in df.columns
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA

    out = df[keep].drop_duplicates(subset=["game_pk"], keep="last").copy()
    return out


def load_live_weather(
    live_weather_path: Path | None = None,
    *,
    config: dict | None = None,
    season: int | None = None,
    date_str: str | None = None,
) -> pd.DataFrame:
    candidates: list[Path] = []
    if config is not None and season is not None and date_str is not None:
        paths = resolve_live_paths(config=config, season=season, date_str=date_str)
        candidates.append(paths["live_weather_path"])
    elif live_weather_path is not None:
        candidates.append(live_weather_path)

    live_weather_path = next((p for p in candidates if p.exists()), None)
    if live_weather_path is None:
        logging.info("live weather source=none")
        return pd.DataFrame(columns=["game_pk", "weather_data_available"])
    logging.info("live weather source=%s", live_weather_path)

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
    if key_cols:
        wx = wx.sort_values(key_cols, kind="mergesort")
        wx = wx.drop_duplicates(subset=key_cols, keep="last")
    else:
        wx = wx.drop_duplicates()

    wx["weather_data_available"] = True
    return wx


def load_live_lineups(
    *,
    config: dict,
    season: int,
    date_str: str,
) -> pd.DataFrame:
    paths = resolve_live_paths(config=config, season=season, date_str=date_str)
    loaded: list[tuple[str, pd.DataFrame]] = []
    for key in ["confirmed_lineups_path", "projected_lineups_path", "live_lineups_path"]:
        p = paths[key]
        if not p.exists():
            continue
        logging.info("live lineup source candidate=%s", p)
        lu = pd.read_parquet(p).copy()
        if lu.empty:
            continue
        team_col = _pick(list(lu.columns), ["batter_team", "team", "team_abbrev", "batting_team"])
        if team_col:
            lu["batter_team"] = lu[team_col].map(normalize_team_abbr)
        if "game_pk" in lu.columns:
            lu["game_pk"] = pd.to_numeric(lu["game_pk"], errors="coerce").astype("Int64")
        if "lineup_slot" in lu.columns:
            lu["lineup_slot"] = pd.to_numeric(lu["lineup_slot"], errors="coerce")
        lu["_lineup_source_key"] = key
        loaded.append((key, lu))
    if loaded:
        by_key = {k: df for k, df in loaded}
        out = by_key.get("confirmed_lineups_path", pd.DataFrame())
        if "projected_lineups_path" in by_key:
            proj = by_key["projected_lineups_path"]
            if out.empty:
                out = proj.copy()
            elif "game_pk" in out.columns and "game_pk" in proj.columns:
                confirmed_games = set(pd.to_numeric(out["game_pk"], errors="coerce").dropna().astype(int).tolist())
                proj_use = proj[~pd.to_numeric(proj["game_pk"], errors="coerce").astype("Int64").isin(list(confirmed_games))]
                out = pd.concat([out, proj_use], ignore_index=True, sort=False)
        if "live_lineups_path" in by_key:
            legacy = by_key["live_lineups_path"]
            if out.empty:
                out = legacy.copy()
            elif "game_pk" in out.columns and "game_pk" in legacy.columns:
                covered_games = set(pd.to_numeric(out["game_pk"], errors="coerce").dropna().astype(int).tolist())
                legacy_use = legacy[~pd.to_numeric(legacy["game_pk"], errors="coerce").astype("Int64").isin(list(covered_games))]
                out = pd.concat([out, legacy_use], ignore_index=True, sort=False)
        logging.info("live lineup source=resolved rows=%s", len(out))
        return out
    logging.info("live lineup source=none")
    return pd.DataFrame()


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

    def pick(tokens: list[str]) -> str | None:
        for c in cols:
            lc = c.lower()
            if any(t in lc for t in tokens):
                return c
        return None

    chosen = {
        "onbase": pick(["obp", "on_base", "bb_rate", "walk", "woba", "reach"]),
        "contact": pick(["contact", "hit_rate", "whiff_inv", "k_inv", "avg", "batting_avg", "contacts"]),
        "power": pick(["iso", "slug", "slg", "xbh", "hard_hit", "barrel", "hr"]),
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

    metric_map = _pick_proxy_columns(br) if not br.empty else {
        "onbase": None,
        "contact": None,
        "power": None,
    }

    roll = pd.DataFrame()
    if not br.empty:
        br_team_col = _pick(list(br.columns), ["batter_team", "team", "team_abbrev", "batting_team"])
        if br_team_col:
            br["batter_team"] = br[br_team_col].map(normalize_team_abbr)

        if bid_col and bid_col in br.columns:
            br["join_batter_id"] = pd.to_numeric(br[bid_col], errors="coerce").astype("Int64")
        else:
            br_bid_col = _pick(list(br.columns), ["batter_id", "batter", "player_id"])
            if br_bid_col:
                br["join_batter_id"] = pd.to_numeric(br[br_bid_col], errors="coerce").astype("Int64")
            else:
                br["join_batter_id"] = pd.Series(pd.NA, index=br.index, dtype="Int64")

        br_name_col = _pick(list(br.columns), ["player_name", "batter_name", "name"])
        if br_name_col:
            br["join_name"] = (
                br[br_name_col]
                .astype(str)
                .str.lower()
                .str.replace(r"[^a-z0-9 ]+", "", regex=True)
                .str.strip()
            )
        else:
            br["join_name"] = ""

        group_cols = ["join_batter_id"] if br["join_batter_id"].notna().any() else ["batter_team", "join_name"]
        keep = group_cols + [c for c in metric_map.values() if c is not None] + ["game_date"]
        roll = (
            br[keep]
            .sort_values("game_date")
            .groupby(group_cols, as_index=False)
            .tail(1)
        )

    if bid_col and bid_col in lu.columns:
        lu["join_batter_id"] = pd.to_numeric(lu[bid_col], errors="coerce").astype("Int64")
    else:
        lu["join_batter_id"] = pd.Series(pd.NA, index=lu.index, dtype="Int64")

    if name_col and name_col in lu.columns:
        lu["join_name"] = (
            lu[name_col]
            .astype(str)
            .str.lower()
            .str.replace(r"[^a-z0-9 ]+", "", regex=True)
            .str.strip()
        )
    else:
        lu["join_name"] = ""

    merged = lu.copy()
    if not roll.empty:
        if merged["join_batter_id"].notna().any() and "join_batter_id" in roll.columns:
            merged = merged.merge(
                roll.drop(columns=["game_date"], errors="ignore"),
                on=["join_batter_id"],
                how="left",
                suffixes=("", "_roll"),
            )
        elif {"batter_team", "join_name"}.issubset(roll.columns):
            merged = merged.merge(
                roll.drop(columns=["game_date"], errors="ignore"),
                on=["batter_team", "join_name"],
                how="left",
                suffixes=("", "_roll"),
            )

    def _side_agg(side_df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for game_pk, g in side_df.groupby("game_pk", dropna=False):
            g = g.sort_values("lineup_slot", kind="mergesort")
            rec = {
                "game_pk": game_pk,
                "lineup_proj_hitters_count": int(g["lineup_slot"].notna().sum()),
            }

            for tag, n in [("top3", 3), ("top5", 5)]:
                sub = g[g["lineup_slot"] <= n] if g["lineup_slot"].notna().any() else g.head(n)
                rec[f"lineup_{tag}_count"] = int(len(sub))
                for mtag, col in metric_map.items():
                    if col and col in sub.columns:
                        rec[f"lineup_{tag}_{mtag}_mean"] = pd.to_numeric(sub[col], errors="coerce").mean()
                    else:
                        rec[f"lineup_{tag}_{mtag}_mean"] = np.nan

            for mtag, col in metric_map.items():
                if col and col in g.columns:
                    rec[f"lineup_full_{mtag}_mean"] = pd.to_numeric(g[col], errors="coerce").mean()
                else:
                    rec[f"lineup_full_{mtag}_mean"] = np.nan

            full_means = [
                rec.get("lineup_full_onbase_mean"),
                rec.get("lineup_full_contact_mean"),
                rec.get("lineup_full_power_mean"),
            ]
            rec["lineup_quality_score"] = np.nanmean(full_means)
            rec["lineup_completeness_score"] = min(rec["lineup_proj_hitters_count"] / 9.0, 1.0)
            records.append(rec)

        return pd.DataFrame(records)

    team_game_cols = [c for c in ["game_pk", "away_team", "home_team"] if c in merged.columns]
    games = merged[team_game_cols].drop_duplicates(subset=["game_pk"], keep="last") if team_game_cols else pd.DataFrame(columns=["game_pk"])

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

    out = (
        games[["game_pk"]]
        .merge(away, on="game_pk", how="left")
        .merge(home, on="game_pk", how="left")
    )

    return out.drop_duplicates(subset=["game_pk"], keep="last")


def _to_series(value, index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.reindex(index)
    if value is None:
        return pd.Series(pd.NA, index=index)
    try:
        if np.isscalar(value):
            return pd.Series([value] * len(index), index=index)
    except Exception:
        pass
    return pd.Series(value, index=index)


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
        out["lineup_source"] = "live_lineups"
    else:
        out["lineup_source"] = pd.NA

    weather_avail = out.get("weather_data_available", False)
    out["has_weather"] = _to_series(weather_avail, out.index).fillna(False).astype(bool)

    away_comp = _to_series(pd.to_numeric(out.get("away_lineup_completeness_score"), errors="coerce"), out.index)
    home_comp = _to_series(pd.to_numeric(out.get("home_lineup_completeness_score"), errors="coerce"), out.index)

    out["has_projected_lineups"] = away_comp.notna() | home_comp.notna()
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

    away_sp = _to_series(pd.to_numeric(ctx.get("away_sp_id"), errors="coerce"), ctx.index)
    home_sp = _to_series(pd.to_numeric(ctx.get("home_sp_id"), errors="coerce"), ctx.index)
    starters = away_sp.notna() & home_sp.notna()

    away_found = int(_to_series(pd.to_numeric(ctx.get("away_lineup_completeness_score"), errors="coerce"), ctx.index).notna().sum())
    home_found = int(_to_series(pd.to_numeric(ctx.get("home_lineup_completeness_score"), errors="coerce"), ctx.index).notna().sum())

    has_weather = _to_series(ctx.get("has_weather", False), ctx.index).fillna(False).astype(bool)
    has_lineups = _to_series(ctx.get("has_projected_lineups", False), ctx.index).fillna(False).astype(bool)

    return {
        "games": n,
        "pct_with_starters": float(starters.mean() * 100.0),
        "pct_with_weather": float(has_weather.mean() * 100.0),
        "pct_with_lineups": float(has_lineups.mean() * 100.0),
        "pct_with_projected_lineups": float(has_lineups.mean() * 100.0),
        "away_lineup_found": away_found,
        "home_lineup_found": home_found,
    }
