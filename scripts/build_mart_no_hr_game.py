from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header

SUPPRESSION_KEYWORDS = [
    "hr",
    "barrel",
    "hard_hit",
    "launch_angle",
    "launch_speed",
    "fb",
    "gb",
    "air",
    "xbh",
    "iso",
    "slug",
    "contact",
    "whiff",
    "chase",
    "bb",
    "k",
]

DANGER_TOKENS = [
    "barrel",
    "hard_hit",
    "hardhit",
    "launch_speed",
    "launch_angle",
    "fly_ball",
    "flyball",
    "fb_rate",
    "air_ball",
    "airball",
    "air_rate",
    "pulled",
    "pulled_air",
    "pull_air",
    "xbh",
    "iso",
    "slug",
    "slg",
]

PITCHER_SUPPRESSION_TOKENS = [
    "barrel",
    "hard_hit",
    "hardhit",
    "launch_speed",
    "launch_angle",
    "fly_ball",
    "flyball",
    "fb_rate",
    "air_ball",
    "airball",
    "air_rate",
    "fly",
    "contact_rate",
    "whiff_rate",
    "chase_rate",
]

STRICT_EXCLUDE_SUBSTRINGS = [
    "game_pk",
    "park_id",
    "pitches",
    "swings",
    "whiffs",
    "contacts",
    "chases",
    "in_zone_pitches",
    "_k_",
    "_bb_",
    "_h_",
    "_hbp_",
    "_hr_",
]

STRICT_EXCLUDE_SUFFIXES = ("_k", "_bb", "_h", "_hbp", "_hr")

TEAM_ABBR_MAP = {
    "arizona diamondbacks": "AZ",
    "atlanta braves": "ATL",
    "baltimore orioles": "BAL",
    "boston red sox": "BOS",
    "chicago cubs": "CHC",
    "chicago white sox": "CWS",
    "cincinnati reds": "CIN",
    "cleveland guardians": "CLE",
    "cleveland indians": "CLE",
    "colorado rockies": "COL",
    "detroit tigers": "DET",
    "houston astros": "HOU",
    "kansas city royals": "KC",
    "los angeles angels": "LAA",
    "anaheim angels": "LAA",
    "los angeles dodgers": "LAD",
    "miami marlins": "MIA",
    "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",
    "new york mets": "NYM",
    "new york yankees": "NYY",
    "oakland athletics": "OAK",
    "athletics": "OAK",
    "philadelphia phillies": "PHI",
    "pittsburgh pirates": "PIT",
    "san diego padres": "SD",
    "san francisco giants": "SF",
    "seattle mariners": "SEA",
    "st. louis cardinals": "STL",
    "st louis cardinals": "STL",
    "tampa bay rays": "TB",
    "texas rangers": "TEX",
    "toronto blue jays": "TOR",
    "washington nationals": "WSH",
}


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _pick_contains(cols: list[str], terms: list[str]) -> str | None:
    terms_l = [t.lower() for t in terms]
    for c in cols:
        lc = c.lower()
        if any(t in lc for t in terms_l):
            return c
    return None


def _normalize_team_value(v: object) -> object:
    if pd.isna(v):
        return v
    s = str(v).strip()
    if len(s) <= 4 and s.upper() == s:
        if s == "SDP":
            return "SD"
        if s == "SFG":
            return "SF"
        if s == "TBR":
            return "TB"
        if s == "WSN":
            return "WSH"
        if s == "ARI":
            return "AZ"
        return s
    return TEAM_ABBR_MAP.get(s.lower(), s)


def _normalize_team_series(series: pd.Series) -> pd.Series:
    return series.map(_normalize_team_value)


def _filter_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.copy()
    if "game_date" in out.columns:
        gd = pd.to_datetime(out["game_date"], errors="coerce")
        out = out[gd.dt.year == season].copy()
    elif "season" in out.columns:
        out = out[pd.to_numeric(out["season"], errors="coerce") == season].copy()
    return out


def _load_spine_for_season(dirs: dict[str, Path], season: int) -> pd.DataFrame:
    hist_path = dirs["processed_dir"] / "model_spine_game.parquet"
    hist = _filter_season(read_parquet(hist_path), season)
    logging.info("historical spine path=%s season=%s rows=%s", hist_path.resolve(), season, len(hist))
    if len(hist):
        logging.info("using historical spine for season=%s", season)
        return hist

    live_dir = dirs["processed_dir"] / "live"
    candidates = sorted(live_dir.glob(f"model_spine_game_{season}_*.parquet"))
    if not candidates:
        logging.warning("no live spine candidates found for season=%s in %s", season, live_dir.resolve())
        return hist

    def _rank_live_path(p: Path) -> tuple[pd.Timestamp, float]:
        m = re.search(rf"model_spine_game_{season}_(\d{{4}}-\d{{2}}-\d{{2}})\.parquet$", p.name)
        ts = pd.to_datetime(m.group(1), errors="coerce") if m else pd.NaT
        return (ts if pd.notna(ts) else pd.Timestamp.min, p.stat().st_mtime)

    live_path = max(candidates, key=_rank_live_path)
    live_spine = read_parquet(live_path)
    live_spine = _filter_season(live_spine, season)
    logging.info("using live spine fallback season=%s path=%s rows=%s", season, live_path.resolve(), len(live_spine))
    return live_spine


def _usable_batter_rows(df: pd.DataFrame) -> pd.DataFrame:
    batter_col = _pick(df, ["batter_id", "batter", "player_id"])
    if batter_col is None or "game_pk" not in df.columns:
        return df.iloc[0:0].copy()
    out = df.copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out[batter_col] = pd.to_numeric(out[batter_col], errors="coerce").astype("Int64")
    out = out[out["game_pk"].notna() & out[batter_col].notna()]
    return out


def _usable_pitcher_rows(df: pd.DataFrame) -> pd.DataFrame:
    pid_col = _pick(df, ["pitcher_id", "pitcher", "mlbam_pitcher_id", "player_id"])
    if pid_col is None or "game_pk" not in df.columns:
        return df.iloc[0:0].copy()
    out = df.copy()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out[pid_col] = pd.to_numeric(out[pid_col], errors="coerce").astype("Int64")
    out = out[out["game_pk"].notna() & out[pid_col].notna()]
    return out


def _enforce_prior_to_game_rolling(df: pd.DataFrame, entity_col: str, source_name: str) -> pd.DataFrame:
    if df.empty:
        logging.info("%s prior-to-game shift skipped: empty dataframe", source_name)
        return df.copy()
    if entity_col not in df.columns:
        logging.warning("%s prior-to-game shift skipped: missing entity column=%s", source_name, entity_col)
        return df.copy()
    if "game_pk" not in df.columns:
        logging.warning("%s prior-to-game shift skipped: missing game_pk column", source_name)
        return df.copy()

    out = df.copy()
    out[entity_col] = pd.to_numeric(out[entity_col], errors="coerce").astype("Int64")
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce") if "game_date" in out.columns else pd.NaT
    out = out[out[entity_col].notna() & out["game_pk"].notna()].copy()

    passthrough_cols = {
        entity_col,
        "game_pk",
        "game_date",
        "season",
        "batter",
        "batter_id",
        "pitcher",
        "pitcher_id",
        "player_id",
        "team",
        "batter_team",
        "batting_team",
        "bat_team",
        "offense_team",
    }
    rolling_stat_cols = []
    for c in out.columns:
        if c in passthrough_cols:
            continue
        lc = c.lower()
        is_numeric = pd.api.types.is_numeric_dtype(out[c])
        is_roll_or_stat = "roll" in lc or any(k in lc for k in SUPPRESSION_KEYWORDS)
        if is_numeric or is_roll_or_stat:
            rolling_stat_cols.append(c)

    if not rolling_stat_cols:
        logging.warning("%s prior-to-game shift found no rolling/stat columns to shift", source_name)
        return out

    out["__source_game_pk"] = out["game_pk"]
    out = out.sort_values([entity_col, "game_date", "game_pk"], ascending=[True, True, True])
    shift_cols = rolling_stat_cols + ["__source_game_pk"]
    out[shift_cols] = out.groupby(entity_col, dropna=False)[shift_cols].shift(1)

    leak_mask = out["__source_game_pk"].notna() & (pd.to_numeric(out["__source_game_pk"], errors="coerce").astype("Int64") == out["game_pk"])
    if bool(leak_mask.any()):
        raise ValueError(f"Leakage detected: rolling features include same-game data ({source_name})")

    null_pct = out[rolling_stat_cols].isna().mean().sort_values(ascending=False)
    logging.info("%s prior-to-game shift rolling columns=%s", source_name, len(rolling_stat_cols))
    logging.info("%s shifted null pct (top 20)=%s", source_name, {k: round(float(v), 4) for k, v in null_pct.head(20).items()})
    sample_cols = [entity_col, "game_date", "game_pk", "__source_game_pk"] + rolling_stat_cols[:6]
    logging.info("%s shift sample rows=%s", source_name, out[sample_cols].head(5).to_dict("records"))
    return out


def _sort_for_latest(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "game_date" in work.columns:
        work["_sort_date"] = pd.to_datetime(work["game_date"], errors="coerce")
    else:
        work["_sort_date"] = pd.NaT
    if "game_pk" in work.columns:
        work["_sort_game_pk"] = pd.to_numeric(work["game_pk"], errors="coerce")
    else:
        work["_sort_game_pk"] = np.nan
    return work


def _select_batter_feature_cols(batter_roll: pd.DataFrame) -> list[str]:
    exclude = {
        "game_pk",
        "season",
        "batter_id",
        "batter",
        "pitcher_id",
        "player_id",
        "total_hr",
        "no_hr_game",
    }
    num_cols = []
    for c in batter_roll.columns:
        if c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(batter_roll[c]):
            continue
        lc = c.lower()
        is_roll_like = "roll" in lc
        is_allowed_family = any(k in lc for k in DANGER_TOKENS)
        has_strict_excluded_substring = any(tok in lc for tok in STRICT_EXCLUDE_SUBSTRINGS)
        has_strict_excluded_suffix = lc.endswith(STRICT_EXCLUDE_SUFFIXES)
        if is_roll_like and is_allowed_family and (not has_strict_excluded_substring) and (not has_strict_excluded_suffix) and "team" not in lc:
            num_cols.append(c)
    logging.info("selected HR-danger batter rolling columns=%s sample=%s", len(num_cols), num_cols[:15])
    return num_cols


def _aggregate_with_top3_mean(work: pd.DataFrame, group_cols: list[str], feat_cols: list[str], prefix: str) -> pd.DataFrame:
    grouped = work.groupby(group_cols, dropna=False)
    mean_df = grouped[feat_cols].mean().rename(columns=lambda c: f"{prefix}{c}_mean")
    max_df = grouped[feat_cols].max().rename(columns=lambda c: f"{prefix}{c}_max")

    top3_parts: list[pd.DataFrame] = []
    for c in feat_cols:
        top3_series = grouped[c].apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().nlargest(3).mean())
        top3_parts.append(top3_series.rename(f"{prefix}{c}_top3_mean").to_frame())
    top3_df = pd.concat(top3_parts, axis=1) if top3_parts else pd.DataFrame(index=mean_df.index)

    out = pd.concat([mean_df, max_df, top3_df], axis=1).reset_index()
    return out


def _team_rollup_by_game(batter_roll: pd.DataFrame) -> pd.DataFrame:
    team_col = _pick(batter_roll, ["batter_team", "batting_team", "bat_team", "team", "offense_team"])
    if team_col is None or "game_pk" not in batter_roll.columns:
        return pd.DataFrame(columns=["game_pk", "team"])
    feat_cols = _select_batter_feature_cols(batter_roll)
    if not feat_cols:
        return pd.DataFrame(columns=["game_pk", "team"])
    work = batter_roll[["game_pk", team_col] + feat_cols].copy()
    work = work.rename(columns={team_col: "team"})
    agg = _aggregate_with_top3_mean(work, ["game_pk", "team"], feat_cols, "team_")
    logging.info("engineered team HR-danger features (by game)=%s", len([c for c in agg.columns if c.startswith("team_")]))
    return agg


def _team_rollup_latest_by_team(batter_roll: pd.DataFrame) -> pd.DataFrame:
    team_col = _pick(batter_roll, ["batter_team", "batting_team", "bat_team", "team", "offense_team"])
    batter_col = _pick(batter_roll, ["batter_id", "batter", "player_id"])
    if team_col is None or batter_col is None:
        return pd.DataFrame(columns=["team"])

    feat_cols = _select_batter_feature_cols(batter_roll)
    if not feat_cols:
        return pd.DataFrame(columns=["team"])

    work = _sort_for_latest(batter_roll)
    work[batter_col] = pd.to_numeric(work[batter_col], errors="coerce").astype("Int64")
    work = work[work[batter_col].notna() & work[team_col].notna()].copy()
    work = work.sort_values(["_sort_date", "_sort_game_pk"], ascending=[True, True])
    latest = work.groupby([team_col, batter_col], dropna=False).tail(1)

    latest = latest.rename(columns={team_col: "team"})
    team_agg = _aggregate_with_top3_mean(latest, ["team"], feat_cols, "team_")
    logging.info("engineered team HR-danger features (latest by team)=%s", len([c for c in team_agg.columns if c.startswith("team_")]))
    return team_agg


def _team_context_from_spine(spine: pd.DataFrame) -> pd.DataFrame:
    if "game_pk" not in spine.columns:
        return pd.DataFrame(columns=["game_pk", "team"])
    parts: list[pd.DataFrame] = []
    if "home_team" in spine.columns:
        h = spine[["game_pk", "home_team"]].rename(columns={"home_team": "team"})
        parts.append(h)
    if "away_team" in spine.columns:
        a = spine[["game_pk", "away_team"]].rename(columns={"away_team": "team"})
        parts.append(a)
    if not parts:
        return pd.DataFrame(columns=["game_pk", "team"])
    out = pd.concat(parts, ignore_index=True).dropna(subset=["game_pk", "team"]).drop_duplicates()
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["team"] = _normalize_team_series(out["team"]).astype("string")
    return out


def _build_team_rollups_with_context(batter_roll: pd.DataFrame, team_context: pd.DataFrame, latest_by_team: bool) -> pd.DataFrame:
    feat_cols = _select_batter_feature_cols(batter_roll)
    batter_col = _pick(batter_roll, ["batter_id", "batter", "player_id"])
    if not feat_cols or batter_col is None or "game_pk" not in batter_roll.columns or team_context.empty:
        return pd.DataFrame(columns=["team"] if latest_by_team else ["game_pk", "team"])

    work = batter_roll[["game_pk", batter_col] + feat_cols + (["game_date"] if "game_date" in batter_roll.columns else [])].copy()
    if "__source_game_pk" in batter_roll.columns:
        work["__source_game_pk"] = pd.to_numeric(batter_roll["__source_game_pk"], errors="coerce").astype("Int64")
    work["game_pk"] = pd.to_numeric(work["game_pk"], errors="coerce").astype("Int64")
    work[batter_col] = pd.to_numeric(work[batter_col], errors="coerce").astype("Int64")
    work = work.merge(team_context[["game_pk", "team"]], on="game_pk", how="inner")
    work = work[work[batter_col].notna() & work["team"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["team"] if latest_by_team else ["game_pk", "team"])
    if "__source_game_pk" in work.columns:
        leak_mask = work["__source_game_pk"].notna() & (work["__source_game_pk"] == work["game_pk"])
        if bool(leak_mask.any()):
            raise ValueError("Leakage detected: rolling features include same-game data")

    if latest_by_team:
        work = _sort_for_latest(work)
        work = work.sort_values(["_sort_date", "_sort_game_pk"], ascending=[True, True])
        latest = work.groupby(["team", batter_col], dropna=False).tail(1)
        agg = _aggregate_with_top3_mean(latest, ["team"], feat_cols, "team_")
        logging.info("engineered team HR-danger features (context latest)=%s", len([c for c in agg.columns if c.startswith("team_")]))
        return agg

    agg = _aggregate_with_top3_mean(work, ["game_pk", "team"], feat_cols, "team_")
    logging.info("engineered team HR-danger features (context by game)=%s", len([c for c in agg.columns if c.startswith("team_")]))
    return agg


def _ensure_starter_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "home_sp_id" not in out.columns:
        for c in ["home_probable_pitcher_id", "home_sp", "home_starter_id", "home_pitcher_id", "home_starting_pitcher_id"]:
            if c in out.columns:
                out["home_sp_id"] = out[c]
                break
    if "away_sp_id" not in out.columns:
        for c in ["away_probable_pitcher_id", "away_sp", "away_starter_id", "away_pitcher_id", "away_starting_pitcher_id"]:
            if c in out.columns:
                out["away_sp_id"] = out[c]
                break
    if "home_sp_id" in out.columns:
        out["home_sp_id"] = pd.to_numeric(out["home_sp_id"], errors="coerce").astype("Int64")
    if "away_sp_id" in out.columns:
        out["away_sp_id"] = pd.to_numeric(out["away_sp_id"], errors="coerce").astype("Int64")
    return out


def _starter_features(spine: pd.DataFrame, pitcher_roll: pd.DataFrame, use_latest_per_pitcher: bool = False) -> pd.DataFrame:
    mart = spine.copy()
    if pitcher_roll.empty:
        logging.warning("starter merge skipped: pitcher rolling dataframe is empty")
        return mart

    pitcher_df = pitcher_roll.copy()
    if "pitcher_id" in pitcher_df.columns:
        pitcher_id_col = "pitcher_id"
    elif "pitcher" in pitcher_df.columns:
        pitcher_id_col = "pitcher"
    else:
        logging.warning("starter merge skipped: no pitcher id column found in pitcher rolling dataframe")
        return mart

    logging.info("starter merge pitcher id column=%s", pitcher_id_col)
    logging.info("starter merge rolling pitcher rows=%s", len(pitcher_df))

    mart = _ensure_starter_id_cols(mart)
    if "home_sp_id" not in mart.columns or "away_sp_id" not in mart.columns:
        cand = [c for c in ["home_sp", "away_sp", "home_probable_pitcher_id", "away_probable_pitcher_id", "home_starter_id", "away_starter_id"] if c in mart.columns]
        logging.warning("starter merge skipped: could not find home_sp_id/away_sp_id columns candidates_present=%s", cand)
        return mart

    logging.info("pre-starter keys home_sp_id_present=%s away_sp_id_present=%s", "home_sp_id" in mart.columns, "away_sp_id" in mart.columns)
    logging.info("pre-starter nonnull counts home_sp_id=%s away_sp_id=%s", int(mart["home_sp_id"].notna().sum()), int(mart["away_sp_id"].notna().sum()))
    pitcher_df[pitcher_id_col] = pd.to_numeric(pitcher_df[pitcher_id_col], errors="coerce").astype("Int64")
    pitcher_df = pitcher_df[pitcher_df[pitcher_id_col].notna()].copy()

    if "game_pk" not in pitcher_df.columns or "game_pk" not in mart.columns:
        logging.warning("starter merge skipped: game_pk required for prior-to-game alignment")
        return mart

    pitcher_df["game_pk"] = pd.to_numeric(pitcher_df["game_pk"], errors="coerce").astype("Int64")
    mart["game_pk"] = pd.to_numeric(mart["game_pk"], errors="coerce").astype("Int64")
    pitcher_df = pitcher_df[pitcher_df["game_pk"].notna()].copy()

    if "__source_game_pk" in pitcher_df.columns:
        pitcher_df["__source_game_pk"] = pd.to_numeric(pitcher_df["__source_game_pk"], errors="coerce").astype("Int64")

    numeric_cols = pitcher_df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) <= 2:
        for c in pitcher_df.columns:
            if c in numeric_cols:
                continue
            coerced = pd.to_numeric(pitcher_df[c], errors="coerce")
            if coerced.notna().any():
                pitcher_df[c] = coerced
                numeric_cols.append(c)
    exclude = {pitcher_id_col, "pitcher", "pitcher_id", "game_pk", "season", "total_hr", "no_hr_game", "__source_game_pk"}
    payload_cols = [c for c in numeric_cols if c not in exclude]

    logging.info("starter merge rolling pitchers=%s", int(pitcher_df[pitcher_id_col].nunique()))
    starter_ids = pd.concat([mart["home_sp_id"], mart["away_sp_id"]], ignore_index=True).dropna().nunique()
    logging.info("starter merge unique starter ids in spine=%s", int(starter_ids))

    if not payload_cols:
        logging.warning("starter merge aborted: no payload columns found")
        return mart

    logging.info("starter payload cols=%s sample=%s", len(payload_cols), payload_cols[:10])

    merge_cols = ["game_pk", pitcher_id_col] + payload_cols + (["__source_game_pk"] if "__source_game_pk" in pitcher_df.columns else [])
    per_game_pitcher = pitcher_df[merge_cols].drop_duplicates(subset=["game_pk", pitcher_id_col], keep="last").copy()

    home_snapshot = per_game_pitcher.copy()
    home_rename = {c: f"home_sp_{c}" for c in payload_cols}
    if "__source_game_pk" in home_snapshot.columns:
        home_rename["__source_game_pk"] = "home_sp_source_game_pk"
    home_snapshot = home_snapshot.rename(columns=home_rename)
    mart = mart.merge(home_snapshot, how="left", left_on=["game_pk", "home_sp_id"], right_on=["game_pk", pitcher_id_col])
    if pitcher_id_col in mart.columns:
        mart = mart.drop(columns=[pitcher_id_col])

    home_cols = [c for c in mart.columns if c.startswith("home_sp_")]
    home_matches = int(mart[home_cols].notna().any(axis=1).sum()) if home_cols else 0
    logging.info("created home_sp columns=%s home starter matches=%s", len(home_cols), home_matches)

    away_snapshot = per_game_pitcher.copy()
    away_rename = {c: f"away_sp_{c}" for c in payload_cols}
    if "__source_game_pk" in away_snapshot.columns:
        away_rename["__source_game_pk"] = "away_sp_source_game_pk"
    away_snapshot = away_snapshot.rename(columns=away_rename)
    mart = mart.merge(away_snapshot, how="left", left_on=["game_pk", "away_sp_id"], right_on=["game_pk", pitcher_id_col])
    if pitcher_id_col in mart.columns:
        mart = mart.drop(columns=[pitcher_id_col])

    away_cols = [c for c in mart.columns if c.startswith("away_sp_")]
    away_matches = int(mart[away_cols].notna().any(axis=1).sum()) if away_cols else 0
    logging.info("created away_sp columns=%s away starter matches=%s", len(away_cols), away_matches)

    starter_cols = [c for c in mart.columns if c.startswith("home_sp_") or c.startswith("away_sp_")]
    games_with_starter_features = int(mart[starter_cols].notna().any(axis=1).sum()) if starter_cols else 0
    logging.info("starter_feature_cols=%s games_with_starter_features=%s", len(starter_cols), games_with_starter_features)
    if "home_sp_source_game_pk" in mart.columns:
        leak_home = mart["home_sp_source_game_pk"].notna() & (mart["home_sp_source_game_pk"] == mart["game_pk"])
        if bool(leak_home.any()):
            raise ValueError("Leakage detected: rolling features include same-game data")
    if "away_sp_source_game_pk" in mart.columns:
        leak_away = mart["away_sp_source_game_pk"].notna() & (mart["away_sp_source_game_pk"] == mart["game_pk"])
        if bool(leak_away.any()):
            raise ValueError("Leakage detected: rolling features include same-game data")
    mart = mart.drop(columns=["home_sp_source_game_pk", "away_sp_source_game_pk"], errors="ignore")
    if not starter_cols:
        logging.warning("starter merge produced zero prefixed columns; check merge keys and payload column selection")
    return mart


def _attach_engineered_features(mart: pd.DataFrame) -> pd.DataFrame:
    out = mart.copy()
    engineered_starter = 0
    engineered_matchup_env = 0

    def _as_numeric_series(df: pd.DataFrame, value: object, default: float) -> pd.Series:
        if isinstance(value, str) and value in df.columns:
            return pd.to_numeric(df[value], errors="coerce")
        if isinstance(value, pd.Series):
            return pd.to_numeric(value.reindex(df.index), errors="coerce")
        return pd.Series(default, index=df.index, dtype="float64")

    temp_col = _pick_contains(list(out.columns), ["temperature", "temp"])
    wind_speed_col = _pick_contains(list(out.columns), ["wind_speed", "wind_mph", "wind"])
    if temp_col or wind_speed_col:
        temp_v = _as_numeric_series(out, temp_col, 70.0)
        wind_v = _as_numeric_series(out, wind_speed_col, 0.0)
        out["env_temp_wind_interaction"] = temp_v * wind_v

    park_hr_col = _pick_contains(list(out.columns), ["park_hr", "hr_factor", "home_run_factor", "hr_park"])
    wind_out_col = _pick_contains(list(out.columns), ["wind_out", "windout"])
    wind_in_col = _pick_contains(list(out.columns), ["wind_in", "windin"])
    if temp_col or wind_speed_col or park_hr_col or wind_out_col or wind_in_col:
        temp_v = _as_numeric_series(out, temp_col, 70.0)
        wind_v = _as_numeric_series(out, wind_speed_col, 0.0)
        park_v = _as_numeric_series(out, park_hr_col, 1.0)
        wind_out_v = _as_numeric_series(out, wind_out_col, 0.0)
        wind_in_v = _as_numeric_series(out, wind_in_col, 0.0)
        out["combined_park_weather_hr_index"] = (temp_v.fillna(70.0) / 70.0) * (1.0 + wind_v.fillna(0.0) / 20.0) * pd.to_numeric(park_v, errors="coerce").fillna(1.0)
        out["env_hr_suppression_proxy"] = (1.0 / out["combined_park_weather_hr_index"].clip(lower=0.25, upper=5.0)) + (wind_in_v.fillna(0.0) * 0.02) - (wind_out_v.fillna(0.0) * 0.02)

    home_sp_col = _pick_contains([c for c in out.columns if c.startswith("home_sp_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])
    away_sp_col = _pick_contains([c for c in out.columns if c.startswith("away_sp_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])
    home_team_power_col = _pick_contains([c for c in out.columns if c.startswith("home_team_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])
    away_team_power_col = _pick_contains([c for c in out.columns if c.startswith("away_team_")], ["hr", "barrel", "hard_hit", "slug", "iso", "xbh"])

    if away_sp_col and home_team_power_col:
        out["starter_hr_suppression_gap_away_vs_home"] = pd.to_numeric(out[away_sp_col], errors="coerce") - pd.to_numeric(out[home_team_power_col], errors="coerce")
        engineered_matchup_env += 1
    if home_sp_col and away_team_power_col:
        out["starter_hr_suppression_gap_home_vs_away"] = pd.to_numeric(out[home_sp_col], errors="coerce") - pd.to_numeric(out[away_team_power_col], errors="coerce")
        engineered_matchup_env += 1

    def _collect_prefixed_numeric_cols(starts_with: str, allow_tokens: list[str]) -> list[str]:
        matched: list[str] = []
        for c in out.columns:
            if not c.startswith(starts_with):
                continue
            lc = c.lower()
            if not any(tok in lc for tok in allow_tokens):
                continue
            if any(tok in lc for tok in STRICT_EXCLUDE_SUBSTRINGS):
                continue
            if lc.endswith(STRICT_EXCLUDE_SUFFIXES):
                continue
            if not pd.api.types.is_numeric_dtype(out[c]):
                continue
            matched.append(c)
        return matched

    home_bad_cols = _collect_prefixed_numeric_cols("home_sp_", ["barrel", "hard_hit", "hardhit", "launch_speed", "launch_angle", "fly_ball", "flyball", "fb_rate", "air_ball", "airball", "air_rate"])
    away_bad_cols = _collect_prefixed_numeric_cols("away_sp_", ["barrel", "hard_hit", "hardhit", "launch_speed", "launch_angle", "fly_ball", "flyball", "fb_rate", "air_ball", "airball", "air_rate"])
    home_good_cols = _collect_prefixed_numeric_cols("home_sp_", ["whiff_rate", "chase_rate"])
    away_good_cols = _collect_prefixed_numeric_cols("away_sp_", ["whiff_rate", "chase_rate"])
    home_contact_cols = _collect_prefixed_numeric_cols("home_sp_", ["contact_rate"])
    away_contact_cols = _collect_prefixed_numeric_cols("away_sp_", ["contact_rate"])

    home_bad_stack = [pd.to_numeric(out[c], errors="coerce") for c in home_bad_cols]
    away_bad_stack = [pd.to_numeric(out[c], errors="coerce") for c in away_bad_cols]
    home_good_stack = [pd.to_numeric(out[c], errors="coerce") for c in home_good_cols] + [(-pd.to_numeric(out[c], errors="coerce")) for c in home_contact_cols]
    away_good_stack = [pd.to_numeric(out[c], errors="coerce") for c in away_good_cols] + [(-pd.to_numeric(out[c], errors="coerce")) for c in away_contact_cols]

    pitcher_supp_cols_used = len(home_bad_cols) + len(away_bad_cols) + len(home_good_cols) + len(away_good_cols) + len(home_contact_cols) + len(away_contact_cols)
    logging.info("pitcher suppression columns selected=%s", pitcher_supp_cols_used)

    if home_bad_stack or home_good_stack:
        home_bad = pd.concat(home_bad_stack, axis=1).mean(axis=1) if home_bad_stack else pd.Series(0.0, index=out.index)
        home_good = pd.concat(home_good_stack, axis=1).mean(axis=1) if home_good_stack else pd.Series(0.0, index=out.index)
        out["home_sp_hr_suppression_score"] = home_good - home_bad
        engineered_starter += 1
    if away_bad_stack or away_good_stack:
        away_bad = pd.concat(away_bad_stack, axis=1).mean(axis=1) if away_bad_stack else pd.Series(0.0, index=out.index)
        away_good = pd.concat(away_good_stack, axis=1).mean(axis=1) if away_good_stack else pd.Series(0.0, index=out.index)
        out["away_sp_hr_suppression_score"] = away_good - away_bad
        engineered_starter += 1

    home_team_danger_col = _pick_contains([c for c in out.columns if c.startswith("home_team_")], DANGER_TOKENS)
    away_team_danger_col = _pick_contains([c for c in out.columns if c.startswith("away_team_")], DANGER_TOKENS)
    if home_team_danger_col and "away_sp_hr_suppression_score" in out.columns:
        out["home_offense_power_vs_away_sp_suppression"] = pd.to_numeric(out[home_team_danger_col], errors="coerce") - pd.to_numeric(out["away_sp_hr_suppression_score"], errors="coerce")
        engineered_matchup_env += 1
    if away_team_danger_col and "home_sp_hr_suppression_score" in out.columns:
        out["away_offense_power_vs_home_sp_suppression"] = pd.to_numeric(out[away_team_danger_col], errors="coerce") - pd.to_numeric(out["home_sp_hr_suppression_score"], errors="coerce")
        engineered_matchup_env += 1

    home_barrel_col = _pick_contains([c for c in out.columns if c.startswith("home_team_")], ["barrel"])
    away_barrel_col = _pick_contains([c for c in out.columns if c.startswith("away_team_")], ["barrel"])
    away_sp_contact_supp = _pick_contains([c for c in out.columns if c.startswith("away_sp_")], ["contact_rate"])
    home_sp_contact_supp = _pick_contains([c for c in out.columns if c.startswith("home_sp_")], ["contact_rate"])
    if home_barrel_col and away_sp_contact_supp:
        out["home_barrel_pressure_vs_away_sp_contact_suppression"] = pd.to_numeric(out[home_barrel_col], errors="coerce") - pd.to_numeric(out[away_sp_contact_supp], errors="coerce")
        engineered_matchup_env += 1
    if away_barrel_col and home_sp_contact_supp:
        out["away_barrel_pressure_vs_home_sp_contact_suppression"] = pd.to_numeric(out[away_barrel_col], errors="coerce") - pd.to_numeric(out[home_sp_contact_supp], errors="coerce")
        engineered_matchup_env += 1

    if home_team_danger_col and "combined_park_weather_hr_index" in out.columns:
        out["home_power_env_interaction"] = pd.to_numeric(out[home_team_danger_col], errors="coerce") * pd.to_numeric(out["combined_park_weather_hr_index"], errors="coerce")
        engineered_matchup_env += 1
    if away_team_danger_col and "combined_park_weather_hr_index" in out.columns:
        out["away_power_env_interaction"] = pd.to_numeric(out[away_team_danger_col], errors="coerce") * pd.to_numeric(out["combined_park_weather_hr_index"], errors="coerce")
        engineered_matchup_env += 1
    if "home_sp_hr_suppression_score" in out.columns and "env_hr_suppression_proxy" in out.columns:
        out["home_sp_env_suppression_interaction"] = pd.to_numeric(out["home_sp_hr_suppression_score"], errors="coerce") * pd.to_numeric(out["env_hr_suppression_proxy"], errors="coerce")
        engineered_matchup_env += 1
    if "away_sp_hr_suppression_score" in out.columns and "env_hr_suppression_proxy" in out.columns:
        out["away_sp_env_suppression_interaction"] = pd.to_numeric(out["away_sp_hr_suppression_score"], errors="coerce") * pd.to_numeric(out["env_hr_suppression_proxy"], errors="coerce")
        engineered_matchup_env += 1

    logging.info("engineered starter suppression features created=%s", engineered_starter)
    logging.info("engineered matchup/environment interaction features created=%s", engineered_matchup_env)

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build No-HR game feature mart")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_mart_no_hr_game.log")
    log_header("scripts/build_mart_no_hr_game.py", repo_root, config_path, dirs)

    logging.info("requested season=%s", args.season)

    spine = _load_spine_for_season(dirs, args.season)
    spine = spine.copy()
    spine["game_pk"] = pd.to_numeric(spine["game_pk"], errors="coerce").astype("Int64")
    for team_col in ["home_team", "away_team"]:
        if team_col in spine.columns:
            before = spine[team_col].astype("string")
            spine[team_col] = _normalize_team_series(before).astype("string")
            changed = int((before.fillna("__NA__") != spine[team_col].fillna("__NA__")).sum())
            if changed:
                logging.info("team normalization column=%s changed_values=%s", team_col, changed)

    all_batter_roll = read_parquet(dirs["processed_dir"] / "batter_game_rolling.parquet")
    all_pitcher_roll = read_parquet(dirs["processed_dir"] / "pitcher_game_rolling.parquet")

    curr_batter = _usable_batter_rows(_filter_season(all_batter_roll, args.season))
    prev_batter = _usable_batter_rows(_filter_season(all_batter_roll, args.season - 1))
    curr_pitcher = _usable_pitcher_rows(_filter_season(all_pitcher_roll, args.season))
    prev_pitcher = _usable_pitcher_rows(_filter_season(all_pitcher_roll, args.season - 1))

    batter_entity_col = _pick(all_batter_roll, ["batter_id", "batter", "player_id"])
    pitcher_entity_col = _pick(all_pitcher_roll, ["pitcher_id", "pitcher", "mlbam_pitcher_id", "player_id"])
    if batter_entity_col is None or pitcher_entity_col is None:
        raise ValueError("Could not determine batter/pitcher entity key columns for prior-to-game shift enforcement")

    curr_batter = _enforce_prior_to_game_rolling(curr_batter, batter_entity_col, "batter_game_rolling_current")
    prev_batter = _enforce_prior_to_game_rolling(prev_batter, batter_entity_col, "batter_game_rolling_prior")
    curr_pitcher = _enforce_prior_to_game_rolling(curr_pitcher, pitcher_entity_col, "pitcher_game_rolling_current")
    prev_pitcher = _enforce_prior_to_game_rolling(prev_pitcher, pitcher_entity_col, "pitcher_game_rolling_prior")

    logging.info(
        "batter rolling usable rows season=%s:%s prior=%s:%s",
        args.season,
        len(curr_batter),
        args.season - 1,
        len(prev_batter),
    )
    logging.info(
        "pitcher rolling usable rows season=%s:%s prior=%s:%s",
        args.season,
        len(curr_pitcher),
        args.season - 1,
        len(prev_pitcher),
    )

    use_batter_fallback = args.season == 2026 and curr_batter.empty and not prev_batter.empty
    use_pitcher_fallback = args.season == 2026 and curr_pitcher.empty and not prev_pitcher.empty

    if args.season == 2026 and curr_batter.empty and prev_batter.empty:
        logging.warning("No usable batter rolling rows found for 2026 or 2025 fallback")
    if args.season == 2026 and curr_pitcher.empty and prev_pitcher.empty:
        logging.warning("No usable pitcher rolling rows found for 2026 or 2025 fallback")

    batter_source = "previous_season_fallback" if use_batter_fallback else "requested_season"
    pitcher_source = "previous_season_fallback" if use_pitcher_fallback else "requested_season"
    batter_snapshot_season = args.season - 1 if use_batter_fallback else args.season
    pitcher_snapshot_season = args.season - 1 if use_pitcher_fallback else args.season

    mart = spine[
        [
            c
            for c in [
                "game_pk",
                "game_date",
                "season",
                "home_team",
                "away_team",
                "park_id",
                "canonical_park_key",
                "home_sp_id",
                "away_sp_id",
                "home_sp",
                "away_sp",
                "home_probable_pitcher_id",
                "away_probable_pitcher_id",
                "home_starter_id",
                "away_starter_id",
                "home_pitcher_id",
                "away_pitcher_id",
                "home_starting_pitcher_id",
                "away_starting_pitcher_id",
            ]
            if c in spine.columns
        ]
    ].copy()
    mart = _ensure_starter_id_cols(mart)

    batter_for_rollup = prev_batter if use_batter_fallback else curr_batter
    if use_batter_fallback:
        prior_spine = _load_spine_for_season(dirs, args.season - 1)
        prior_spine = prior_spine.copy()
        if "home_team" in prior_spine.columns:
            prior_spine["home_team"] = _normalize_team_series(prior_spine["home_team"]).astype("string")
        if "away_team" in prior_spine.columns:
            prior_spine["away_team"] = _normalize_team_series(prior_spine["away_team"]).astype("string")
        team_context = _team_context_from_spine(prior_spine)
        team_roll = _build_team_rollups_with_context(batter_for_rollup, team_context, latest_by_team=True)
        logging.info("team fallback snapshot team_profiles=%s team_key_column=team", len(team_roll))
        if not team_roll.empty and {"home_team", "away_team"}.issubset(mart.columns):
            home_roll = team_roll.rename(columns={c: f"home_{c}" for c in team_roll.columns if c != "team"})
            away_roll = team_roll.rename(columns={c: f"away_{c}" for c in team_roll.columns if c != "team"})
            mart = mart.merge(home_roll, left_on="home_team", right_on="team", how="left").drop(columns=["team"], errors="ignore")
            mart = mart.merge(away_roll, left_on="away_team", right_on="team", how="left").drop(columns=["team"], errors="ignore")
            home_cols = [c for c in mart.columns if c.startswith("home_team_")]
            away_cols = [c for c in mart.columns if c.startswith("away_team_")]
            home_matches = int(mart[home_cols].notna().any(axis=1).sum()) if home_cols else 0
            away_matches = int(mart[away_cols].notna().any(axis=1).sum()) if away_cols else 0
            logging.info("team fallback merge matches home=%s away=%s", home_matches, away_matches)
    else:
        team_context = _team_context_from_spine(spine)
        team_roll = _build_team_rollups_with_context(batter_for_rollup, team_context, latest_by_team=False)
        if not team_roll.empty and {"home_team", "away_team"}.issubset(mart.columns):
            home_roll = team_roll.rename(columns={c: f"home_{c}" for c in team_roll.columns if c not in {"game_pk", "team"}})
            away_roll = team_roll.rename(columns={c: f"away_{c}" for c in team_roll.columns if c not in {"game_pk", "team"}})
            mart = mart.merge(home_roll, left_on=["game_pk", "home_team"], right_on=["game_pk", "team"], how="left")
            mart = mart.drop(columns=["team"], errors="ignore")
            mart = mart.merge(away_roll, left_on=["game_pk", "away_team"], right_on=["game_pk", "team"], how="left")
            mart = mart.drop(columns=["team"], errors="ignore")

    pitcher_for_merge = prev_pitcher if use_pitcher_fallback else curr_pitcher
    pre_cols = set(mart.columns)
    starter_key_candidates = [c for c in ["home_sp_id", "away_sp_id", "home_sp", "away_sp", "home_probable_pitcher_id", "away_probable_pitcher_id", "home_starter_id", "away_starter_id"] if c in mart.columns]
    logging.info(
        "pre-starter-call keys home_sp_id_present=%s away_sp_id_present=%s home_sp_id_nonnull=%s away_sp_id_nonnull=%s candidate_cols=%s",
        "home_sp_id" in pre_cols,
        "away_sp_id" in pre_cols,
        int(mart["home_sp_id"].notna().sum()) if "home_sp_id" in mart.columns else 0,
        int(mart["away_sp_id"].notna().sum()) if "away_sp_id" in mart.columns else 0,
        starter_key_candidates,
    )
    mart = _starter_features(mart, pitcher_for_merge, use_latest_per_pitcher=use_pitcher_fallback)
    starter_cols_after_merge = [c for c in mart.columns if c.startswith("home_sp_") or c.startswith("away_sp_")]
    logging.info("post-starter-merge starter column count=%s", len(starter_cols_after_merge))

    weather_path = dirs["processed_dir"] / "weather_game.parquet"
    if weather_path.exists():
        weather = _filter_season(read_parquet(weather_path), args.season)
        if "game_pk" in weather.columns:
            weather["game_pk"] = pd.to_numeric(weather["game_pk"], errors="coerce").astype("Int64")
            weather_cols = [
                c
                for c in weather.columns
                if c == "game_pk" or c.startswith(("temp", "wind", "weather", "humidity", "pressure"))
            ]
            if len(weather_cols) > 1:
                mart = mart.merge(weather[weather_cols].drop_duplicates(subset=["game_pk"]), on="game_pk", how="left")

    parks_path = dirs["processed_dir"] / "parks.parquet"
    if parks_path.exists() and "park_id" in mart.columns:
        parks = read_parquet(parks_path)
        pid = _pick(parks, ["park_id", "venue_id", "id"])
        if pid is not None:
            parks = parks.copy()
            parks["park_id"] = parks[pid]
            park_cols = [c for c in parks.columns if c == "park_id" or "hr" in c.lower() or "park_factor" in c.lower()]
            if len(park_cols) > 1:
                mart = mart.merge(parks[park_cols].drop_duplicates(subset=["park_id"]), on="park_id", how="left")

    target_path = dirs["processed_dir"] / "targets" / "game" / f"targets_no_hr_game_{args.season}.parquet"
    if target_path.exists():
        targets = read_parquet(target_path)
        targets["game_pk"] = pd.to_numeric(targets["game_pk"], errors="coerce").astype("Int64")
        merge_cols = [c for c in ["game_pk", "total_hr", "no_hr_game"] if c in targets.columns]
        mart = mart.merge(targets[merge_cols], on="game_pk", how="left")
    else:
        logging.info("Targets file missing for season=%s (expected for live scoring): %s", args.season, target_path.resolve())

    if "total_hr" not in mart.columns:
        mart["total_hr"] = pd.Series([pd.NA] * len(mart), dtype="Int64")
    if "no_hr_game" not in mart.columns:
        mart["no_hr_game"] = pd.Series([pd.NA] * len(mart), dtype="Int64")
    else:
        mart["no_hr_game"] = pd.to_numeric(mart["no_hr_game"], errors="coerce").astype("Int64")

    mart["batter_feature_source"] = batter_source
    mart["pitcher_feature_source"] = pitcher_source
    mart["fallback_used"] = bool(use_batter_fallback or use_pitcher_fallback)
    mart["feature_snapshot_season"] = np.where(use_batter_fallback or use_pitcher_fallback, args.season - 1, args.season)
    mart["batter_feature_snapshot_season"] = batter_snapshot_season
    mart["pitcher_feature_snapshot_season"] = pitcher_snapshot_season

    mart = _attach_engineered_features(mart)

    out_dir = dirs["processed_dir"] / "marts" / "no_hr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"no_hr_game_features_{args.season}.parquet"
    if out_path.exists() and not args.force:
        logging.info("exists and force=False: %s", out_path.resolve())
    else:
        write_parquet(mart, out_path)

    home_feat_cols = [c for c in mart.columns if c.startswith("home_team_")]
    away_feat_cols = [c for c in mart.columns if c.startswith("away_team_")]
    starter_cols = [c for c in mart.columns if c.startswith("home_sp_") or c.startswith("away_sp_")]

    home_games_with_feats = int(mart[home_feat_cols].notna().any(axis=1).sum()) if home_feat_cols else 0
    away_games_with_feats = int(mart[away_feat_cols].notna().any(axis=1).sum()) if away_feat_cols else 0
    games_with_starter = int(mart[starter_cols].notna().any(axis=1).sum()) if starter_cols else 0

    logging.info(
        "fallback diagnostics requested_season=%s batter_source=%s pitcher_source=%s fallback_used=%s",
        args.season,
        batter_source,
        pitcher_source,
        use_batter_fallback or use_pitcher_fallback,
    )
    logging.info(
        "no_hr_mart rows=%s unique_games=%s games_with_home_features=%s games_with_away_features=%s games_with_starter_features=%s path=%s",
        len(mart),
        int(mart["game_pk"].nunique()) if "game_pk" in mart.columns else 0,
        home_games_with_feats,
        away_games_with_feats,
        games_with_starter,
        out_path.resolve(),
    )
    print(f"no_hr_mart -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
