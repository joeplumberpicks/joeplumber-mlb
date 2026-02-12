from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def _first_existing(columns: pd.Index, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def _normalize_hand(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    return s.where(s.isin(["R", "L"]), "UNK")


def _load_optional(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")


def _discover_pitcher_rolling(season: int) -> Path | None:
    attempted = [
        Path(f"data/processed/pitcher_rolling_{season}.parquet"),
        Path("data/processed/pitcher_rolling.parquet"),
    ]
    for p in attempted:
        if p.exists():
            return p
    return None


def _build_starter_rows(spine: pd.DataFrame, season: int, start: str | None, end: str | None) -> pd.DataFrame:
    req = {"game_date", "game_pk", "home_team", "away_team", "home_sp_id", "away_sp_id"}
    miss = req - set(spine.columns)
    if miss:
        raise ValueError(f"spine missing required columns: {sorted(miss)}")

    sp = spine.copy()
    sp["game_date"] = pd.to_datetime(sp["game_date"], errors="coerce").dt.normalize()
    sp["game_pk"] = pd.to_numeric(sp["game_pk"], errors="coerce").astype("Int64")
    sp["home_sp_id"] = pd.to_numeric(sp["home_sp_id"], errors="coerce").astype("Int64")
    sp["away_sp_id"] = pd.to_numeric(sp["away_sp_id"], errors="coerce").astype("Int64")

    if "season" in sp.columns:
        sp = sp.loc[pd.to_numeric(sp["season"], errors="coerce") == int(season)].copy()
    else:
        sp = sp.loc[sp["game_date"].dt.year == int(season)].copy()

    if start:
        sp = sp.loc[sp["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        sp = sp.loc[sp["game_date"] <= pd.to_datetime(end).normalize()].copy()

    sp = sp.sort_values(["game_date", "game_pk"], kind="mergesort").drop_duplicates(subset=["game_pk"], keep="first")

    home = pd.DataFrame(
        {
            "game_date": sp["game_date"],
            "game_pk": sp["game_pk"],
            "pitcher_id": sp["home_sp_id"],
            "team": sp["home_team"].astype("string"),
            "opponent_team": sp["away_team"].astype("string"),
            "starter_side": "home",
        }
    )
    away = pd.DataFrame(
        {
            "game_date": sp["game_date"],
            "game_pk": sp["game_pk"],
            "pitcher_id": sp["away_sp_id"],
            "team": sp["away_team"].astype("string"),
            "opponent_team": sp["home_team"].astype("string"),
            "starter_side": "away",
        }
    )
    out = pd.concat([home, away], ignore_index=True)
    out = out.dropna(subset=["game_pk", "pitcher_id"]).reset_index(drop=True)
    return out


def _prepare_context(context: pd.DataFrame, season: int, start: str | None, end: str | None) -> pd.DataFrame:
    req = {"game_date", "game_pk", "pitcher_id"}
    miss = req - set(context.columns)
    if miss:
        raise ValueError(f"context missing required columns: {sorted(miss)}")

    c = context.copy()
    c["game_date"] = pd.to_datetime(c["game_date"], errors="coerce").dt.normalize()
    c["game_pk"] = pd.to_numeric(c["game_pk"], errors="coerce").astype("Int64")
    c["pitcher_id"] = pd.to_numeric(c["pitcher_id"], errors="coerce").astype("Int64")

    c = c.loc[c["game_date"].dt.year == int(season)].copy()
    if start:
        c = c.loc[c["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        c = c.loc[c["game_date"] <= pd.to_datetime(end).normalize()].copy()

    hand_col = _first_existing(c.columns, ["pitcher_hand", "pitcher_throws"])
    if hand_col is not None:
        c["pitcher_hand"] = _normalize_hand(c[hand_col])
    else:
        c["pitcher_hand"] = "UNK"

    keep = [col for col in c.columns if col not in {"side", "pitcher_team", "opponent_team"}]
    return c[keep].drop_duplicates(subset=["game_pk", "pitcher_id"], keep="first")


def _merge_offense(starters: pd.DataFrame, offense: pd.DataFrame | None) -> pd.DataFrame:
    if offense is None:
        return starters

    req = {"game_date", "team", "vs_pitcher_hand"}
    miss = req - set(offense.columns)
    if miss:
        raise ValueError(f"offense discipline missing required columns: {sorted(miss)}")

    od = offense.copy()
    od["game_date"] = pd.to_datetime(od["game_date"], errors="coerce").dt.normalize()
    od["team"] = od["team"].astype("string")
    od["vs_pitcher_hand"] = _normalize_hand(od["vs_pitcher_hand"])

    out = starters.copy()
    out["pitcher_hand"] = _normalize_hand(out.get("pitcher_hand", pd.Series("UNK", index=out.index)))

    metric_cols = [c for c in od.columns if c not in {"game_date", "team", "vs_pitcher_hand"}]
    rename = {c: f"opp_off_{c}" for c in metric_cols}
    od = od.rename(columns=rename)

    out = out.merge(
        od,
        left_on=["game_date", "opponent_team", "pitcher_hand"],
        right_on=["game_date", "team", "vs_pitcher_hand"],
        how="left",
    )
    for col in ["team_y", "vs_pitcher_hand"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    if "team_x" in out.columns:
        out = out.rename(columns={"team_x": "team"})
    return out


def _merge_bullpen(starters: pd.DataFrame, bullpen: pd.DataFrame | None) -> pd.DataFrame:
    if bullpen is None:
        return starters

    req = {"game_date", "team"}
    miss = req - set(bullpen.columns)
    if miss:
        raise ValueError(f"bullpen context missing required columns: {sorted(miss)}")

    bp = bullpen.copy()
    bp["game_date"] = pd.to_datetime(bp["game_date"], errors="coerce").dt.normalize()
    bp["team"] = bp["team"].astype("string")
    metric_cols = [c for c in bp.columns if c not in {"game_date", "team"}]
    bp = bp.rename(columns={c: f"team_bp_{c}" for c in metric_cols})

    out = starters.merge(bp, on=["game_date", "team"], how="left")
    return out


def _merge_pitcher_rolling(starters: pd.DataFrame, rolling: pd.DataFrame | None) -> pd.DataFrame:
    if rolling is None or rolling.empty:
        return starters

    date_col = _first_existing(rolling.columns, ["game_date", "date", "game_dt"])
    pid_col = _first_existing(rolling.columns, ["pitcher_id", "pitcher", "player_id", "mlb_id"])
    if not date_col or not pid_col:
        return starters

    r = rolling.copy()
    r["game_date"] = pd.to_datetime(r[date_col], errors="coerce").dt.normalize()
    r["pitcher_id"] = pd.to_numeric(r[pid_col], errors="coerce").astype("Int64")

    numeric_cols = [c for c in r.columns if pd.api.types.is_numeric_dtype(r[c]) and c not in {"pitcher_id", "game_pk"}]
    keep = ["game_date", "pitcher_id"] + numeric_cols
    r = r[keep].copy().drop_duplicates(subset=["game_date", "pitcher_id"], keep="last")
    r = r.rename(columns={c: f"roll_{c}" for c in numeric_cols})

    return starters.merge(r, on=["game_date", "pitcher_id"], how="left")


def build_pitcher_game_features(
    season: int,
    spine: pd.DataFrame,
    context: pd.DataFrame,
    *,
    offense: pd.DataFrame | None = None,
    bullpen: pd.DataFrame | None = None,
    pitcher_rolling: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    allow_partial: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)

    starters = _build_starter_rows(spine, season, start, end)
    expected_rows = len(starters)

    ctx = _prepare_context(context, season, start, end)
    out = starters.merge(ctx, on=["game_date", "game_pk", "pitcher_id"], how="left", suffixes=("", "_ctx"))

    matched = int(out["pitches_thrown"].notna().sum()) if "pitches_thrown" in out.columns else 0
    match_rate = matched / expected_rows if expected_rows else 0.0
    log.info("pitcher_features_expected_starter_rows=%d", expected_rows)
    log.info("pitcher_features_actual_rows=%d", len(out))
    log.info("pitcher_features_context_match_rate=%.4f (%d/%d)", match_rate, matched, expected_rows)

    if match_rate < 0.90 and not allow_partial:
        raise SystemExit(2)
    if match_rate < 0.90 and allow_partial:
        log.warning("pitcher_features_context_match_rate_below_threshold=%.4f", match_rate)

    out = _merge_offense(out, offense)
    out = _merge_bullpen(out, bullpen)
    out = _merge_pitcher_rolling(out, pitcher_rolling)

    fixed = ["game_date", "game_pk", "pitcher_id", "team", "opponent_team", "starter_side"]
    cols = fixed + [c for c in out.columns if c not in fixed]
    out = out[cols].sort_values(["game_date", "game_pk", "pitcher_id"], kind="mergesort").reset_index(drop=True)
    return out


def build_and_write_pitcher_game_features(
    season: int,
    spine_path: Path,
    context_path: Path,
    output_path: Path,
    *,
    offense_path: Path | None = None,
    bullpen_path: Path | None = None,
    pitcher_rolling_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    allow_partial: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)
    if not spine_path.exists():
        raise FileNotFoundError(f"Missing spine parquet: {spine_path}")
    if not context_path.exists():
        raise FileNotFoundError(f"Missing context parquet: {context_path}")

    rolling_path = pitcher_rolling_path if pitcher_rolling_path and pitcher_rolling_path.exists() else _discover_pitcher_rolling(season)

    out = build_pitcher_game_features(
        season=season,
        spine=pd.read_parquet(spine_path, engine="pyarrow"),
        context=pd.read_parquet(context_path, engine="pyarrow"),
        offense=_load_optional(offense_path),
        bullpen=_load_optional(bullpen_path),
        pitcher_rolling=_load_optional(rolling_path),
        start=start,
        end=end,
        allow_partial=allow_partial,
        logger=log,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_pitcher_game_features season=%s rows=%d path=%s", season, len(out), output_path)
    return out
