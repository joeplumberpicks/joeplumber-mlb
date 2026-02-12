from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


def _safe_match_rate(matched: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return matched / total


def _load_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} parquet: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def _normalize_hand(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.upper()
    return s.where(s.isin(["R", "L"]), "UNK")


def _rename_context_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep_keys = ["game_pk", "pitcher_id"]
    rename_map: dict[str, str] = {}
    for col in df.columns:
        if col in keep_keys:
            continue
        rename_map[col] = f"{prefix}{col}"
    return df.rename(columns=rename_map)


def _merge_optional_bullpen(out: pd.DataFrame, bullpen: pd.DataFrame) -> pd.DataFrame:
    req = {"game_date", "team"}
    miss = req - set(bullpen.columns)
    if miss:
        raise ValueError(f"bullpen context missing required columns: {sorted(miss)}")

    bp = bullpen.copy()
    bp["game_date"] = pd.to_datetime(bp["game_date"], errors="coerce").dt.normalize()
    bp["team"] = bp["team"].astype("string")

    metric_cols = [c for c in bp.columns if c not in {"game_date", "team"}]

    home_bp = bp.rename(columns={c: f"home_{c}" for c in metric_cols})
    out = out.merge(
        home_bp,
        left_on=["game_date", "home_team"],
        right_on=["game_date", "team"],
        how="left",
        suffixes=("", ""),
    )
    if "team" in out.columns:
        out = out.drop(columns=["team"])

    away_bp = bp.rename(columns={c: f"away_{c}" for c in metric_cols})
    out = out.merge(
        away_bp,
        left_on=["game_date", "away_team"],
        right_on=["game_date", "team"],
        how="left",
        suffixes=("", ""),
    )
    if "team" in out.columns:
        out = out.drop(columns=["team"])

    return out


def _derive_pitcher_hand_lookup(pitches: pd.DataFrame | None) -> pd.Series:
    if pitches is None or pitches.empty:
        return pd.Series(dtype="string")

    p = pitches.copy()
    pitcher_col = "pitcher_id" if "pitcher_id" in p.columns else ("pitcher" if "pitcher" in p.columns else None)
    hand_col = "pitcher_throws" if "pitcher_throws" in p.columns else ("pitcher_hand" if "pitcher_hand" in p.columns else None)
    if pitcher_col is None or hand_col is None:
        return pd.Series(dtype="string")

    p["pitcher_id"] = pd.to_numeric(p[pitcher_col], errors="coerce").astype("Int64")
    p["pitcher_hand"] = _normalize_hand(p[hand_col])
    p = p.dropna(subset=["pitcher_id"])

    mode = (
        p.groupby("pitcher_id", observed=False)["pitcher_hand"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "UNK")
        .astype("string")
    )
    return mode


def _merge_optional_offense(
    out: pd.DataFrame,
    offense: pd.DataFrame,
    *,
    hand_lookup: pd.Series | None = None,
) -> pd.DataFrame:
    req = {"game_date", "team", "vs_pitcher_hand"}
    miss = req - set(offense.columns)
    if miss:
        raise ValueError(f"offense discipline missing required columns: {sorted(miss)}")

    od = offense.copy()
    od["game_date"] = pd.to_datetime(od["game_date"], errors="coerce").dt.normalize()
    od["team"] = od["team"].astype("string")
    od["vs_pitcher_hand"] = _normalize_hand(od["vs_pitcher_hand"])

    if "home_pitcher_hand" in out.columns:
        out["home_pitcher_hand"] = _normalize_hand(out["home_pitcher_hand"])
    else:
        out["home_pitcher_hand"] = "UNK"

    if "away_pitcher_hand" in out.columns:
        out["away_pitcher_hand"] = _normalize_hand(out["away_pitcher_hand"])
    else:
        out["away_pitcher_hand"] = "UNK"

    if hand_lookup is not None and not hand_lookup.empty:
        home_missing = out["home_pitcher_hand"].eq("UNK")
        away_missing = out["away_pitcher_hand"].eq("UNK")
        out.loc[home_missing, "home_pitcher_hand"] = out.loc[home_missing, "home_sp_id"].map(hand_lookup).fillna("UNK")
        out.loc[away_missing, "away_pitcher_hand"] = out.loc[away_missing, "away_sp_id"].map(hand_lookup).fillna("UNK")

    metric_cols = [c for c in od.columns if c not in {"game_date", "team", "vs_pitcher_hand"}]

    home_od = od.rename(columns={c: f"home_off_{c}" for c in metric_cols})
    out = out.merge(
        home_od,
        left_on=["game_date", "home_team", "away_pitcher_hand"],
        right_on=["game_date", "team", "vs_pitcher_hand"],
        how="left",
        suffixes=("", ""),
    )
    for col in ["team", "vs_pitcher_hand"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    away_od = od.rename(columns={c: f"away_off_{c}" for c in metric_cols})
    out = out.merge(
        away_od,
        left_on=["game_date", "away_team", "home_pitcher_hand"],
        right_on=["game_date", "team", "vs_pitcher_hand"],
        how="left",
        suffixes=("", ""),
    )
    for col in ["team", "vs_pitcher_hand"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    return out


def _merge_optional_weather_game(out: pd.DataFrame, weather_game: pd.DataFrame) -> pd.DataFrame:
    req = {"game_date", "game_pk"}
    miss = req - set(weather_game.columns)
    if miss:
        raise ValueError(f"weather_game missing required columns: {sorted(miss)}")

    wx = weather_game.copy()
    wx["game_date"] = pd.to_datetime(wx["game_date"], errors="coerce").dt.normalize()
    wx["game_pk"] = pd.to_numeric(wx["game_pk"], errors="coerce").astype("Int64")
    wx = wx.sort_values(["game_date", "game_pk"], kind="mergesort")
    wx = wx.drop_duplicates(subset=["game_pk"], keep="first")

    drop_cols = [c for c in ["home_team", "away_team"] if c in wx.columns]
    if drop_cols:
        wx = wx.drop(columns=drop_cols)
    return out.merge(wx, on=["game_date", "game_pk"], how="left")


def _merge_optional_weather_factors(out: pd.DataFrame, weather_factors: pd.DataFrame) -> pd.DataFrame:
    req = {"game_date", "game_pk"}
    miss = req - set(weather_factors.columns)
    if miss:
        raise ValueError(f"weather_factors_game missing required columns: {sorted(miss)}")

    wf = weather_factors.copy()
    wf["game_date"] = pd.to_datetime(wf["game_date"], errors="coerce").dt.normalize()
    wf["game_pk"] = pd.to_numeric(wf["game_pk"], errors="coerce").astype("Int64")
    wf = wf.sort_values(["game_date", "game_pk"], kind="mergesort")
    wf = wf.drop_duplicates(subset=["game_pk"], keep="first")
    return out.merge(wf, on=["game_date", "game_pk"], how="left")




def _merge_optional_park_game(out: pd.DataFrame, park_game: pd.DataFrame) -> pd.DataFrame:
    req = {"game_date", "game_pk", "park_id"}
    miss = req - set(park_game.columns)
    if miss:
        raise ValueError(f"park_game missing required columns: {sorted(miss)}")

    pg = park_game.copy()
    pg["game_date"] = pd.to_datetime(pg["game_date"], errors="coerce").dt.normalize()
    pg["game_pk"] = pd.to_numeric(pg["game_pk"], errors="coerce").astype("Int64")
    pg = pg.sort_values(["game_date", "game_pk"], kind="mergesort")
    pg = pg.drop_duplicates(subset=["game_pk"], keep="first")

    keep_cols = [
        c
        for c in [
            "game_date",
            "game_pk",
            "park_id",
            "roof_type",
            "cf_bearing_deg",
            "lat",
            "lon",
            "timezone",
            "stadium_name",
            "city",
            "state",
        ]
        if c in pg.columns
    ]
    return out.merge(pg[keep_cols], on=["game_date", "game_pk"], how="left")


def _merge_optional_park_factors(out: pd.DataFrame, park_factors: pd.DataFrame) -> pd.DataFrame:
    req = {"game_date", "game_pk"}
    miss = req - set(park_factors.columns)
    if miss:
        raise ValueError(f"park_factors_game missing required columns: {sorted(miss)}")

    pf = park_factors.copy()
    pf["game_date"] = pd.to_datetime(pf["game_date"], errors="coerce").dt.normalize()
    pf["game_pk"] = pd.to_numeric(pf["game_pk"], errors="coerce").astype("Int64")
    pf = pf.sort_values(["game_date", "game_pk"], kind="mergesort")
    pf = pf.drop_duplicates(subset=["game_pk"], keep="first")

    keep_cols = [
        c
        for c in ["game_date", "game_pk", "park_games_prior", "park_hr_mult", "park_runs_delta", "park_yrfi_delta"]
        if c in pf.columns
    ]
    return out.merge(pf[keep_cols], on=["game_date", "game_pk"], how="left")

def build_model_features_game(
    season: int,
    spine: pd.DataFrame,
    context: pd.DataFrame,
    *,
    bullpen: pd.DataFrame | None = None,
    offense: pd.DataFrame | None = None,
    weather_game: pd.DataFrame | None = None,
    weather_factors: pd.DataFrame | None = None,
    park_game: pd.DataFrame | None = None,
    park_factors: pd.DataFrame | None = None,
    hand_lookup: pd.Series | None = None,
    start: str | None = None,
    end: str | None = None,
    allow_partial: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Merge canonical game spine with starter context plus optional bullpen/offense context."""
    log = logger or logging.getLogger(__name__)

    required_spine = {"game_date", "game_pk", "home_team", "away_team", "home_sp_id", "away_sp_id"}
    missing_spine = required_spine - set(spine.columns)
    if missing_spine:
        raise ValueError(f"spine missing required columns: {sorted(missing_spine)}")

    required_context = {"game_date", "game_pk", "side", "pitcher_id"}
    missing_context = required_context - set(context.columns)
    if missing_context:
        raise ValueError(f"context missing required columns: {sorted(missing_context)}")

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

    sp = sp.sort_values(["game_date", "game_pk"], kind="mergesort")
    sp = sp.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    ctx = context.copy()
    ctx["game_date"] = pd.to_datetime(ctx["game_date"], errors="coerce").dt.normalize()
    ctx["game_pk"] = pd.to_numeric(ctx["game_pk"], errors="coerce").astype("Int64")
    ctx["pitcher_id"] = pd.to_numeric(ctx["pitcher_id"], errors="coerce").astype("Int64")
    ctx["side"] = ctx["side"].astype("string").str.strip().str.lower()

    if start:
        ctx = ctx.loc[ctx["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        ctx = ctx.loc[ctx["game_date"] <= pd.to_datetime(end).normalize()].copy()
    ctx = ctx.loc[ctx["game_date"].dt.year == int(season)].copy()

    home_side = _rename_context_columns(ctx.loc[ctx["side"] == "home"].copy(), "home_")
    away_side = _rename_context_columns(ctx.loc[ctx["side"] == "away"].copy(), "away_")

    out = sp.merge(home_side, left_on=["game_pk", "home_sp_id"], right_on=["game_pk", "pitcher_id"], how="left")
    if "pitcher_id" in out.columns:
        out = out.drop(columns=["pitcher_id"])

    out = out.merge(away_side, left_on=["game_pk", "away_sp_id"], right_on=["game_pk", "pitcher_id"], how="left")
    if "pitcher_id" in out.columns:
        out = out.drop(columns=["pitcher_id"])

    total_games = len(sp)
    home_match_col = "home_pitches_thrown"
    away_match_col = "away_pitches_thrown"
    home_matched = int(out[home_match_col].notna().sum()) if home_match_col in out.columns else 0
    away_matched = int(out[away_match_col].notna().sum()) if away_match_col in out.columns else 0
    home_rate = _safe_match_rate(home_matched, total_games)
    away_rate = _safe_match_rate(away_matched, total_games)

    log.info("spine_games=%d", total_games)
    log.info("home_join_match_rate=%.4f (%d/%d)", home_rate, home_matched, total_games)
    log.info("away_join_match_rate=%.4f (%d/%d)", away_rate, away_matched, total_games)

    min_rate = min(home_rate, away_rate)
    if min_rate < 0.90:
        message = f"Starter context join match rate below 90%: home={home_rate:.4f}, away={away_rate:.4f}"
        if allow_partial:
            log.warning(message)
        else:
            raise SystemExit(2)

    if bullpen is not None:
        out = _merge_optional_bullpen(out, bullpen)

    if offense is not None:
        out = _merge_optional_offense(out, offense, hand_lookup=hand_lookup)

    if weather_game is not None:
        out = _merge_optional_weather_game(out, weather_game)

    if weather_factors is not None:
        out = _merge_optional_weather_factors(out, weather_factors)

    if park_game is not None:
        out = _merge_optional_park_game(out, park_game)

    if park_factors is not None:
        out = _merge_optional_park_factors(out, park_factors)

    if park_game is not None:
        park_match = float(out["park_id"].notna().mean()) if "park_id" in out.columns and len(out) else 0.0
        if park_match < 0.95:
            msg = f"Park game merge match rate below 95%: {park_match:.4f}"
            if allow_partial:
                log.warning(msg)
            else:
                raise SystemExit(2)

    if park_factors is not None:
        pf_col = "park_hr_mult" if "park_hr_mult" in out.columns else None
        pf_match = float(out[pf_col].notna().mean()) if pf_col and len(out) else 0.0
        if pf_match < 0.95:
            msg = f"Park factors merge match rate below 95%: {pf_match:.4f}"
            if allow_partial:
                log.warning(msg)
            else:
                raise SystemExit(2)

    out = out.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    return out


def build_and_write_model_features_game(
    season: int,
    spine_path: Path,
    context_path: Path,
    output_path: Path,
    *,
    bullpen_path: Path | None = None,
    offense_path: Path | None = None,
    weather_game_path: Path | None = None,
    weather_factors_path: Path | None = None,
    park_game_path: Path | None = None,
    park_factors_path: Path | None = None,
    pitches_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    allow_partial: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Load spine/context, merge game-level features, and write parquet output."""
    log = logger or logging.getLogger(__name__)
    spine = _load_required(spine_path, "canonical spine")
    context = _load_required(context_path, "statcast game context")
    bullpen = _load_required(bullpen_path, "bullpen context") if bullpen_path is not None else None
    offense = _load_required(offense_path, "offense discipline") if offense_path is not None else None
    weather_game = _load_required(weather_game_path, "weather game") if weather_game_path is not None else None
    weather_factors = (
        _load_required(weather_factors_path, "weather factors game") if weather_factors_path is not None else None
    )
    park_game = _load_required(park_game_path, "park game") if park_game_path is not None else None
    park_factors = _load_required(park_factors_path, "park factors game") if park_factors_path is not None else None

    hand_lookup = pd.Series(dtype="string")
    if pitches_path is not None and pitches_path.exists():
        hand_lookup = _derive_pitcher_hand_lookup(pd.read_parquet(pitches_path, engine="pyarrow"))

    out = build_model_features_game(
        season=season,
        spine=spine,
        context=context,
        bullpen=bullpen,
        offense=offense,
        weather_game=weather_game,
        weather_factors=weather_factors,
        park_game=park_game,
        park_factors=park_factors,
        hand_lookup=hand_lookup,
        start=start,
        end=end,
        allow_partial=allow_partial,
        logger=log,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_model_features_game season=%s rows=%d path=%s", season, len(out), output_path)
    return out
