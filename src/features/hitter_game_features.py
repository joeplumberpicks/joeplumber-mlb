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


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _normalize_hand(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip().str.upper()
    return x.where(x.isin(["R", "L"]), "UNK")


def _load_optional(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")


def _discover_batter_rolling(season: int) -> Path | None:
    for p in [Path(f"data/processed/batter_rolling_{season}.parquet"), Path("data/processed/batter_rolling.parquet")]:
        if p.exists():
            return p
    return None


def _build_participation_spine(events: pd.DataFrame, season: int, start: str | None, end: str | None) -> pd.DataFrame:
    cols = events.columns
    date_col = _first_existing(cols, ["game_date", "date", "game_dt"])
    gpk_col = _first_existing(cols, ["game_pk", "game_id", "mlb_game_id"])
    batter_col = _first_existing(cols, ["batter_id", "batter", "hitter_id", "player_id"])
    team_col = _first_existing(cols, ["batting_team", "bat_team", "team", "team_id"])
    if not all([date_col, gpk_col, batter_col, team_col]):
        raise ValueError("events_pa missing columns required for hitter participation spine")

    sp = events[[date_col, gpk_col, batter_col, team_col]].copy()
    sp.columns = ["game_date", "game_pk", "batter_id", "team"]
    sp["game_date"] = _to_date(sp["game_date"])
    sp["game_pk"] = _to_int(sp["game_pk"])
    sp["batter_id"] = _to_int(sp["batter_id"])
    sp["team"] = sp["team"].astype("string")

    sp = sp.loc[sp["game_date"].dt.year == int(season)].copy()
    if start:
        sp = sp.loc[sp["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        sp = sp.loc[sp["game_date"] <= pd.to_datetime(end).normalize()].copy()

    return sp.dropna(subset=["game_date", "game_pk", "batter_id", "team"]).drop_duplicates(["game_pk", "batter_id"], keep="first")


def _attach_team_meta(spine: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    cols = games.columns
    gpk = _first_existing(cols, ["game_pk", "game_id", "mlb_game_id"])
    home = _first_existing(cols, ["home_team", "home_team_name", "home_name"])
    away = _first_existing(cols, ["away_team", "away_team_name", "away_name"])
    park = _first_existing(cols, ["park", "park_id", "venue", "venue_id"])
    if not all([gpk, home, away]):
        return spine.assign(opponent_team=pd.NA, park=pd.NA)

    g = games[[gpk, home, away] + ([park] if park else [])].copy().drop_duplicates(subset=[gpk], keep="first")
    rename = {gpk: "game_pk", home: "home_team", away: "away_team"}
    if park:
        rename[park] = "park"
    g = g.rename(columns=rename)
    g["game_pk"] = _to_int(g["game_pk"])

    out = spine.merge(g, on="game_pk", how="left")
    out["opponent_team"] = np.where(out["team"] == out["home_team"], out["away_team"], np.where(out["team"] == out["away_team"], out["home_team"], pd.NA))
    if "park" not in out.columns:
        out["park"] = pd.NA
    return out


def _prepare_game_features(game_features: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    gf = game_features.copy()
    gf["game_pk"] = _to_int(gf["game_pk"])

    sp = spine[["game_pk", "home_sp_id", "away_sp_id", "home_team", "away_team"]].drop_duplicates(subset=["game_pk"], keep="first").copy()
    sp["home_sp_id"] = _to_int(sp["home_sp_id"])
    sp["away_sp_id"] = _to_int(sp["away_sp_id"])

    out = gf.merge(sp, on="game_pk", how="left", suffixes=("", "_sp"))
    return out


def _merge_shared_game_features(base: pd.DataFrame, game_features: pd.DataFrame) -> pd.DataFrame:
    gf = game_features.copy()
    gf["game_pk"] = _to_int(gf["game_pk"])
    out = base.merge(gf, on=["game_date", "game_pk"], how="left", suffixes=("", "_game"))

    # choose opponent starter and hand by batter team
    out["opp_starter_id"] = np.where(out["team"] == out.get("home_team"), out.get("away_sp_id"), out.get("home_sp_id"))
    out["opp_starter_hand"] = np.where(out["team"] == out.get("home_team"), out.get("away_pitcher_hand", "UNK"), out.get("home_pitcher_hand", "UNK"))
    out["opp_starter_hand"] = _normalize_hand(pd.Series(out["opp_starter_hand"]))
    return out


def _merge_ppmi(base: pd.DataFrame, ppmi: pd.DataFrame | None) -> tuple[pd.DataFrame, float]:
    if ppmi is None:
        return base, 0.0
    p = ppmi.copy()
    p["batter_id"] = _to_int(p["batter_id"])
    p["pitcher_id"] = _to_int(p["pitcher_id"])
    if "game_pk" in p.columns:
        p["game_pk"] = _to_int(p["game_pk"])
        if "game_date" in p.columns:
            p["game_date"] = _to_date(p["game_date"])
            merged = base.merge(
                p,
                left_on=["game_date", "game_pk", "batter_id", "opp_starter_id"],
                right_on=["game_date", "game_pk", "batter_id", "pitcher_id"],
                how="left",
            )
        else:
            merged = base.merge(
                p,
                left_on=["game_pk", "batter_id", "opp_starter_id"],
                right_on=["game_pk", "batter_id", "pitcher_id"],
                how="left",
            )
    else:
        p = p.sort_values([c for c in ["game_date", "pitcher_id", "batter_id"] if c in p.columns], kind="mergesort")
        p = p.drop_duplicates(subset=["batter_id", "pitcher_id"], keep="last")
        merged = base.merge(p, left_on=["batter_id", "opp_starter_id"], right_on=["batter_id", "pitcher_id"], how="left")
    rate = float(merged["ppmi_xwoba"].notna().mean()) if "ppmi_xwoba" in merged.columns and len(merged) else 0.0
    return merged, rate


def _aggregate_hitter_pitchtype(hpt: pd.DataFrame) -> pd.DataFrame:
    h = hpt.copy()
    h["batter_id"] = _to_int(h["batter_id"])
    h["pitcher_throws"] = _normalize_hand(h["pitcher_throws"]) if "pitcher_throws" in h.columns else "UNK"
    weight = pd.to_numeric(h.get("pitch_usage_pct"), errors="coerce").fillna(0.0)
    metrics = ["whiff_rate", "swing_rate", "hr_per_bip", "hard_hit_rate", "xwoba_on_contact"]
    for m in metrics:
        h[m] = pd.to_numeric(h.get(m), errors="coerce")
    agg_rows = []
    for (bid, hand), g in h.groupby(["batter_id", "pitcher_throws"], dropna=False, observed=False):
        w = pd.to_numeric(g.get("pitch_usage_pct"), errors="coerce").fillna(0.0)
        row = {"batter_id": bid, "pitcher_throws": hand}
        den = float(w.sum())
        for m in metrics:
            row[f"hpt_{m}"] = float((g[m] * w).sum() / den) if den > 0 else np.nan
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


def _aggregate_pitcher_mix(pm: pd.DataFrame) -> pd.DataFrame:
    p = pm.copy()
    p["pitcher_id"] = _to_int(p["pitcher_id"])
    p["batter_stand"] = p.get("batter_stand", "UNK").astype("string").str.upper()
    p["pitch_usage_pct"] = pd.to_numeric(p.get("pitch_usage_pct"), errors="coerce")
    p["whiff_rate"] = pd.to_numeric(p.get("whiff_rate"), errors="coerce")
    p["zone_pct"] = pd.to_numeric(p.get("zone_pct"), errors="coerce")

    rows=[]
    for (pid, stand), g in p.groupby(["pitcher_id", "batter_stand"], dropna=False, observed=False):
        g=g.sort_values(["pitch_usage_pct","pitch_type"], ascending=[False,True], kind="mergesort")
        top=g.iloc[0] if len(g) else None
        rows.append({
            "pitcher_id": pid,
            "batter_stand": stand,
            "starter_top1_usage": float(top["pitch_usage_pct"]) if top is not None else np.nan,
            "starter_top1_whiff": float(top["whiff_rate"]) if top is not None else np.nan,
            "starter_top1_zone_pct": float(top["zone_pct"]) if top is not None else np.nan,
        })
    return pd.DataFrame(rows)


def _merge_batter_rolling(base: pd.DataFrame, roll: pd.DataFrame | None) -> pd.DataFrame:
    if roll is None or roll.empty:
        return base
    date_col = _first_existing(roll.columns, ["game_date", "date", "game_dt"])
    bid_col = _first_existing(roll.columns, ["batter_id", "batter", "hitter_id", "player_id"])
    if not date_col or not bid_col:
        return base
    r = roll.copy()
    r["game_date"] = _to_date(r[date_col])
    r["batter_id"] = _to_int(r[bid_col])
    num_cols = [c for c in r.columns if pd.api.types.is_numeric_dtype(r[c]) and c not in {"batter_id", "game_pk"}]
    keep = ["game_date", "batter_id"] + num_cols
    r = r[keep].drop_duplicates(["game_date", "batter_id"], keep="last")
    r = r.rename(columns={c: f"roll_{c}" for c in num_cols})
    return base.merge(r, on=["game_date", "batter_id"], how="left")


def build_hitter_game_features(
    season: int,
    events_pa: pd.DataFrame,
    games: pd.DataFrame,
    spine: pd.DataFrame,
    game_features: pd.DataFrame,
    *,
    ppmi: pd.DataFrame | None = None,
    hitter_pitchtype: pd.DataFrame | None = None,
    pitcher_mix: pd.DataFrame | None = None,
    batter_rolling: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    allow_partial: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)

    base = _build_participation_spine(events_pa, season, start, end)
    base = _attach_team_meta(base, games)

    gf = _prepare_game_features(game_features, spine)
    base = _merge_shared_game_features(base, gf)

    # optional joins
    ppmi_rate = 0.0
    base, ppmi_rate = _merge_ppmi(base, ppmi)

    if hitter_pitchtype is not None:
        hpt = _aggregate_hitter_pitchtype(hitter_pitchtype)
        base = base.merge(hpt, left_on=["batter_id", "opp_starter_hand"], right_on=["batter_id", "pitcher_throws"], how="left")
        if "pitcher_throws" in base.columns:
            base = base.drop(columns=["pitcher_throws"])
    hpt_rate = float(base.filter(like="hpt_").notna().any(axis=1).mean()) if len(base) else 0.0

    if pitcher_mix is not None:
        pm = _aggregate_pitcher_mix(pitcher_mix)
        batter_stand = _normalize_hand(pd.Series(np.where(base["team"] == base.get("home_team"), "R", "L"), index=base.index))
        base["_tmp_batter_stand"] = batter_stand
        base = base.merge(pm, left_on=["opp_starter_id", "_tmp_batter_stand"], right_on=["pitcher_id", "batter_stand"], how="left")
        for c in ["pitcher_id", "batter_stand", "_tmp_batter_stand"]:
            if c in base.columns:
                base = base.drop(columns=[c])

    base = _merge_batter_rolling(base, batter_rolling)

    log.info("hitter_game_rows_created=%d", len(base))
    log.info("hitter_game_ppmi_match_rate=%.4f", ppmi_rate)
    log.info("hitter_game_hitter_pitchtype_match_rate=%.4f", hpt_rate)

    if (ppmi_rate < 0.3 or hpt_rate < 0.3) and not allow_partial:
        raise SystemExit(2)
    if (ppmi_rate < 0.3 or hpt_rate < 0.3) and allow_partial:
        log.warning("hitter_game_low_match_rates ppmi=%.4f hpt=%.4f", ppmi_rate, hpt_rate)

    keep_fixed = ["game_date", "game_pk", "batter_id", "team", "opponent_team", "park"]
    cols = keep_fixed + [c for c in base.columns if c not in keep_fixed]
    return base[cols].sort_values(["game_date", "game_pk", "batter_id"], kind="mergesort").reset_index(drop=True)


def build_and_write_hitter_game_features(
    season: int,
    events_path: Path,
    games_path: Path,
    spine_path: Path,
    game_features_path: Path,
    output_path: Path,
    *,
    ppmi_path: Path | None = None,
    hitter_pitchtype_path: Path | None = None,
    pitcher_mix_path: Path | None = None,
    batter_rolling_path: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    allow_partial: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)
    for p, label in [(events_path, "events_pa"), (games_path, "games"), (spine_path, "spine"), (game_features_path, "game features")]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} parquet: {p}")

    br = batter_rolling_path if batter_rolling_path and batter_rolling_path.exists() else _discover_batter_rolling(season)

    out = build_hitter_game_features(
        season=season,
        events_pa=pd.read_parquet(events_path, engine="pyarrow"),
        games=pd.read_parquet(games_path, engine="pyarrow"),
        spine=pd.read_parquet(spine_path, engine="pyarrow"),
        game_features=pd.read_parquet(game_features_path, engine="pyarrow"),
        ppmi=_load_optional(ppmi_path),
        hitter_pitchtype=_load_optional(hitter_pitchtype_path),
        pitcher_mix=_load_optional(pitcher_mix_path),
        batter_rolling=_load_optional(br),
        start=start,
        end=end,
        allow_partial=allow_partial,
        logger=log,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_hitter_game_features season=%s rows=%d path=%s", season, len(out), output_path)
    return out
