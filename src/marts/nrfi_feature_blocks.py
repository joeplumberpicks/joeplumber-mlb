from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def _load_pa(raw_dir: Path, season: int | None) -> pd.DataFrame:
    if season is None:
        return pd.DataFrame()
    p = raw_dir / "by_season" / f"pa_{season}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    return df


def _infer_runs(pa: pd.DataFrame) -> pd.Series:
    for c in ["rbi", "runs_scored", "bat_score_diff", "delta_bat_score"]:
        if c in pa.columns:
            return pd.to_numeric(pa[c], errors="coerce").fillna(0.0).clip(lower=0)
    return pd.Series(np.zeros(len(pa)), index=pa.index, dtype="float64")


def _build_starter_fi_features(spine: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    if spine.empty or pa.empty or "game_pk" not in spine.columns:
        return pd.DataFrame(columns=["game_pk"])

    inning_col = _pick(pa, ["inning"])
    half_col = _pick(pa, ["inning_topbot", "inning_half", "topbot", "inning_top_bot"])
    pit_col = _pick(pa, ["pitcher", "pitcher_id"])
    if not inning_col or not half_col or not pit_col:
        return pd.DataFrame(columns=["game_pk"])

    p = pa.copy()
    p["game_pk"] = pd.to_numeric(p.get("game_pk"), errors="coerce").astype("Int64")
    p = p[p["game_pk"].notna()].copy()
    p[inning_col] = pd.to_numeric(p[inning_col], errors="coerce")
    p = p[p[inning_col] == 1].copy()
    p["_half"] = p[half_col].astype(str).str.lower()
    p["pitcher_id"] = pd.to_numeric(p[pit_col], errors="coerce").astype("Int64")
    p["fi_runs"] = _infer_runs(p)

    game_runs = (
        p.groupby(["game_pk", "pitcher_id", "_half"], dropna=False)
        .agg(fi_runs_allowed=("fi_runs", "sum"), fi_pa=("game_pk", "size"))
        .reset_index()
    )
    game_runs["fi_allowed_yrfi"] = (game_runs["fi_runs_allowed"] >= 1).astype(float)

    s = spine[[c for c in ["game_pk", "game_date", "home_sp_id", "away_sp_id"] if c in spine.columns]].copy()
    s["game_pk"] = pd.to_numeric(s.get("game_pk"), errors="coerce").astype("Int64")
    s["game_date"] = pd.to_datetime(s.get("game_date"), errors="coerce")

    away = s[["game_pk", "game_date", "away_sp_id"]].rename(columns={"away_sp_id": "pitcher_id"})
    away["side"] = "away"
    away["_half"] = "bot"
    home = s[["game_pk", "game_date", "home_sp_id"]].rename(columns={"home_sp_id": "pitcher_id"})
    home["side"] = "home"
    home["_half"] = "top"
    starts = pd.concat([away, home], ignore_index=True)
    starts["pitcher_id"] = pd.to_numeric(starts["pitcher_id"], errors="coerce").astype("Int64")

    m = starts.merge(game_runs, on=["game_pk", "pitcher_id", "_half"], how="left")
    m["fi_runs_allowed"] = pd.to_numeric(m["fi_runs_allowed"], errors="coerce").fillna(0.0)
    m["fi_pa"] = pd.to_numeric(m["fi_pa"], errors="coerce").fillna(0.0)
    m["fi_allowed_yrfi"] = pd.to_numeric(m["fi_allowed_yrfi"], errors="coerce").fillna(0.0)

    m = m.sort_values(["pitcher_id", "game_date"], kind="mergesort")
    windows = [15, 30]
    lg = float(m["fi_allowed_yrfi"].mean()) if len(m) else 0.27
    for w in windows:
        grp = m.groupby("pitcher_id", dropna=False)
        starts_prev = grp["fi_allowed_yrfi"].shift(1).rolling(w, min_periods=1).count()
        yrfi_sum_prev = grp["fi_allowed_yrfi"].shift(1).rolling(w, min_periods=1).sum()
        runs_prev = grp["fi_runs_allowed"].shift(1).rolling(w, min_periods=1).mean()
        pa_prev = grp["fi_pa"].shift(1).rolling(w, min_periods=1).sum()
        m[f"sp_fi_yrfi_rate_w{w}"] = (yrfi_sum_prev / starts_prev).fillna(0.0)
        m[f"sp_fi_runs_per_start_w{w}"] = runs_prev.fillna(0.0)
        m[f"sp_fi_pa_w{w}"] = pa_prev.fillna(0.0)
        m[f"sp_fi_yrfi_rate_shrunk_w{w}"] = ((yrfi_sum_prev.fillna(0.0) + 10.0 * lg) / (starts_prev.fillna(0.0) + 10.0)).fillna(lg)

    keep = ["game_pk", "side"] + [c for c in m.columns if c.startswith("sp_fi_")]
    m = m[keep].copy()
    away_f = m[m["side"] == "away"].drop(columns=["side"]).rename(columns={c: f"away_sp_{c}" for c in m.columns if c.startswith("sp_fi_")})
    home_f = m[m["side"] == "home"].drop(columns=["side"]).rename(columns={c: f"home_sp_{c}" for c in m.columns if c.startswith("sp_fi_")})
    out = away_f.merge(home_f, on="game_pk", how="outer")
    return out


def _rate(num: pd.Series, den: pd.Series) -> pd.Series:
    den2 = den.replace(0, np.nan)
    return (num / den2).fillna(0.0)


def _build_top3_vs_hand(spine: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    if spine.empty or pa.empty:
        return pd.DataFrame(columns=["game_pk"])
    bat_col = _pick(pa, ["batter", "batter_id"])
    pit_col = _pick(pa, ["pitcher", "pitcher_id"])
    hand_col = _pick(pa, ["p_throws", "pitcher_throws"])
    half_col = _pick(pa, ["inning_topbot", "inning_half", "topbot", "inning_top_bot"])
    inning_col = _pick(pa, ["inning"])
    if not bat_col or not pit_col or not hand_col or not half_col or not inning_col:
        return pd.DataFrame(columns=["game_pk"])

    p = pa.copy()
    p["game_pk"] = pd.to_numeric(p.get("game_pk"), errors="coerce").astype("Int64")
    p["game_date"] = pd.to_datetime(p.get("game_date"), errors="coerce")
    p["batter_id"] = pd.to_numeric(p[bat_col], errors="coerce").astype("Int64")
    p["pitcher_id"] = pd.to_numeric(p[pit_col], errors="coerce").astype("Int64")
    p["opp_hand"] = p[hand_col].astype(str).str.upper().where(p[hand_col].notna())
    p["pa"] = 1.0
    p["bb"] = p.get("events", pd.Series(index=p.index, dtype=object)).astype(str).str.lower().isin(["walk", "intent_walk"]).astype(float)
    p["so"] = p.get("events", pd.Series(index=p.index, dtype=object)).astype(str).str.lower().isin(["strikeout", "strikeout_double_play"]).astype(float)
    p["h"] = p.get("events", pd.Series(index=p.index, dtype=object)).astype(str).str.lower().isin(["single", "double", "triple", "home_run"]).astype(float)
    p["tb"] = (
        p.get("events", pd.Series(index=p.index, dtype=object)).astype(str).str.lower().map({"single": 1, "double": 2, "triple": 3, "home_run": 4}).fillna(0).astype(float)
    )

    bs = (
        p.groupby(["game_pk", "game_date", "batter_id", "opp_hand"], dropna=False)
        .agg(pa=("pa", "sum"), bb=("bb", "sum"), so=("so", "sum"), h=("h", "sum"), tb=("tb", "sum"))
        .reset_index()
    )
    bs = bs.sort_values(["batter_id", "opp_hand", "game_date"], kind="mergesort")
    for w in [30]:
        grp = bs.groupby(["batter_id", "opp_hand"], dropna=False)
        pa_prev = grp["pa"].shift(1).rolling(w, min_periods=1).sum()
        bb_prev = grp["bb"].shift(1).rolling(w, min_periods=1).sum()
        so_prev = grp["so"].shift(1).rolling(w, min_periods=1).sum()
        h_prev = grp["h"].shift(1).rolling(w, min_periods=1).sum()
        tb_prev = grp["tb"].shift(1).rolling(w, min_periods=1).sum()
        bs[f"woba_vs_hand_w{w}"] = _rate(0.69 * bb_prev + 0.89 * h_prev, pa_prev)
        bs[f"iso_vs_hand_w{w}"] = _rate(tb_prev - h_prev, pa_prev)
        bs[f"k_rate_vs_hand_w{w}"] = _rate(so_prev, pa_prev)
        bs[f"bb_rate_vs_hand_w{w}"] = _rate(bb_prev, pa_prev)
        bs[f"sample_pa_w{w}"] = pa_prev.fillna(0.0)

    # top3 from first inning appearances
    p1 = p[pd.to_numeric(p[inning_col], errors="coerce") == 1].copy()
    p1["_half"] = p1[half_col].astype(str).str.lower()
    order_col = _pick(p1, ["at_bat_number", "pitch_number"])
    if order_col:
        p1[order_col] = pd.to_numeric(p1[order_col], errors="coerce")
        p1 = p1.sort_values(["game_pk", "_half", order_col], kind="mergesort")
    p1 = p1.dropna(subset=["batter_id", "game_pk"])
    p1 = p1.drop_duplicates(subset=["game_pk", "_half", "batter_id"], keep="first")
    p1["rn"] = p1.groupby(["game_pk", "_half"]).cumcount() + 1
    top3 = p1[p1["rn"] <= 3][["game_pk", "batter_id", "_half"]].copy()

    s = spine[[c for c in ["game_pk", "away_sp_id", "home_sp_id"] if c in spine.columns]].copy()
    s["game_pk"] = pd.to_numeric(s.get("game_pk"), errors="coerce").astype("Int64")
    hand_map = p[["pitcher_id", "opp_hand"]].dropna().drop_duplicates(subset=["pitcher_id"], keep="last")
    s = s.merge(hand_map.rename(columns={"pitcher_id": "home_sp_id", "opp_hand": "home_sp_hand"}), on="home_sp_id", how="left")
    s = s.merge(hand_map.rename(columns={"pitcher_id": "away_sp_id", "opp_hand": "away_sp_hand"}), on="away_sp_id", how="left")

    away_top = top3[top3["_half"].str.contains("top", na=False)].copy()
    away_top = away_top.merge(s[["game_pk", "home_sp_hand"]], on="game_pk", how="left")
    away_top = away_top.rename(columns={"home_sp_hand": "opp_hand"})

    home_top = top3[top3["_half"].str.contains("bot", na=False)].copy()
    home_top = home_top.merge(s[["game_pk", "away_sp_hand"]], on="game_pk", how="left")
    home_top = home_top.rename(columns={"away_sp_hand": "opp_hand"})

    top = pd.concat([away_top.assign(side="away"), home_top.assign(side="home")], ignore_index=True)
    use = bs[["game_pk", "batter_id", "opp_hand", "woba_vs_hand_w30", "iso_vs_hand_w30", "k_rate_vs_hand_w30", "bb_rate_vs_hand_w30", "sample_pa_w30"]]
    top = top.merge(use, on=["game_pk", "batter_id", "opp_hand"], how="left")
    agg = (
        top.groupby(["game_pk", "side"], dropna=False)
        .agg(
            top3_woba_vs_sp_hand_w30=("woba_vs_hand_w30", "mean"),
            top3_iso_vs_sp_hand_w30=("iso_vs_hand_w30", "mean"),
            top3_k_rate_vs_sp_hand_w30=("k_rate_vs_hand_w30", "mean"),
            top3_bb_rate_vs_sp_hand_w30=("bb_rate_vs_hand_w30", "mean"),
            top3_pa_w30=("sample_pa_w30", "sum"),
        )
        .reset_index()
    )
    away = agg[agg["side"] == "away"].drop(columns=["side"]).rename(columns={c: f"away_{c}" for c in agg.columns if c.startswith("top3_")})
    home = agg[agg["side"] == "home"].drop(columns=["side"]).rename(columns={c: f"home_{c}" for c in agg.columns if c.startswith("top3_")})
    return away.merge(home, on="game_pk", how="outer")


def build_nrfi_feature_blocks(dirs: dict[str, Path], spine: pd.DataFrame, season: int | None) -> pd.DataFrame:
    pa = _load_pa(dirs["raw_dir"], season)
    if pa.empty:
        logging.warning("NRFI feature blocks skipped: missing raw PA for season=%s", season)
        return pd.DataFrame(columns=["game_pk"])
    fi = _build_starter_fi_features(spine, pa)
    top3 = _build_top3_vs_hand(spine, pa)
    out = fi.merge(top3, on="game_pk", how="outer") if not fi.empty or not top3.empty else pd.DataFrame(columns=["game_pk"])
    logging.info("NRFI feature blocks built rows=%s cols=%s", len(out), len(out.columns))
    return out
