from __future__ import annotations

import hashlib
import pandas as pd


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def _safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    return out


def _normalize_is_home(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "y": True,
                "n": False,
                "yes": True,
                "no": False,
                "home": True,
                "away": False,
                "@": False,
                "vs": True,
            }
        )
        .astype("boolean")
    )


def _to_int64_nullable(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _stable_noise(*values: object, scale: float = 0.008) -> float:
    key = "|".join("" if v is None else str(v) for v in values)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    n = int(digest[:8], 16) / 0xFFFFFFFF
    return (n - 0.5) * 2 * scale


def _latest_rows_by_id(df: pd.DataFrame, id_candidates: list[str], slate_date: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    out = _safe_copy(df)
    id_col = _pick_col(out, id_candidates, required=True)
    date_col = _pick_col(out, ["game_date"], required=True)

    out[id_col] = _to_int64_nullable(out[id_col])
    out = out[out[date_col].notna()].copy()
    out = out[out[date_col] <= slate_date].copy()

    sort_cols = [c for c in [id_col, date_col, "game_pk"] if c in out.columns]
    out = out.sort_values(sort_cols, kind="stable")
    out = out.drop_duplicates(subset=[id_col], keep="last").reset_index(drop=True)
    return out, id_col


def _attach_opposing_pitcher(lineups: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    lu = _safe_copy(lineups)
    sp = _safe_copy(spine)

    game_pk_col = _pick_col(lu, ["game_pk"], required=True)
    lu_is_home_col = _pick_col(lu, ["is_home"], required=True)
    lu[lu_is_home_col] = _normalize_is_home(lu[lu_is_home_col])

    sp_game_pk = _pick_col(sp, ["game_pk"], required=True)
    home_sp_col = _pick_col(sp, ["home_sp_id", "home_starting_pitcher_id", "home_pitcher_id", "home_starter_pitcher_id"], required=True)
    away_sp_col = _pick_col(sp, ["away_sp_id", "away_starting_pitcher_id", "away_pitcher_id", "away_starter_pitcher_id"], required=True)
    home_team_col = _pick_col(sp, ["home_team"], required=False)
    away_team_col = _pick_col(sp, ["away_team"], required=False)
    weather_cols = [c for c in ["temperature_f", "wind_mph", "weather_wind_out", "weather_wind_in", "weather_crosswind"] if c in sp.columns]

    sp_join = sp[[c for c in [sp_game_pk, home_sp_col, away_sp_col, home_team_col, away_team_col] if c is not None] + weather_cols].copy()
    sp_join = sp_join.rename(columns={sp_game_pk: game_pk_col})

    sp_join[home_sp_col] = _to_int64_nullable(sp_join[home_sp_col])
    sp_join[away_sp_col] = _to_int64_nullable(sp_join[away_sp_col])

    lu = lu.merge(sp_join, on=game_pk_col, how="left")

    lu["opp_pitcher_id"] = pd.Series(pd.NA, index=lu.index, dtype="Int64")
    lu.loc[lu[lu_is_home_col].eq(True), "opp_pitcher_id"] = _to_int64_nullable(lu.loc[lu[lu_is_home_col].eq(True), away_sp_col])
    lu.loc[lu[lu_is_home_col].eq(False), "opp_pitcher_id"] = _to_int64_nullable(lu.loc[lu[lu_is_home_col].eq(False), home_sp_col])

    opp_col = _pick_col(lu, ["opponent"], required=False)
    if opp_col is None:
        lu["opponent"] = pd.NA
        opp_col = "opponent"

    if away_team_col is not None and home_team_col is not None:
        lu.loc[lu[lu_is_home_col].eq(True), opp_col] = lu.loc[lu[lu_is_home_col].eq(True), away_team_col]
        lu.loc[lu[lu_is_home_col].eq(False), opp_col] = lu.loc[lu[lu_is_home_col].eq(False), home_team_col]

    return lu


def _build_team_context(lineups: pd.DataFrame, batter_roll: pd.DataFrame, slate_date: pd.Timestamp) -> pd.DataFrame:
    lu = _safe_copy(lineups)
    br = _safe_copy(batter_roll)

    team_col = _pick_col(lu, ["team", "team_abbr", "batting_team"], required=True)
    batter_id_col = _pick_col(lu, ["player_id", "batter_id"], required=True)
    game_pk_col = _pick_col(lu, ["game_pk"], required=True)
    game_date_col = _pick_col(lu, ["game_date"], required=True)
    slot_col = _pick_col(lu, ["batting_order", "lineup_slot", "order_spot", "slot"], required=False)

    lu[batter_id_col] = _to_int64_nullable(lu[batter_id_col])

    br_latest, br_batter_id = _latest_rows_by_id(br, ["batter_id", "player_id"], slate_date)
    br_keep = [c for c in br_latest.columns if c == br_batter_id or "roll" in c]

    lu = lu.merge(br_latest[br_keep].copy(), left_on=batter_id_col, right_on=br_batter_id, how="left")

    if slot_col is not None:
        lu[slot_col] = pd.to_numeric(lu[slot_col], errors="coerce")
        ahead = lu.loc[lu[slot_col].le(5)].copy()
    else:
        ahead = lu.copy()

    agg_cols = {}
    for c in ahead.columns:
        if any(x in c for x in ["bb_rate_roll", "rbi_roll", "tb_roll", "hardhit_rate_roll", "barrel_rate_roll", "ev_mean_roll"]):
            agg_cols[c] = "mean"

    if not agg_cols:
        return (
            ahead.groupby([game_pk_col, game_date_col, team_col], dropna=False)
            .size()
            .reset_index(name="team_context_count")
        )

    out = (
        ahead.groupby([game_pk_col, game_date_col, team_col], dropna=False)
        .agg(agg_cols)
        .reset_index()
    )
    out["team_context_count"] = (
        ahead.groupby([game_pk_col, game_date_col, team_col], dropna=False)[batter_id_col]
        .count()
        .values
    )
    return out


def build_rbi_features(
    spine: pd.DataFrame,
    lineups: pd.DataFrame,
    batter_roll: pd.DataFrame,
    pitcher_roll: pd.DataFrame,
) -> pd.DataFrame:
    sp = _safe_copy(spine)
    slate_date = pd.to_datetime(sp["game_date"], errors="coerce").dropna().max()

    lu = _attach_opposing_pitcher(lineups, sp)
    br = _safe_copy(batter_roll)
    pr = _safe_copy(pitcher_roll)

    lu_batter_id = _pick_col(lu, ["player_id", "batter_id"], required=True)
    lu_game_date = _pick_col(lu, ["game_date"], required=True)
    lu_team_col = _pick_col(lu, ["team", "team_abbr", "batting_team"], required=True)
    lu_game_pk = _pick_col(lu, ["game_pk"], required=True)
    lu_name = _pick_col(lu, ["player_name", "batter_name", "name"], required=True)
    slot_col = _pick_col(lu, ["batting_order", "lineup_slot", "order_spot", "slot"], required=False)

    lu[lu_batter_id] = _to_int64_nullable(lu[lu_batter_id])

    if slot_col is not None:
        slot = pd.to_numeric(lu[slot_col], errors="coerce")
        lu["lineup_weight"] = slot.map({
            1: 1.08,
            2: 1.10,
            3: 1.15,
            4: 1.18,
            5: 1.10,
            6: 1.02,
            7: 0.96,
            8: 0.92,
            9: 0.88,
        }).fillna(1.0)
    else:
        lu["lineup_weight"] = 1.0

    br_latest, br_batter_id = _latest_rows_by_id(br, ["batter_id", "player_id"], slate_date)
    br_keep = [c for c in br_latest.columns if c == br_batter_id or "roll" in c]
    df = lu.merge(br_latest[br_keep].copy(), left_on=lu_batter_id, right_on=br_batter_id, how="left")

    pr_latest, pr_pitcher_id = _latest_rows_by_id(pr, ["pitcher_id"], slate_date)
    pr_keep = [c for c in pr_latest.columns if c == pr_pitcher_id or "roll" in c]
    pr_use = pr_latest[pr_keep].copy()

    rename_map = {}
    for c in pr_use.columns:
        if c == pr_pitcher_id:
            rename_map[c] = "opp_pitcher_id"
        else:
            rename_map[c] = f"opp_{c}"
    pr_use = pr_use.rename(columns=rename_map)
    pr_use["opp_pitcher_id"] = _to_int64_nullable(pr_use["opp_pitcher_id"])
    df["opp_pitcher_id"] = _to_int64_nullable(df["opp_pitcher_id"])

    df = df.merge(pr_use, on="opp_pitcher_id", how="left")

    team_context = _build_team_context(lu, br, slate_date)
    team_context_team = _pick_col(team_context, ["team", "team_abbr", "batting_team"], required=True)

    team_context = team_context.add_prefix("team_ctx_").rename(
        columns={
            "team_ctx_game_pk": lu_game_pk,
            "team_ctx_game_date": lu_game_date,
            f"team_ctx_{team_context_team}": lu_team_col,
        }
    )

    df = df.merge(team_context, on=[lu_game_pk, lu_game_date, lu_team_col], how="left")

    # Hitter-side lineup weighting
    for c in ["rbi_roll15", "rbi_roll30", "tb_roll15", "tb_roll30", "hardhit_rate_roll15", "barrel_rate_roll15", "bb_rate_roll15"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce") * df["lineup_weight"]

    # Batter vs pitcher interaction
    if "rbi_roll15" in df.columns and "opp_bb_rate_roll15" in df.columns:
        df["rbi_walk_pressure_roll15"] = (
            pd.to_numeric(df["rbi_roll15"], errors="coerce")
            * pd.to_numeric(df["opp_bb_rate_roll15"], errors="coerce")
        )

    if "tb_roll15" in df.columns and "opp_hr_allowed_roll15" in df.columns:
        df["rbi_power_pressure_roll15"] = (
            pd.to_numeric(df["tb_roll15"], errors="coerce")
            * pd.to_numeric(df["opp_hr_allowed_roll15"], errors="coerce")
        )

    if "hardhit_rate_roll15" in df.columns and "opp_hardhit_rate_allowed_roll15" in df.columns:
        df["rbi_contact_pressure_roll15"] = (
            pd.to_numeric(df["hardhit_rate_roll15"], errors="coerce")
            * pd.to_numeric(df["opp_hardhit_rate_allowed_roll15"], errors="coerce")
        )

    if "team_ctx_bb_rate_roll15" in df.columns and "opp_bb_rate_roll15" in df.columns:
        df["rbi_team_onbase_pressure"] = (
            pd.to_numeric(df["team_ctx_bb_rate_roll15"], errors="coerce")
            - pd.to_numeric(df["opp_bb_rate_roll15"], errors="coerce")
        )

    df["tie_break_noise"] = [
        _stable_noise(df.at[i, lu_name], df.at[i, lu_team_col], df.at[i, "opponent"], scale=0.008)
        for i in df.index
    ]

    return df