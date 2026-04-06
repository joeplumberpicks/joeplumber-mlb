from __future__ import annotations

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


def _attach_opposing_pitcher(lineups: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    lu = _safe_copy(lineups)
    sp = _safe_copy(spine)

    game_pk_col = _pick_col(lu, ["game_pk"], required=True)
    lu_is_home_col = _pick_col(lu, ["is_home"], required=False)
    if lu_is_home_col is None:
        raise KeyError("Lineups must include is_home to attach opposing pitcher.")

    lu[lu_is_home_col] = _normalize_is_home(lu[lu_is_home_col])

    sp_game_pk = _pick_col(sp, ["game_pk"], required=True)
    home_sp_col = _pick_col(
        sp,
        [
            "home_sp_id",
            "home_starting_pitcher_id",
            "home_pitcher_id",
            "home_starter_pitcher_id",
        ],
        required=True,
    )
    away_sp_col = _pick_col(
        sp,
        [
            "away_sp_id",
            "away_starting_pitcher_id",
            "away_pitcher_id",
            "away_starter_pitcher_id",
        ],
        required=True,
    )
    home_team_col = _pick_col(sp, ["home_team"], required=False)
    away_team_col = _pick_col(sp, ["away_team"], required=False)
    weather_cols = [
        c for c in ["temperature_f", "wind_mph", "weather_wind_out", "weather_wind_in", "weather_crosswind"]
        if c in sp.columns
    ]

    sp_join = sp[
        [c for c in [sp_game_pk, home_sp_col, away_sp_col, home_team_col, away_team_col] if c is not None]
        + weather_cols
    ].copy()
    sp_join = sp_join.rename(columns={sp_game_pk: game_pk_col})

    lu = lu.merge(sp_join, on=game_pk_col, how="left")

    lu["opp_pitcher_id"] = pd.NA
    lu.loc[lu[lu_is_home_col].eq(True), "opp_pitcher_id"] = lu.loc[lu[lu_is_home_col].eq(True), away_sp_col]
    lu.loc[lu[lu_is_home_col].eq(False), "opp_pitcher_id"] = lu.loc[lu[lu_is_home_col].eq(False), home_sp_col]

    opp_col = _pick_col(lu, ["opponent"], required=False)
    team_col = _pick_col(lu, ["team", "team_abbr", "batting_team"], required=False)

    if opp_col is None:
        lu["opponent"] = pd.NA
        opp_col = "opponent"

    if away_team_col is not None and home_team_col is not None and team_col is not None:
        lu.loc[lu[lu_is_home_col].eq(True), opp_col] = lu.loc[lu[lu_is_home_col].eq(True), away_team_col]
        lu.loc[lu[lu_is_home_col].eq(False), opp_col] = lu.loc[lu[lu_is_home_col].eq(False), home_team_col]

    return lu


def _build_team_context(lineups: pd.DataFrame, batter_roll: pd.DataFrame) -> pd.DataFrame:
    lu = _safe_copy(lineups)
    br = _safe_copy(batter_roll)

    team_col = _pick_col(lu, ["team", "team_abbr", "batting_team"], required=True)
    batter_id_col = _pick_col(lu, ["player_id", "batter_id"], required=True)
    game_pk_col = _pick_col(lu, ["game_pk"], required=True)
    game_date_col = _pick_col(lu, ["game_date"], required=True)
    slot_col = _pick_col(lu, ["batting_order", "lineup_slot", "order_spot", "slot"], required=False)

    br_batter_id = _pick_col(br, ["batter_id", "player_id"], required=True)
    br_game_date = _pick_col(br, ["game_date"], required=False)

    left_on = [batter_id_col]
    right_on = [br_batter_id]
    if br_game_date is not None:
        left_on.append(game_date_col)
        right_on.append(br_game_date)

    br_keep = [c for c in br.columns if c in set(right_on) or "roll" in c]
    lu = lu.merge(
        br[br_keep].copy(),
        left_on=left_on,
        right_on=right_on,
        how="left",
    )

    if slot_col is not None:
        lu[slot_col] = pd.to_numeric(lu[slot_col], errors="coerce")
        ahead = lu.loc[lu[slot_col].le(4)].copy()
    else:
        ahead = lu.copy()

    agg_cols = {}
    for c in ahead.columns:
        if any(x in c for x in ["bb_rate_roll", "rbi_roll", "tb_roll", "hardhit_rate_roll", "barrel_rate_roll"]):
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
    """
    Build one-row-per-batter RBI feature table.
    """
    sp = _safe_copy(spine)
    lu = _attach_opposing_pitcher(lineups, sp)
    br = _safe_copy(batter_roll)
    pr = _safe_copy(pitcher_roll)

    lu_batter_id = _pick_col(lu, ["player_id", "batter_id"], required=True)
    lu_game_date = _pick_col(lu, ["game_date"], required=True)
    lu_team_col = _pick_col(lu, ["team", "team_abbr", "batting_team"], required=True)
    lu_game_pk = _pick_col(lu, ["game_pk"], required=True)

    br_batter_id = _pick_col(br, ["batter_id", "player_id"], required=True)
    br_game_date = _pick_col(br, ["game_date"], required=False)

    left_on = [lu_batter_id]
    right_on = [br_batter_id]
    if br_game_date is not None:
        left_on.append(lu_game_date)
        right_on.append(br_game_date)

    br_keep = [c for c in br.columns if c in set(right_on) or "roll" in c]
    df = lu.merge(
        br[br_keep].copy(),
        left_on=left_on,
        right_on=right_on,
        how="left",
    )

    pr_pitcher_id = _pick_col(pr, ["pitcher_id"], required=True)
    pr_game_date = _pick_col(pr, ["game_date"], required=False)

    left_on = ["opp_pitcher_id"]
    right_on = [pr_pitcher_id]
    if pr_game_date is not None:
        left_on.append(lu_game_date)
        right_on.append(pr_game_date)

    pr_keep = [c for c in pr.columns if c in set(right_on) or "roll" in c]
    pr_use = pr[pr_keep].copy()

    rename_map = {}
    for c in pr_use.columns:
        if c == pr_pitcher_id:
            rename_map[c] = "opp_pitcher_id"
        elif c != pr_game_date:
            rename_map[c] = f"opp_{c}"
    pr_use = pr_use.rename(columns=rename_map)

    df = df.merge(
        pr_use,
        left_on=left_on,
        right_on=[rename_map.get(c, c) for c in right_on],
        how="left",
    )

    team_context = _build_team_context(lu, br)
    team_context_team = _pick_col(team_context, ["team", "team_abbr", "batting_team"], required=True)

    team_context = team_context.add_prefix("team_ctx_").rename(
        columns={
            "team_ctx_game_pk": lu_game_pk,
            "team_ctx_game_date": lu_game_date,
            f"team_ctx_{team_context_team}": lu_team_col,
        }
    )

    df = df.merge(team_context, on=[lu_game_pk, lu_game_date, lu_team_col], how="left")

    for bat_c, pit_c, out_c in [
        ("rbi_roll15", "opp_bb_rate_roll15", "rbi_walk_pressure_roll15"),
        ("tb_roll15", "opp_hr_allowed_roll15", "rbi_power_pressure_roll15"),
        ("hardhit_rate_roll15", "opp_hardhit_rate_allowed_roll15", "rbi_contact_pressure_roll15"),
    ]:
        if bat_c in df.columns and pit_c in df.columns:
            df[out_c] = (
                pd.to_numeric(df[bat_c], errors="coerce")
                + pd.to_numeric(df[pit_c], errors="coerce")
            )

    return df