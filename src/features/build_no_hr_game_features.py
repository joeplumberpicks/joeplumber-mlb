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


def _to_int64_nullable(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _latest_rows_by_id(
    df: pd.DataFrame,
    id_candidates: list[str],
    slate_date: pd.Timestamp,
) -> tuple[pd.DataFrame, str]:
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


def _build_team_power_features(
    lineups: pd.DataFrame,
    batter_roll: pd.DataFrame,
    slate_date: pd.Timestamp,
) -> pd.DataFrame:
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

    lu = lu.merge(
        br_latest[br_keep].copy(),
        left_on=batter_id_col,
        right_on=br_batter_id,
        how="left",
    )

    if slot_col is not None:
        lu[slot_col] = pd.to_numeric(lu[slot_col], errors="coerce")
        core = lu.loc[lu[slot_col].le(6)].copy()
    else:
        core = lu.copy()

    agg_map: dict[str, str] = {}
    for c in core.columns:
        if any(x in c for x in ["hr_roll", "tb_roll", "barrel_rate_roll", "hardhit_rate_roll", "ev_mean_roll", "k_rate_roll"]):
            agg_map[c] = "mean"

    if not agg_map:
        return (
            core.groupby([game_pk_col, game_date_col, team_col], dropna=False)
            .size()
            .reset_index(name="lineup_count")
        )

    out = (
        core.groupby([game_pk_col, game_date_col, team_col], dropna=False)
        .agg(agg_map)
        .reset_index()
    )

    out["lineup_count"] = (
        core.groupby([game_pk_col, game_date_col, team_col], dropna=False)[batter_id_col]
        .count()
        .values
    )
    return out


def build_no_hr_game_features(
    spine: pd.DataFrame,
    lineups: pd.DataFrame,
    batter_roll: pd.DataFrame,
    pitcher_roll: pd.DataFrame,
) -> pd.DataFrame:
    sp = _safe_copy(spine)
    lu = _safe_copy(lineups)
    br = _safe_copy(batter_roll)
    pr = _safe_copy(pitcher_roll)

    game_pk_col = _pick_col(sp, ["game_pk"], required=True)
    game_date_col = _pick_col(sp, ["game_date"], required=True)
    home_team_col = _pick_col(sp, ["home_team"], required=True)
    away_team_col = _pick_col(sp, ["away_team"], required=True)

    slate_date = pd.to_datetime(sp[game_date_col], errors="coerce").dropna().max()
    if pd.isna(slate_date):
        raise ValueError("Unable to determine slate_date from spine.game_date")

    home_sp_col = _pick_col(
        sp,
        ["home_sp_id", "home_starting_pitcher_id", "home_pitcher_id", "home_starter_pitcher_id"],
        required=True,
    )
    away_sp_col = _pick_col(
        sp,
        ["away_sp_id", "away_starting_pitcher_id", "away_pitcher_id", "away_starter_pitcher_id"],
        required=True,
    )

    sp[home_sp_col] = _to_int64_nullable(sp[home_sp_col])
    sp[away_sp_col] = _to_int64_nullable(sp[away_sp_col])

    pr_latest, pr_pitcher_id = _latest_rows_by_id(pr, ["pitcher_id"], slate_date)
    pr_keep = [c for c in pr_latest.columns if c == pr_pitcher_id or "roll" in c]

    home_pr = pr_latest[pr_keep].copy().rename(
        columns={
            pr_pitcher_id: home_sp_col,
            **{c: f"home_sp_{c}" for c in pr_keep if c != pr_pitcher_id},
        }
    )
    away_pr = pr_latest[pr_keep].copy().rename(
        columns={
            pr_pitcher_id: away_sp_col,
            **{c: f"away_sp_{c}" for c in pr_keep if c != pr_pitcher_id},
        }
    )

    sp = sp.merge(home_pr, on=home_sp_col, how="left")
    sp = sp.merge(away_pr, on=away_sp_col, how="left")

    team_power = _build_team_power_features(lu, br, slate_date)
    team_power_team = _pick_col(team_power, ["team", "team_abbr", "batting_team"], required=True)

    home_team_power = team_power.copy().add_prefix("home_team_").rename(
        columns={
            "home_team_game_pk": game_pk_col,
            "home_team_game_date": game_date_col,
            f"home_team_{team_power_team}": home_team_col,
        }
    )
    away_team_power = team_power.copy().add_prefix("away_team_").rename(
        columns={
            "away_team_game_pk": game_pk_col,
            "away_team_game_date": game_date_col,
            f"away_team_{team_power_team}": away_team_col,
        }
    )

    sp = sp.merge(home_team_power, on=[game_pk_col, game_date_col, home_team_col], how="left")
    sp = sp.merge(away_team_power, on=[game_pk_col, game_date_col, away_team_col], how="left")

    # Combined suppression and power-pressure features
    if "home_sp_hr_allowed_roll15" in sp.columns and "away_sp_hr_allowed_roll15" in sp.columns:
        sp["game_sp_hr_allowed_sum_roll15"] = (
            pd.to_numeric(sp["home_sp_hr_allowed_roll15"], errors="coerce").fillna(0.0)
            + pd.to_numeric(sp["away_sp_hr_allowed_roll15"], errors="coerce").fillna(0.0)
        )

    if "home_sp_barrel_rate_allowed_roll15" in sp.columns and "away_sp_barrel_rate_allowed_roll15" in sp.columns:
        sp["game_sp_barrel_allowed_sum_roll15"] = (
            pd.to_numeric(sp["home_sp_barrel_rate_allowed_roll15"], errors="coerce").fillna(0.0)
            + pd.to_numeric(sp["away_sp_barrel_rate_allowed_roll15"], errors="coerce").fillna(0.0)
        )

    if "home_sp_hardhit_rate_allowed_roll15" in sp.columns and "away_sp_hardhit_rate_allowed_roll15" in sp.columns:
        sp["game_sp_hardhit_allowed_sum_roll15"] = (
            pd.to_numeric(sp["home_sp_hardhit_rate_allowed_roll15"], errors="coerce").fillna(0.0)
            + pd.to_numeric(sp["away_sp_hardhit_rate_allowed_roll15"], errors="coerce").fillna(0.0)
        )

    if "home_sp_k_rate_roll15" in sp.columns and "away_sp_k_rate_roll15" in sp.columns:
        sp["game_sp_k_sum_roll15"] = (
            pd.to_numeric(sp["home_sp_k_rate_roll15"], errors="coerce").fillna(0.0)
            + pd.to_numeric(sp["away_sp_k_rate_roll15"], errors="coerce").fillna(0.0)
        )

    for metric in ["hr_roll15", "barrel_rate_roll15", "hardhit_rate_roll15", "tb_roll15", "ev_mean_roll15"]:
        home_c = f"home_team_{metric}"
        away_c = f"away_team_{metric}"
        if home_c in sp.columns and away_c in sp.columns:
            sp[f"game_lineup_{metric}_sum"] = (
                pd.to_numeric(sp[home_c], errors="coerce").fillna(0.0)
                + pd.to_numeric(sp[away_c], errors="coerce").fillna(0.0)
            )

    if "weather_wind_out" not in sp.columns:
        sp["weather_wind_out"] = 0.0
    if "weather_wind_in" not in sp.columns:
        sp["weather_wind_in"] = 0.0
    if "temperature_f" not in sp.columns:
        sp["temperature_f"] = pd.NA

    # Missing-data helpers
    sp["missing_home_starter"] = sp[home_sp_col].isna()
    sp["missing_away_starter"] = sp[away_sp_col].isna()

    if "home_team_lineup_count" in sp.columns:
        sp["missing_home_lineup_core"] = pd.to_numeric(sp["home_team_lineup_count"], errors="coerce").fillna(0).lt(6)
    else:
        sp["missing_home_lineup_core"] = True

    if "away_team_lineup_count" in sp.columns:
        sp["missing_away_lineup_core"] = pd.to_numeric(sp["away_team_lineup_count"], errors="coerce").fillna(0).lt(6)
    else:
        sp["missing_away_lineup_core"] = True

    sp = sp.drop_duplicates(subset=[game_pk_col]).reset_index(drop=True)
    return sp