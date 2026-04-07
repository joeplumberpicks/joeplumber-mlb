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


def _build_top3_team_features(
    lineups: pd.DataFrame,
    batter_roll: pd.DataFrame,
    slate_date: pd.Timestamp,
) -> pd.DataFrame:
    lu = _safe_copy(lineups)
    br = _safe_copy(batter_roll)

    team_col = _pick_col(lu, ["team", "team_abbr", "batting_team"], required=True)
    slot_col = _pick_col(lu, ["batting_order", "lineup_slot", "order_spot", "slot"])
    batter_id_col = _pick_col(lu, ["player_id", "batter_id"], required=True)
    game_pk_col = _pick_col(lu, ["game_pk"], required=True)
    game_date_col = _pick_col(lu, ["game_date"], required=True)

    lu[batter_id_col] = _to_int64_nullable(lu[batter_id_col])

    br_latest, br_batter_id = _latest_rows_by_id(br, ["batter_id", "player_id"], slate_date)

    top3 = lu.copy()
    if slot_col is not None:
        top3[slot_col] = pd.to_numeric(top3[slot_col], errors="coerce")
        top3 = top3.loc[top3[slot_col].le(3)].copy()

    br_keep = [c for c in br_latest.columns if c == br_batter_id or "roll" in c]
    top3 = top3.merge(
        br_latest[br_keep].copy(),
        left_on=batter_id_col,
        right_on=br_batter_id,
        how="left",
    )

    agg_map: dict[str, str] = {}
    for c in top3.columns:
        if "roll" in c:
            agg_map[c] = "mean"

    if not agg_map:
        return (
            top3.groupby([game_pk_col, game_date_col, team_col], dropna=False)
            .size()
            .reset_index(name="top3_count")
        )

    top3_team = (
        top3.groupby([game_pk_col, game_date_col, team_col], dropna=False)
        .agg(agg_map)
        .reset_index()
    )
    top3_team["top3_count"] = (
        top3.groupby([game_pk_col, game_date_col, team_col], dropna=False)[batter_id_col]
        .count()
        .values
    )
    return top3_team


def build_nrfi_features(
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

    team_top3 = _build_top3_team_features(lu, br, slate_date)
    team_top3_team = _pick_col(team_top3, ["team", "team_abbr", "batting_team"], required=True)

    home_top3 = team_top3.copy().add_prefix("home_top3_").rename(
        columns={
            "home_top3_game_pk": game_pk_col,
            "home_top3_game_date": game_date_col,
            f"home_top3_{team_top3_team}": home_team_col,
        }
    )
    away_top3 = team_top3.copy().add_prefix("away_top3_").rename(
        columns={
            "away_top3_game_pk": game_pk_col,
            "away_top3_game_date": game_date_col,
            f"away_top3_{team_top3_team}": away_team_col,
        }
    )

    sp = sp.merge(home_top3, on=[game_pk_col, game_date_col, home_team_col], how="left")
    sp = sp.merge(away_top3, on=[game_pk_col, game_date_col, away_team_col], how="left")

    for metric in [
        "k_rate_roll7",
        "bb_rate_roll7",
        "hr_rate_roll7",
        "barrel_rate_allowed_roll7",
        "hardhit_rate_allowed_roll7",
        "ev_mean_roll7",
    ]:
        home_c = f"home_sp_{metric}"
        away_c = f"away_sp_{metric}"
        if home_c in sp.columns and away_c in sp.columns:
            sp[f"sp_diff_{metric}"] = (
                pd.to_numeric(sp[home_c], errors="coerce")
                - pd.to_numeric(sp[away_c], errors="coerce")
            )

    sp = sp.drop_duplicates(subset=[game_pk_col]).reset_index(drop=True)
    return sp