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


def _build_team_lineup_features(lineups: pd.DataFrame, batter_roll: pd.DataFrame) -> pd.DataFrame:
    lu = _safe_copy(lineups)
    br = _safe_copy(batter_roll)

    team_col = _pick_col(lu, ["team", "team_abbr", "batting_team"], required=True)
    batter_id_col = _pick_col(lu, ["batter_id", "player_id"], required=True)
    game_pk_col = _pick_col(lu, ["game_pk"], required=True)
    game_date_col = _pick_col(lu, ["game_date"], required=True)

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

    agg_map = {}
    for c in lu.columns:
        if "roll" in c:
            agg_map[c] = "mean"

    if not agg_map:
        out = (
            lu.groupby([game_pk_col, game_date_col, team_col], dropna=False)
            .size()
            .reset_index(name="lineup_count")
        )
        return out

    out = (
        lu.groupby([game_pk_col, game_date_col, team_col], dropna=False)
        .agg(agg_map)
        .reset_index()
    )
    out["lineup_count"] = (
        lu.groupby([game_pk_col, game_date_col, team_col], dropna=False)[batter_id_col]
        .count()
        .values
    )
    return out


def build_moneyline_features(
    spine: pd.DataFrame,
    lineups: pd.DataFrame,
    batter_roll: pd.DataFrame,
    pitcher_roll: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one-row-per-game Moneyline feature table.
    """
    sp = _safe_copy(spine)
    lu = _safe_copy(lineups)
    br = _safe_copy(batter_roll)
    pr = _safe_copy(pitcher_roll)

    game_pk_col = _pick_col(sp, ["game_pk"], required=True)
    game_date_col = _pick_col(sp, ["game_date"], required=True)
    home_team_col = _pick_col(sp, ["home_team"], required=True)
    away_team_col = _pick_col(sp, ["away_team"], required=True)

    home_sp_col = _pick_col(
        sp,
        ["home_sp_id", "home_starting_pitcher_id", "home_pitcher_id"],
        required=True,
    )
    away_sp_col = _pick_col(
        sp,
        ["away_sp_id", "away_starting_pitcher_id", "away_pitcher_id"],
        required=True,
    )

    pr_pitcher_id = _pick_col(pr, ["pitcher_id"], required=True)

    pr_keep = [c for c in pr.columns if c == pr_pitcher_id or "roll" in c]
    pr_use = pr[pr_keep].copy()

    # home SP rolling features
    home_pr = pr_use.rename(
        columns={
            pr_pitcher_id: home_sp_col,
            **{c: f"home_sp_{c}" for c in pr_use.columns if c != pr_pitcher_id},
        }
    )
    sp = sp.merge(home_pr, on=home_sp_col, how="left")

    # away SP rolling features
    away_pr = pr_use.rename(
        columns={
            pr_pitcher_id: away_sp_col,
            **{c: f"away_sp_{c}" for c in pr_use.columns if c != pr_pitcher_id},
        }
    )
    sp = sp.merge(away_pr, on=away_sp_col, how="left")

    # team offensive form from full projected/confirmed lineup
    team_form = _build_team_lineup_features(lu, br)

    team_form_team_col = _pick_col(team_form, ["team", "team_abbr", "batting_team"], required=True)

    home_team_form = team_form.copy().add_prefix("home_team_").rename(
        columns={
            "home_team_game_pk": game_pk_col,
            "home_team_game_date": game_date_col,
            f"home_team_{team_form_team_col}": home_team_col,
        }
    )

    away_team_form = team_form.copy().add_prefix("away_team_").rename(
        columns={
            "away_team_game_pk": game_pk_col,
            "away_team_game_date": game_date_col,
            f"away_team_{team_form_team_col}": away_team_col,
        }
    )

    sp = sp.merge(home_team_form, on=[game_pk_col, game_date_col, home_team_col], how="left")
    sp = sp.merge(away_team_form, on=[game_pk_col, game_date_col, away_team_col], how="left")

    # convenience deltas
    for metric in [
        "k_rate_roll7",
        "bb_rate_roll7",
        "hr_rate_roll7",
        "runs_rate_roll7",
        "barrel_rate_roll7",
        "hardhit_rate_roll7",
        "ev_mean_roll7",
    ]:
        home_c = f"home_sp_{metric}"
        away_c = f"away_sp_{metric}"
        if home_c in sp.columns and away_c in sp.columns:
            sp[f"sp_diff_{metric}"] = pd.to_numeric(sp[home_c], errors="coerce") - pd.to_numeric(
                sp[away_c], errors="coerce"
            )

    for metric in [
        "rbi_roll7",
        "tb_roll7",
        "barrel_rate_roll7",
        "hardhit_rate_roll7",
        "ev_mean_roll7",
        "bb_rate_roll7",
        "k_rate_roll7",
    ]:
        home_c = f"home_team_{metric}"
        away_c = f"away_team_{metric}"
        if home_c in sp.columns and away_c in sp.columns:
            sp[f"off_diff_{metric}"] = pd.to_numeric(sp[home_c], errors="coerce") - pd.to_numeric(
                sp[away_c], errors="coerce"
            )

    return sp