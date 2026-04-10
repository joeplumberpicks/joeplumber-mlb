"""
Plate appearance ingestion for Joe Plumber MLB Engine.

Purpose
-------
Normalize plate appearance / event-level records into a clean one-row-per-PA
Layer 1 ingest table.

This module is Layer 1 only:
- raw truth only
- no modeling logic
- no feature engineering
- no target creation
- no rolling stats

Design
------
This file is provider-agnostic at the normalization layer.

Public function
---------------
- build_plate_appearances(...)

Input format
------------
The builder expects a list[dict] or DataFrame with event records already pulled
from a source such as MLB StatsAPI / Statcast / custom event exports.

The normalization layer tolerates multiple possible source column names.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _as_dataframe(records: pd.DataFrame | list[dict[str, Any]] | None) -> pd.DataFrame:
    """Convert supported record inputs into a DataFrame."""
    if records is None:
        return pd.DataFrame()
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, list):
        return pd.DataFrame(records)
    raise TypeError(f"Unsupported records type: {type(records)!r}")


def _coalesce_series(df: pd.DataFrame, candidates: list[str], default: Any = None) -> pd.Series:
    """
    Return the first available column from candidates, otherwise a default-valued series.
    """
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _to_nullable_int(series: pd.Series) -> pd.Series:
    """Convert a series to pandas nullable Int64."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _to_nullable_float(series: pd.Series) -> pd.Series:
    """Convert a series to float."""
    return pd.to_numeric(series, errors="coerce")


def _to_string(series: pd.Series) -> pd.Series:
    """Convert a series to pandas string dtype."""
    return series.astype("string")


def _to_boolean(series: pd.Series, default: bool | None = None) -> pd.Series:
    """
    Convert common truthy/falsy values to pandas nullable boolean.
    """
    s = series.copy()

    if default is not None:
        s = s.fillna(default)

    true_values = {True, 1, 1.0, "1", "true", "True", "TRUE", "y", "Y", "yes", "YES"}
    false_values = {False, 0, 0.0, "0", "false", "False", "FALSE", "n", "N", "no", "NO"}

    def _map_value(x: Any) -> Any:
        if pd.isna(x):
            return pd.NA
        if x in true_values:
            return True
        if x in false_values:
            return False
        return pd.NA

    return s.map(_map_value).astype("boolean")


def _normalize_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build standard event indicator flags.

    Priority:
    1. Explicit source columns if present
    2. Fallback from normalized event_type text
    """
    event_type = _coalesce_series(df, ["event_type", "event", "events", "result", "pa_result"], default=None)
    event_type = event_type.astype("string").str.strip().str.lower()

    out = pd.DataFrame(index=df.index)

    explicit_map = {
        "is_pa": ["is_pa"],
        "is_ab": ["is_ab"],
        "is_hit": ["is_hit"],
        "is_1b": ["is_1b", "single_flag"],
        "is_2b": ["is_2b", "double_flag"],
        "is_3b": ["is_3b", "triple_flag"],
        "is_hr": ["is_hr", "home_run_flag"],
        "is_bb": ["is_bb", "walk_flag"],
        "is_hbp": ["is_hbp", "hit_by_pitch_flag"],
        "is_so": ["is_so", "strikeout_flag"],
        "is_rbi": ["is_rbi", "rbi_flag"],
        "is_sac_fly": ["is_sac_fly", "sac_fly_flag"],
        "is_reached_on_error": ["is_reached_on_error", "roe_flag", "reached_on_error_flag"],
    }

    fallback_rules = {
        "is_pa": pd.Series([True] * len(df), index=df.index),
        "is_ab": event_type.isin(
            {
                "single",
                "double",
                "triple",
                "home_run",
                "strikeout",
                "field_out",
                "force_out",
                "grounded_into_double_play",
                "fielders_choice",
                "field_error",
                "double_play",
                "triple_play",
                "lineout",
                "flyout",
                "pop_out",
            }
        ),
        "is_hit": event_type.isin({"single", "double", "triple", "home_run"}),
        "is_1b": event_type.eq("single"),
        "is_2b": event_type.eq("double"),
        "is_3b": event_type.eq("triple"),
        "is_hr": event_type.isin({"home_run", "home run"}),
        "is_bb": event_type.isin({"walk", "intent_walk", "intentional_walk"}),
        "is_hbp": event_type.isin({"hit_by_pitch"}),
        "is_so": event_type.isin({"strikeout", "strikeout_double_play"}),
        "is_rbi": _to_nullable_int(_coalesce_series(df, ["rbi", "rbi_on_play"], default=0)).fillna(0).gt(0),
        "is_sac_fly": event_type.eq("sac_fly"),
        "is_reached_on_error": event_type.isin({"field_error", "reached_on_error"}),
    }

    for out_col, source_cols in explicit_map.items():
        source_series = None
        for col in source_cols:
            if col in df.columns:
                source_series = _to_boolean(df[col])
                break

        if source_series is None:
            source_series = fallback_rules[out_col].astype("boolean")

        out[out_col] = source_series

    return out


def _empty_plate_appearances_df() -> pd.DataFrame:
    """Return an empty normalized PA DataFrame."""
    return pd.DataFrame(
        columns=[
            "game_pk",
            "game_date",
            "season",
            "inning",
            "inning_topbot",
            "batting_team",
            "fielding_team",
            "batter_id",
            "batter_name",
            "pitcher_id",
            "pitcher_name",
            "pa_index",
            "pitch_number_start",
            "pitch_number_end",
            "outs_before_pa",
            "outs_after_pa",
            "base_state_before",
            "base_state_after",
            "runs_scored_on_pa",
            "rbi",
            "event_type",
            "event_text",
            "is_pa",
            "is_ab",
            "is_hit",
            "is_1b",
            "is_2b",
            "is_3b",
            "is_hr",
            "is_bb",
            "is_hbp",
            "is_so",
            "is_rbi",
            "is_sac_fly",
            "is_reached_on_error",
            "launch_speed",
            "launch_angle",
            "hit_distance_sc",
            "bb_type",
            "hc_x",
            "hc_y",
            "source",
            "source_pull_ts",
        ]
    )


def normalize_plate_appearance_records(
    records: pd.DataFrame | list[dict[str, Any]] | None,
    source: str | None = None,
) -> pd.DataFrame:
    """
    Normalize raw event / PA records into a one-row-per-plate-appearance table.
    """
    df = _as_dataframe(records)

    if df.empty:
        return _empty_plate_appearances_df()

    flags = _normalize_event_flags(df)

    game_pk = _to_nullable_int(_coalesce_series(df, ["game_pk", "game_id"]))
    game_date = pd.to_datetime(
        _coalesce_series(df, ["game_date", "date", "official_date"]),
        errors="coerce",
    ).dt.date.astype("string")
    season = _to_nullable_int(_coalesce_series(df, ["season"]))

    inning = _to_nullable_int(_coalesce_series(df, ["inning", "inning_number"]))
    inning_half_raw = _coalesce_series(df, ["inning_topbot", "inning_half", "half_inning", "top_bottom"])
    inning_topbot = (
        inning_half_raw.astype("string").str.strip().str.upper().replace({"TOP": "TOP", "BOT": "BOT", "BOTTOM": "BOT"})
    )

    batting_team = _to_string(_coalesce_series(df, ["batting_team", "offense_team", "team_at_bat", "posteam"]))
    fielding_team = _to_string(_coalesce_series(df, ["fielding_team", "defense_team", "team_in_field"]))

    batter_id = _to_nullable_int(_coalesce_series(df, ["batter_id", "batter", "hitter_id", "player_id"]))
    batter_name = _to_string(_coalesce_series(df, ["batter_name", "batter", "hitter_name", "player_name"]))
    pitcher_id = _to_nullable_int(_coalesce_series(df, ["pitcher_id", "pitcher"]))
    pitcher_name = _to_string(_coalesce_series(df, ["pitcher_name", "pitcher"]))

    pa_index = _to_nullable_int(_coalesce_series(df, ["pa_index", "plate_appearance_index", "at_bat_number", "ab_number"]))
    pitch_number_start = _to_nullable_int(_coalesce_series(df, ["pitch_number_start", "pitch_index_start"]))
    pitch_number_end = _to_nullable_int(_coalesce_series(df, ["pitch_number_end", "pitch_index_end", "pitch_number"]))

    outs_before_pa = _to_nullable_int(_coalesce_series(df, ["outs_before_pa", "outs_when_up", "outs_before"]))
    outs_after_pa = _to_nullable_int(_coalesce_series(df, ["outs_after_pa", "outs_after"]))

    base_state_before = _to_string(_coalesce_series(df, ["base_state_before", "on_base_start", "base_state_start"]))
    base_state_after = _to_string(_coalesce_series(df, ["base_state_after", "on_base_end", "base_state_end"]))

    runs_scored_on_pa = _to_nullable_int(_coalesce_series(df, ["runs_scored_on_pa", "runs_scored", "runs_on_play"], default=0))
    rbi = _to_nullable_int(_coalesce_series(df, ["rbi", "rbi_on_play"], default=0))

    event_type = (
        _to_string(_coalesce_series(df, ["event_type", "event", "events", "result", "pa_result"]))
        .str.strip()
        .str.lower()
    )
    event_text = _to_string(_coalesce_series(df, ["event_text", "description", "play_description", "result_text"]))

    # Preserve Statcast contact-quality fields for downstream EV/LA and batted-ball features
    launch_speed = _to_nullable_float(
        _coalesce_series(
            df,
            [
                "launch_speed",
                "hit_speed",
            ],
            default=None,
        )
    )

    launch_angle = _to_nullable_float(
        _coalesce_series(
            df,
            [
                "launch_angle",
                "hit_angle",
            ],
            default=None,
        )
    )

    hit_distance_sc = _to_nullable_float(
        _coalesce_series(
            df,
            [
                "hit_distance_sc",
                "hit_distance",
            ],
            default=None,
        )
    )

    bb_type = _to_string(
        _coalesce_series(
            df,
            [
                "bb_type",
                "batted_ball_type",
            ],
            default=None,
        )
    )

    hc_x = _to_nullable_float(
        _coalesce_series(
            df,
            [
                "hc_x",
                "hit_coordinate_x",
            ],
            default=None,
        )
    )

    hc_y = _to_nullable_float(
        _coalesce_series(
            df,
            [
                "hc_y",
                "hit_coordinate_y",
            ],
            default=None,
        )
    )

    out = pd.DataFrame(
        {
            "game_pk": game_pk,
            "game_date": game_date,
            "season": season,
            "inning": inning,
            "inning_topbot": inning_topbot,
            "batting_team": batting_team,
            "fielding_team": fielding_team,
            "batter_id": batter_id,
            "batter_name": batter_name,
            "pitcher_id": pitcher_id,
            "pitcher_name": pitcher_name,
            "pa_index": pa_index,
            "pitch_number_start": pitch_number_start,
            "pitch_number_end": pitch_number_end,
            "outs_before_pa": outs_before_pa,
            "outs_after_pa": outs_after_pa,
            "base_state_before": base_state_before,
            "base_state_after": base_state_after,
            "runs_scored_on_pa": runs_scored_on_pa,
            "rbi": rbi,
            "event_type": event_type,
            "event_text": event_text,
            "is_pa": flags["is_pa"],
            "is_ab": flags["is_ab"],
            "is_hit": flags["is_hit"],
            "is_1b": flags["is_1b"],
            "is_2b": flags["is_2b"],
            "is_3b": flags["is_3b"],
            "is_hr": flags["is_hr"],
            "is_bb": flags["is_bb"],
            "is_hbp": flags["is_hbp"],
            "is_so": flags["is_so"],
            "is_rbi": flags["is_rbi"],
            "is_sac_fly": flags["is_sac_fly"],
            "is_reached_on_error": flags["is_reached_on_error"],
            "launch_speed": launch_speed,
            "launch_angle": launch_angle,
            "hit_distance_sc": hit_distance_sc,
            "bb_type": bb_type,
            "hc_x": hc_x,
            "hc_y": hc_y,
            "source": _to_string(_coalesce_series(df, ["source"], default=source if source is not None else "unknown")),
            "source_pull_ts": pd.to_datetime(
                _coalesce_series(df, ["source_pull_ts"], default=pd.Timestamp.utcnow()),
                utc=True,
                errors="coerce",
            ),
        }
    )

    missing_pa_index = out["pa_index"].isna()
    if missing_pa_index.any():
        out.loc[missing_pa_index, "pa_index"] = (
            out.loc[missing_pa_index]
            .groupby(["game_pk"], dropna=False)
            .cumcount()
            .add(1)
            .astype("Int64")
        )

    out["pa_index"] = _to_nullable_int(out["pa_index"])

    out = out.sort_values(
        ["game_date", "game_pk", "inning", "inning_topbot", "pa_index"],
        kind="stable",
    ).reset_index(drop=True)

    return out


def validate_plate_appearances(df: pd.DataFrame) -> None:
    """
    Validate normalized plate appearances output.

    Raises
    ------
    ValueError
        If required columns are missing or key constraints fail.
    """
    required_columns = {
        "game_pk",
        "game_date",
        "batter_name",
        "pitcher_name",
        "pa_index",
        "event_type",
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"plate appearances validation failed; missing required columns: {missing}")

    if df["game_pk"].isna().any():
        raise ValueError(
            f"plate appearances validation failed; null game_pk count={int(df['game_pk'].isna().sum())}"
        )

    if df["pa_index"].isna().any():
        raise ValueError(
            f"plate appearances validation failed; null pa_index count={int(df['pa_index'].isna().sum())}"
        )

    dupes = int(df.duplicated(subset=["game_pk", "pa_index"]).sum())
    if dupes:
        raise ValueError(
            "plate appearances validation failed; duplicate key count="
            f"{dupes} on ['game_pk', 'pa_index']"
        )


def summarize_plate_appearances(df: pd.DataFrame, label: str = "plate_appearances") -> None:
    """Print a compact ingest summary."""
    row_count = len(df)
    distinct_games = df["game_pk"].nunique(dropna=True) if "game_pk" in df.columns else 0
    min_date = df["game_date"].min() if "game_date" in df.columns and row_count else None
    max_date = df["game_date"].max() if "game_date" in df.columns and row_count else None

    print(f"Row count [{label}]: {row_count:,}")
    print(f"Distinct game_pk: {distinct_games:,}")
    print(f"Min game_date: {min_date}")
    print(f"Max game_date: {max_date}")

    for col in [
        "game_pk",
        "pa_index",
        "batter_name",
        "pitcher_name",
        "event_type",
        "launch_speed",
        "launch_angle",
    ]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")

    if {"game_pk", "pa_index"}.issubset(df.columns):
        print(
            "Duplicates on ['game_pk', 'pa_index']: "
            f"{int(df.duplicated(subset=['game_pk', 'pa_index']).sum()):,}"
        )


def build_plate_appearances(
    records: pd.DataFrame | list[dict[str, Any]] | None,
    source: str = "unknown",
    validate: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build normalized plate appearance table.
    """
    df = normalize_plate_appearance_records(records=records, source=source)

    if validate and not df.empty:
        validate_plate_appearances(df)

    if verbose:
        summarize_plate_appearances(df, label="plate_appearances")

    return df