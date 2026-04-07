from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)
PUNCT_RE = re.compile(r"[^a-z0-9\s]")
SPACE_RE = re.compile(r"\s+")


def normalize_name(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None

    s = str(value).strip().lower()
    if not s:
        return None

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = SUFFIX_RE.sub("", s)
    s = PUNCT_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()

    return s or None


def normalize_last_name(value: object) -> str | None:
    s = normalize_name(value)
    if not s:
        return None
    parts = s.split()
    if not parts:
        return None
    return parts[-1]


def normalize_initial_last(value: object) -> str | None:
    s = normalize_name(value)
    if not s:
        return None
    parts = s.split()
    if len(parts) < 2:
        return None
    first = parts[0]
    last = parts[-1]
    if not first or not last:
        return None
    return f"{first[0]} {last}"


def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def _collect_history_frames(processed_dir: Path) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []

    candidates = [
        processed_dir / "batter_game_statcast.parquet",
        processed_dir / "pitcher_game_statcast.parquet",
        processed_dir / "pa.parquet",
        processed_dir / "by_season" / "pa_2025.parquet",
        processed_dir / "by_season" / "pa_2026.parquet",
    ]

    for path in candidates:
        df = _read_if_exists(path)
        if df is not None and not df.empty:
            frames.append(df)

    return frames


def _team_candidates(df: pd.DataFrame) -> list[str]:
    return [c for c in ["team", "team_abbr", "batting_team", "pitching_team"] if c in df.columns]


def _build_lookup_from_history(
    frames: Iterable[pd.DataFrame],
    entity: str,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    if entity == "batter":
        id_candidates = ["batter_id", "player_id"]
        name_candidates = ["batter_name", "player_name", "name"]
        out_id_col = "player_id"
    elif entity == "pitcher":
        id_candidates = ["pitcher_id", "player_id"]
        name_candidates = ["pitcher_name", "player_name", "name"]
        out_id_col = "pitcher_id"
    else:
        raise ValueError(f"Unsupported entity: {entity}")

    for df in frames:
        id_col = next((c for c in id_candidates if c in df.columns), None)
        name_col = next((c for c in name_candidates if c in df.columns), None)
        team_cols = _team_candidates(df)

        if id_col is None or name_col is None:
            continue

        keep = [id_col, name_col] + team_cols
        tmp = df[keep].copy()

        rename_map = {id_col: out_id_col, name_col: "player_name"}
        if team_cols:
            rename_map[team_cols[0]] = "team"
        tmp = tmp.rename(columns=rename_map)

        if "team" not in tmp.columns:
            tmp["team"] = pd.NA

        tmp[out_id_col] = pd.to_numeric(tmp[out_id_col], errors="coerce").astype("Int64")
        tmp["player_name"] = tmp["player_name"].astype("string").str.strip()
        tmp["team"] = tmp["team"].astype("string").str.upper()
        tmp = tmp.dropna(subset=[out_id_col, "player_name"]).drop_duplicates()

        if not tmp.empty:
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(
            columns=[
                "normalized_name",
                "normalized_initial_last",
                "normalized_last_name",
                "team",
                out_id_col,
                "resolution_source",
                "id_variant_count",
            ]
        )

    base = pd.concat(rows, ignore_index=True).drop_duplicates()
    base["normalized_name"] = base["player_name"].map(normalize_name)
    base["normalized_initial_last"] = base["player_name"].map(normalize_initial_last)
    base["normalized_last_name"] = base["player_name"].map(normalize_last_name)
    base = base.dropna(subset=["normalized_name", out_id_col]).copy()

    exact = (
        base.groupby(["normalized_name", "team"], dropna=False)[out_id_col]
        .agg(lambda s: sorted(pd.Series(s).dropna().astype("Int64").unique().tolist()))
        .reset_index(name="id_list")
    )
    exact["id_variant_count"] = exact["id_list"].map(len)
    exact = exact[exact["id_variant_count"] == 1].copy()
    exact[out_id_col] = exact["id_list"].map(lambda x: x[0]).astype("Int64")

    initial_last = (
        base.dropna(subset=["normalized_initial_last"])
        .groupby(["normalized_initial_last", "team"], dropna=False)[out_id_col]
        .agg(lambda s: sorted(pd.Series(s).dropna().astype("Int64").unique().tolist()))
        .reset_index(name="id_list")
    )
    initial_last["id_variant_count"] = initial_last["id_list"].map(len)
    initial_last = initial_last[initial_last["id_variant_count"] == 1].copy()
    initial_last[out_id_col] = initial_last["id_list"].map(lambda x: x[0]).astype("Int64")

    last_only = (
        base.dropna(subset=["normalized_last_name"])
        .groupby(["normalized_last_name", "team"], dropna=False)[out_id_col]
        .agg(lambda s: sorted(pd.Series(s).dropna().astype("Int64").unique().tolist()))
        .reset_index(name="id_list")
    )
    last_only["id_variant_count"] = last_only["id_list"].map(len)
    last_only = last_only[last_only["id_variant_count"] == 1].copy()
    last_only[out_id_col] = last_only["id_list"].map(lambda x: x[0]).astype("Int64")

    lookup = exact.rename(columns={"normalized_name": "lookup_key"})
    lookup["lookup_type"] = "exact_full_name"

    init_lookup = initial_last.rename(columns={"normalized_initial_last": "lookup_key"})
    init_lookup["lookup_type"] = "initial_last_team"

    last_lookup = last_only.rename(columns={"normalized_last_name": "lookup_key"})
    last_lookup["lookup_type"] = "last_name_team"

    out = pd.concat([lookup, init_lookup, last_lookup], ignore_index=True, sort=False)
    out["resolution_source"] = "historical_statcast_name_lookup"

    keep_cols = ["lookup_key", "lookup_type", "team", out_id_col, "resolution_source", "id_variant_count"]
    return out[keep_cols].drop_duplicates().reset_index(drop=True)


def build_and_save_lookups(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = _collect_history_frames(processed_dir)

    batter_lookup = _build_lookup_from_history(frames, entity="batter")
    pitcher_lookup = _build_lookup_from_history(frames, entity="pitcher")

    batter_out = processed_dir / "player_id_lookup_batters.parquet"
    pitcher_out = processed_dir / "player_id_lookup_pitchers.parquet"

    batter_lookup.to_parquet(batter_out, index=False)
    pitcher_lookup.to_parquet(pitcher_out, index=False)

    print(f"Saved batter lookup: {batter_out} rows={len(batter_lookup):,}")
    print(f"Saved pitcher lookup: {pitcher_out} rows={len(pitcher_lookup):,}")

    return batter_lookup, pitcher_lookup


def _load_or_build_batter_lookup(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "player_id_lookup_batters.parquet"
    if path.exists():
        return pd.read_parquet(path)
    batter_lookup, _ = build_and_save_lookups(processed_dir)
    return batter_lookup


def _load_or_build_pitcher_lookup(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "player_id_lookup_pitchers.parquet"
    if path.exists():
        return pd.read_parquet(path)
    _, pitcher_lookup = build_and_save_lookups(processed_dir)
    return pitcher_lookup


def _resolve_ids_generic(
    df: pd.DataFrame,
    processed_dir: Path,
    name_col: str,
    id_col: str,
    lookup_loader,
    resolution_method_col: str,
) -> pd.DataFrame:
    out = df.copy()

    if name_col not in out.columns:
        return out

    if id_col not in out.columns:
        out[id_col] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out[id_col] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")

    if "team" not in out.columns:
        out["team"] = pd.NA

    out["team"] = out["team"].astype("string").str.upper()
    out["normalized_name_full"] = out[name_col].map(normalize_name)
    out["normalized_name_initial_last"] = out[name_col].map(normalize_initial_last)
    out["normalized_name_last"] = out[name_col].map(normalize_last_name)

    lookup = lookup_loader(processed_dir)
    if lookup.empty:
        out[resolution_method_col] = "unresolved_no_lookup"
        return out

    out[resolution_method_col] = pd.Series("unresolved", index=out.index, dtype="string")

    # exact full name + team
    exact = lookup[lookup["lookup_type"] == "exact_full_name"].rename(columns={"lookup_key": "normalized_name_full"})
    merged = out.merge(exact[["normalized_name_full", "team", id_col, "resolution_source"]], on=["normalized_name_full", "team"], how="left", suffixes=("", "_lkp"))
    fill_mask = merged[id_col].isna() & merged[f"{id_col}_lkp"].notna()
    merged.loc[fill_mask, id_col] = merged.loc[fill_mask, f"{id_col}_lkp"].astype("Int64")
    merged.loc[fill_mask, resolution_method_col] = "exact_full_name_team"
    merged = merged.drop(columns=[f"{id_col}_lkp", "resolution_source"])

    # initial + last + team
    init_lkp = lookup[lookup["lookup_type"] == "initial_last_team"].rename(columns={"lookup_key": "normalized_name_initial_last"})
    merged = merged.merge(init_lkp[["normalized_name_initial_last", "team", id_col, "resolution_source"]], on=["normalized_name_initial_last", "team"], how="left", suffixes=("", "_lkp"))
    fill_mask = merged[id_col].isna() & merged[f"{id_col}_lkp"].notna()
    merged.loc[fill_mask, id_col] = merged.loc[fill_mask, f"{id_col}_lkp"].astype("Int64")
    merged.loc[fill_mask, resolution_method_col] = "initial_last_team"
    merged = merged.drop(columns=[f"{id_col}_lkp", "resolution_source"])

    # last name + team fallback
    last_lkp = lookup[lookup["lookup_type"] == "last_name_team"].rename(columns={"lookup_key": "normalized_name_last"})
    merged = merged.merge(last_lkp[["normalized_name_last", "team", id_col, "resolution_source"]], on=["normalized_name_last", "team"], how="left", suffixes=("", "_lkp"))
    fill_mask = merged[id_col].isna() & merged[f"{id_col}_lkp"].notna()
    merged.loc[fill_mask, id_col] = merged.loc[fill_mask, f"{id_col}_lkp"].astype("Int64")
    merged.loc[fill_mask, resolution_method_col] = "last_name_team"
    merged = merged.drop(columns=[f"{id_col}_lkp", "resolution_source"])

    merged = merged.drop(columns=["normalized_name_full", "normalized_name_initial_last", "normalized_name_last"])
    return merged


def resolve_lineup_player_ids(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    return _resolve_ids_generic(
        df=df,
        processed_dir=processed_dir,
        name_col="player_name",
        id_col="player_id",
        lookup_loader=_load_or_build_batter_lookup,
        resolution_method_col="player_id_resolution_method",
    )


def resolve_starting_pitcher_ids(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    return _resolve_ids_generic(
        df=df,
        processed_dir=processed_dir,
        name_col="pitcher_name",
        id_col="pitcher_id",
        lookup_loader=_load_or_build_pitcher_lookup,
        resolution_method_col="pitcher_id_resolution_method",
    )