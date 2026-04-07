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

        if id_col is None or name_col is None:
            continue

        tmp = df[[id_col, name_col]].copy()
        tmp = tmp.rename(columns={id_col: out_id_col, name_col: "player_name"})
        tmp[out_id_col] = pd.to_numeric(tmp[out_id_col], errors="coerce").astype("Int64")
        tmp["player_name"] = tmp["player_name"].astype("string").str.strip()
        tmp = tmp.dropna(subset=[out_id_col, "player_name"]).drop_duplicates()

        if not tmp.empty:
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["normalized_name", out_id_col, "resolution_source", "id_variant_count"])

    base = pd.concat(rows, ignore_index=True).drop_duplicates()
    base["normalized_name"] = base["player_name"].map(normalize_name)
    base = base.dropna(subset=["normalized_name", out_id_col]).copy()

    grouped = (
        base.groupby("normalized_name", dropna=False)[out_id_col]
        .agg(lambda s: sorted(pd.Series(s).dropna().astype("Int64").unique().tolist()))
        .reset_index(name="id_list")
    )

    grouped["id_variant_count"] = grouped["id_list"].map(len)
    grouped = grouped[grouped["id_variant_count"] == 1].copy()
    grouped[out_id_col] = grouped["id_list"].map(lambda x: x[0]).astype("Int64")
    grouped["resolution_source"] = "historical_statcast_name_lookup"

    return grouped[["normalized_name", out_id_col, "resolution_source", "id_variant_count"]].reset_index(drop=True)


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


def resolve_lineup_player_ids(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    out = df.copy()

    if "player_name" not in out.columns:
        return out

    if "player_id" not in out.columns:
        out["player_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("Int64")

    lookup = _load_or_build_batter_lookup(processed_dir)
    if lookup.empty:
        out["player_id_resolution_method"] = "unresolved_no_lookup"
        return out

    out["normalized_name"] = out["player_name"].map(normalize_name)
    merged = out.merge(lookup, on="normalized_name", how="left", suffixes=("", "_lkp"))

    fill_mask = merged["player_id"].isna() & merged["player_id_lkp"].notna()
    merged.loc[fill_mask, "player_id"] = merged.loc[fill_mask, "player_id_lkp"].astype("Int64")

    merged["player_id_resolution_method"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged.loc[fill_mask, "player_id_resolution_method"] = merged.loc[fill_mask, "resolution_source"].astype("string")
    merged.loc[merged["player_id"].isna(), "player_id_resolution_method"] = "unresolved"

    drop_cols = [c for c in ["player_id_lkp", "resolution_source", "id_variant_count", "normalized_name"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    return merged


def resolve_starting_pitcher_ids(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    out = df.copy()

    if "pitcher_name" not in out.columns:
        return out

    if "pitcher_id" not in out.columns:
        out["pitcher_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")

    lookup = _load_or_build_pitcher_lookup(processed_dir)
    if lookup.empty:
        out["pitcher_id_resolution_method"] = "unresolved_no_lookup"
        return out

    out["normalized_name"] = out["pitcher_name"].map(normalize_name)
    merged = out.merge(lookup, on="normalized_name", how="left", suffixes=("", "_lkp"))

    fill_mask = merged["pitcher_id"].isna() & merged["pitcher_id_lkp"].notna()
    merged.loc[fill_mask, "pitcher_id"] = merged.loc[fill_mask, "pitcher_id_lkp"].astype("Int64")

    merged["pitcher_id_resolution_method"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged.loc[fill_mask, "pitcher_id_resolution_method"] = merged.loc[fill_mask, "resolution_source"].astype("string")
    merged.loc[merged["pitcher_id"].isna(), "pitcher_id_resolution_method"] = "unresolved"

    drop_cols = [c for c in ["pitcher_id_lkp", "resolution_source", "id_variant_count", "normalized_name"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    return merged