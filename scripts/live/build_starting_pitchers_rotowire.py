#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.ingest.io import log_kv, log_section, write_parquet
from src.ingest.lineups import build_starting_pitchers
from src.providers import rotowire
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs

from scripts.live._live_lineup_helpers import (
    enrich_with_schedule,
    load_schedule_for_date,
    print_quality,
)

ROTOWIRE_FALLBACK_URL = "https://www.rotowire.com/baseball/daily-lineups.php"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build starting pitchers from Rotowire.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _normalize_name(name: object) -> str | None:
    if name is None or pd.isna(name):
        return None
    s = str(name).strip()
    if not s:
        return None
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _normalize_team(team: object) -> str | None:
    if team is None or pd.isna(team):
        return None
    s = str(team).strip().upper()
    if not s:
        return None
    aliases = {
        "AZ": "ARI",
        "ARZ": "ARI",
        "ATH": "OAK",
        "CWS": "CHW",
        "KCR": "KC",
        "KAN": "KC",
        "SDP": "SD",
        "SFG": "SF",
        "TBR": "TB",
        "TAM": "TB",
        "WAS": "WSH",
    }
    return aliases.get(s, s)


def _build_pitcher_lookup(processed_dir: Path) -> pd.DataFrame:
    lookup_path = processed_dir / "player_id_lookup_pitchers.parquet"
    if not lookup_path.exists():
        raise FileNotFoundError(f"Missing pitcher lookup: {lookup_path}")

    df = pd.read_parquet(lookup_path).copy()

    id_col = None
    for c in ["pitcher_id", "player_id"]:
        if c in df.columns:
            id_col = c
            break
    if id_col is None:
        raise ValueError("Pitcher lookup missing pitcher_id/player_id column")

    team_col = "team" if "team" in df.columns else None

    # Preferred shape: lookup_key + lookup_type
    if "lookup_key" in df.columns and "lookup_type" in df.columns:
        out = df.copy()
        out["lookup_key"] = out["lookup_key"].map(_normalize_name)
        out[id_col] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
        if team_col is not None:
            out[team_col] = out[team_col].map(_normalize_team)
        keep = ["lookup_key", "lookup_type"] + ([team_col] if team_col else []) + [id_col]
        return out[keep].dropna(subset=["lookup_key", id_col]).drop_duplicates()

    # Fallback: build lookup from player_name
    name_col = None
    for c in ["player_name", "name", "full_name"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        raise ValueError("Pitcher lookup missing lookup_key and player_name/name/full_name")

    out = df.copy()
    out["name_norm"] = out[name_col].map(_normalize_name)
    out[id_col] = pd.to_numeric(out[id_col], errors="coerce").astype("Int64")
    if team_col is not None:
        out[team_col] = out[team_col].map(_normalize_team)

    exact = out.copy()
    exact["lookup_key"] = exact["name_norm"]
    exact["lookup_type"] = "exact_full_name"

    init = out.copy()
    init["lookup_key"] = init["name_norm"].map(
        lambda s: f"{s.split()[0][0]} {s.split()[-1]}" if isinstance(s, str) and len(s.split()) >= 2 else None
    )
    init["lookup_type"] = "initial_last_team"

    last = out.copy()
    last["lookup_key"] = last["name_norm"].map(
        lambda s: s.split()[-1] if isinstance(s, str) and len(s.split()) >= 1 else None
    )
    last["lookup_type"] = "last_name_team"

    built = pd.concat([exact, init, last], ignore_index=True)
    keep = ["lookup_key", "lookup_type"] + ([team_col] if team_col else []) + [id_col]
    return built[keep].dropna(subset=["lookup_key", id_col]).drop_duplicates()


def _resolve_pitcher_ids(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    if df.empty:
        df = df.copy()
        df["pitcher_id_resolution_method"] = pd.Series(dtype="string")
        return df

    lookup = _build_pitcher_lookup(processed_dir)
    id_col = "pitcher_id" if "pitcher_id" in lookup.columns else "player_id"
    has_team = "team" in lookup.columns

    out = df.copy()
    out["team_norm"] = out["team"].map(_normalize_team)
    out["pitcher_name_norm"] = out["pitcher_name"].map(_normalize_name)
    out["pitcher_name_initial_last"] = out["pitcher_name_norm"].map(
        lambda s: f"{s.split()[0][0]} {s.split()[-1]}" if isinstance(s, str) and len(s.split()) >= 2 else None
    )
    out["pitcher_name_last"] = out["pitcher_name_norm"].map(
        lambda s: s.split()[-1] if isinstance(s, str) and len(s.split()) >= 1 else None
    )

    if "pitcher_id" not in out.columns:
        out["pitcher_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["pitcher_id"] = pd.to_numeric(out["pitcher_id"], errors="coerce").astype("Int64")

    out["pitcher_id_resolution_method"] = pd.Series("unresolved", index=out.index, dtype="string")

    # exact full name + team
    exact = lookup[lookup["lookup_type"] == "exact_full_name"].copy()
    exact = exact.rename(columns={"lookup_key": "pitcher_name_norm", id_col: "pitcher_id_lkp"})
    if has_team:
        exact = exact.rename(columns={"team": "team_norm"})
        merge_keys = ["pitcher_name_norm", "team_norm"]
    else:
        merge_keys = ["pitcher_name_norm"]

    out = out.merge(exact[merge_keys + ["pitcher_id_lkp"]].drop_duplicates(), on=merge_keys, how="left")
    mask = out["pitcher_id"].isna() & out["pitcher_id_lkp"].notna()
    out.loc[mask, "pitcher_id"] = out.loc[mask, "pitcher_id_lkp"].astype("Int64")
    out.loc[mask, "pitcher_id_resolution_method"] = "exact_full_name_team"
    out = out.drop(columns=["pitcher_id_lkp"])

    # initial + last + team
    init = lookup[lookup["lookup_type"] == "initial_last_team"].copy()
    if not init.empty:
        init = init.rename(columns={"lookup_key": "pitcher_name_initial_last", id_col: "pitcher_id_lkp"})
        if has_team:
            init = init.rename(columns={"team": "team_norm"})
            merge_keys = ["pitcher_name_initial_last", "team_norm"]
        else:
            merge_keys = ["pitcher_name_initial_last"]

        out = out.merge(init[merge_keys + ["pitcher_id_lkp"]].drop_duplicates(), on=merge_keys, how="left")
        mask = out["pitcher_id"].isna() & out["pitcher_id_lkp"].notna()
        out.loc[mask, "pitcher_id"] = out.loc[mask, "pitcher_id_lkp"].astype("Int64")
        out.loc[mask, "pitcher_id_resolution_method"] = "initial_last_team"
        out = out.drop(columns=["pitcher_id_lkp"])

    # last name + team
    last = lookup[lookup["lookup_type"] == "last_name_team"].copy()
    if not last.empty:
        last = last.rename(columns={"lookup_key": "pitcher_name_last", id_col: "pitcher_id_lkp"})
        if has_team:
            last = last.rename(columns={"team": "team_norm"})
            merge_keys = ["pitcher_name_last", "team_norm"]
        else:
            merge_keys = ["pitcher_name_last"]

        out = out.merge(last[merge_keys + ["pitcher_id_lkp"]].drop_duplicates(), on=merge_keys, how="left")
        mask = out["pitcher_id"].isna() & out["pitcher_id_lkp"].notna()
        out.loc[mask, "pitcher_id"] = out.loc[mask, "pitcher_id_lkp"].astype("Int64")
        out.loc[mask, "pitcher_id_resolution_method"] = "last_name_team"
        out = out.drop(columns=["pitcher_id_lkp"])

    return out.drop(columns=["team_norm", "pitcher_name_norm", "pitcher_name_initial_last", "pitcher_name_last"])


def _pull_starters(config: dict) -> pd.DataFrame:
    rw_cfg = config.get("rotowire", {})

    url = (
        str(rw_cfg.get("starting_pitchers_url", "")).strip()
        or str(rw_cfg.get("confirmed_lineups_url", "")).strip()
        or str(rw_cfg.get("lineups_url", "")).strip()
        or str(rw_cfg.get("daily_lineups_url", "")).strip()
        or ROTOWIRE_FALLBACK_URL
    )

    request_timeout = int(rw_cfg.get("request_timeout", 30))

    tables = rotowire.read_html_tables(
        url=url,
        request_timeout=request_timeout,
        verbose=True,
    )

    frames = []
    for idx, table in enumerate(tables):
        try:
            extracted = rotowire.extract_starting_pitchers(table, starter_status="probable")
        except Exception:
            continue

        if extracted is None or extracted.empty:
            continue

        extracted = extracted.copy()
        extracted["source_table_idx"] = idx
        frames.append(extracted)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config_path = (repo_root / args.config).resolve()

    log_section("scripts/live/build_starting_pitchers_rotowire.py")
    log_kv("repo_root", repo_root)
    log_kv("config_path", config_path)

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    raw_live_dir = Path(dirs["raw_dir"]) / "live"
    processed_dir = Path(dirs["processed_dir"])
    schedule_df = load_schedule_for_date(raw_live_dir, args.season, args.date)

    provider_df = _pull_starters(config)
    provider_df = enrich_with_schedule(provider_df, schedule_df, args.date)

    if not provider_df.empty:
        provider_df = _resolve_pitcher_ids(provider_df, processed_dir)

    out_df = build_starting_pitchers(
        records=provider_df,
        starter_status="probable",
        source="rotowire",
        validate=True,
        verbose=True,
    )

    print_quality(out_df, "starting_pitchers_out")

    if "pitcher_id_resolution_method" in provider_df.columns:
        print(provider_df["pitcher_id_resolution_method"].fillna("missing").value_counts(dropna=False).to_string())

    latest_out = raw_live_dir / f"starting_pitchers_{args.season}.parquet"
    dated_out = raw_live_dir / f"starting_pitchers_{args.season}_{args.date}.parquet"
    debug_provider = raw_live_dir / f"DEBUG_starting_pitchers_provider_{args.season}_{args.date}.parquet"
    debug_enriched = raw_live_dir / f"DEBUG_starting_pitchers_enriched_{args.season}_{args.date}.parquet"

    provider_df.to_parquet(debug_provider, index=False)
    out_df.to_parquet(debug_enriched, index=False)

    write_parquet(out_df, latest_out)
    write_parquet(out_df, dated_out)

    print(f"starting_pitchers_out={dated_out}")
    print(f"debug_provider_out={debug_provider}")
    print(f"debug_enriched_out={debug_enriched}")


if __name__ == "__main__":
    main()