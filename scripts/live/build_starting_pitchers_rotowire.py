#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import unicodedata
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.providers import rotowire
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


ROTOWIRE_FALLBACK_URL = "https://www.rotowire.com/baseball/daily-lineups.php"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build starting pitchers from Rotowire.")
    parser.add_argument("--season", type=str, required=True)
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
    return s or None


def _to_int64_nullable(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _load_pitcher_lookup(processed_dir: Path) -> pd.DataFrame:
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

    if "lookup_key" in df.columns and "lookup_type" in df.columns:
        base = df.copy()
        if team_col is not None:
            base[team_col] = base[team_col].map(_normalize_team)
        base[id_col] = pd.to_numeric(base[id_col], errors="coerce").astype("Int64")
        base["lookup_key"] = base["lookup_key"].map(_normalize_name)
        return base[["lookup_key", "lookup_type"] + ([team_col] if team_col else []) + [id_col]].dropna(
            subset=["lookup_key", id_col]
        )

    name_col = None
    for c in ["player_name", "name", "full_name"]:
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        raise ValueError("Pitcher lookup missing lookup_key and also missing player_name/name/full_name")

    base = df.copy()
    base["lookup_key"] = base[name_col].map(_normalize_name)
    if team_col is not None:
        base[team_col] = base[team_col].map(_normalize_team)
    base[id_col] = pd.to_numeric(base[id_col], errors="coerce").astype("Int64")

    exact = base[["lookup_key"] + ([team_col] if team_col else []) + [id_col]].copy()
    exact["lookup_type"] = "exact_full_name"

    def _initial_last(s: object) -> str | None:
        n = _normalize_name(s)
        if not n:
            return None
        parts = n.split()
        if len(parts) < 2:
            return None
        return f"{parts[0][0]} {parts[-1]}"

    def _last_name(s: object) -> str | None:
        n = _normalize_name(s)
        if not n:
            return None
        parts = n.split()
        return parts[-1] if parts else None

    initial_last = base.copy()
    initial_last["lookup_key"] = initial_last[name_col].map(_initial_last)
    initial_last["lookup_type"] = "initial_last_team"
    initial_last = initial_last[initial_last["lookup_key"].notna()]

    last_only = base.copy()
    last_only["lookup_key"] = last_only[name_col].map(_last_name)
    last_only["lookup_type"] = "last_name_team"
    last_only = last_only[last_only["lookup_key"].notna()]

    out = pd.concat(
        [
            exact[["lookup_key", "lookup_type"] + ([team_col] if team_col else []) + [id_col]],
            initial_last[["lookup_key", "lookup_type"] + ([team_col] if team_col else []) + [id_col]],
            last_only[["lookup_key", "lookup_type"] + ([team_col] if team_col else []) + [id_col]],
        ],
        ignore_index=True,
    )
    out = out.dropna(subset=["lookup_key", id_col]).drop_duplicates()
    return out


def _resolve_pitcher_ids(df: pd.DataFrame, processed_dir: Path) -> pd.DataFrame:
    out = df.copy()

    if out.empty:
        out["pitcher_id"] = pd.Series(dtype="Int64")
        out["pitcher_id_resolution_method"] = pd.Series(dtype="string")
        return out

    lookup = _load_pitcher_lookup(processed_dir)

    id_col = "pitcher_id" if "pitcher_id" in lookup.columns else "player_id"
    team_col_lookup = "team" if "team" in lookup.columns else None

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

    exact = lookup[lookup["lookup_type"] == "exact_full_name"].copy()
    exact = exact.rename(columns={"lookup_key": "pitcher_name_norm", id_col: "pitcher_id_lkp"})
    merge_keys = ["pitcher_name_norm"] + (["team"] if team_col_lookup else [])
    if team_col_lookup:
        exact = exact.rename(columns={"team": "team_norm"})
        merge_keys = ["pitcher_name_norm", "team_norm"]

    out = out.merge(
        exact[merge_keys + ["pitcher_id_lkp"]].drop_duplicates(),
        on=merge_keys,
        how="left",
    )
    mask = out["pitcher_id"].isna() & out["pitcher_id_lkp"].notna()
    out.loc[mask, "pitcher_id"] = out.loc[mask, "pitcher_id_lkp"].astype("Int64")
    out.loc[mask, "pitcher_id_resolution_method"] = "exact_full_name_team"
    out = out.drop(columns=["pitcher_id_lkp"])

    init_df = lookup[lookup["lookup_type"] == "initial_last_team"].copy()
    if not init_df.empty:
        init_df = init_df.rename(columns={"lookup_key": "pitcher_name_initial_last", id_col: "pitcher_id_lkp"})
        merge_keys = ["pitcher_name_initial_last"] + (["team"] if team_col_lookup else [])
        if team_col_lookup:
            init_df = init_df.rename(columns={"team": "team_norm"})
            merge_keys = ["pitcher_name_initial_last", "team_norm"]

        out = out.merge(
            init_df[merge_keys + ["pitcher_id_lkp"]].drop_duplicates(),
            on=merge_keys,
            how="left",
        )
        mask = out["pitcher_id"].isna() & out["pitcher_id_lkp"].notna()
        out.loc[mask, "pitcher_id"] = out.loc[mask, "pitcher_id_lkp"].astype("Int64")
        out.loc[mask, "pitcher_id_resolution_method"] = "initial_last_team"
        out = out.drop(columns=["pitcher_id_lkp"])

    last_df = lookup[lookup["lookup_type"] == "last_name_team"].copy()
    if not last_df.empty:
        last_df = last_df.rename(columns={"lookup_key": "pitcher_name_last", id_col: "pitcher_id_lkp"})
        merge_keys = ["pitcher_name_last"] + (["team"] if team_col_lookup else [])
        if team_col_lookup:
            last_df = last_df.rename(columns={"team": "team_norm"})
            merge_keys = ["pitcher_name_last", "team_norm"]

        out = out.merge(
            last_df[merge_keys + ["pitcher_id_lkp"]].drop_duplicates(),
            on=merge_keys,
            how="left",
        )
        mask = out["pitcher_id"].isna() & out["pitcher_id_lkp"].notna()
        out.loc[mask, "pitcher_id"] = out.loc[mask, "pitcher_id_lkp"].astype("Int64")
        out.loc[mask, "pitcher_id_resolution_method"] = "last_name_team"
        out = out.drop(columns=["pitcher_id_lkp"])

    out = out.drop(columns=["team_norm", "pitcher_name_norm", "pitcher_name_initial_last", "pitcher_name_last"])
    return out


def _load_schedule_for_date(raw_dir: Path, season: str, date_str: str) -> pd.DataFrame:
    schedule_path = raw_dir / "live" / f"games_schedule_{season}_{date_str}.parquet"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Missing schedule file: {schedule_path}")
    df = pd.read_parquet(schedule_path).copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    return df


def _pull_provider_df(config: dict) -> pd.DataFrame:
    rw_cfg = config.get("rotowire", {}) if isinstance(config, dict) else {}

    url = (
        rw_cfg.get("lineups_url")
        or rw_cfg.get("daily_lineups_url")
        or rw_cfg.get("confirmed_lineups_url")
        or rw_cfg.get("url")
        or ROTOWIRE_FALLBACK_URL
    )

    provider_df = rotowire.fetch_lineups(url=url)
    if provider_df is None or provider_df.empty:
        return pd.DataFrame()

    if "record_type" in provider_df.columns:
        provider_df = provider_df[provider_df["record_type"].astype("string").eq("starter")].copy()

    return provider_df.reset_index(drop=True)


def _enrich_provider_df(
    provider_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    season: str,
    target_date: str,
    processed_dir: Path,
) -> pd.DataFrame:
    if provider_df is None or provider_df.empty:
        return pd.DataFrame(
            columns=[
                "game_pk",
                "game_date",
                "season",
                "team",
                "opponent",
                "is_home",
                "pitcher_id",
                "pitcher_name",
                "throws",
                "starter_status",
                "source",
                "source_pull_ts",
                "pitcher_id_resolution_method",
                "rotowire_id",
            ]
        )

    df = provider_df.copy()

    rename_map = {}
    if "player_name" in df.columns and "pitcher_name" not in df.columns:
        rename_map["player_name"] = "pitcher_name"
    if "handedness_throw" in df.columns and "throws" not in df.columns:
        rename_map["handedness_throw"] = "throws"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "starter_status" not in df.columns:
        df["starter_status"] = "confirmed"
    if "source" not in df.columns:
        df["source"] = "rotowire"
    if "source_pull_ts" not in df.columns:
        df["source_pull_ts"] = pd.Timestamp.now(tz="UTC")
    if "pitcher_id" not in df.columns:
        df["pitcher_id"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    for col in ["team", "opponent"]:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = df[col].astype("string").str.upper()

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()
    else:
        df["game_date"] = pd.Timestamp(target_date).normalize()

    df = _resolve_pitcher_ids(df, processed_dir=processed_dir)

    sched = schedule_df.copy()
    sched["away_team"] = sched["away_team"].astype("string").str.upper()
    sched["home_team"] = sched["home_team"].astype("string").str.upper()

    away = sched[["game_pk", "game_date", "away_team", "home_team"]].copy()
    away = away.rename(columns={"away_team": "team", "home_team": "opponent"})
    away["is_home"] = False

    home = sched[["game_pk", "game_date", "home_team", "away_team"]].copy()
    home = home.rename(columns={"home_team": "team", "away_team": "opponent"})
    home["is_home"] = True

    game_map = pd.concat([away, home], ignore_index=True)

    df = df.drop(columns=[c for c in ["game_pk", "is_home"] if c in df.columns], errors="ignore")
    df = df.merge(game_map, on=["team", "opponent"], how="left", suffixes=("", "_sched"))

    if "game_date_sched" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date_sched"], errors="coerce").fillna(df["game_date"])
        df = df.drop(columns=["game_date_sched"])

    df["season"] = str(season)
    df["pitcher_id"] = pd.to_numeric(df["pitcher_id"], errors="coerce").astype("Int64")

    if "rotowire_id" not in df.columns:
        df["rotowire_id"] = pd.NA

    keep_cols = [
        "game_pk",
        "game_date",
        "season",
        "team",
        "opponent",
        "is_home",
        "pitcher_id",
        "pitcher_name",
        "throws",
        "starter_status",
        "source",
        "source_pull_ts",
        "pitcher_id_resolution_method",
        "rotowire_id",
    ]

    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df[keep_cols].copy()


def _print_summary(df: pd.DataFrame) -> None:
    print(f"Row count [starting_pitchers_out]: {len(df):,}")
    print(f"Distinct game_pk: {df['game_pk'].nunique(dropna=True) if 'game_pk' in df.columns else 0}")
    print(f"Min game_date: {df['game_date'].min() if 'game_date' in df.columns and not df.empty else None}")
    print(f"Max game_date: {df['game_date'].max() if 'game_date' in df.columns and not df.empty else None}")
    for c in ["game_pk", "team", "opponent", "pitcher_id"]:
        if c in df.columns:
            print(f"Nulls [{c}]: {int(df[c].isna().sum())}")
    if "pitcher_id_resolution_method" in df.columns:
        print(df["pitcher_id_resolution_method"].fillna("missing").value_counts(dropna=False).to_string())


def main() -> None:
    args = parse_args()

    repo_root = REPO_ROOT
    config_path = (repo_root / args.config).resolve()

    print("========== scripts/live/build_starting_pitchers_rotowire.py =========")
    print(f"repo_root: {repo_root}")
    print(f"config_path: {config_path}")

    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    raw_dir = Path(dirs["raw_dir"])
    processed_dir = Path(dirs["processed_dir"])
    live_dir = raw_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    schedule_df = _load_schedule_for_date(raw_dir=raw_dir, season=args.season, date_str=args.date)
    provider_df = _pull_provider_df(config)
    enriched_df = _enrich_provider_df(
        provider_df=provider_df,
        schedule_df=schedule_df,
        season=args.season,
        target_date=args.date,
        processed_dir=processed_dir,
    )

    out_latest = live_dir / f"starting_pitchers_{args.season}.parquet"
    out_dated = live_dir / f"starting_pitchers_{args.season}_{args.date}.parquet"
    debug_provider = live_dir / f"DEBUG_starting_pitchers_provider_{args.season}_{args.date}.parquet"
    debug_enriched = live_dir / f"DEBUG_starting_pitchers_enriched_{args.season}_{args.date}.parquet"

    provider_df.to_parquet(debug_provider, index=False)
    enriched_df.to_parquet(debug_enriched, index=False)
    enriched_df.to_parquet(out_latest, index=False)
    enriched_df.to_parquet(out_dated, index=False)

    _print_summary(enriched_df)
    print(f"Writing to: {out_latest}")
    print(f"Writing to: {out_dated}")
    print(f"starting_pitchers_out={out_dated}")
    print(f"debug_provider_out={debug_provider}")
    print(f"debug_enriched_out={debug_enriched}")


if __name__ == "__main__":
    main()
    