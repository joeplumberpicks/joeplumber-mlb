from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ingest.io import log_dataframe_summary, read_dataset


TEAM_ALIASES = {
    "AZ": "ARI",
    "ARZ": "ARI",
    "ATH": "OAK",
    "CWS": "CHW",
    "CHW": "CHW",
    "KCR": "KC",
    "KAN": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "TAM": "TB",
    "WSH": "WSH",
    "WAS": "WSH",
}


def normalize_team_abbr(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().upper()
    return TEAM_ALIASES.get(s, s)


def load_schedule_for_date(raw_live_dir: Path, season: int, slate_date: str) -> pd.DataFrame:
    path = raw_live_dir / f"games_schedule_{season}_{slate_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing schedule file: {path}")
    df = read_dataset(path)
    if df.empty:
        raise ValueError(f"Schedule file is empty: {path}")
    return df


def build_schedule_team_lookup(schedule_df: pd.DataFrame) -> pd.DataFrame:
    away = schedule_df[["game_pk", "game_date", "season", "away_team", "home_team"]].copy()
    away["team"] = away["away_team"].map(normalize_team_abbr)
    away["opponent"] = away["home_team"].map(normalize_team_abbr)
    away["is_home"] = False

    home = schedule_df[["game_pk", "game_date", "season", "away_team", "home_team"]].copy()
    home["team"] = home["home_team"].map(normalize_team_abbr)
    home["opponent"] = home["away_team"].map(normalize_team_abbr)
    home["is_home"] = True

    out = pd.concat(
        [
            away[["game_pk", "game_date", "season", "team", "opponent", "is_home"]],
            home[["game_pk", "game_date", "season", "team", "opponent", "is_home"]],
        ],
        ignore_index=True,
    )

    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype("string")
    out["team"] = out["team"].astype("string")
    out["opponent"] = out["opponent"].astype("string")
    out["is_home"] = out["is_home"].astype("boolean")
    return out


def prep_provider_frame(df: pd.DataFrame, slate_date: str) -> pd.DataFrame:
    out = df.copy()

    if "team" in out.columns:
        out["team"] = out["team"].map(normalize_team_abbr).astype("string")
    if "opponent" in out.columns:
        out["opponent"] = out["opponent"].map(normalize_team_abbr).astype("string")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype("string")
        out["game_date"] = out["game_date"].fillna(slate_date)
    else:
        out["game_date"] = pd.Series([slate_date] * len(out), dtype="string")

    if "is_home" in out.columns:
        out["is_home"] = (
            out["is_home"]
            .astype("string")
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

    return out


def enrich_with_schedule(
    provider_df: pd.DataFrame,
    schedule_df: pd.DataFrame,
    slate_date: str,
) -> pd.DataFrame:
    if provider_df is None or provider_df.empty:
        return pd.DataFrame()

    provider = prep_provider_frame(provider_df, slate_date)
    lookup = build_schedule_team_lookup(schedule_df)

    # strongest join: date + team + opp + is_home
    if {"team", "opponent", "is_home"}.issubset(provider.columns):
        merged = provider.merge(
            lookup,
            on=["game_date", "team", "opponent", "is_home"],
            how="left",
            suffixes=("", "_sched"),
        )
    # fallback: date + team + opp
    elif {"team", "opponent"}.issubset(provider.columns):
        merged = provider.merge(
            lookup.drop(columns=["is_home"]),
            on=["game_date", "team", "opponent"],
            how="left",
            suffixes=("", "_sched"),
        )
    # weakest fallback: date + team
    else:
        merged = provider.merge(
            lookup.drop(columns=["opponent", "is_home"]),
            on=["game_date", "team"],
            how="left",
            suffixes=("", "_sched"),
        )

    if "game_pk" not in merged.columns:
        merged["game_pk"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    else:
        merged["game_pk"] = pd.to_numeric(merged["game_pk"], errors="coerce").astype("Int64")

    if "season" not in merged.columns:
        merged["season"] = pd.Series(schedule_df["season"].iloc[0], index=merged.index, dtype="Int64")
    else:
        merged["season"] = pd.to_numeric(merged["season"], errors="coerce").astype("Int64")

    if "is_home" not in merged.columns:
        merged["is_home"] = pd.Series(pd.NA, index=merged.index, dtype="boolean")
    else:
        merged["is_home"] = merged["is_home"].astype("boolean")

    return merged


def print_quality(df: pd.DataFrame, label: str) -> None:
    log_dataframe_summary(df, label=label)
    for col in ["game_pk", "team", "opponent", "player_id", "pitcher_id"]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")
