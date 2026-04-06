from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.ingest.io import log_dataframe_summary, read_dataset
from src.providers import rotowire


TEAM_ALIASES = {
    "AZ": "ARI",
    "ARZ": "ARI",
    "ATH": "OAK",
    "CWS": "CHW",
    "CHW": "CHW",
    "KCR": "KC",
    "KAN": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "NYY": "NYY",
    "NYM": "NYM",
    "SD": "SD",
    "SDP": "SD",
    "SFG": "SF",
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
    s = TEAM_ALIASES.get(s, s)
    return s or None


def load_schedule_for_date(raw_live_dir: Path, season: int, date_str: str) -> pd.DataFrame:
    schedule_path = raw_live_dir / f"games_schedule_{season}_{date_str}.parquet"
    if not schedule_path.exists():
        raise FileNotFoundError(f"Missing schedule file: {schedule_path}")
    df = read_dataset(schedule_path)
    if df.empty:
        raise ValueError(f"Schedule file is empty: {schedule_path}")
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

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype("string")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    out["team"] = out["team"].astype("string")
    out["opponent"] = out["opponent"].astype("string")
    out["is_home"] = out["is_home"].astype("boolean")
    return out


def _coerce_provider_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "team" in out.columns:
        out["team"] = out["team"].map(normalize_team_abbr).astype("string")
    if "opponent" in out.columns:
        out["opponent"] = out["opponent"].map(normalize_team_abbr).astype("string")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.date.astype("string")

    if "is_home" in out.columns:
        mapped = (
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
        )
        out["is_home"] = mapped.astype("boolean")

    return out


def enrich_provider_rows_with_schedule(provider_df: pd.DataFrame, schedule_df: pd.DataFrame, slate_date: str) -> pd.DataFrame:
    """
    Attach game_pk / season / normalized game metadata to provider rows.
    """
    if provider_df.empty:
        return provider_df.copy()

    provider = _coerce_provider_flags(provider_df)
    provider["game_date"] = provider["game_date"].fillna(slate_date).astype("string")

    lookup = build_schedule_team_lookup(schedule_df)

    by_team_opp_home = provider.merge(
        lookup,
        on=["game_date", "team", "opponent", "is_home"],
        how="left",
        suffixes=("", "_sched"),
    )

    missing_mask = by_team_opp_home["game_pk"].isna()
    if missing_mask.any():
        fallback = provider.loc[missing_mask].drop(columns=[c for c in ["game_pk", "season"] if c in provider.columns], errors="ignore").merge(
            lookup.drop(columns=["opponent"]),
            on=["game_date", "team", "is_home"],
            how="left",
            suffixes=("", "_sched"),
        )

        by_team_opp_home.loc[missing_mask, "game_pk"] = fallback["game_pk"].values
        by_team_opp_home.loc[missing_mask, "season"] = fallback["season"].values
        if "opponent" in fallback.columns:
            by_team_opp_home.loc[missing_mask, "opponent"] = by_team_opp_home.loc[missing_mask, "opponent"].fillna(fallback["opponent"].astype("string")).values

    if "game_pk" not in by_team_opp_home.columns:
        by_team_opp_home["game_pk"] = pd.Series(pd.NA, index=by_team_opp_home.index, dtype="Int64")
    else:
        by_team_opp_home["game_pk"] = pd.to_numeric(by_team_opp_home["game_pk"], errors="coerce").astype("Int64")

    if "season" not in by_team_opp_home.columns:
        by_team_opp_home["season"] = schedule_df["season"].iloc[0]
    by_team_opp_home["season"] = pd.to_numeric(by_team_opp_home["season"], errors="coerce").astype("Int64")

    return by_team_opp_home


def pull_rotowire_tables(
    *,
    url: str,
    request_timeout: int = 30,
) -> list[pd.DataFrame]:
    if not url:
        raise ValueError("Missing Rotowire URL. Populate configs/project.yaml under rotowire:*_url")

    tables = rotowire.read_html_tables(
        url=url,
        request_timeout=request_timeout,
        verbose=True,
    )
    return tables


def combine_provider_tables(
    tables: list[pd.DataFrame],
    extractor,
    *,
    status_label: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for idx, table in enumerate(tables):
        try:
            extracted = extractor(table, lineup_status=status_label) if "lineup" in extractor.__name__ else extractor(table, starter_status=status_label)
        except Exception:
            continue

        if extracted is None or extracted.empty:
            continue

        extracted = extracted.copy()
        extracted["source_table_idx"] = idx
        frames.append(extracted)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def print_join_quality(df: pd.DataFrame, label: str) -> None:
    log_dataframe_summary(df, label=label)

    if "team" in df.columns:
        print(f"Distinct team [{label}]: {df['team'].dropna().astype('string').nunique()}")

    for col in ["game_pk", "team", "opponent", "player_id", "pitcher_id"]:
        if col in df.columns:
            print(f"Nulls [{col}]: {int(df[col].isna().sum()):,}")
