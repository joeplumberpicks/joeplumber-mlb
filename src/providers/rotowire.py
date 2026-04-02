"""
Rotowire Data Provider

Purpose
-------
Handles raw data pulls and light parsing for Rotowire-backed lineup data.

This is a LOW-LEVEL provider:
- No normalization contracts
- No schema enforcement
- No feature logic
- No modeling logic

Returns raw or lightly standardized DataFrames.

Intended uses
-------------
- projected lineups
- confirmed lineups
- starting pitchers

Notes
-----
This provider is intentionally conservative and table-based.
It avoids hard-coding brittle page selectors into the rest of the repo.

Used by:
- lineups.py
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

REQUEST_TIMEOUT = 30
USER_AGENT = "joeplumber-mlb/1.0"


def _request_text(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    request_timeout: int = REQUEST_TIMEOUT,
) -> str:
    """
    Internal helper for HTTP GET requests that return text content.
    """
    response = requests.get(
        url,
        params=params,
        timeout=request_timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.text


def _standardize_rotowire_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply light provider-level cleanup to a Rotowire DataFrame.

    This is intentionally minimal:
    - strip column whitespace
    - drop fully empty rows
    - normalize obvious date columns if present
    """
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    out = out.dropna(axis=0, how="all").reset_index(drop=True)

    for col in ["Date", "date", "GameDate", "game_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.date.astype("string")

    return out


def read_html_tables(
    url: str,
    *,
    request_timeout: int = REQUEST_TIMEOUT,
    match: str | None = None,
    verbose: bool = True,
) -> list[pd.DataFrame]:
    """
    Read HTML tables from a Rotowire page.

    Parameters
    ----------
    url:
        HTML page URL.
    request_timeout:
        HTTP timeout in seconds.
    match:
        Optional string/regex passed to pandas.read_html to target a table.
    verbose:
        Whether to print a summary.

    Returns
    -------
    list[pandas.DataFrame]
        List of lightly standardized DataFrames.
    """
    html = _request_text(url, request_timeout=request_timeout)
    tables = pd.read_html(StringIO(html), match=match)

    out = [_standardize_rotowire_dataframe(df) for df in tables]

    if verbose:
        print(f"Tables found [rotowire_html]: {len(out):,}")
        for idx, df in enumerate(out):
            print(f"  Table {idx}: {len(df):,} rows")

    return out


def read_html_text(
    url: str,
    *,
    request_timeout: int = REQUEST_TIMEOUT,
) -> str:
    """
    Read raw HTML text from a Rotowire page.
    """
    return _request_text(url, request_timeout=request_timeout)


def read_csv_file(
    path: str | Path,
    *,
    verbose: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a local CSV file exported or saved from a Rotowire workflow.

    Parameters
    ----------
    path:
        Local CSV path.
    verbose:
        Whether to print a row-count summary.
    **kwargs:
        Additional keyword arguments passed to pandas.read_csv.

    Returns
    -------
    pandas.DataFrame
        Lightly standardized DataFrame.
    """
    in_path = Path(path)
    df = pd.read_csv(in_path, **kwargs)
    df = _standardize_rotowire_dataframe(df)

    if verbose:
        print(f"Row count [rotowire_file_{in_path.name}]: {len(df):,}")

    return df


def _pick(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """
    Return the first available column from candidates.
    """
    for col in candidates:
        if col in df.columns:
            return df[col]
    return pd.Series(pd.NA, index=df.index)


def extract_lineups(df: pd.DataFrame, *, lineup_status: str = "projected") -> pd.DataFrame:
    """
    Lightly map a Rotowire lineup-style table into a provider-level shape.

    This is still provider-layer logic, not final Layer 1 schema.

    Returns
    -------
    pandas.DataFrame
        Provider-level lineup DataFrame.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team",
                "opponent",
                "is_home",
                "player_name",
                "player_id",
                "batting_order",
                "handedness_bat",
                "handedness_throw",
                "position",
                "lineup_status",
                "source",
            ]
        )

    out = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                _pick(df, "Date", "date", "GameDate", "game_date"),
                errors="coerce",
            ).dt.date.astype("string"),
            "team": _pick(df, "Team", "team", "Tm", "team_abbr").astype("string"),
            "opponent": _pick(df, "Opp", "Opponent", "opponent", "opp").astype("string"),
            "is_home": _pick(df, "IsHome", "is_home", "HomeFlag", "home_flag").astype("string"),
            "player_name": _pick(df, "Player", "Name", "player_name").astype("string"),
            "player_id": pd.to_numeric(
                _pick(df, "player_id", "PlayerID", "mlbam_id", "MLBAMID"),
                errors="coerce",
            ).astype("Int64"),
            "batting_order": pd.to_numeric(
                _pick(df, "Order", "BatOrder", "LineupOrder", "batting_order"),
                errors="coerce",
            ).astype("Int64"),
            "handedness_bat": _pick(df, "Bat", "Bats", "handedness_bat").astype("string"),
            "handedness_throw": _pick(df, "Throws", "handedness_throw").astype("string"),
            "position": _pick(df, "Pos", "Position", "position").astype("string"),
            "lineup_status": pd.Series([lineup_status] * len(df), index=df.index, dtype="string"),
            "source": pd.Series(["rotowire"] * len(df), index=df.index, dtype="string"),
        }
    )

    # Normalize common home/away encodings to simple booleans where possible
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

    return out.reset_index(drop=True)


def extract_starting_pitchers(df: pd.DataFrame, *, starter_status: str = "probable") -> pd.DataFrame:
    """
    Lightly map a Rotowire probable-pitcher-style table into a provider-level shape.

    This is still provider-layer logic, not final Layer 1 schema.

    Returns
    -------
    pandas.DataFrame
        Provider-level starting pitcher DataFrame.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "team",
                "opponent",
                "is_home",
                "pitcher_name",
                "pitcher_id",
                "throws",
                "starter_status",
                "source",
            ]
        )

    out = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                _pick(df, "Date", "date", "GameDate", "game_date"),
                errors="coerce",
            ).dt.date.astype("string"),
            "team": _pick(df, "Team", "team", "Tm", "team_abbr").astype("string"),
            "opponent": _pick(df, "Opp", "Opponent", "opponent", "opp").astype("string"),
            "is_home": _pick(df, "IsHome", "is_home", "HomeFlag", "home_flag").astype("string"),
            "pitcher_name": _pick(df, "Pitcher", "Player", "Name", "pitcher_name").astype("string"),
            "pitcher_id": pd.to_numeric(
                _pick(df, "pitcher_id", "PlayerID", "mlbam_id", "MLBAMID"),
                errors="coerce",
            ).astype("Int64"),
            "throws": _pick(df, "Throws", "handedness_throw").astype("string"),
            "starter_status": pd.Series([starter_status] * len(df), index=df.index, dtype="string"),
            "source": pd.Series(["rotowire"] * len(df), index=df.index, dtype="string"),
        }
    )

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

    return out.reset_index(drop=True)


def test_connection(
    url: str,
    *,
    request_timeout: int = REQUEST_TIMEOUT,
) -> bool:
    """
    Simple health check to confirm the target Rotowire page is reachable.
    """
    try:
        response = requests.get(
            url,
            timeout=request_timeout,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        return True
    except Exception:
        return False


__all__ = [
    "read_html_tables",
    "read_html_text",
    "read_csv_file",
    "extract_lineups",
    "extract_starting_pitchers",
    "test_connection",
]
