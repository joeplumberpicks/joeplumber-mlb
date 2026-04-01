"""
FanGraphs Data Provider

Purpose
-------
Handles raw data pulls and light parsing for FanGraphs-backed data sources.

This is a LOW-LEVEL provider:
- No normalization contracts
- No schema enforcement
- No feature logic
- No modeling logic

Returns raw or lightly standardized DataFrames.

Intended uses
-------------
- projected lineups
- leaderboard exports
- park factor exports
- batter/pitcher tables saved from FanGraphs

Notes
-----
FanGraphs does not provide a stable public API in the same style as MLB StatsAPI.
This provider is designed to work with:
- direct CSV export URLs
- saved CSV files
- HTML tables when needed

Used by:
- lineups.py
- parks.py
- future batter/pitcher reference ingests
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


def _standardize_fangraphs_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply light provider-level cleanup to a FanGraphs DataFrame.

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


def read_csv_export(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    request_timeout: int = REQUEST_TIMEOUT,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read a FanGraphs CSV export URL into a DataFrame.

    Parameters
    ----------
    url:
        CSV export URL.
    params:
        Optional query parameters.
    request_timeout:
        HTTP timeout in seconds.
    verbose:
        Whether to print a row-count summary.

    Returns
    -------
    pandas.DataFrame
        Lightly standardized DataFrame.
    """
    text = _request_text(url, params=params, request_timeout=request_timeout)
    df = pd.read_csv(StringIO(text))
    df = _standardize_fangraphs_dataframe(df)

    if verbose:
        print(f"Row count [fangraphs_csv]: {len(df):,}")

    return df


def read_html_tables(
    url: str,
    *,
    request_timeout: int = REQUEST_TIMEOUT,
    match: str | None = None,
    verbose: bool = True,
) -> list[pd.DataFrame]:
    """
    Read HTML tables from a FanGraphs page.

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

    out = [_standardize_fangraphs_dataframe(df) for df in tables]

    if verbose:
        print(f"Tables found [fangraphs_html]: {len(out):,}")
        for idx, df in enumerate(out):
            print(f"  Table {idx}: {len(df):,} rows")

    return out


def read_csv_file(
    path: str | Path,
    *,
    verbose: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read a local CSV file exported from FanGraphs.

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
    df = _standardize_fangraphs_dataframe(df)

    if verbose:
        print(f"Row count [fangraphs_file_{in_path.name}]: {len(df):,}")

    return df


def extract_projected_lineups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightly map a FanGraphs projected-lineups-style table into a provider-level shape.

    This is still provider-layer logic, not final Layer 1 schema.

    Expected-ish source columns may include:
    - Team
    - Opp
    - Player
    - Bat
    - Pos
    - Order
    - Date

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
                "player_name",
                "batting_order",
                "handedness_bat",
                "position",
                "source",
            ]
        )

    def _pick(*candidates: str) -> pd.Series:
        for col in candidates:
            if col in df.columns:
                return df[col]
        return pd.Series(pd.NA, index=df.index)

    out = pd.DataFrame(
        {
            "game_date": pd.to_datetime(
                _pick("Date", "date", "GameDate", "game_date"),
                errors="coerce",
            ).dt.date.astype("string"),
            "team": _pick("Team", "team", "Tm").astype("string"),
            "opponent": _pick("Opp", "Opponent", "opponent").astype("string"),
            "player_name": _pick("Player", "Name", "player_name").astype("string"),
            "batting_order": pd.to_numeric(
                _pick("Order", "BatOrder", "batting_order", "LineupOrder"),
                errors="coerce",
            ).astype("Int64"),
            "handedness_bat": _pick("Bat", "Bats", "handedness_bat").astype("string"),
            "position": _pick("Pos", "Position", "position").astype("string"),
            "source": pd.Series(["fangraphs"] * len(df), index=df.index, dtype="string"),
        }
    )

    return out.reset_index(drop=True)


def extract_park_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightly map a FanGraphs park-factor-style table into a provider-level shape.

    This is still provider-layer logic, not final Layer 1 schema.

    Returns
    -------
    pandas.DataFrame
        Provider-level park factor DataFrame.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "venue_name",
                "park_factor_runs",
                "park_factor_hr",
                "park_factor_hits",
                "park_factor_2b",
                "park_factor_3b",
                "park_factor_bb",
                "source",
            ]
        )

    def _pick(*candidates: str) -> pd.Series:
        for col in candidates:
            if col in df.columns:
                return df[col]
        return pd.Series(pd.NA, index=df.index)

    out = pd.DataFrame(
        {
            "team": _pick("Team", "team", "Tm").astype("string"),
            "venue_name": _pick("Park", "venue_name", "Venue", "Stadium").astype("string"),
            "park_factor_runs": pd.to_numeric(_pick("R", "Runs", "park_factor_runs"), errors="coerce"),
            "park_factor_hr": pd.to_numeric(_pick("HR", "park_factor_hr"), errors="coerce"),
            "park_factor_hits": pd.to_numeric(_pick("H", "Hits", "park_factor_hits"), errors="coerce"),
            "park_factor_2b": pd.to_numeric(_pick("2B", "park_factor_2b"), errors="coerce"),
            "park_factor_3b": pd.to_numeric(_pick("3B", "park_factor_3b"), errors="coerce"),
            "park_factor_bb": pd.to_numeric(_pick("BB", "park_factor_bb"), errors="coerce"),
            "source": pd.Series(["fangraphs"] * len(df), index=df.index, dtype="string"),
        }
    )

    return out.reset_index(drop=True)


def test_connection(
    url: str = "https://www.fangraphs.com",
    *,
    request_timeout: int = REQUEST_TIMEOUT,
) -> bool:
    """
    Simple health check to confirm FanGraphs is reachable.
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
    "read_csv_export",
    "read_html_tables",
    "read_csv_file",
    "extract_projected_lineups",
    "extract_park_factors",
    "test_connection",
]
