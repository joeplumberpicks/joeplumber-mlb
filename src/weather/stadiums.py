from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "team_abbr",
    "season_start",
    "season_end",
    "park_id",
    "stadium_name",
    "city",
    "state",
    "lat",
    "lon",
    "timezone",
    "roof_type",
    "cf_bearing_deg",
]

TEAM_ALIASES = {
    "TB": "TBR",
    "TBR": "TBR",
    "OAK": "ATH",
    "ATH": "ATH",
}

# Keep fallback minimal but schema-complete. Primary source should be data/reference/mlb_stadiums.csv.
FALLBACK_STADIUMS: list[dict[str, object]] = [
    {
        "team_abbr": "ATH",
        "season_start": 2025,
        "season_end": 2026,
        "park_id": "SUTTER_HEALTH_PARK",
        "stadium_name": "Sutter Health Park",
        "city": "West Sacramento",
        "state": "CA",
        "lat": 38.5806,
        "lon": -121.5138,
        "timezone": "America/Los_Angeles",
        "roof_type": "open",
        "cf_bearing_deg": 0.0,
    },
    {
        "team_abbr": "TBR",
        "season_start": 2025,
        "season_end": 2025,
        "park_id": "STEINBRENNER_FIELD",
        "stadium_name": "George M. Steinbrenner Field",
        "city": "Tampa",
        "state": "FL",
        "lat": 27.9800,
        "lon": -82.5060,
        "timezone": "America/New_York",
        "roof_type": "open",
        "cf_bearing_deg": 0.0,
    },
    {
        "team_abbr": "TBR",
        "season_start": 2026,
        "season_end": 2099,
        "park_id": "TROPICANA_FIELD",
        "stadium_name": "Tropicana Field",
        "city": "St. Petersburg",
        "state": "FL",
        "lat": 27.7683,
        "lon": -82.6534,
        "timezone": "America/New_York",
        "roof_type": "dome",
        "cf_bearing_deg": 0.0,
    },
]


def normalize_team_abbr(team: object) -> str:
    value = str(team or "").strip().upper()
    return TEAM_ALIASES.get(value, value)


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Legacy compatibility mapping.
    rename_candidates = {
        "team": "team_abbr",
        "park": "stadium_name",
    }
    for src, dest in rename_candidates.items():
        if src in out.columns and dest not in out.columns:
            out = out.rename(columns={src: dest})

    if "park_id" not in out.columns and "stadium_name" in out.columns:
        out["park_id"] = (
            out["stadium_name"].astype("string").str.upper().str.replace(r"[^A-Z0-9]+", "_", regex=True).str.strip("_")
        )

    if "season_start" not in out.columns:
        out["season_start"] = 1900
    if "season_end" not in out.columns:
        out["season_end"] = 2099
    if "city" not in out.columns:
        out["city"] = pd.NA
    if "state" not in out.columns:
        out["state"] = pd.NA
    if "roof_type" not in out.columns:
        out["roof_type"] = "unknown"
    if "cf_bearing_deg" not in out.columns:
        out["cf_bearing_deg"] = pd.NA

    missing = set(REQUIRED_COLUMNS) - set(out.columns)
    if missing:
        raise ValueError(f"Stadium reference missing required columns {sorted(missing)}")

    out = out[REQUIRED_COLUMNS].copy()
    out["team_abbr"] = out["team_abbr"].map(normalize_team_abbr)
    out["season_start"] = pd.to_numeric(out["season_start"], errors="coerce").astype("Int64")
    out["season_end"] = pd.to_numeric(out["season_end"], errors="coerce").astype("Int64")
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out["cf_bearing_deg"] = pd.to_numeric(out["cf_bearing_deg"], errors="coerce")
    out["park_id"] = out["park_id"].astype("string").str.upper().str.strip()
    out["stadium_name"] = out["stadium_name"].astype("string").str.strip()
    out["timezone"] = out["timezone"].astype("string").str.strip()
    out["roof_type"] = out["roof_type"].astype("string").str.lower().fillna("unknown")
    return out


def load_stadium_reference(path: Path | None = None, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Load stadium coordinates and season-aware park mapping."""
    log = logger or logging.getLogger(__name__)
    ref_path = path or Path("data/reference/mlb_stadiums.csv")
    if ref_path.exists():
        return _coerce_schema(pd.read_csv(ref_path))

    log.warning("Stadium reference not found at %s; using embedded fallback table.", ref_path)
    return _coerce_schema(pd.DataFrame(FALLBACK_STADIUMS, columns=REQUIRED_COLUMNS))


def load_park_overrides(path: Path | None = None) -> pd.DataFrame:
    override_path = path or Path("data/reference/park_overrides.csv")
    if not override_path.exists():
        return pd.DataFrame(columns=["game_pk", "park_id_override", "notes"])
    overrides = pd.read_csv(override_path)
    if "game_pk" not in overrides.columns or "park_id_override" not in overrides.columns:
        raise ValueError(f"park_overrides must include columns ['game_pk','park_id_override']: {override_path}")
    if "notes" not in overrides.columns:
        overrides["notes"] = pd.NA
    overrides["game_pk"] = pd.to_numeric(overrides["game_pk"], errors="coerce").astype("Int64")
    overrides["park_id_override"] = overrides["park_id_override"].astype("string").str.upper().str.strip()
    return overrides[["game_pk", "park_id_override", "notes"]]


def _resolve_by_explicit_venue(game_row: pd.Series, stadiums_df: pd.DataFrame) -> str | None:
    for col in ["park_id", "venue_id", "venue", "venue_name", "park"]:
        if col not in game_row.index:
            continue
        value = game_row.get(col)
        if pd.isna(value) or value is None:
            continue
        text = str(value).strip()
        if not text:
            continue

        park_id_guess = text.upper().replace(" ", "_")
        if (stadiums_df["park_id"] == park_id_guess).any():
            return park_id_guess

        name_matches = stadiums_df[stadiums_df["stadium_name"].str.lower() == text.lower()]
        if not name_matches.empty:
            return str(name_matches.iloc[0]["park_id"])
    return None


def _resolve_by_team_season(
    team_abbr: str,
    season: int,
    stadiums_df: pd.DataFrame,
    logger: logging.Logger,
    game_pk: object,
) -> str | None:
    team = normalize_team_abbr(team_abbr)
    cand = stadiums_df[
        (stadiums_df["team_abbr"] == team)
        & (stadiums_df["season_start"] <= int(season))
        & (stadiums_df["season_end"] >= int(season))
    ].copy()
    if cand.empty:
        return None

    if len(cand) > 1:
        cand["span"] = cand["season_end"] - cand["season_start"]
        cand = cand.sort_values(["span", "season_start"], ascending=[True, False], kind="mergesort")
        logger.warning("park_resolution_tiebreak game_pk=%s team=%s season=%s candidates=%d", game_pk, team, season, len(cand))
    return str(cand.iloc[0]["park_id"])


def resolve_park_for_game(
    game_row: pd.Series,
    season: int,
    stadiums_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> tuple[str | None, str]:
    """Resolve park_id for one game row; returns (park_id, source)."""
    log = logger or logging.getLogger(__name__)
    game_pk = pd.to_numeric(game_row.get("game_pk"), errors="coerce")

    if "game_pk" in overrides_df.columns and not pd.isna(game_pk):
        ov = overrides_df[overrides_df["game_pk"] == int(game_pk)]
        if not ov.empty:
            return str(ov.iloc[0]["park_id_override"]), "override"

    explicit = _resolve_by_explicit_venue(game_row, stadiums_df)
    if explicit is not None:
        return explicit, "explicit_venue"

    team = game_row.get("home_team") or game_row.get("home_team_abbr") or game_row.get("team_abbr")
    if team is None or pd.isna(team):
        log.warning("park_resolution_missing_team game_pk=%s", game_pk)
        return None, "unresolved"

    mapped = _resolve_by_team_season(str(team), season, stadiums_df, log, game_pk=game_pk)
    if mapped is None:
        return None, "unresolved"
    return mapped, "team_season"


def save_default_stadium_reference(path: Path) -> Path:
    """Write fallback stadium rows to CSV for local editing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(FALLBACK_STADIUMS, columns=REQUIRED_COLUMNS).to_csv(path, index=False)
    return path


def validate_resolution_examples(logger: logging.Logger | None = None) -> None:
    """Lightweight no-network validation for season-aware park resolution."""
    log = logger or logging.getLogger(__name__)
    stadiums = load_stadium_reference()
    overrides = pd.DataFrame(
        {
            "game_pk": pd.Series([999004], dtype="Int64"),
            "park_id_override": pd.Series(["LAS_VEGAS_BALLPARK"], dtype="string"),
            "notes": ["test"],
        }
    )

    samples = pd.DataFrame(
        [
            {"game_pk": 999001, "game_date": "2026-05-01", "home_team": "ATH"},
            {"game_pk": 999002, "game_date": "2025-06-01", "home_team": "TBR"},
            {"game_pk": 999003, "game_date": "2026-04-10", "home_team": "TBR"},
            {"game_pk": 999004, "game_date": "2026-06-10", "home_team": "ATH"},
        ]
    )

    expected = {
        999001: "SUTTER_HEALTH_PARK",
        999002: "STEINBRENNER_FIELD",
        999003: "TROPICANA_FIELD",
        999004: "LAS_VEGAS_BALLPARK",
    }
    for row in samples.to_dict(orient="records"):
        season = pd.to_datetime(row["game_date"]).year
        park_id, _ = resolve_park_for_game(pd.Series(row), season, stadiums, overrides, logger=log)
        assert park_id == expected[row["game_pk"]], (row["game_pk"], park_id, expected[row["game_pk"]])
