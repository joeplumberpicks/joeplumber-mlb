from __future__ import annotations

"""Create parks reference artifacts for ingest scaffolding."""

from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount
from src.utils.io import write_csv, write_parquet

PARKS_MASTER_COLUMNS = ["park_id", "park_name", "lat", "lon"]
PARKS_SEASON_COLUMNS = ["park_id", "park_name", "park_factor", "season"]


def build_parks_reference(dirs: dict[str, Path], season: int, repo_root: Path) -> tuple[Path, Path]:
    """Write a GitHub-safe parks master CSV and seasonal raw parks parquet."""
    parks_master_path = repo_root / "data" / "reference" / "parks_master.csv"
    parks_master_df = pd.DataFrame(columns=PARKS_MASTER_COLUMNS)
    print("WARNING: No parks source provided; writing empty parks master with headers.")
    print_rowcount("parks_master", parks_master_df)
    print(f"Writing to: {parks_master_path.resolve()}")
    write_csv(parks_master_df, parks_master_path)

    parks_season_df = pd.DataFrame(columns=PARKS_SEASON_COLUMNS)
    parks_season_df["season"] = pd.Series(dtype="Int64")
    parks_path = dirs["raw_dir"] / "by_season" / f"parks_{season}.parquet"
    print_rowcount(f"parks_{season}", parks_season_df)
    print(f"Writing to: {parks_path.resolve()}")
    write_parquet(parks_season_df, parks_path)

    return parks_master_path, parks_path
