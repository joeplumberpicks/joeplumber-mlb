from __future__ import annotations

"""Create parks reference artifacts for ingest scaffolding."""

from pathlib import Path

import pandas as pd

from src.utils.checks import print_rowcount
from src.utils.io import write_csv

PARKS_MASTER_COLUMNS = ["park_id", "park_name", "lat", "lon"]


def build_parks_reference(dirs: dict[str, Path], season: int, repo_root: Path) -> Path:
    """Write GitHub-safe parks master CSV and avoid clobbering raw season parks parquet."""
    _ = (dirs, season)
    parks_master_path = repo_root / "data" / "reference" / "parks_master.csv"
    if parks_master_path.exists():
        return parks_master_path

    parks_master_df = pd.DataFrame(columns=PARKS_MASTER_COLUMNS)
    print("WARNING: No parks source provided for repo reference; writing empty parks master headers.")
    print_rowcount("parks_master", parks_master_df)
    print(f"Writing to: {parks_master_path.resolve()}")
    write_csv(parks_master_df, parks_master_path)
    return parks_master_path
