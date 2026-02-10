from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str = "config/project.yaml") -> dict:
    """Load YAML project configuration from disk."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read parquet using the pyarrow engine."""
    return pd.read_parquet(path, engine="pyarrow")


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """Write parquet using the pyarrow engine."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
