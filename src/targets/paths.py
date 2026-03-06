from __future__ import annotations

from pathlib import Path


def target_output_path(processed_dir: Path, market: str, season: int) -> Path:
    out_dir = processed_dir / "targets" / market
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"targets_{market}_{season}.parquet"


def target_input_candidates(processed_dir: Path, market: str, season: int) -> list[Path]:
    return [
        processed_dir / "targets" / market / f"targets_{market}_{season}.parquet",
        processed_dir / f"targets_{market}_{season}.parquet",
    ]
