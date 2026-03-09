from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import write_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build park factors reference table.")
    p.add_argument("--season-start", type=int, default=2019)
    p.add_argument("--season-end", type=int, default=2025)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], cands: list[str]) -> str | None:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "build_park_factors.log")
    log_header("scripts/reference/build_park_factors.py", repo_root, config_path, dirs)

    pa_frames: list[pd.DataFrame] = []
    for season in range(args.season_start, args.season_end + 1):
        pa_path = dirs["processed_dir"] / "by_season" / f"pa_{season}.parquet"
        if not pa_path.exists():
            continue
        pa = pd.read_parquet(pa_path).copy()
        park_col = _pick(list(pa.columns), ["canonical_park_key", "venue_id", "park_id"])
        if park_col is None:
            continue
        ev_col = _pick(list(pa.columns), ["events", "event_type"])
        if ev_col is None:
            continue
        pa["canonical_park_key"] = pa[park_col].astype(str)
        ev = pa[ev_col].astype(str).str.lower().str.strip()
        pa["_is_hit"] = ev.isin({"single", "double", "triple", "home_run"}).astype(float)
        pa["_is_hr"] = (ev == "home_run").astype(float)
        if "rbi" in pa.columns:
            pa["_runs"] = pd.to_numeric(pa["rbi"], errors="coerce").fillna(0.0)
        else:
            pa["_runs"] = 0.0
        pa_frames.append(pa[["canonical_park_key", "_is_hit", "_is_hr", "_runs"]])

    if not pa_frames:
        raise FileNotFoundError("No processed PA seasonal files found to build park factors")

    all_pa = pd.concat(pa_frames, ignore_index=True, sort=False)
    lg_hit = float(all_pa["_is_hit"].mean()) if len(all_pa) else 0.0
    lg_hr = float(all_pa["_is_hr"].mean()) if len(all_pa) else 0.0
    lg_runs = float(all_pa["_runs"].mean()) if len(all_pa) else 0.0

    park = all_pa.groupby("canonical_park_key", as_index=False).agg(
        n_pa=("canonical_park_key", "size"),
        hit_rate=("_is_hit", "mean"),
        hr_rate=("_is_hr", "mean"),
        runs_pa=("_runs", "mean"),
    )
    park["park_factor_hits"] = park["hit_rate"] / (lg_hit if lg_hit > 0 else 1.0)
    park["park_factor_hr"] = park["hr_rate"] / (lg_hr if lg_hr > 0 else 1.0)
    park["park_factor_runs"] = park["runs_pa"] / (lg_runs if lg_runs > 0 else 1.0)

    out = park[["canonical_park_key", "park_factor_hits", "park_factor_hr", "park_factor_runs", "n_pa"]]
    out_path = dirs["reference_dir"] / "parks.parquet"
    write_parquet(out, out_path)
    logging.info("parks reference rows=%s path=%s", len(out), out_path)
    print(f"parks_out={out_path}")


if __name__ == "__main__":
    main()
