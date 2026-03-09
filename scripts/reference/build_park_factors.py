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


HIT_EVENTS = {"single", "double", "triple", "home_run"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build historical park factors reference table.")
    p.add_argument("--season-start", type=int, default=2019)
    p.add_argument("--season-end", type=int, default=2025)
    p.add_argument("--prior-pa", type=float, default=4000.0, help="Shrinkage prior PA for historical park factors")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], cands: list[str]) -> str | None:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def _load_pa_events(dirs: dict[str, Path], season_start: int, season_end: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for season in range(season_start, season_end + 1):
        path = dirs["processed_dir"] / "by_season" / f"pa_{season}.parquet"
        if not path.exists():
            continue
        pa = pd.read_parquet(path)
        park_col = _pick(list(pa.columns), ["canonical_park_key", "venue_id", "park_id"])
        ev_col = _pick(list(pa.columns), ["events", "event_type", "event"])
        if park_col is None or ev_col is None:
            continue
        ev = pa[ev_col].astype(str).str.lower().str.strip()
        out = pd.DataFrame({
            "canonical_park_key": pa[park_col].astype(str),
            "_is_hit": ev.isin(HIT_EVENTS).astype(float),
            "_is_hr": (ev == "home_run").astype(float),
            "_runs": pd.to_numeric(pa["rbi"], errors="coerce").fillna(0.0) if "rbi" in pa.columns else 0.0,
        })
        frames.append(out)
    if not frames:
        raise FileNotFoundError("No PA seasonal files found under processed/by_season to build parks.parquet")
    return pd.concat(frames, ignore_index=True, sort=False)


def _shrunk_factor(rate: pd.Series, lg_rate: float, pa: pd.Series, prior_pa: float) -> pd.Series:
    if lg_rate <= 0:
        return pd.Series(1.0, index=rate.index, dtype="float64")
    raw = rate / lg_rate
    weight = pa / (pa + prior_pa)
    return (1.0 - weight) * 1.0 + weight * raw


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "build_park_factors.log")
    log_header("scripts/reference/build_park_factors.py", repo_root, config_path, dirs)

    all_pa = _load_pa_events(dirs, args.season_start, args.season_end)
    lg_hit = float(all_pa["_is_hit"].mean())
    lg_hr = float(all_pa["_is_hr"].mean())
    lg_runs = float(all_pa["_runs"].mean())

    park = all_pa.groupby("canonical_park_key", as_index=False).agg(
        park_pa_hist=("canonical_park_key", "size"),
        _hit_rate=("_is_hit", "mean"),
        _hr_rate=("_is_hr", "mean"),
        _runs_rate=("_runs", "mean"),
    )
    park["park_factor_hits_hist"] = _shrunk_factor(park["_hit_rate"], lg_hit, park["park_pa_hist"], args.prior_pa)
    park["park_factor_hr_hist"] = _shrunk_factor(park["_hr_rate"], lg_hr, park["park_pa_hist"], args.prior_pa)
    park["park_factor_runs_hist"] = _shrunk_factor(park["_runs_rate"], lg_runs, park["park_pa_hist"], args.prior_pa)

    out = park[[
        "canonical_park_key",
        "park_pa_hist",
        "park_factor_hits_hist",
        "park_factor_hr_hist",
        "park_factor_runs_hist",
    ]].copy()
    out_path = dirs["reference_dir"] / "parks.parquet"
    write_parquet(out, out_path)
    logging.info("historical parks built rows=%s path=%s", len(out), out_path)
    print(f"parks_out={out_path}")


if __name__ == "__main__":
    main()
