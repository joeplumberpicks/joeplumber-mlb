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
    p = argparse.ArgumentParser(description="Build dynamic 2026 park factors through a date.")
    p.add_argument("--through-date", required=True)
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--current-prior-pa", type=float, default=2500.0)
    p.add_argument("--max-current-weight", type=float, default=0.50)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], cands: list[str]) -> str | None:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def _load_current_pa(dirs: dict[str, Path], season: int, through_date: pd.Timestamp) -> pd.DataFrame:
    path = dirs["processed_dir"] / "by_season" / f"pa_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing current-season PA file: {path}")
    pa = pd.read_parquet(path)
    park_col = _pick(list(pa.columns), ["canonical_park_key", "venue_id", "park_id"])
    ev_col = _pick(list(pa.columns), ["events", "event_type", "event"])
    gd_col = _pick(list(pa.columns), ["game_date", "date"])
    if park_col is None or ev_col is None:
        raise ValueError("Current-season PA file missing park/event columns")
    if gd_col:
        gd = pd.to_datetime(pa[gd_col], errors="coerce")
        pa = pa[gd < through_date].copy()
    ev = pa[ev_col].astype(str).str.lower().str.strip()
    return pd.DataFrame({
        "canonical_park_key": pa[park_col].astype(str),
        "_is_hit": ev.isin(HIT_EVENTS).astype(float),
        "_is_hr": (ev == "home_run").astype(float),
        "_runs": pd.to_numeric(pa["rbi"], errors="coerce").fillna(0.0) if "rbi" in pa.columns else 0.0,
    })


def _rate_table(df: pd.DataFrame, pa_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["canonical_park_key", pa_col, "_hit_rate", "_hr_rate", "_runs_rate"])
    return df.groupby("canonical_park_key", as_index=False).agg(
        **{pa_col: ("canonical_park_key", "size")},
        _hit_rate=("_is_hit", "mean"),
        _hr_rate=("_is_hr", "mean"),
        _runs_rate=("_runs", "mean"),
    )


def main() -> None:
    args = parse_args()
    through_date = pd.to_datetime(args.through_date, errors="raise")

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "build_dynamic_park_factors.log")
    log_header("scripts/reference/build_dynamic_park_factors.py", repo_root, config_path, dirs)

    hist_path = dirs["reference_dir"] / "parks.parquet"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing historical parks baseline: {hist_path}")
    hist = pd.read_parquet(hist_path).copy()

    current_pa = _load_current_pa(dirs, args.season, through_date)
    cur = _rate_table(current_pa, "park_pa_2026")

    lg_hit_cur = float(current_pa["_is_hit"].mean()) if len(current_pa) else np.nan
    lg_hr_cur = float(current_pa["_is_hr"].mean()) if len(current_pa) else np.nan
    lg_runs_cur = float(current_pa["_runs"].mean()) if len(current_pa) else np.nan

    if len(cur):
        cur["park_factor_hits_2026_roll"] = 1.0 + (cur["park_pa_2026"] / (cur["park_pa_2026"] + args.current_prior_pa)) * ((cur["_hit_rate"] / max(lg_hit_cur, 1e-9)) - 1.0)
        cur["park_factor_hr_2026_roll"] = 1.0 + (cur["park_pa_2026"] / (cur["park_pa_2026"] + args.current_prior_pa)) * ((cur["_hr_rate"] / max(lg_hr_cur, 1e-9)) - 1.0)
        cur["park_factor_runs_2026_roll"] = 1.0 + (cur["park_pa_2026"] / (cur["park_pa_2026"] + args.current_prior_pa)) * ((cur["_runs_rate"] / max(lg_runs_cur, 1e-9)) - 1.0)

    out = hist.merge(cur[[c for c in ["canonical_park_key", "park_pa_2026", "park_factor_hits_2026_roll", "park_factor_hr_2026_roll", "park_factor_runs_2026_roll"] if c in cur.columns]], on="canonical_park_key", how="left")
    out["through_date"] = through_date.normalize()
    out["park_pa_2026"] = pd.to_numeric(out.get("park_pa_2026"), errors="coerce").fillna(0.0)
    w_cur = (out["park_pa_2026"] / (out["park_pa_2026"] + args.current_prior_pa)).clip(0.0, args.max_current_weight)

    out["park_factor_hits_blend"] = (1.0 - w_cur) * pd.to_numeric(out["park_factor_hits_hist"], errors="coerce") + w_cur * pd.to_numeric(out.get("park_factor_hits_2026_roll"), errors="coerce").fillna(pd.to_numeric(out["park_factor_hits_hist"], errors="coerce"))
    out["park_factor_hr_blend"] = (1.0 - w_cur) * pd.to_numeric(out["park_factor_hr_hist"], errors="coerce") + w_cur * pd.to_numeric(out.get("park_factor_hr_2026_roll"), errors="coerce").fillna(pd.to_numeric(out["park_factor_hr_hist"], errors="coerce"))
    out["park_factor_runs_blend"] = (1.0 - w_cur) * pd.to_numeric(out["park_factor_runs_hist"], errors="coerce") + w_cur * pd.to_numeric(out.get("park_factor_runs_2026_roll"), errors="coerce").fillna(pd.to_numeric(out["park_factor_runs_hist"], errors="coerce"))

    out_path = dirs["reference_dir"] / "parks_dynamic_2026.parquet"
    write_parquet(out, out_path)
    logging.info("dynamic parks built rows=%s through_date=%s path=%s", len(out), through_date.date(), out_path)
    print(f"parks_dynamic_out={out_path}")


if __name__ == "__main__":
    main()
