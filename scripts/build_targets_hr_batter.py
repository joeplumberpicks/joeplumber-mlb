from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.paths import target_output_path
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build HR batter targets")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_targets_hr_batter.log")
    log_header("scripts/build_targets_hr_batter.py", repo_root, config_path, dirs)

    pa_path = dirs["processed_dir"] / "by_season" / f"pa_{args.season}.parquet"
    pa = read_parquet(pa_path)

    batter_col = _pick(pa, ["batter", "batter_id", "mlbam_batter_id", "player_id"])
    event_col = _pick(pa, ["events", "event_type"])
    if batter_col is None or event_col is None:
        raise ValueError(f"Missing batter/event columns in PA. cols={sorted(pa.columns)}")

    pa = pa.copy()
    pa["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")
    pa["batter_id"] = pd.to_numeric(pa[batter_col], errors="coerce").astype("Int64")
    ev = pa[event_col].astype(str).str.lower()
    pa["_hr"] = (ev == "home_run").astype(int)

    agg = pa.groupby(["game_pk", "batter_id"], dropna=False).agg(target_hr=("_hr", "max")).reset_index()
    meta_cols = [c for c in ["game_pk", "game_date", "home_team", "away_team"] if c in pa.columns]
    meta = pa[meta_cols].drop_duplicates(subset=["game_pk"]) if "game_pk" in meta_cols else pd.DataFrame(columns=["game_pk"])
    out = agg.merge(meta, on="game_pk", how="left")
    out = out[[c for c in ["game_pk", "game_date", "batter_id", "target_hr"] if c in out.columns]]

    out_path = target_output_path(dirs["processed_dir"], "hr_batter", args.season)
    if not out_path.exists() or args.force:
        write_parquet(out, out_path)

    logging.info(
        "targets_hr_batter rows=%s unique_keys=%s date_min=%s date_max=%s null_rate=%.4f pos_rate=%.4f path=%s",
        len(out), int(out[["game_pk", "batter_id"]].drop_duplicates().shape[0]) if len(out) else 0,
        pd.to_datetime(out.get("game_date"), errors="coerce").min() if len(out) else pd.NaT,
        pd.to_datetime(out.get("game_date"), errors="coerce").max() if len(out) else pd.NaT,
        float(out["target_hr"].isna().mean()) if len(out) else 0.0,
        float(pd.to_numeric(out["target_hr"], errors="coerce").fillna(0).mean()) if len(out) else 0.0,
        out_path.resolve(),
    )
    print(f"targets_hr_batter -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
