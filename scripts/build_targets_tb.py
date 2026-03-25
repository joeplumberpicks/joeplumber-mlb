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

_BATTER_ID_CANDS = ["batter", "batter_id", "mlbam_batter_id", "player_id"]
_EVENT_CANDS = ["events", "event_type"]


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TB targets by game_pk+batter_id.")
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

    configure_logging(dirs["logs_dir"] / "build_targets_tb.log")
    log_header("scripts/build_targets_tb.py", repo_root, config_path, dirs)

    pa_path = dirs["processed_dir"] / "by_season" / f"pa_{args.season}.parquet"
    out_path = target_output_path(dirs["processed_dir"], "tb", args.season)
    if out_path.exists() and not args.force:
        logging.info("targets_tb exists and force=False: %s", out_path.resolve())
        print(f"targets_tb -> {out_path.resolve()}")
        return

    pa = read_parquet(pa_path)
    batter_col = _pick(pa, _BATTER_ID_CANDS)
    event_col = _pick(pa, _EVENT_CANDS)
    if batter_col is None or event_col is None or "game_pk" not in pa.columns:
        raise ValueError(f"PA missing required columns game_pk/batter/events. available={sorted(pa.columns)}")

    pa = pa.copy()
    pa["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")
    pa["batter_id"] = pd.to_numeric(pa[batter_col], errors="coerce").astype("Int64")
    pa["game_date"] = pd.to_datetime(pa.get("game_date"), errors="coerce")

    ev = pa[event_col].astype(str).str.lower().str.strip()
    pa["_tb"] = (
        (ev == "single").astype(int)
        + 2 * (ev == "double").astype(int)
        + 3 * (ev == "triple").astype(int)
        + 4 * (ev == "home_run").astype(int)
    )

    out = (
        pa.groupby(["game_pk", "batter_id"], dropna=False)
        .agg(game_date=("game_date", "min"), target_tb=("_tb", "sum"))
        .reset_index()
    )
    out["target_tb"] = pd.to_numeric(out["target_tb"], errors="coerce").fillna(0.0)
    out = out[["game_pk", "game_date", "batter_id", "target_tb"]]

    if len(out) == 0:
        raise ValueError(f"targets_tb output is empty for season={args.season} from {pa_path.resolve()}")

    null_diag = {
        "game_pk_null_rate": float(out["game_pk"].isna().mean()),
        "batter_id_null_rate": float(out["batter_id"].isna().mean()),
        "game_date_null_rate": float(out["game_date"].isna().mean()),
        "target_tb_null_rate": float(out["target_tb"].isna().mean()),
    }

    write_parquet(out, out_path)
    logging.info(
        "targets_tb rows=%s unique_keys=%s date_min=%s date_max=%s null_diag=%s path=%s",
        len(out),
        int(out[["game_pk", "batter_id"]].drop_duplicates().shape[0]),
        pd.to_datetime(out["game_date"], errors="coerce").min(),
        pd.to_datetime(out["game_date"], errors="coerce").max(),
        null_diag,
        out_path.resolve(),
    )
    print(f"targets_tb -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
