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

_HIT_EVENTS = {"single", "double", "triple", "home_run"}
_WALK_EVENTS = {"walk", "intent_walk", "intentional_walk"}
_BATTER_ID_CANDS = ["batter", "batter_id", "mlbam_batter_id", "player_id"]


def _pick(df: pd.DataFrame, cands: list[str]) -> str | None:
    for c in cands:
        if c in df.columns:
            return c
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build hitter prop targets by game_pk+batter_id.")
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

    configure_logging(dirs["logs_dir"] / "build_targets_hitter_props.log")
    log_header("scripts/build_targets_hitter_props.py", repo_root, config_path, dirs)

    pa_path = dirs["processed_dir"] / "by_season" / f"pa_{args.season}.parquet"
    out_path = target_output_path(dirs["processed_dir"], "hitter_props", args.season)
    if out_path.exists() and not args.force:
        logging.info("targets_hitter_props exists and force=False: %s", out_path.resolve())
        print(f"targets_hitter_props -> {out_path.resolve()}")
        return

    pa = read_parquet(pa_path)
    batter_col = _pick(pa, _BATTER_ID_CANDS)
    event_col = _pick(pa, ["events", "event_type"])
    if batter_col is None or event_col is None or "game_pk" not in pa.columns:
        raise ValueError(f"PA missing required columns game_pk/batter/events. available={sorted(pa.columns)}")

    pa = pa.copy()
    pa["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")
    pa["batter_id"] = pd.to_numeric(pa[batter_col], errors="coerce").astype("Int64")
    pa["game_date"] = pd.to_datetime(pa.get("game_date"), errors="coerce")
    ev = pa[event_col].astype(str).str.lower().str.strip()

    pa["_hit"] = ev.isin(_HIT_EVENTS).astype(int)
    pa["_tb"] = (
        (ev == "single").astype(int)
        + 2 * (ev == "double").astype(int)
        + 3 * (ev == "triple").astype(int)
        + 4 * (ev == "home_run").astype(int)
    )
    pa["_bb"] = ev.isin(_WALK_EVENTS).astype(int)

    if "rbi" in pa.columns:
        pa["_rbi"] = pd.to_numeric(pa["rbi"], errors="coerce").fillna(0).clip(lower=0)
    elif {"bat_score", "post_bat_score"}.issubset(pa.columns):
        pre = pd.to_numeric(pa["bat_score"], errors="coerce")
        post = pd.to_numeric(pa["post_bat_score"], errors="coerce")
        pa["_rbi"] = (post - pre).fillna(0).clip(lower=0, upper=4)
    else:
        raise ValueError(
            "Cannot derive RBI target. Need either 'rbi' column or score progression columns "
            f"('bat_score','post_bat_score'). available={sorted(pa.columns)}"
        )

    agg = (
        pa.groupby(["game_pk", "batter_id"], dropna=False)
        .agg(
            game_date=("game_date", "min"),
            hits=("_hit", "sum"),
            tb=("_tb", "sum"),
            bb=("_bb", "sum"),
            rbi=("_rbi", "sum"),
        )
        .reset_index()
    )

    out = agg[["game_pk", "game_date", "batter_id"]].copy()
    out["target_hit1p"] = (agg["hits"] >= 1).astype("Int64")
    out["target_tb2p"] = (agg["tb"] >= 2).astype("Int64")
    out["target_rbi1p"] = (agg["rbi"] >= 1).astype("Int64")
    out["target_bb1p"] = (agg["bb"] >= 1).astype("Int64")
    out = out[["game_pk", "game_date", "batter_id", "target_hit1p", "target_tb2p", "target_rbi1p", "target_bb1p"]]

    if len(out) == 0:
        raise ValueError(f"targets_hitter_props output is empty for season={args.season} from {pa_path.resolve()}")

    null_rates = {c: float(out[c].isna().mean()) for c in ["target_hit1p", "target_tb2p", "target_rbi1p", "target_bb1p"]}
    bad = {k: v for k, v in null_rates.items() if v > 0}
    if bad:
        raise ValueError(f"targets_hitter_props has nulls in targets: {bad}")

    write_parquet(out, out_path)

    logging.info(
        "targets_hitter_props rows=%s unique_keys=%s date_min=%s date_max=%s hit_null=%.4f tb_null=%.4f rbi_null=%.4f bb_null=%.4f path=%s",
        len(out),
        int(out[["game_pk", "batter_id"]].drop_duplicates().shape[0]),
        pd.to_datetime(out["game_date"], errors="coerce").min(),
        pd.to_datetime(out["game_date"], errors="coerce").max(),
        null_rates["target_hit1p"],
        null_rates["target_tb2p"],
        null_rates["target_rbi1p"],
        null_rates["target_bb1p"],
        out_path.resolve(),
    )
    print(f"targets_hitter_props -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
