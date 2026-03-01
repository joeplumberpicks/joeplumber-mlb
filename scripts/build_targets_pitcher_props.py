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
    p = argparse.ArgumentParser(description="Build pitcher prop targets")
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

    configure_logging(dirs["logs_dir"] / "build_targets_pitcher_props.log")
    log_header("scripts/build_targets_pitcher_props.py", repo_root, config_path, dirs)

    pa = read_parquet(dirs["processed_dir"] / "by_season" / f"pa_{args.season}.parquet")
    pitcher_col = _pick(pa, ["pitcher_id", "pitcher", "mlbam_pitcher_id", "player_id"])
    event_col = _pick(pa, ["events", "event_type"])
    if pitcher_col is None or event_col is None:
        raise ValueError(f"Missing pitcher/event columns in PA. cols={sorted(pa.columns)}")

    pa = pa.copy()
    pa["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")
    pa["pitcher_id"] = pd.to_numeric(pa[pitcher_col], errors="coerce").astype("Int64")
    ev = pa[event_col].astype(str).str.lower()
    pa["_k"] = ev.isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa["_bb"] = ev.isin({"walk", "intent_walk"}).astype(int)
    one = {"strikeout","field_out","force_out","ground_out","fly_out","line_out","pop_out","sac_fly","sac_bunt","fielders_choice_out","sac_fly_error"}
    two = {"double_play","grounded_into_double_play","strikeout_double_play","sac_fly_double_play"}
    three = {"triple_play"}
    pa["_outs"] = 0
    pa.loc[ev.isin(one), "_outs"] = 1
    pa.loc[ev.isin(two), "_outs"] = 2
    pa.loc[ev.isin(three), "_outs"] = 3

    er_col = _pick(pa, ["earned_runs", "er", "pitcher_er"])
    if er_col is None:
        pa["_er"] = pd.NA
        logging.warning("ER column missing in PA; target_er will be null")
    else:
        pa["_er"] = pd.to_numeric(pa[er_col], errors="coerce")

    agg = pa.groupby(["game_pk", "pitcher_id"], dropna=False).agg(
        target_k=("_k", "sum"),
        target_outs=("_outs", "sum"),
        target_bb=("_bb", "sum"),
        target_er=("_er", "max"),
    ).reset_index()

    meta_cols = [c for c in ["game_pk", "game_date"] if c in pa.columns]
    meta = pa[meta_cols].drop_duplicates(subset=["game_pk"]) if "game_pk" in meta_cols else pd.DataFrame(columns=["game_pk"])
    out = agg.merge(meta, on="game_pk", how="left")
    out = out[[c for c in ["game_pk", "game_date", "pitcher_id", "target_k", "target_outs", "target_er", "target_bb"] if c in out.columns]]

    out_path = target_output_path(dirs["processed_dir"], "pitcher_props", args.season)
    if not out_path.exists() or args.force:
        write_parquet(out, out_path)

    logging.info(
        "targets_pitcher_props rows=%s unique_keys=%s date_min=%s date_max=%s k_null=%.4f outs_null=%.4f er_null=%.4f bb_null=%.4f path=%s",
        len(out), int(out[["game_pk", "pitcher_id"]].drop_duplicates().shape[0]) if len(out) else 0,
        pd.to_datetime(out.get("game_date"), errors="coerce").min() if len(out) else pd.NaT,
        pd.to_datetime(out.get("game_date"), errors="coerce").max() if len(out) else pd.NaT,
        float(out["target_k"].isna().mean()) if len(out) else 0.0,
        float(out["target_outs"].isna().mean()) if len(out) else 0.0,
        float(out["target_er"].isna().mean()) if len(out) else 0.0,
        float(out["target_bb"].isna().mean()) if len(out) else 0.0,
        out_path.resolve(),
    )
    print(f"targets_pitcher_props -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
