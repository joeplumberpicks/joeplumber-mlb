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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build NRFI/YRFI targets")
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

    configure_logging(dirs["logs_dir"] / "build_targets_nrfi.log")
    log_header("scripts/build_targets_nrfi.py", repo_root, config_path, dirs)

    games_path = dirs["processed_dir"] / "by_season" / f"games_{args.season}.parquet"
    games = read_parquet(games_path)
    need = ["game_pk", "game_date", "home_team", "away_team"]
    out = games[[c for c in need if c in games.columns]].drop_duplicates(subset=["game_pk"]).copy()

    home1 = next((c for c in ["home_score_1", "home_runs_1st", "home_1st_inning_runs"] if c in games.columns), None)
    away1 = next((c for c in ["away_score_1", "away_runs_1st", "away_1st_inning_runs"] if c in games.columns), None)

    if home1 is not None and away1 is not None:
        h = pd.to_numeric(games[home1], errors="coerce")
        a = pd.to_numeric(games[away1], errors="coerce")
        first = games[["game_pk"]].copy()
        first["first_runs"] = (h.fillna(0) + a.fillna(0)).astype(float)
        first = first.drop_duplicates(subset=["game_pk"])
    else:
        pa_path = dirs["processed_dir"] / "by_season" / f"pa_{args.season}.parquet"
        pa = read_parquet(pa_path)
        required = {"game_pk", "inning", "bat_score", "post_bat_score"}
        if not required.issubset(pa.columns):
            raise ValueError(f"Cannot compute NRFI targets; need one of inning runs in games or PA score progression columns. Available PA columns={sorted(pa.columns)}")
        pa = pa.copy()
        pa["inning"] = pd.to_numeric(pa["inning"], errors="coerce")
        pa = pa[pa["inning"] == 1].copy()
        pre = pd.to_numeric(pa["bat_score"], errors="coerce")
        post = pd.to_numeric(pa["post_bat_score"], errors="coerce")
        pa["runs"] = (post - pre).fillna(0).clip(lower=0, upper=4)
        first = pa.groupby("game_pk", dropna=False)["runs"].sum().reset_index(name="first_runs")

    out["game_pk"] = pd.to_numeric(out["game_pk"], errors="coerce").astype("Int64")
    first["game_pk"] = pd.to_numeric(first["game_pk"], errors="coerce").astype("Int64")
    out = out.merge(first, on="game_pk", how="left")
    out["target_nrfi"] = (pd.to_numeric(out["first_runs"], errors="coerce").fillna(0) == 0).astype("Int64")
    out["target_yrfi"] = (pd.to_numeric(out["first_runs"], errors="coerce").fillna(0) >= 1).astype("Int64")
    out = out[["game_pk", "game_date", "home_team", "away_team", "target_nrfi", "target_yrfi"]]

    out_path = target_output_path(dirs["processed_dir"], "nrfi", args.season)
    if not out_path.exists() or args.force:
        write_parquet(out, out_path)

    logging.info(
        "targets_nrfi rows=%s unique_games=%s date_min=%s date_max=%s nrfi_null=%.4f nrfi_pos=%.4f yrfi_null=%.4f yrfi_pos=%.4f path=%s",
        len(out), int(out["game_pk"].nunique()) if "game_pk" in out.columns else 0,
        pd.to_datetime(out.get("game_date"), errors="coerce").min() if len(out) else pd.NaT,
        pd.to_datetime(out.get("game_date"), errors="coerce").max() if len(out) else pd.NaT,
        float(out["target_nrfi"].isna().mean()) if len(out) else 0.0,
        float(pd.to_numeric(out["target_nrfi"], errors="coerce").fillna(0).mean()) if len(out) else 0.0,
        float(out["target_yrfi"].isna().mean()) if len(out) else 0.0,
        float(pd.to_numeric(out["target_yrfi"], errors="coerce").fillna(0).mean()) if len(out) else 0.0,
        out_path.resolve(),
    )
    print(f"targets_nrfi -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
