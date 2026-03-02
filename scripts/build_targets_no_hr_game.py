from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    p = argparse.ArgumentParser(description="Build no-HR game-level targets")
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

    configure_logging(dirs["logs_dir"] / "build_targets_no_hr_game.log")
    log_header("scripts/build_targets_no_hr_game.py", repo_root, config_path, dirs)

    spine = read_parquet(dirs["processed_dir"] / "model_spine_game.parquet")
    events = read_parquet(dirs["processed_dir"] / "events_pa.parquet")

    if "game_pk" not in spine.columns or "game_pk" not in events.columns:
        raise ValueError("game_pk required in both spine and events_pa")

    spine = spine.copy()
    spine["game_pk"] = pd.to_numeric(spine["game_pk"], errors="coerce").astype("Int64")
    spine_dates = pd.to_datetime(spine.get("game_date"), errors="coerce") if "game_date" in spine.columns else pd.Series(pd.NaT, index=spine.index)
    season_mask = spine_dates.dt.year == args.season
    if "season" in spine.columns:
        season_mask = season_mask | (pd.to_numeric(spine["season"], errors="coerce") == args.season)
    spine = spine.loc[season_mask].copy()

    events = events.copy()
    events["game_pk"] = pd.to_numeric(events["game_pk"], errors="coerce").astype("Int64")
    ev_col = _pick(events, ["events", "event_type"])
    if ev_col is None:
        raise ValueError(f"events_pa missing events/event_type column. cols={sorted(events.columns)}")

    ev = events[ev_col].astype(str).str.lower()
    hr = events.loc[ev == "home_run", ["game_pk"]].copy()
    hr["_hr"] = 1
    total_hr = hr.groupby("game_pk", dropna=False)["_hr"].sum().reset_index(name="total_hr")

    out = spine[[c for c in ["game_pk", "game_date"] if c in spine.columns]].drop_duplicates(subset=["game_pk"]).copy()
    out = out.merge(total_hr, on="game_pk", how="left")
    out["total_hr"] = pd.to_numeric(out["total_hr"], errors="coerce").fillna(0).astype("Int64")
    out["no_hr_game"] = (out["total_hr"] == 0).astype("Int64")

    out_dir = dirs["processed_dir"] / "targets" / "game"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"targets_no_hr_game_{args.season}.parquet"
    if out_path.exists() and not args.force:
        logging.info("exists and force=False: %s", out_path.resolve())
    else:
        write_parquet(out, out_path)

    logging.info(
        "targets_no_hr_game rows=%s unique_games=%s date_min=%s date_max=%s no_hr_rate=%.4f null_rate=%.4f path=%s",
        len(out),
        int(out["game_pk"].nunique()) if "game_pk" in out.columns else 0,
        pd.to_datetime(out.get("game_date"), errors="coerce").min() if len(out) else pd.NaT,
        pd.to_datetime(out.get("game_date"), errors="coerce").max() if len(out) else pd.NaT,
        float(pd.to_numeric(out["no_hr_game"], errors="coerce").mean()) if len(out) else 0.0,
        float(out["no_hr_game"].isna().mean()) if len(out) else 0.0,
        out_path.resolve(),
    )
    print(f"targets_no_hr_game -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
