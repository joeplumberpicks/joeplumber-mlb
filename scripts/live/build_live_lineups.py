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
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build standardized live lineups for hit props.")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], candidates: list[str]) -> str | None:
    cset = set(cols)
    for c in candidates:
        if c in cset:
            return c
    return None


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_live_lineups.log")
    log_header("scripts/live/build_live_lineups.py", repo_root, config_path, dirs)

    spine_path = dirs["processed_dir"] / "live" / f"model_spine_game_{args.season}_{args.date}.parquet"
    if not spine_path.exists():
        raise FileNotFoundError(
            f"Live spine not found: {spine_path}. Build it first with scripts/live/build_spine_from_schedule.py"
        )

    live_dir = dirs["processed_dir"] / "live"
    lineup_candidates = [
        live_dir / f"confirmed_lineups_{args.season}_{args.date}.parquet",
        live_dir / f"projected_lineups_{args.season}_{args.date}.parquet",
        live_dir / f"lineups_{args.season}_{args.date}.parquet",
    ]
    src = next((p for p in lineup_candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError(
            f"No lineup source found for {args.season} {args.date}. Expected one of: {[str(p) for p in lineup_candidates]}"
        )

    lu = pd.read_parquet(src).copy()
    spine = pd.read_parquet(spine_path).copy()

    game_pk_col = _pick(list(lu.columns), ["game_pk", "game_id"])
    batter_id_col = _pick(list(lu.columns), ["batter_id", "batter", "player_id"])
    name_col = _pick(list(lu.columns), ["player_name", "batter_name", "name"])
    team_col = _pick(list(lu.columns), ["batter_team", "team", "batting_team", "team_abbrev", "team_name"])
    slot_col = _pick(list(lu.columns), ["lineup_slot", "batting_order", "order", "lineup_position"])

    if batter_id_col is None or team_col is None:
        raise ValueError(
            f"Lineup source missing required batter/team columns. columns={sorted(lu.columns)}"
        )

    out = pd.DataFrame()
    out["game_pk"] = pd.to_numeric(lu[game_pk_col], errors="coerce").astype("Int64") if game_pk_col else pd.NA
    out["batter_id"] = pd.to_numeric(lu[batter_id_col], errors="coerce").astype("Int64")
    out["player_name"] = lu[name_col] if name_col else out["batter_id"].astype(str)
    out["batter_team"] = lu[team_col]
    out["lineup_slot"] = pd.to_numeric(lu[slot_col], errors="coerce") if slot_col else np.nan

    spine_view = spine[[c for c in ["game_pk", "game_date", "season", "home_team", "away_team"] if c in spine.columns]].copy()
    spine_view["game_pk"] = pd.to_numeric(spine_view["game_pk"], errors="coerce").astype("Int64")
    out = out.merge(spine_view.drop_duplicates(subset=["game_pk"], keep="last"), on="game_pk", how="left")

    # fallback game_pk from team match if missing in lineup source
    if out["game_pk"].isna().any() and {"home_team", "away_team"}.issubset(spine_view.columns):
        team_games = pd.concat(
            [
                spine_view[["game_pk", "home_team", "away_team"]].rename(columns={"home_team": "batter_team"}),
                spine_view[["game_pk", "away_team", "home_team"]].rename(columns={"away_team": "batter_team", "home_team": "away_team"}),
            ],
            ignore_index=True,
            sort=False,
        )
        missing_mask = out["game_pk"].isna()
        if missing_mask.any():
            m = out.loc[missing_mask, ["batter_team"]].merge(team_games.drop_duplicates(subset=["batter_team"], keep="last"), on="batter_team", how="left")
            out.loc[missing_mask, "game_pk"] = pd.to_numeric(m["game_pk"], errors="coerce").astype("Int64").values
        out = out.drop(columns=[c for c in ["home_team", "away_team"] if c in out.columns], errors="ignore")
        out = out.merge(spine_view.drop_duplicates(subset=["game_pk"], keep="last"), on="game_pk", how="left")

    out["home_away"] = np.where(
        out["batter_team"].astype(str) == out.get("home_team", pd.Series(index=out.index, dtype="object")).astype(str),
        1.0,
        np.where(
            out["batter_team"].astype(str) == out.get("away_team", pd.Series(index=out.index, dtype="object")).astype(str),
            0.0,
            np.nan,
        ),
    )
    out["opponent_team"] = np.where(out["home_away"] == 1.0, out.get("away_team"), out.get("home_team"))
    out["game_date"] = pd.to_datetime(out.get("game_date"), errors="coerce").dt.strftime("%Y-%m-%d")
    out["season"] = pd.to_numeric(out.get("season"), errors="coerce").fillna(args.season).astype("Int64")

    cols = [
        "game_pk",
        "game_date",
        "season",
        "batter_id",
        "player_name",
        "batter_team",
        "opponent_team",
        "lineup_slot",
        "home_away",
    ]
    out = out[cols].dropna(subset=["game_pk", "batter_id", "batter_team"]).drop_duplicates(subset=["game_pk", "batter_id"], keep="last")

    out_path = live_dir / f"lineups_{args.season}_{args.date}.parquet"
    out.to_parquet(out_path, index=False)

    pct_slot = float(pd.to_numeric(out["lineup_slot"], errors="coerce").notna().mean()) if len(out) else 0.0
    logging.info(
        "live_lineups built rows=%s unique_games=%s unique_hitters=%s pct_with_lineup_slot=%.4f src=%s out=%s",
        len(out),
        int(out["game_pk"].nunique()) if len(out) else 0,
        int(out["batter_id"].nunique()) if len(out) else 0,
        pct_slot,
        src,
        out_path,
    )
    print(f"lineups_out={out_path}")


if __name__ == "__main__":
    main()
