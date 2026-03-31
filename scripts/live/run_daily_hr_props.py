from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.daily_context import (
    build_game_level_lineup_features,
    load_live_lineups,
    load_live_weather,
    merge_live_context,
    run_live_preflight,
    summarize_live_context,
)
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily HR props board.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (defaults to latest date in mart)")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--auto-build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-lineups", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--permissive-live-context", action="store_true")
    return parser.parse_args()


def _latest_model_path(model_dir: Path) -> Path:
    models = sorted(model_dir.glob("hr_model_*.joblib"))
    if not models:
        raise FileNotFoundError(f"No hr_model artifacts in {model_dir.resolve()}")
    return models[-1]


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "run_daily_hr_props.log")
    log_header("scripts/live/run_daily_hr_props.py", repo_root, config_path, dirs)

    mart_path = dirs["marts_dir"] / "hr_batter_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart: {mart_path.resolve()}")
    df = read_parquet(mart_path)
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
    run_date = pd.to_datetime(args.date) if args.date else df["game_date"].max()
    season = args.season or int(run_date.year)
    run_live_preflight(
        repo_root=repo_root,
        config_path=config_path,
        season=season,
        date_str=run_date.strftime("%Y-%m-%d"),
        auto_build=bool(args.auto_build),
        force_spine=True,
        build_lineups=not args.skip_lineups,
        build_weather=not args.skip_weather,
        permissive_live_context=bool(args.permissive_live_context),
    )
    board = df[df["game_date"] == run_date].copy()
    if board.empty:
        print(f"No rows for date={run_date.date()} in {mart_path.resolve()}")
        return
    game_frame = board[[c for c in ["game_pk", "game_date", "away_team", "home_team"] if c in board.columns]].drop_duplicates()
    live_weather = load_live_weather(config=config, season=season, date_str=run_date.strftime("%Y-%m-%d")) if not args.skip_weather else pd.DataFrame()
    live_lineups = load_live_lineups(config=config, season=season, date_str=run_date.strftime("%Y-%m-%d")) if not args.skip_lineups else pd.DataFrame()
    batter_roll_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    batter_roll = pd.read_parquet(batter_roll_path).copy() if batter_roll_path.exists() else pd.DataFrame()
    lineup_game = build_game_level_lineup_features(live_lineups, batter_roll, run_date)
    game_context = merge_live_context(game_frame, live_weather, lineup_game)
    smoke = summarize_live_context(game_context)
    logging.info(
        "live_feature_smoke games=%s pct_with_weather=%.2f pct_with_lineups=%.2f away_lineup_found=%s home_lineup_found=%s",
        smoke["games"],
        smoke["pct_with_weather"],
        smoke["pct_with_lineups"],
        smoke["away_lineup_found"],
        smoke["home_lineup_found"],
    )
    board = board.merge(game_context.drop(columns=["game_date", "away_team", "home_team"], errors="ignore"), on="game_pk", how="left")

    model_blob = load(_latest_model_path(dirs["models_dir"] / "hr_model"))
    model = model_blob["model"]
    imputer = model_blob["imputer"]
    features = model_blob["features"]

    X = board.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    X_imp = imputer.transform(X)
    board["HR_PROB"] = model.predict_proba(X_imp)[:, 1]
    board["implied_prob"] = pd.to_numeric(board.get("implied_prob"), errors="coerce").fillna(0.0)
    board["EDGE"] = board["HR_PROB"] - board["implied_prob"]
    board = board.sort_values("EDGE", ascending=False).reset_index(drop=True)
    board["RANK"] = board.index + 1

    player_col = "player_name" if "player_name" in board.columns else ("batter_name" if "batter_name" in board.columns else "batter_id")
    out = board[[player_col, "batter_team", "HR_PROB", "EDGE", "RANK"]].rename(
        columns={player_col: "PLAYER", "batter_team": "TEAM"}
    )
    out = out.head(args.top_n)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
