from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import load

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily HR props board.")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD (defaults to latest date in mart)")
    parser.add_argument("--top-n", type=int, default=30)
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

    mart_path = dirs["marts_dir"] / "hr_batter_features.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart: {mart_path.resolve()}")
    df = read_parquet(mart_path)
    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce")
    run_date = pd.to_datetime(args.date) if args.date else df["game_date"].max()
    board = df[df["game_date"] == run_date].copy()
    if board.empty:
        print(f"No rows for date={run_date.date()} in {mart_path.resolve()}")
        return

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
