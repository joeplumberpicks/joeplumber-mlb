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


def _filter_season(df: pd.DataFrame, season: int) -> pd.DataFrame:
    out = df.copy()
    if "game_date" in out.columns:
        gd = pd.to_datetime(out["game_date"], errors="coerce")
        out = out[gd.dt.year == season].copy()
    elif "season" in out.columns:
        out = out[pd.to_numeric(out["season"], errors="coerce") == season].copy()
    return out


def _team_rollup(batter_roll: pd.DataFrame) -> pd.DataFrame:
    team_col = _pick(batter_roll, ["batter_team", "batting_team", "bat_team", "team", "offense_team"])
    if team_col is None or "game_pk" not in batter_roll.columns:
        return pd.DataFrame(columns=["game_pk", "team"])
    numeric = [
        c for c in batter_roll.select_dtypes(include=["number"]).columns
        if c not in {"game_pk", "season", "batter_id", "batter", "pitcher_id"}
    ]
    if not numeric:
        return pd.DataFrame(columns=["game_pk", "team"])
    work = batter_roll[["game_pk", team_col] + numeric].copy()
    work = work.rename(columns={team_col: "team"})
    agg = work.groupby(["game_pk", "team"], dropna=False)[numeric].mean().reset_index()
    rename = {c: f"team_{c}" for c in numeric}
    return agg.rename(columns=rename)


def _starter_features(spine: pd.DataFrame, pitcher_roll: pd.DataFrame) -> pd.DataFrame:
    if pitcher_roll.empty or "game_pk" not in pitcher_roll.columns:
        return spine

    pr = pitcher_roll.copy()
    pr["game_pk"] = pd.to_numeric(pr["game_pk"], errors="coerce").astype("Int64")
    pid_col = _pick(pr, ["pitcher_id", "pitcher", "mlbam_pitcher_id", "player_id"])
    if pid_col is None:
        return spine

    pr["pitcher_id"] = pd.to_numeric(pr[pid_col], errors="coerce").astype("Int64")
    num_cols = [
        c for c in pr.select_dtypes(include=["number"]).columns
        if c not in {"game_pk", "pitcher_id", "season"}
    ]
    if not num_cols:
        return spine

    out = spine.copy()
    home_sp = _pick(out, ["home_sp_id", "home_starter_id", "home_pitcher_id", "home_starting_pitcher_id"])
    away_sp = _pick(out, ["away_sp_id", "away_starter_id", "away_pitcher_id", "away_starting_pitcher_id"])
    if home_sp is None or away_sp is None:
        return out

    out["home_sp_id"] = pd.to_numeric(out[home_sp], errors="coerce").astype("Int64")
    out["away_sp_id"] = pd.to_numeric(out[away_sp], errors="coerce").astype("Int64")

    slim = pr[["game_pk", "pitcher_id"] + num_cols].drop_duplicates(subset=["game_pk", "pitcher_id"])
    h = slim.rename(columns={c: f"home_sp_{c}" for c in num_cols})
    a = slim.rename(columns={c: f"away_sp_{c}" for c in num_cols})
    out = out.merge(h, left_on=["game_pk", "home_sp_id"], right_on=["game_pk", "pitcher_id"], how="left")
    out = out.drop(columns=["pitcher_id"], errors="ignore")
    out = out.merge(a, left_on=["game_pk", "away_sp_id"], right_on=["game_pk", "pitcher_id"], how="left")
    out = out.drop(columns=["pitcher_id"], errors="ignore")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build No-HR game feature mart")
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

    configure_logging(dirs["logs_dir"] / "build_mart_no_hr_game.log")
    log_header("scripts/build_mart_no_hr_game.py", repo_root, config_path, dirs)

    spine = _filter_season(read_parquet(dirs["processed_dir"] / "model_spine_game.parquet"), args.season)
    spine = spine.copy()
    spine["game_pk"] = pd.to_numeric(spine["game_pk"], errors="coerce").astype("Int64")

    targets = read_parquet(dirs["processed_dir"] / "targets" / "game" / f"targets_no_hr_game_{args.season}.parquet")
    targets["game_pk"] = pd.to_numeric(targets["game_pk"], errors="coerce").astype("Int64")

    batter_roll = _filter_season(read_parquet(dirs["processed_dir"] / "batter_game_rolling.parquet"), args.season)
    pitcher_roll = _filter_season(read_parquet(dirs["processed_dir"] / "pitcher_game_rolling.parquet"), args.season)

    mart = spine[[c for c in ["game_pk", "game_date", "season", "home_team", "away_team", "park_id", "canonical_park_key"] if c in spine.columns]].copy()

    team_roll = _team_rollup(batter_roll)
    if not team_roll.empty and {"home_team", "away_team"}.issubset(mart.columns):
        home_roll = team_roll.rename(columns={c: f"home_{c}" for c in team_roll.columns if c not in {"game_pk", "team"}})
        away_roll = team_roll.rename(columns={c: f"away_{c}" for c in team_roll.columns if c not in {"game_pk", "team"}})
        mart = mart.merge(home_roll, left_on=["game_pk", "home_team"], right_on=["game_pk", "team"], how="left")
        mart = mart.drop(columns=["team"], errors="ignore")
        mart = mart.merge(away_roll, left_on=["game_pk", "away_team"], right_on=["game_pk", "team"], how="left")
        mart = mart.drop(columns=["team"], errors="ignore")

    mart = _starter_features(mart, pitcher_roll)

    weather_path = dirs["processed_dir"] / "weather_game.parquet"
    if weather_path.exists():
        weather = _filter_season(read_parquet(weather_path), args.season)
        if "game_pk" in weather.columns:
            weather["game_pk"] = pd.to_numeric(weather["game_pk"], errors="coerce").astype("Int64")
            weather_cols = [c for c in weather.columns if c == "game_pk" or c.startswith(("temp", "wind", "weather", "humidity", "pressure"))]
            if len(weather_cols) > 1:
                mart = mart.merge(weather[weather_cols].drop_duplicates(subset=["game_pk"]), on="game_pk", how="left")

    parks_path = dirs["processed_dir"] / "parks.parquet"
    if parks_path.exists() and "park_id" in mart.columns:
        parks = read_parquet(parks_path)
        pid = _pick(parks, ["park_id", "venue_id", "id"])
        if pid is not None:
            parks = parks.copy()
            parks["park_id"] = parks[pid]
            park_cols = [c for c in parks.columns if c == "park_id" or "hr" in c.lower() or "park_factor" in c.lower()]
            if len(park_cols) > 1:
                mart = mart.merge(parks[park_cols].drop_duplicates(subset=["park_id"]), on="park_id", how="left")

    mart = mart.merge(targets[["game_pk", "total_hr", "no_hr_game"]], on="game_pk", how="left")
    mart["no_hr_game"] = pd.to_numeric(mart["no_hr_game"], errors="coerce").astype("Int64")

    out_dir = dirs["processed_dir"] / "marts" / "no_hr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"no_hr_game_features_{args.season}.parquet"
    if out_path.exists() and not args.force:
        logging.info("exists and force=False: %s", out_path.resolve())
    else:
        write_parquet(mart, out_path)

    logging.info(
        "no_hr_mart rows=%s unique_games=%s target_null_rate=%.4f path=%s",
        len(mart),
        int(mart["game_pk"].nunique()) if "game_pk" in mart.columns else 0,
        float(mart["no_hr_game"].isna().mean()) if "no_hr_game" in mart.columns and len(mart) else 0.0,
        out_path.resolve(),
    )
    print(f"no_hr_mart -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
