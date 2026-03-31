from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

# Make repo root importable when running from scripts/live/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.daily_context import (
    build_game_level_lineup_features,
    load_live_lineups,
    load_live_spine,
    load_live_weather,
    merge_live_context,
    resolve_live_paths,
    run_live_preflight,
    summarize_live_context,
)
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import read_parquet, write_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run daily No-HR v1 scoring.")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--auto-build", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-lineups", action="store_true")
    p.add_argument("--skip-weather", action="store_true")
    p.add_argument("--permissive-live-context", action="store_true")
    p.add_argument("--force-mart", action="store_true")
    p.add_argument("--outdir-name", default="no_hr")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--no-board", action="store_true")
    p.add_argument("--board-top", type=int, default=15)
    return p.parse_args()


def _fmt_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "--.-%"
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "--.-%"


def _grade_from_prob(p: float) -> str:
    if p >= 0.18:
        return "A+"
    if p >= 0.16:
        return "A"
    if p >= 0.145:
        return "A-"
    if p >= 0.13:
        return "B+"
    if p >= 0.12:
        return "B"
    if p >= 0.11:
        return "B-"
    if p >= 0.10:
        return "C+"
    if p >= 0.09:
        return "C"
    return "D"


def _print_board(board: pd.DataFrame, date_str: str, top_n: int = 15) -> None:
    if board.empty:
        print(f"\nJOE PLUMBER NO-HR BOARD — {date_str}")
        print("-----------------------------------")
        print("(no games)\n")
        return

    view = board.copy()
    view["p_no_hr"] = pd.to_numeric(view["p_no_hr"], errors="coerce")
    view["delta_vs_baseline"] = pd.to_numeric(view["delta_vs_baseline"], errors="coerce")
    view = view.sort_values(["p_no_hr", "delta_vs_baseline"], ascending=[False, False]).head(top_n)

    print(f"\nJOE PLUMBER NO-HR BOARD — {date_str}")
    print("-----------------------------------")
    for _, r in view.iterrows():
        matchup = f"{str(r.get('away_team', '') or '')} @ {str(r.get('home_team', '') or '')}"
        prob = float(r["p_no_hr"]) if pd.notna(r["p_no_hr"]) else float("nan")
        delta = float(r["delta_vs_baseline"]) if pd.notna(r["delta_vs_baseline"]) else float("nan")
        grade = str(r.get("grade", "") or "")
        delta_str = f"{100.0 * delta:+.1f}%" if pd.notna(delta) else "--.-%"
        print(f"{matchup:<28} {_fmt_pct(prob):>6}   {delta_str:>7}   {grade}")
    print("")


def _run_cmd(cmd: list[str], repo_root: Path) -> None:
    logging.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def _resolve_model_path(model_dir: Path, requested: Path | None) -> Path:
    if requested is not None:
        if not requested.exists():
            raise FileNotFoundError(f"Model path not found: {requested}")
        return requested.resolve()

    candidates = sorted(model_dir.glob("no_hr_model_*.pkl"))
    if not candidates:
        raise FileNotFoundError(f"No no_hr model artifacts found in: {model_dir.resolve()}")
    return candidates[-1].resolve()


def _safe_game_filter(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    out = df.copy()
    if "game_date" not in out.columns:
        return out

    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
    target_date = pd.to_datetime(date_str, errors="coerce")
    filtered = out[out["game_date"] == target_date].copy()
    return filtered if not filtered.empty else out


def _load_model_bundle(model_path: Path) -> tuple[object, list[str], float]:
    bundle = load(model_path)

    if isinstance(bundle, dict):
        model = bundle.get("model")
        feat_cols = list(bundle.get("features", []))
        baseline = float(bundle.get("baseline_no_hr_rate", 0.1085))
    else:
        raise ValueError(
            f"Unexpected no-hr model artifact format at {model_path}. Expected dict-like bundle."
        )

    if model is None:
        raise ValueError(f"Model bundle missing 'model' key: {model_path}")

    if not feat_cols:
        raise ValueError(f"Model bundle missing/empty 'features' key: {model_path}")

    return model, feat_cols, baseline


def main() -> None:
    args = parse_args()
    season = args.season or int(str(args.date)[:4])

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "run_daily_no_hr_v1.log")
    log_header("scripts/live/run_daily_no_hr_v1.py", repo_root, config_path, dirs)
    logging.info("requested season=%s date=%s", season, args.date)

    # Build live context: schedule/spine + projected lineups + weather
    preflight = run_live_preflight(
        repo_root=repo_root,
        config_path=config_path,
        season=season,
        date_str=args.date,
        auto_build=args.auto_build,
        force_spine=True,
        build_lineups=not args.skip_lineups,
        build_weather=not args.skip_weather,
        permissive_live_context=args.permissive_live_context,
    )
    logging.info("live preflight status=%s", preflight)
    live_paths = resolve_live_paths(config=config, season=season, date_str=args.date)
    live_spine = load_live_spine(live_paths["live_spine_path"])
    live_weather = load_live_weather(config=config, season=season, date_str=args.date) if not args.skip_weather else pd.DataFrame()
    live_lineups = load_live_lineups(config=config, season=season, date_str=args.date) if not args.skip_lineups else pd.DataFrame()
    batter_roll_path = dirs["processed_dir"] / "batter_game_rolling.parquet"
    batter_roll = pd.read_parquet(batter_roll_path).copy() if batter_roll_path.exists() else pd.DataFrame()
    lineup_game = build_game_level_lineup_features(live_lineups, batter_roll, pd.to_datetime(args.date, errors="coerce"))
    live_context = merge_live_context(live_spine, live_weather, lineup_game)
    smoke = summarize_live_context(live_context)
    logging.info(
        "live_feature_smoke games=%s pct_with_starters=%.2f pct_with_weather=%.2f pct_with_lineups=%.2f away_lineup_found=%s home_lineup_found=%s",
        smoke["games"],
        smoke["pct_with_starters"],
        smoke["pct_with_weather"],
        smoke["pct_with_lineups"],
        smoke["away_lineup_found"],
        smoke["home_lineup_found"],
    )

    # Build the no-hr mart for the requested season
    mart_cmd = [
        sys.executable,
        "scripts/build_mart_no_hr_game.py",
        "--season",
        str(season),
        "--config",
        str(config_path),
    ]
    if args.force_mart or True:
        mart_cmd.append("--force")
    _run_cmd(mart_cmd, repo_root)

    model_dir = dirs["models_dir"] / "no_hr"
    model_path = _resolve_model_path(model_dir, args.model_path)
    model, feat_cols, baseline = _load_model_bundle(model_path)

    logging.info("using model_path=%s", model_path)
    logging.info("using baseline_no_hr_rate=%.6f", baseline)
    logging.info("model feature count=%s", len(feat_cols))

    mart_path = dirs["processed_dir"] / "marts" / "no_hr" / f"no_hr_game_features_{season}.parquet"
    if not mart_path.exists():
        raise FileNotFoundError(f"No-HR mart not found: {mart_path.resolve()}")

    mart = read_parquet(mart_path)
    mart = _safe_game_filter(mart, args.date)
    if mart.empty:
        raise ValueError(f"No rows found in no-HR mart for season={season} date={args.date}")
    mart = merge_live_context(mart, live_weather, lineup_game)

    # Prepare scoring matrix in exact training-feature order
    X = pd.DataFrame(index=mart.index)
    missing_features: list[str] = []
    for feature in feat_cols:
        if feature in mart.columns:
            X[feature] = pd.to_numeric(mart[feature], errors="coerce")
        else:
            X[feature] = np.nan
            missing_features.append(feature)

    X = X.fillna(0.0).astype("float32")

    pred_delta = model.predict(X)
    pred_prob = np.clip(baseline + pred_delta, 0.01, 0.50)

    out_cols = [c for c in ["game_pk", "game_date", "away_team", "home_team"] if c in mart.columns]
    out = mart[out_cols].copy()
    out["baseline_no_hr_rate"] = baseline
    out["p_no_hr"] = pd.to_numeric(pred_prob, errors="coerce")
    out["delta_vs_baseline"] = out["p_no_hr"] - baseline
    out["grade"] = out["p_no_hr"].apply(_grade_from_prob)
    out = out.sort_values(["p_no_hr", "delta_vs_baseline"], ascending=[False, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)

    outputs_dir = dirs["outputs_dir"] / args.outdir_name
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = outputs_dir / f"no_hr_board_{season}_{args.date}.csv"
    parquet_path = outputs_dir / f"no_hr_board_{season}_{args.date}.parquet"
    out.to_csv(csv_path, index=False)
    write_parquet(out, parquet_path)

    logging.info(
        "daily_no_hr rows=%s p_min=%.6f p_max=%.6f board_csv=%s",
        len(out),
        float(out["p_no_hr"].min()) if len(out) else 0.0,
        float(out["p_no_hr"].max()) if len(out) else 0.0,
        csv_path.resolve(),
    )
    if missing_features:
        logging.warning(
            "Missing model features in live mart (filled with 0.0): count=%s sample=%s",
            len(missing_features),
            missing_features[:10],
        )

    print(f"daily_no_hr_csv -> {csv_path.resolve()}")
    print(f"daily_no_hr_parquet -> {parquet_path.resolve()}")

    if not args.no_board:
        _print_board(out, args.date, top_n=args.board_top)


if __name__ == "__main__":
    main()
