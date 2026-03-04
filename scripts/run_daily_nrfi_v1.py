from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

MODEL_VERSION = "nrfi_xgb_v1.0"
A_TIER = {"A+", "A", "A-"}
GRADE_THRESHOLDS = [
    ("A+", 0.70),
    ("A", 0.67),
    ("A-", 0.64),
    ("B+", 0.61),
    ("B", 0.58),
    ("B-", 0.55),
    ("C+", 0.53),
    ("C", 0.51),
    ("C-", 0.50),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run daily NRFI v1.0 scoring with optional schedule-spine auto-build.")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--season", type=int, default=None)
    p.add_argument("--auto-build", action="store_true")
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _grade_from_prob(prob: float) -> str:
    p = float(prob)
    for grade, threshold in GRADE_THRESHOLDS:
        if p >= threshold:
            return grade
    return "C-"


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    logging.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def _has_targets(df: pd.DataFrame) -> bool:
    return "target_nrfi" in df.columns


def _load_features(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature list not found: {path}")
    feats = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not feats:
        raise ValueError(f"Feature list empty: {path}")
    return feats


def main() -> None:
    args = parse_args()
    date_ts = pd.to_datetime(args.date, format="%Y-%m-%d", errors="raise")
    season = int(args.season) if args.season is not None else int(date_ts.year)

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "run_daily_nrfi_v1_root.log")
    log_header("scripts/run_daily_nrfi_v1.py", repo_root, config_path, dirs)

    data_root = dirs["data_root"]

    if args.auto_build:
        _run_cmd(
            [
                sys.executable,
                "scripts/ingest/ingest_schedule_games.py",
                "--date",
                args.date,
                "--season",
                str(season),
                "--game-types",
                "S,R",
                "--config",
                str(args.config),
            ],
            cwd=repo_root,
        )
        _run_cmd(
            [
                sys.executable,
                "scripts/live/build_spine_from_schedule.py",
                "--season",
                str(season),
                "--date",
                args.date,
                "--config",
                str(args.config),
                "--force",
            ],
            cwd=repo_root,
        )

    mart_path = data_root / "marts" / "by_season" / f"nrfi_features_{season}.parquet"
    if not mart_path.exists():
        fallback = data_root / "marts" / "by_season" / "nrfi_features_2025.parquet"
        if fallback.exists():
            logging.warning("Season mart missing for %s, using fallback: %s", season, fallback)
            mart_path = fallback
        else:
            raise FileNotFoundError(f"No mart found for season={season} and no 2025 fallback at {fallback}")

    model_path = data_root / "models" / "nrfi_xgb" / "releases" / "v1.0" / "nrfi_model.json"
    features_path = data_root / "models" / "nrfi_xgb" / "releases" / "v1.0" / "features_240.txt"

    daily_out_path = data_root / "outputs" / "nrfi_xgb" / "v1.0" / "daily" / f"{args.date}_predictions.csv"
    public_out_path = data_root / "outputs" / "nrfi_xgb" / "v1.0" / "public" / f"{args.date}_A_tier_picks.csv"
    ledger_path = data_root / "public_ledgers" / "nrfi_xgb" / "v1.0" / "ledger.csv"

    daily_out_path.parent.mkdir(parents=True, exist_ok=True)
    public_out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    mart = pd.read_parquet(mart_path)
    if "game_date" not in mart.columns:
        raise ValueError(f"Mart missing game_date column: {mart_path}")
    mart["game_date"] = pd.to_datetime(mart["game_date"], errors="coerce")
    daily = mart[mart["game_date"].dt.date == date_ts.date()].copy()

    if daily.empty:
        raise RuntimeError(f"No games found in mart={mart_path} for date={args.date}")

    features = _load_features(features_path)
    X = daily.reindex(columns=features).copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmat = xgb.DMatrix(X, feature_names=features)

    daily["p_nrfi"] = booster.predict(dmat)
    daily["p_yrfi"] = 1.0 - daily["p_nrfi"]
    daily["pick"] = daily["p_nrfi"].ge(0.5).map({True: "NRFI", False: "YRFI"})
    daily["pick_prob"] = daily[["p_nrfi", "p_yrfi"]].max(axis=1)
    daily["grade"] = daily["pick_prob"].map(_grade_from_prob)

    if _has_targets(daily):
        daily["actual"] = daily["target_nrfi"].map(lambda x: "NRFI" if pd.notna(x) and int(x) == 1 else "YRFI")
        daily["win"] = (daily["pick"] == daily["actual"]).astype(int)
    else:
        daily["actual"] = ""
        daily["win"] = ""

    daily["game_date"] = daily["game_date"].dt.strftime("%Y-%m-%d")

    required = [
        "game_date",
        "game_pk",
        "away_team",
        "home_team",
        "p_nrfi",
        "p_yrfi",
        "pick",
        "pick_prob",
        "grade",
        "actual",
        "win",
    ]
    for c in required:
        if c not in daily.columns:
            daily[c] = ""
    daily_out = daily[required].copy()
    daily_out.to_csv(daily_out_path, index=False)

    public_cols = ["game_date", "away_team", "home_team", "pick", "grade", "pick_prob", "p_nrfi", "p_yrfi", "game_pk"]
    a_tier = daily[daily["grade"].isin(A_TIER)].copy()
    for c in public_cols:
        if c not in a_tier.columns:
            a_tier[c] = ""
    a_tier = a_tier[public_cols].sort_values("pick_prob", ascending=False, kind="mergesort")
    a_tier.to_csv(public_out_path, index=False)

    ledger_add = daily_out.copy()
    ledger_add["run_date"] = args.date
    ledger_add["model_version"] = MODEL_VERSION
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        ledger = pd.concat([ledger, ledger_add], ignore_index=True, sort=False)
    else:
        ledger = ledger_add

    ledger = ledger.drop_duplicates(subset=["model_version", "game_pk", "game_date"], keep="last")
    ledger["_gd"] = pd.to_datetime(ledger["game_date"], errors="coerce")
    ledger["_pk"] = pd.to_numeric(ledger["game_pk"], errors="coerce")
    ledger = ledger.sort_values(["_gd", "_pk"], kind="mergesort").drop(columns=["_gd", "_pk"])
    ledger.to_csv(ledger_path, index=False)

    logging.info(
        "daily nrfi run complete date=%s season=%s rows=%s daily_out=%s public_out=%s ledger=%s",
        args.date,
        season,
        len(daily_out),
        daily_out_path,
        public_out_path,
        ledger_path,
    )
    print(f"daily_out={daily_out_path}")
    print(f"public_out={public_out_path}")
    print(f"ledger_out={ledger_path}")


if __name__ == "__main__":
    main()
