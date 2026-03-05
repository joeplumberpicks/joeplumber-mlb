from __future__ import annotations

"""
Usage:
  python scripts/live/run_daily_nrfi_v1.py --date 2026-03-31
  python scripts/live/run_daily_nrfi_v1.py --date 2026-03-31 --auto-build
  python scripts/live/run_daily_nrfi_v1.py --date 2026-03-31 --min-grade A
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.logging import configure_logging, log_header

DRIVE_ROOT = Path("/content/drive/MyDrive/joeplumber-mlb")
MODEL_VERSION = "nrfi_xgb_v1.0"
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
GRADE_ORDER = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run daily NRFI/YRFI v1.0 predictions and public picks.")
    parser.add_argument("--date", required=True, help="Scoring date in YYYY-MM-DD")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--auto-build", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-grade", default="A-")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return parser.parse_args()


def _validate_date(date_str: str) -> pd.Timestamp:
    try:
        dt = pd.to_datetime(date_str, format="%Y-%m-%d", errors="raise")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid --date format: {date_str}. Expected YYYY-MM-DD") from exc
    return dt


def _load_features(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Locked feature list not found: {path}")
    features = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not features:
        raise ValueError(f"Locked feature list is empty: {path}")
    return features


def _grade_from_prob(prob: float) -> str:
    p = float(prob)
    for grade, threshold in GRADE_THRESHOLDS:
        if p >= threshold:
            return grade
    return "C-"


def _grade_at_or_above(grade: str, min_grade: str) -> bool:
    rank = {g: i for i, g in enumerate(GRADE_ORDER)}
    if grade not in rank or min_grade not in rank:
        raise ValueError(f"Unknown grade. grade={grade} min_grade={min_grade} supported={GRADE_ORDER}")
    return rank[grade] <= rank[min_grade]




def _log_nrfi_block_sanity(daily: pd.DataFrame) -> None:
    fi_cols = sorted([c for c in daily.columns if c.startswith(("away_sp_sp_fi_", "home_sp_sp_fi_"))])
    top3_cols = sorted([c for c in daily.columns if "_top3_" in c and (c.startswith("away_") or c.startswith("home_"))])
    check_cols = fi_cols + top3_cols
    if not check_cols:
        logging.warning("NRFI feature-block sanity: no SP-FI or top3 columns found in daily mart slice")
        return
    stats = []
    for c in check_cols:
        nn = float(pd.to_numeric(daily[c], errors="coerce").notna().mean()) if c in daily.columns else 0.0
        stats.append(f"{c}={nn:.3f}")
    logging.info("NRFI feature-block sanity non-null ratios | %s", " | ".join(stats))
def _load_booster(path: Path) -> xgb.Booster:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def _run_cmd(cmd: list[str], cwd: Path) -> bool:
    logging.info("auto-build: running %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        logging.warning("auto-build command failed (exit=%s): %s", proc.returncode, " ".join(cmd))
        return False
    return True


def _auto_build(repo_root: Path, season: int, date_str: str) -> None:
    cmds = [
        [
            sys.executable,
            "scripts/ingest/run_ingest_season.py",
            "--season",
            str(season),
            "--start",
            date_str,
            "--end",
            date_str,
            "--chunk-days",
            "5",
        ],
        [sys.executable, "scripts/build_spine.py", "--season", str(season), "--force"],
        [sys.executable, "scripts/build_marts.py", "--season", str(season), "--force"],
    ]
    for cmd in cmds:
        _run_cmd(cmd, cwd=repo_root)


def main() -> None:
    args = parse_args()
    run_date = _validate_date(args.date)
    season = int(args.season) if args.season is not None else int(run_date.year)

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "run_daily_nrfi_v1.log")
    log_header("scripts/live/run_daily_nrfi_v1.py", repo_root, config_path, dirs)

    release_dir = DRIVE_ROOT / "data/models/nrfi_xgb/releases/v1.0"
    nrfi_model_path = release_dir / "nrfi_model.json"
    yrfi_model_path = release_dir / "yrfi_model.json"
    features_path = release_dir / "features_240.txt"

    mart_path = DRIVE_ROOT / f"data/marts/by_season/nrfi_features_{season}.parquet"

    out_dir = DRIVE_ROOT / "data/outputs/nrfi_xgb/v1.0/daily"
    pub_dir = DRIVE_ROOT / "data/outputs/nrfi_xgb/v1.0/public"
    ledger_path = DRIVE_ROOT / "data/public_ledgers/nrfi_xgb/v1.0/ledger.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    pub_dir.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    daily_out = out_dir / f"{args.date}_predictions.csv"
    public_out = pub_dir / f"{args.date}_{args.min_grade}_tier_picks.csv"
    a_tier_out = pub_dir / f"{args.date}_A_tier_picks.csv"

    if not args.force:
        for path in [daily_out, public_out, a_tier_out]:
            if path.exists():
                raise FileExistsError(f"Output exists (use --force to overwrite): {path}")

    if args.auto_build:
        _auto_build(repo_root, season, args.date)

    if not mart_path.exists():
        raise FileNotFoundError(
            f"Mart not found for season={season}: {mart_path}. "
            "Run with --auto-build or build marts upstream first."
        )

    df = pd.read_parquet(mart_path)
    if "game_date" not in df.columns:
        raise ValueError(f"Expected game_date column in mart: {mart_path}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    daily = df[df["game_date"].dt.date == run_date.date()].copy()

    if daily.empty:
        logging.error("No games found for date=%s in mart=%s", args.date, mart_path)
        raise SystemExit(1)

    _log_nrfi_block_sanity(daily)

    features = _load_features(features_path)
    X = daily.reindex(columns=features).copy()
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    dmat = xgb.DMatrix(X, feature_names=features)
    nrfi_booster = _load_booster(nrfi_model_path)

    p_nrfi = nrfi_booster.predict(dmat)
    p_yrfi = 1.0 - p_nrfi

    if yrfi_model_path.exists():
        logging.info("Found yrfi_model.json at %s (default scoring uses 1-p_nrfi).", yrfi_model_path)
    else:
        logging.info("yrfi_model.json not present; using p_yrfi = 1 - p_nrfi.")

    daily["p_nrfi"] = p_nrfi
    daily["p_yrfi"] = p_yrfi
    daily["pick"] = daily["p_nrfi"].ge(0.5).map({True: "NRFI", False: "YRFI"})
    daily["pick_prob"] = daily[["p_nrfi", "p_yrfi"]].max(axis=1)
    daily["grade"] = daily["pick_prob"].map(_grade_from_prob)

    if "target_nrfi" in daily.columns:
        daily["actual"] = daily["target_nrfi"].map(lambda x: "NRFI" if pd.notna(x) and int(x) == 1 else "YRFI")
        daily["win"] = (daily["pick"] == daily["actual"]).astype(int)
    else:
        daily["actual"] = ""
        daily["win"] = ""

    daily["game_date"] = daily["game_date"].dt.strftime("%Y-%m-%d")

    required_cols = [
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
    for c in required_cols:
        if c not in daily.columns:
            daily[c] = ""
    daily_out_df = daily[required_cols].copy()
    daily_out_df.to_csv(daily_out, index=False)

    pub_cols = ["game_date", "away_team", "home_team", "pick", "grade", "pick_prob", "p_nrfi", "p_yrfi", "game_pk"]
    for c in pub_cols:
        if c not in daily.columns:
            daily[c] = ""

    public_df = daily[daily["grade"].map(lambda g: _grade_at_or_above(g, args.min_grade))][pub_cols].copy()
    public_df.to_csv(public_out, index=False)

    a_tier_min = "A-"
    a_tier_df = daily[daily["grade"].map(lambda g: _grade_at_or_above(g, a_tier_min))][pub_cols].copy()
    a_tier_df.to_csv(a_tier_out, index=False)

    ledger_add = daily_out_df.copy()
    ledger_add["run_date"] = args.date
    ledger_add["model_version"] = MODEL_VERSION

    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        ledger = pd.concat([ledger, ledger_add], ignore_index=True, sort=False)
    else:
        ledger = ledger_add

    ledger = ledger.drop_duplicates(subset=["model_version", "game_pk", "game_date"], keep="last")
    ledger["_game_date_sort"] = pd.to_datetime(ledger["game_date"], errors="coerce")
    ledger["_game_pk_sort"] = pd.to_numeric(ledger["game_pk"], errors="coerce")
    ledger = ledger.sort_values(["_game_date_sort", "_game_pk_sort"], kind="mergesort")
    ledger = ledger.drop(columns=["_game_date_sort", "_game_pk_sort"])
    ledger.to_csv(ledger_path, index=False)

    logging.info(
        "NRFI v1.0 daily run complete | date=%s season=%s rows=%s daily_csv=%s public_csv=%s a_tier_csv=%s ledger=%s a_tier_count=%s",
        args.date,
        season,
        len(daily_out_df),
        daily_out,
        public_out,
        a_tier_out,
        ledger_path,
        len(a_tier_df),
    )

    print(f"date={args.date}")
    print(f"season={season}")
    print(f"rows={len(daily_out_df)}")
    print(f"daily_csv={daily_out}")
    print(f"public_csv={public_out}")
    print(f"a_tier_csv={a_tier_out}")
    print(f"ledger_csv={ledger_path}")
    print(f"a_tier_count={len(a_tier_df)}")


if __name__ == "__main__":
    main()
