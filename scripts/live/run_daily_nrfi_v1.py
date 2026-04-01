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
from src.live.daily_context import load_live_lineups, load_live_spine, load_live_weather, resolve_live_paths, run_live_preflight
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
    parser.add_argument("--auto-build", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-grade", default="A-")
    parser.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--skip-lineups", action="store_true")
    parser.add_argument("--skip-weather", action="store_true")
    parser.add_argument("--permissive-live-context", action="store_true")
    parser.add_argument("--board-top", type=int, default=15)
    parser.add_argument("--suppress-preflight-summary", action="store_true")
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


def _resolve_live_mart_path(
    *,
    marts_by_season_dir: Path,
    requested_season: int,
    run_date: pd.Timestamp,
    preferred_fallback_season: int = 2025,
) -> tuple[Path, int]:
    def _date_rows(path: Path) -> int:
        if not path.exists():
            return 0
        try:
            df = pd.read_parquet(path, columns=["game_date"])
        except Exception:
            df = pd.read_parquet(path)
        if "game_date" not in df.columns or df.empty:
            return 0
        d = pd.to_datetime(df["game_date"], errors="coerce")
        return int((d.dt.date == run_date.date()).sum())

    requested_path = marts_by_season_dir / f"nrfi_features_{requested_season}.parquet"
    requested_rows = _date_rows(requested_path)
    if requested_rows > 0:
        return requested_path, requested_season

    candidate_seasons: list[int] = []
    if preferred_fallback_season != requested_season:
        candidate_seasons.append(preferred_fallback_season)
    candidate_seasons.extend(range(requested_season - 1, 2014, -1))
    seen: set[int] = set()
    for season in candidate_seasons:
        if season in seen:
            continue
        seen.add(season)
        path = marts_by_season_dir / f"nrfi_features_{season}.parquet"
        if _date_rows(path) > 0:
            logging.info(
                "live scoring mart fallback: requested season=%s using mart season=%s because %s mart missing/empty for date=%s",
                requested_season,
                season,
                requested_season,
                run_date.strftime("%Y-%m-%d"),
            )
            return path, season
    raise FileNotFoundError(
        f"No usable NRFI mart found for date={run_date.strftime('%Y-%m-%d')} requested_season={requested_season}"
    )


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
    preflight = run_live_preflight(
        repo_root=repo_root,
        config_path=config_path,
        season=season,
        date_str=args.date,
        auto_build=bool(args.auto_build),
        force_spine=True,
        build_lineups=not args.skip_lineups,
        build_weather=not args.skip_weather,
        permissive_live_context=bool(args.permissive_live_context),
        emit_summary_line=not args.suppress_preflight_summary,
    )
    live_paths = resolve_live_paths(config=config, season=season, date_str=args.date)
    live_spine = load_live_spine(live_paths["live_spine_path"])
    live_game_pks = set(pd.to_numeric(live_spine["game_pk"], errors="coerce").dropna().astype(int).tolist())
    if not args.skip_weather:
        _ = load_live_weather(config=config, season=season, date_str=args.date)
    if not args.skip_lineups:
        _ = load_live_lineups(config=config, season=season, date_str=args.date)

    release_dir = DRIVE_ROOT / "data/models/nrfi_xgb/releases/v1.0"
    nrfi_model_path = release_dir / "nrfi_model.json"
    yrfi_model_path = release_dir / "yrfi_model.json"
    features_path = release_dir / "features_240.txt"

    marts_by_season_dir = DRIVE_ROOT / "data/marts/by_season"
    mart_path, mart_season = _resolve_live_mart_path(
        marts_by_season_dir=marts_by_season_dir,
        requested_season=season,
        run_date=run_date,
    )

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

    df = pd.read_parquet(mart_path)
    if "game_date" not in df.columns:
        raise ValueError(f"Expected game_date column in mart: {mart_path}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    daily = df[df["game_date"].dt.date == run_date.date()].copy()
    if live_game_pks:
        daily["game_pk"] = pd.to_numeric(daily.get("game_pk"), errors="coerce").astype("Int64")
        daily = daily[daily["game_pk"].isin(list(live_game_pks))].copy()

    if daily.empty:
        logging.error("No games found for date=%s in mart=%s", args.date, mart_path)
        raise SystemExit(1)

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
        "NRFI v1.0 daily run complete | date=%s season=%s mart_season=%s rows=%s slate_games=%s daily_csv=%s public_csv=%s a_tier_csv=%s ledger=%s a_tier_count=%s preflight_spine_rows=%s",
        args.date,
        season,
        mart_season,
        len(daily_out_df),
        len(live_game_pks),
        daily_out,
        public_out,
        a_tier_out,
        ledger_path,
        len(a_tier_df),
        preflight.get("final_game_spine_row_count"),
    )

    print(f"date={args.date}")
    print(f"season={season}")
    print(f"mart_season={mart_season}")
    print(f"rows={len(daily_out_df)}")
    print(f"daily_csv={daily_out}")
    print(f"public_csv={public_out}")
    print(f"a_tier_csv={a_tier_out}")
    print(f"ledger_csv={ledger_path}")
    print(f"a_tier_count={len(a_tier_df)}")


if __name__ == "__main__":
    main()
