from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.live.daily_context import run_live_preflight
from src.utils.config import get_repo_root, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run official daily game-level engine suite (NRFI + Moneyline).")
    p.add_argument("--date", required=True)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    p.add_argument("--engines", default="nrfi,moneyline")
    p.add_argument("--auto-build", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-lineups", action="store_true")
    p.add_argument("--skip-weather", action="store_true")
    p.add_argument("--permissive-live-context", action="store_true")
    p.add_argument("--fallback-season", type=int, default=2025)
    p.add_argument("--board-top", type=int, default=15)
    return p.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit={proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    _ = load_config(config_path)

    preflight = run_live_preflight(
        repo_root=repo_root,
        config_path=config_path,
        season=args.season,
        date_str=args.date,
        auto_build=bool(args.auto_build),
        force_spine=True,
        build_lineups=not args.skip_lineups,
        build_weather=not args.skip_weather,
        permissive_live_context=bool(args.permissive_live_context),
    )
    print(
        "preflight_complete "
        f"slate_games={preflight.get('slate_game_count', 0)} "
        f"final_spine_rows={preflight.get('final_game_spine_row_count', 0)}"
    )

    selected = [x.strip() for x in args.engines.split(",") if x.strip()]
    produced: list[str] = []

    for engine in selected:
        print(f"\n===== DAILY ENGINE: {engine.upper()} =====")
        if engine == "nrfi":
            cmd = [
                sys.executable,
                "scripts/live/run_daily_nrfi_v1.py",
                "--date",
                args.date,
                "--season",
                str(args.season),
                "--config",
                str(args.config),
                "--board-top",
                str(args.board_top),
                "--no-auto-build",
            ]
            if args.skip_lineups:
                cmd.append("--skip-lineups")
            if args.skip_weather:
                cmd.append("--skip-weather")
            if args.permissive_live_context:
                cmd.append("--permissive-live-context")
            _run(cmd, repo_root)
            produced.append(f"nrfi_board_{args.season}_{args.date}")
        elif engine == "moneyline":
            cmd = [
                sys.executable,
                "scripts/live/run_daily_moneyline_v1.py",
                "--date",
                args.date,
                "--season",
                str(args.season),
                "--config",
                str(args.config),
                "--fallback-season",
                str(args.fallback_season),
                "--no-auto-build",
            ]
            if args.skip_lineups:
                cmd.append("--skip-lineups")
            if args.skip_weather:
                cmd.append("--skip-weather")
            if args.permissive_live_context:
                cmd.append("--permissive-live-context")
            _run(cmd, repo_root)
            produced.append(f"moneyline_board_{args.season}_{args.date}")
        else:
            raise ValueError(f"Unknown engine: {engine}")

    print("\n=== Daily engine run summary ===")
    for p in produced:
        print(f"produced={p}")


if __name__ == "__main__":
    main()
