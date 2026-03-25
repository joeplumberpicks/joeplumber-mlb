from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.paths import target_input_candidates
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import write_parquet
from src.utils.logging import configure_logging, log_header


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TB prop mart from hit prop feature spine + TB targets.")
    p.add_argument("--season-start", type=int, required=True)
    p.add_argument("--season-end", type=int, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _load_hit_spine_for_season(dirs: dict[str, Path], season: int) -> pd.DataFrame:
    by_season = dirs["marts_dir"] / "by_season" / f"hit_prop_features_{season}.parquet"
    if by_season.exists():
        return pd.read_parquet(by_season).copy()

    full = dirs["marts_dir"] / "hit_prop_features.parquet"
    if not full.exists():
        raise FileNotFoundError(f"Missing hit_prop mart sources: {by_season} and {full}")

    df = pd.read_parquet(full).copy()
    season_series = pd.to_numeric(df.get("season"), errors="coerce")
    if not season_series.notna().any():
        season_series = pd.to_datetime(df.get("game_date"), errors="coerce").dt.year
    return df[season_series == season].copy()


def _load_tb_targets(processed_dir: Path, season: int) -> pd.DataFrame:
    for path in target_input_candidates(processed_dir, "tb", season):
        if path.exists():
            t = pd.read_parquet(path).copy()
            if "target_tb" not in t.columns:
                continue
            batter_col = next((c for c in ["batter_id", "batter", "player_id", "mlbam_batter_id"] if c in t.columns), None)
            if batter_col is None:
                continue
            t["game_pk"] = pd.to_numeric(t.get("game_pk"), errors="coerce").astype("Int64")
            t["batter_id"] = pd.to_numeric(t[batter_col], errors="coerce").astype("Int64")
            keep = [c for c in ["game_pk", "batter_id", "game_date", "target_tb"] if c in t.columns]
            return t[keep].drop_duplicates(subset=["game_pk", "batter_id"], keep="last")
    return pd.DataFrame(columns=["game_pk", "batter_id", "game_date", "target_tb"])


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_tb_prop_mart.log")
    log_header("scripts/marts/build_tb_prop_mart.py", repo_root, config_path, dirs)

    out_by_season = dirs["marts_dir"] / "by_season"
    out_by_season.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    for season in range(args.season_start, args.season_end + 1):
        features = _load_hit_spine_for_season(dirs, season)
        if features.empty:
            logging.warning("tb_prop_mart season=%s has no hit_prop features; skipping", season)
            continue

        features = features.copy()
        features["game_pk"] = pd.to_numeric(features.get("game_pk"), errors="coerce").astype("Int64")
        features["batter_id"] = pd.to_numeric(features.get("batter_id"), errors="coerce").astype("Int64")
        pre_rows = len(features)

        targets = _load_tb_targets(dirs["processed_dir"], season)
        target_rows = len(targets)
        if "game_date" not in features.columns and "game_date" in targets.columns:
            features = features.merge(targets[["game_pk", "batter_id", "game_date"]], on=["game_pk", "batter_id"], how="left")

        merged = features.merge(targets[["game_pk", "batter_id", "target_tb"]], on=["game_pk", "batter_id"], how="left")
        if len(merged) != pre_rows:
            raise ValueError(
                f"tb_prop_mart row count changed after left merge season={season} pre={pre_rows} post={len(merged)}"
            )

        matched_rows = int(pd.to_numeric(merged.get("target_tb"), errors="coerce").notna().sum())
        target_mean = float(pd.to_numeric(merged.get("target_tb"), errors="coerce").mean()) if matched_rows else 0.0
        logging.info(
            "tb_prop_mart season=%s feature_rows=%s target_rows=%s matched_rows=%s target_mean=%.6f",
            season,
            pre_rows,
            target_rows,
            matched_rows,
            target_mean,
        )

        season_path = out_by_season / f"tb_prop_features_{season}.parquet"
        write_parquet(merged, season_path)
        all_frames.append(merged)

    if not all_frames:
        raise ValueError("tb_prop_mart produced no seasonal outputs")

    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    combined_path = dirs["marts_dir"] / "tb_prop_features.parquet"
    write_parquet(combined, combined_path)
    print(f"tb_prop_mart_out={combined_path}")


if __name__ == "__main__":
    main()
