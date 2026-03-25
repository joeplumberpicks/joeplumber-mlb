from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.targets.paths import target_input_candidates
from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import write_parquet
from src.utils.logging import configure_logging, log_header

LINEUP_PA_MAP = {1: 4.65, 2: 4.55, 3: 4.45, 4: 4.35, 5: 4.25, 6: 4.10, 7: 3.95, 8: 3.80, 9: 3.70}


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


def _coalesce_numeric(df: pd.DataFrame, out_col: str, candidates: list[str]) -> None:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for c in candidates:
        if c in df.columns:
            out = out.fillna(pd.to_numeric(df[c], errors="coerce"))
    df[out_col] = out


def _add_tb_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # lineup + expected volume helpers
    _coalesce_numeric(out, "lineup_slot_numeric", ["lineup_slot_numeric", "lineup_slot", "bat_order", "batting_order", "lineup_position", "order"])
    if "expected_batting_order_pa" not in out.columns or pd.to_numeric(out["expected_batting_order_pa"], errors="coerce").isna().all():
        out["expected_batting_order_pa"] = pd.to_numeric(out["lineup_slot_numeric"], errors="coerce").map(LINEUP_PA_MAP)
    _coalesce_numeric(out, "bat_ab_per_game_roll15", ["bat_ab_per_game_roll15", "ab_per_game_roll15", "bat_ab_roll15"])
    _coalesce_numeric(out, "bat_pa_per_game_roll15", ["bat_pa_per_game_roll15", "pa_per_game_roll15", "bat_pa_roll15"])
    if "expected_ab_proxy" not in out.columns or pd.to_numeric(out["expected_ab_proxy"], errors="coerce").isna().all():
        out["expected_ab_proxy"] = (
            0.65 * pd.to_numeric(out["bat_ab_per_game_roll15"], errors="coerce")
            + 0.35 * pd.to_numeric(out["expected_batting_order_pa"], errors="coerce")
        )

    # weather/environment normalization
    _coalesce_numeric(out, "temperature", ["temperature", "temp_f", "game_temp", "weather_temp"])
    _coalesce_numeric(out, "wind_speed", ["wind_speed", "weather_wind", "wind_mph", "wind"])
    _coalesce_numeric(out, "weather_wind_out", ["weather_wind_out", "wind_out", "wind_out_flag"])
    _coalesce_numeric(out, "weather_wind_in", ["weather_wind_in", "wind_in", "wind_in_flag"])

    # TB-oriented rolling proxies
    _coalesce_numeric(out, "tb_hits_roll15", ["bat_h_roll15", "h_roll15", "bat_hits_roll15", "hits_roll15"])
    _coalesce_numeric(out, "tb_hr_roll15", ["bat_hr_roll15", "hr_roll15", "bat_home_runs_roll15", "home_runs_roll15"])
    _coalesce_numeric(out, "tb_2b_roll15", ["bat_double_roll15", "double_roll15", "bat_2b_roll15", "2b_roll15", "bat_doubles_roll15", "doubles_roll15"])
    _coalesce_numeric(out, "tb_3b_roll15", ["bat_triple_roll15", "triple_roll15", "bat_3b_roll15", "3b_roll15", "bat_triples_roll15", "triples_roll15"])
    _coalesce_numeric(out, "tb_ab_roll15", ["bat_ab_roll15", "ab_roll15", "at_bats_roll15", "bat_ab_per_game_roll15"])
    _coalesce_numeric(out, "tb_pa_roll15", ["bat_pa_roll15", "pa_roll15", "plate_appearances_roll15", "bat_pa_per_game_roll15"])
    _coalesce_numeric(out, "tb_launch_speed_roll15", ["launch_speed_mean_roll15", "bat_launch_speed_mean_roll15"])
    _coalesce_numeric(out, "tb_launch_angle_roll15", ["launch_angle_mean_roll15", "bat_launch_angle_mean_roll15"])
    _coalesce_numeric(out, "tb_whiff_rate_roll30", ["whiff_rate_roll30", "bat_whiff_rate_roll30", "diff_off_whiff_rate_roll30"])
    _coalesce_numeric(out, "tb_contact_rate_roll30", ["contact_rate_roll30", "bat_contact_rate_roll30", "diff_off_contact_rate_roll30"])
    _coalesce_numeric(out, "tb_contact_rate_roll15", ["contact_rate_roll15", "bat_contact_rate_roll15", "tb_contact_rate_roll30"])

    # rate/proxy helpers
    out["tb_xbh_roll15"] = (
        pd.to_numeric(out["tb_2b_roll15"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["tb_3b_roll15"], errors="coerce").fillna(0.0)
        + pd.to_numeric(out["tb_hr_roll15"], errors="coerce").fillna(0.0)
    )
    pa15 = pd.to_numeric(out["tb_pa_roll15"], errors="coerce")
    ab15 = pd.to_numeric(out["tb_ab_roll15"], errors="coerce")
    denom = pa15.where(pa15 > 0, ab15)
    denom = denom.where(denom > 0, np.nan)
    h15 = pd.to_numeric(out["tb_hits_roll15"], errors="coerce").fillna(0.0)
    hr15 = pd.to_numeric(out["tb_hr_roll15"], errors="coerce").fillna(0.0)
    dbl15 = pd.to_numeric(out["tb_2b_roll15"], errors="coerce").fillna(0.0)
    trp15 = pd.to_numeric(out["tb_3b_roll15"], errors="coerce").fillna(0.0)
    xbh15 = pd.to_numeric(out["tb_xbh_roll15"], errors="coerce").fillna(0.0)
    sing15 = (h15 - xbh15).clip(lower=0)
    out["tb_hit_rate_proxy"] = (h15 / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["tb_xbh_rate"] = (xbh15 / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["tb_weighted_base_event_proxy"] = (
        (sing15 + 2.0 * dbl15 + 3.0 * trp15 + 4.0 * hr15) / denom
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["tb_slugging_proxy"] = out["tb_weighted_base_event_proxy"]
    out["tb_slug_proxy"] = out["tb_slugging_proxy"]
    out["tb_iso_proxy"] = (out["tb_slugging_proxy"] - out["tb_hit_rate_proxy"]).clip(lower=0.0)
    out["tb_hr_rate"] = (hr15 / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["tb_power_boost"] = out["tb_xbh_rate"] * out["tb_hr_rate"]
    out["tb_contact_power_interaction"] = (
        pd.to_numeric(out["tb_contact_rate_roll15"], errors="coerce").fillna(0.0)
        * pd.to_numeric(out["tb_slugging_proxy"], errors="coerce").fillna(0.0)
    )

    for c in [
        "tb_slugging_proxy",
        "tb_iso_proxy",
        "tb_xbh_rate",
        "tb_hr_rate",
        "tb_power_boost",
        "tb_contact_power_interaction",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.5)

    # lineup bucket encoding (numeric-only)
    ls = pd.to_numeric(out["lineup_slot_numeric"], errors="coerce")
    out["lineup_bucket_top"] = ls.between(1, 3).astype(float)
    out["lineup_bucket_mid"] = ls.between(4, 6).astype(float)
    out["lineup_bucket_bottom"] = ls.between(7, 9).astype(float)

    # interactions
    out["contact_quality_x_volume"] = pd.to_numeric(out["tb_launch_speed_roll15"], errors="coerce") * pd.to_numeric(out["expected_ab_proxy"], errors="coerce")
    fallback_cq = pd.to_numeric(out["tb_contact_rate_roll15"], errors="coerce").fillna(0.0) * pd.to_numeric(out["tb_pa_roll15"], errors="coerce").fillna(0.0)
    out["contact_quality_x_volume"] = pd.to_numeric(out["contact_quality_x_volume"], errors="coerce").fillna(fallback_cq)
    out["xbh_proxy_x_expected_ab"] = pd.to_numeric(out["tb_xbh_rate"], errors="coerce") * pd.to_numeric(out["expected_ab_proxy"], errors="coerce")
    out["xbh_proxy_x_expected_ab"] = pd.to_numeric(out["xbh_proxy_x_expected_ab"], errors="coerce").fillna(
        pd.to_numeric(out["tb_xbh_rate"], errors="coerce") * pd.to_numeric(out["tb_pa_roll15"], errors="coerce")
    )
    _coalesce_numeric(out, "park_factor_xbh_proxy", ["park_factor_xbh_blend", "park_factor_xbh_hist_shrunk", "park_factor_hits_blend"])
    out["launch_speed_x_park_factor"] = pd.to_numeric(out["tb_launch_speed_roll15"], errors="coerce") * pd.to_numeric(out["park_factor_xbh_proxy"], errors="coerce")
    feature_cov_cols = [
        "tb_hit_rate_proxy",
        "tb_xbh_rate",
        "tb_hr_rate",
        "tb_power_boost",
        "tb_contact_power_interaction",
        "tb_slugging_proxy",
        "tb_iso_proxy",
        "tb_weighted_base_event_proxy",
        "contact_quality_x_volume",
        "xbh_proxy_x_expected_ab",
    ]
    cov = {c: float(pd.to_numeric(out.get(c), errors="coerce").notna().mean()) for c in feature_cov_cols if c in out.columns}
    sample = out[feature_cov_cols].head(5).to_dict("records") if all(c in out.columns for c in feature_cov_cols) else []
    slug_stats = pd.to_numeric(out.get("tb_slugging_proxy"), errors="coerce")
    logging.info(
        "tb_prop_mart slugging_proxy_stats mean=%.6f std=%.6f min=%.6f max=%.6f",
        float(slug_stats.mean()),
        float(slug_stats.std()),
        float(slug_stats.min()),
        float(slug_stats.max()),
    )
    logging.info("tb_prop_mart engineered_feature_coverage=%s sample=%s", cov, sample)
    return out


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

        merged = _add_tb_features(merged)
        matched_rows = int(pd.to_numeric(merged.get("target_tb"), errors="coerce").notna().sum())
        target_mean = float(pd.to_numeric(merged.get("target_tb"), errors="coerce").mean()) if matched_rows else 0.0
        engineered = [
            "lineup_slot_numeric", "expected_batting_order_pa", "expected_ab_proxy", "temperature", "wind_speed",
            "tb_hit_rate_proxy", "tb_xbh_rate", "tb_iso_proxy", "tb_weighted_base_event_proxy", "tb_slugging_proxy",
            "tb_hr_rate", "tb_power_boost", "tb_contact_power_interaction",
            "contact_quality_x_volume", "xbh_proxy_x_expected_ab", "launch_speed_x_park_factor",
        ]
        engineered_non_null = {c: float(pd.to_numeric(merged.get(c), errors="coerce").notna().mean()) for c in engineered if c in merged.columns}
        logging.info(
            "tb_prop_mart season=%s feature_rows=%s target_rows=%s matched_rows=%s target_mean=%.6f engineered_non_null=%s",
            season,
            pre_rows,
            target_rows,
            matched_rows,
            target_mean,
            engineered_non_null,
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
