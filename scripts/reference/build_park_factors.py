from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_repo_root, load_config
from src.utils.drive import resolve_data_dirs
from src.utils.io import write_parquet
from src.utils.logging import configure_logging, log_header

HIT_EVENTS = {"single", "double", "triple", "home_run"}
BIP_OUT_EVENTS = {
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "double_play",
    "triple_play",
    "fielders_choice",
    "fielders_choice_out",
    "sac_fly",
    "sac_fly_double_play",
    "sac_bunt",
    "sac_bunt_double_play",
    "lineout",
    "flyout",
    "groundout",
    "pop_out",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build rich historical park factor profiles.")
    p.add_argument("--season-start", type=int, default=2019)
    p.add_argument("--season-end", type=int, default=2025)
    p.add_argument("--k-hits", type=float, default=4000.0)
    p.add_argument("--k-runs", type=float, default=4500.0)
    p.add_argument("--k-hr", type=float, default=7000.0)
    p.add_argument("--k-xbh", type=float, default=6500.0)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], cands: list[str]) -> str | None:
    s = set(cols)
    for c in cands:
        if c in s:
            return c
    return None


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(pd.to_numeric(den, errors="coerce") > 0, pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce"), np.nan)


def _mode_or_na(s: pd.Series) -> object:
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else pd.NA


def _get_hard_hit(pa: pd.DataFrame) -> pd.Series:
    hard_col = _pick(list(pa.columns), ["hard_hit", "is_hard_hit", "hard_hit_flag"])
    if hard_col is not None:
        hard = pd.to_numeric(pa[hard_col], errors="coerce")
        if hard.notna().any():
            return (hard > 0).astype(float)
    ls_col = _pick(list(pa.columns), ["launch_speed", "launch_speed_mean", "launch_speed_mph"])  # PA-level expected
    if ls_col is not None:
        ls = pd.to_numeric(pa[ls_col], errors="coerce")
        return (ls >= 95.0).astype(float)
    return pd.Series(np.nan, index=pa.index, dtype="float64")


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)
    configure_logging(dirs["logs_dir"] / "build_park_factors.log")
    log_header("scripts/reference/build_park_factors.py", repo_root, config_path, dirs)

    frames: list[pd.DataFrame] = []
    included_seasons: list[int] = []
    total_rows = 0
    dropped_no_park = 0

    for season in range(args.season_start, args.season_end + 1):
        pa_path = dirs["processed_dir"] / "by_season" / f"pa_{season}.parquet"
        if not pa_path.exists():
            logging.warning("missing PA file for season=%s path=%s", season, pa_path)
            continue

        pa = pd.read_parquet(pa_path)
        if "game_pk" not in pa.columns:
            logging.warning("skipping season=%s (missing game_pk)", season)
            continue

        park_key_col = _pick(list(pa.columns), ["canonical_park_key", "park_key", "venue_id", "park_id"])
        if park_key_col is None:
            logging.warning("skipping season=%s (missing park key)", season)
            continue

        event_col = _pick(list(pa.columns), ["events", "event_type", "event"])
        if event_col is None:
            logging.warning("skipping season=%s (missing event column)", season)
            continue

        total_rows += len(pa)
        park_key = pa[park_key_col].astype("string")
        bad_park = park_key.isna() | (park_key.str.strip() == "") | (park_key.str.lower() == "nan")
        dropped_no_park += int(bad_park.sum())
        pa = pa[~bad_park].copy()
        if pa.empty:
            continue

        included_seasons.append(season)
        ev = pa[event_col].astype(str).str.lower().str.strip()

        launch_speed_col = _pick(list(pa.columns), ["launch_speed", "launch_speed_mean", "launch_speed_mph"])
        launch_angle_col = _pick(list(pa.columns), ["launch_angle", "launch_angle_mean"])
        park_id_col = _pick(list(pa.columns), ["park_id", "venue_id", "stadium_id"])
        park_name_col = _pick(list(pa.columns), ["park_name", "venue_name", "stadium_name"])

        out = pd.DataFrame(index=pa.index)
        out["canonical_park_key"] = park_key[~bad_park].astype(str)
        out["season"] = season
        out["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")
        out["park_id"] = pa[park_id_col] if park_id_col else pd.NA
        out["park_name"] = pa[park_name_col] if park_name_col else pd.NA

        out["pa_n"] = 1.0
        out["hit_n"] = ev.isin(HIT_EVENTS).astype(float)
        out["1b_n"] = (ev == "single").astype(float)
        out["2b_n"] = (ev == "double").astype(float)
        out["3b_n"] = (ev == "triple").astype(float)
        out["hr_n"] = (ev == "home_run").astype(float)
        out["xbh_n"] = ev.isin({"double", "triple", "home_run"}).astype(float)
        out["bb_n"] = ev.isin({"walk", "intent_walk"}).astype(float)
        out["k_n"] = (ev == "strikeout").astype(float)

        bip_hit = out["1b_n"] + out["2b_n"] + out["3b_n"]
        out["bip_n"] = (bip_hit > 0).astype(float) + ev.isin(BIP_OUT_EVENTS).astype(float)
        out["hit_on_bip_n"] = bip_hit

        out["runs_n"] = pd.to_numeric(pa["rbi"], errors="coerce").fillna(0.0) if "rbi" in pa.columns else 0.0
        out["rbi_n"] = pd.to_numeric(pa["rbi"], errors="coerce").fillna(0.0) if "rbi" in pa.columns else 0.0

        out["launch_speed_sum"] = pd.to_numeric(pa[launch_speed_col], errors="coerce") if launch_speed_col else np.nan
        out["launch_speed_obs_n"] = pd.to_numeric(pa[launch_speed_col], errors="coerce").notna().astype(float) if launch_speed_col else 0.0
        out["launch_angle_sum"] = pd.to_numeric(pa[launch_angle_col], errors="coerce") if launch_angle_col else np.nan
        out["launch_angle_obs_n"] = pd.to_numeric(pa[launch_angle_col], errors="coerce").notna().astype(float) if launch_angle_col else 0.0
        out["hard_hit_n"] = _get_hard_hit(pa)
        out["hard_hit_obs_n"] = pd.to_numeric(out["hard_hit_n"], errors="coerce").notna().astype(float)
        out["hard_hit_n"] = pd.to_numeric(out["hard_hit_n"], errors="coerce").fillna(0.0)

        frames.append(out)

    if not frames:
        raise FileNotFoundError("No valid PA historical rows found to build park factors")

    all_pa = pd.concat(frames, ignore_index=True, sort=False)

    park = all_pa.groupby("canonical_park_key", as_index=False).agg(
        representative_park_id=("park_id", _mode_or_na),
        representative_park_name=("park_name", _mode_or_na),
        seasons_covered=("season", "nunique"),
        games_n=("game_pk", "nunique"),
        pa_n=("pa_n", "sum"),
        bat_balls_in_play_n=("bip_n", "sum"),
        hit_n=("hit_n", "sum"),
        one_b_n=("1b_n", "sum"),
        two_b_n=("2b_n", "sum"),
        three_b_n=("3b_n", "sum"),
        hr_n=("hr_n", "sum"),
        xbh_n=("xbh_n", "sum"),
        runs_n=("runs_n", "sum"),
        rbi_n=("rbi_n", "sum"),
        bb_n=("bb_n", "sum"),
        k_n=("k_n", "sum"),
        hit_on_bip_n=("hit_on_bip_n", "sum"),
        launch_speed_sum=("launch_speed_sum", "sum"),
        launch_speed_obs_n=("launch_speed_obs_n", "sum"),
        launch_angle_sum=("launch_angle_sum", "sum"),
        launch_angle_obs_n=("launch_angle_obs_n", "sum"),
        hard_hit_n=("hard_hit_n", "sum"),
        hard_hit_obs_n=("hard_hit_obs_n", "sum"),
    )

    lg = {
        "hits_per_pa": float(all_pa["hit_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "1b_per_pa": float(all_pa["1b_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "2b_per_pa": float(all_pa["2b_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "3b_per_pa": float(all_pa["3b_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "hr_per_pa": float(all_pa["hr_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "xbh_per_pa": float(all_pa["xbh_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "runs_per_pa": float(all_pa["runs_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "rbi_per_pa": float(all_pa["rbi_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "bb_per_pa": float(all_pa["bb_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "k_per_pa": float(all_pa["k_n"].sum() / max(1.0, all_pa["pa_n"].sum())),
        "babip": float(all_pa["hit_on_bip_n"].sum() / max(1.0, all_pa["bip_n"].sum())),
        "avg_launch_speed": float(pd.to_numeric(all_pa["launch_speed_sum"], errors="coerce").sum() / max(1.0, pd.to_numeric(all_pa["launch_speed_obs_n"], errors="coerce").sum())),
        "avg_launch_angle": float(pd.to_numeric(all_pa["launch_angle_sum"], errors="coerce").sum() / max(1.0, pd.to_numeric(all_pa["launch_angle_obs_n"], errors="coerce").sum())),
        "hard_hit_rate": float(pd.to_numeric(all_pa["hard_hit_n"], errors="coerce").sum() / max(1.0, pd.to_numeric(all_pa["hard_hit_obs_n"], errors="coerce").sum())),
    }

    park["park_factor_hits_hist"] = _safe_ratio(_safe_ratio(park["hit_n"], park["pa_n"]), lg["hits_per_pa"])
    park["park_factor_1b_hist"] = _safe_ratio(_safe_ratio(park["one_b_n"], park["pa_n"]), lg["1b_per_pa"])
    park["park_factor_2b_hist"] = _safe_ratio(_safe_ratio(park["two_b_n"], park["pa_n"]), lg["2b_per_pa"])
    park["park_factor_3b_hist"] = _safe_ratio(_safe_ratio(park["three_b_n"], park["pa_n"]), lg["3b_per_pa"])
    park["park_factor_hr_hist"] = _safe_ratio(_safe_ratio(park["hr_n"], park["pa_n"]), lg["hr_per_pa"])
    park["park_factor_xbh_hist"] = _safe_ratio(_safe_ratio(park["xbh_n"], park["pa_n"]), lg["xbh_per_pa"])
    park["park_factor_runs_hist"] = _safe_ratio(_safe_ratio(park["runs_n"], park["pa_n"]), lg["runs_per_pa"])
    park["park_factor_rbi_hist"] = _safe_ratio(_safe_ratio(park["rbi_n"], park["pa_n"]), lg["rbi_per_pa"])
    park["park_factor_bb_hist"] = _safe_ratio(_safe_ratio(park["bb_n"], park["pa_n"]), lg["bb_per_pa"])
    park["park_factor_k_hist"] = _safe_ratio(_safe_ratio(park["k_n"], park["pa_n"]), lg["k_per_pa"])

    park["park_factor_babip_hist"] = _safe_ratio(_safe_ratio(park["hit_on_bip_n"], park["bat_balls_in_play_n"]), lg["babip"])
    park["park_factor_avg_launch_speed_hist"] = _safe_ratio(_safe_ratio(park["launch_speed_sum"], park["launch_speed_obs_n"]), lg["avg_launch_speed"])
    park["park_factor_avg_launch_angle_hist"] = _safe_ratio(_safe_ratio(park["launch_angle_sum"], park["launch_angle_obs_n"]), lg["avg_launch_angle"])
    park["park_factor_hard_hit_hist"] = _safe_ratio(_safe_ratio(park["hard_hit_n"], park["hard_hit_obs_n"]), lg["hard_hit_rate"])

    park["park_factor_singles_contact_hist"] = _safe_ratio(_safe_ratio(park["one_b_n"], park["bat_balls_in_play_n"]), _safe_ratio(all_pa["1b_n"].sum(), all_pa["bip_n"].sum()))
    park["park_factor_extra_base_contact_hist"] = _safe_ratio(_safe_ratio(park["xbh_n"], park["bat_balls_in_play_n"]), _safe_ratio(all_pa["xbh_n"].sum(), all_pa["bip_n"].sum()))

    weight_hits = park["pa_n"] / (park["pa_n"] + args.k_hits)
    weight_runs = park["pa_n"] / (park["pa_n"] + args.k_runs)
    weight_hr = park["pa_n"] / (park["pa_n"] + args.k_hr)
    weight_xbh = park["pa_n"] / (park["pa_n"] + args.k_xbh)

    park["park_factor_hits_hist_shrunk"] = (weight_hits * park["park_factor_hits_hist"]) + ((1.0 - weight_hits) * 1.0)
    park["park_factor_hr_hist_shrunk"] = (weight_hr * park["park_factor_hr_hist"]) + ((1.0 - weight_hr) * 1.0)
    park["park_factor_runs_hist_shrunk"] = (weight_runs * park["park_factor_runs_hist"]) + ((1.0 - weight_runs) * 1.0)
    park["park_factor_xbh_hist_shrunk"] = (weight_xbh * park["park_factor_xbh_hist"]) + ((1.0 - weight_xbh) * 1.0)

    park = park.rename(columns={
        "representative_park_id": "park_id_mode",
        "representative_park_name": "park_name_mode",
    })

    output_cols = [
        "canonical_park_key",
        "park_id_mode",
        "park_name_mode",
        "seasons_covered",
        "games_n",
        "pa_n",
        "bat_balls_in_play_n",
        "park_factor_hits_hist",
        "park_factor_1b_hist",
        "park_factor_2b_hist",
        "park_factor_3b_hist",
        "park_factor_hr_hist",
        "park_factor_xbh_hist",
        "park_factor_runs_hist",
        "park_factor_rbi_hist",
        "park_factor_bb_hist",
        "park_factor_k_hist",
        "park_factor_babip_hist",
        "park_factor_avg_launch_speed_hist",
        "park_factor_avg_launch_angle_hist",
        "park_factor_hard_hit_hist",
        "park_factor_singles_contact_hist",
        "park_factor_extra_base_contact_hist",
        "park_factor_hits_hist_shrunk",
        "park_factor_hr_hist_shrunk",
        "park_factor_runs_hist_shrunk",
        "park_factor_xbh_hist_shrunk",
    ]
    out = park[[c for c in output_cols if c in park.columns]].copy()

    out_path = dirs["reference_dir"] / "parks.parquet"
    write_parquet(out, out_path)

    logging.info("park_factors seasons_included=%s", sorted(set(included_seasons)))
    logging.info("park_factors source_rows_total=%s included_rows=%s dropped_missing_park=%s", total_rows, len(all_pa), dropped_no_park)
    logging.info("park_factors unique_parks=%s", out["canonical_park_key"].nunique())
    logging.info("park_factors sample_lowest_pa=%s", out.nsmallest(10, "pa_n")[["canonical_park_key", "pa_n"]].to_dict(orient="records"))
    logging.info("park_factors sample_highest_pa=%s", out.nlargest(10, "pa_n")[["canonical_park_key", "pa_n"]].to_dict(orient="records"))

    for metric in ["park_factor_hits_hist_shrunk", "park_factor_hr_hist_shrunk", "park_factor_runs_hist_shrunk"]:
        if metric in out.columns:
            logging.info("park_factors top10_%s=%s", metric, out.nlargest(10, metric)[["canonical_park_key", metric, "pa_n"]].to_dict(orient="records"))
            logging.info("park_factors bottom10_%s=%s", metric, out.nsmallest(10, metric)[["canonical_park_key", metric, "pa_n"]].to_dict(orient="records"))

    null_rates = {c: float(out[c].isna().mean()) for c in out.columns}
    logging.info("park_factors null_rates=%s", null_rates)
    logging.info("park_factors sample_rows=%s", out.head(10).to_dict(orient="records"))
    logging.info("park_factors complete rows=%s path=%s", len(out), out_path)
    print(f"parks_out={out_path}")


if __name__ == "__main__":
    main()
