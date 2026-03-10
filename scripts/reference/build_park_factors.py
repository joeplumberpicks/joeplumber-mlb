from __future__ import annotations

import argparse
import logging
import re
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
    "field_out", "force_out", "grounded_into_double_play", "double_play", "triple_play",
    "fielders_choice", "fielders_choice_out", "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play", "lineout", "flyout", "groundout", "pop_out",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build rich historical park factor profiles from joined PA+games.")
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


def _safe_div(num, den):
    num_s = pd.to_numeric(num, errors="coerce")
    den_s = pd.to_numeric(den, errors="coerce")
    return np.where(den_s > 0, num_s / den_s, np.nan)


def _mode_or_na(s: pd.Series):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else pd.NA


def _norm_text(v: object) -> str:
    txt = str(v).strip().lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt or "unknown_park"


def _build_norm_key(df: pd.DataFrame) -> pd.Series:
    venue = pd.to_numeric(df.get("venue_id"), errors="coerce")
    park = pd.to_numeric(df.get("park_id"), errors="coerce")
    raw = df.get("canonical_park_key", pd.Series(pd.NA, index=df.index, dtype="object")).astype("string")
    name = df.get("park_name", pd.Series(pd.NA, index=df.index, dtype="object")).map(_norm_text)
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out = out.where(~venue.notna(), "venue:" + venue.astype("Int64").astype(str))
    out = out.fillna(pd.Series(np.where(park.notna(), "park:" + park.astype("Int64").astype(str), pd.NA), index=df.index))
    out = out.fillna(raw.where(raw.notna() & (raw.str.strip() != "") & (raw.str.lower() != "nan")))
    out = out.fillna("parkname:" + name.astype(str))
    return out.astype(str)


def _hard_hit_flag(df: pd.DataFrame) -> pd.Series:
    hard_col = _pick(list(df.columns), ["hard_hit", "is_hard_hit", "hard_hit_flag"])
    if hard_col:
        hard = pd.to_numeric(df[hard_col], errors="coerce")
        if hard.notna().any():
            return (hard > 0).astype(float)
    ls_col = _pick(list(df.columns), ["launch_speed", "launch_speed_mean", "launch_speed_mph"])
    if ls_col:
        return (pd.to_numeric(df[ls_col], errors="coerce") >= 95.0).astype(float)
    return pd.Series(np.nan, index=df.index, dtype="float64")


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
    total_pa_rows = 0
    total_joined_rows = 0
    total_missing_game_pk = 0

    for season in range(args.season_start, args.season_end + 1):
        pa_path = dirs["processed_dir"] / "by_season" / f"pa_{season}.parquet"
        games_path = dirs["processed_dir"] / "by_season" / f"games_{season}.parquet"
        if not pa_path.exists() or not games_path.exists():
            logging.warning("season=%s skipped missing file pa_exists=%s games_exists=%s", season, pa_path.exists(), games_path.exists())
            continue
        try:
            pa = pd.read_parquet(pa_path)
            games = pd.read_parquet(games_path)
        except Exception:
            logging.exception("season=%s failed to load pa/games", season)
            continue

        pa_rows = len(pa)
        games_rows = len(games)
        total_pa_rows += pa_rows

        if "game_pk" not in pa.columns or "game_pk" not in games.columns:
            logging.warning("season=%s skipped missing game_pk in pa or games", season)
            continue

        event_col = _pick(list(pa.columns), ["events", "event_type", "event"])
        if event_col is None:
            logging.warning("season=%s skipped missing events/event_type in pa", season)
            continue

        keep_games = [c for c in ["game_pk", "canonical_park_key", "park_id", "venue_id", "park_name"] if c in games.columns]
        if len(keep_games) < 2:
            logging.warning("season=%s skipped insufficient park identity columns in games", season)
            continue

        pa["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")
        games["game_pk"] = pd.to_numeric(games["game_pk"], errors="coerce").astype("Int64")
        missing_game_pk = int(pa["game_pk"].isna().sum())
        total_missing_game_pk += missing_game_pk
        pa = pa[pa["game_pk"].notna()].copy()

        joined = pa.merge(games[keep_games].drop_duplicates(subset=["game_pk"], keep="last"), on="game_pk", how="left")
        joined_rows = len(joined)
        total_joined_rows += joined_rows

        joined["canonical_park_key_norm"] = _build_norm_key(joined)
        pct_norm = float(joined["canonical_park_key_norm"].notna().mean()) if joined_rows else 0.0
        logging.info(
            "season=%s pa_rows=%s games_rows=%s joined_rows=%s pct_with_canonical_park_key_after_join=%.4f",
            season, pa_rows, games_rows, joined_rows, pct_norm,
        )

        joined = joined[joined["canonical_park_key_norm"].notna()].copy()
        if joined.empty:
            continue

        included_seasons.append(season)
        ev = joined[event_col].astype(str).str.lower().str.strip()

        out = pd.DataFrame(index=joined.index)
        out["canonical_park_key"] = joined.get("canonical_park_key", pd.NA)
        out["canonical_park_key_norm"] = joined["canonical_park_key_norm"].astype(str)
        out["season"] = season
        out["game_pk"] = joined["game_pk"]
        out["park_id"] = joined.get("park_id", pd.NA)
        out["venue_id"] = joined.get("venue_id", pd.NA)
        out["park_name"] = joined.get("park_name", pd.NA)

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

        # Run environment proxy: prefer RBI if present; fallback to positive delta_run_exp.
        rbi_series = joined["rbi"] if "rbi" in joined.columns else pd.Series(np.nan, index=joined.index)
        dre_series = joined["delta_run_exp"] if "delta_run_exp" in joined.columns else pd.Series(np.nan, index=joined.index)
        rbi_raw = pd.to_numeric(rbi_series, errors="coerce")
        dre_raw = pd.to_numeric(dre_series, errors="coerce")
        rbi_sum = float(rbi_raw.fillna(0.0).sum())
        has_rbi_signal = bool(rbi_raw.notna().any() and rbi_sum > 0)
        if has_rbi_signal:
            out["runs_proxy_n"] = rbi_raw.fillna(0.0)
            out["rbi_n"] = rbi_raw.fillna(0.0)
            out["runs_proxy_source"] = "rbi"
        else:
            # fallback run-environment proxy from positive change in run expectancy
            run_proxy = np.clip(dre_raw.fillna(0.0), 0.0, None)
            out["runs_proxy_n"] = run_proxy
            out["rbi_n"] = np.nan
            out["runs_proxy_source"] = "delta_run_exp_positive"

        ls_col = _pick(list(joined.columns), ["launch_speed", "launch_speed_mean", "launch_speed_mph"])
        la_col = _pick(list(joined.columns), ["launch_angle", "launch_angle_mean"])
        out["launch_speed_sum"] = pd.to_numeric(joined[ls_col], errors="coerce") if ls_col else np.nan
        out["launch_speed_obs_n"] = pd.to_numeric(joined[ls_col], errors="coerce").notna().astype(float) if ls_col else 0.0
        out["launch_angle_sum"] = pd.to_numeric(joined[la_col], errors="coerce") if la_col else np.nan
        out["launch_angle_obs_n"] = pd.to_numeric(joined[la_col], errors="coerce").notna().astype(float) if la_col else 0.0

        frames.append(out)

    if not frames:
        raise FileNotFoundError("No valid PA historical rows found to build park factors")

    all_pa = pd.concat(frames, ignore_index=True, sort=False)

    raw_to_norm = all_pa[["canonical_park_key", "canonical_park_key_norm"]].dropna().drop_duplicates()
    collapsed_n = int(max(0, raw_to_norm["canonical_park_key"].nunique() - raw_to_norm["canonical_park_key_norm"].nunique()))
    examples = (
        raw_to_norm.groupby("canonical_park_key_norm")["canonical_park_key"]
        .nunique()
        .reset_index(name="raw_key_n")
        .query("raw_key_n > 1")
        .head(10)
        .to_dict(orient="records")
    )

    park = all_pa.groupby("canonical_park_key_norm", as_index=False).agg(
        canonical_park_key=("canonical_park_key", _mode_or_na),
        representative_park_id=("park_id", _mode_or_na),
        representative_venue_id=("venue_id", _mode_or_na),
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
        runs_proxy_n=("runs_proxy_n", "sum"),
        rbi_n=("rbi_n", "sum"),
        bb_n=("bb_n", "sum"),
        k_n=("k_n", "sum"),
        hit_on_bip_n=("hit_on_bip_n", "sum"),
        launch_speed_sum=("launch_speed_sum", "sum"),
        launch_speed_obs_n=("launch_speed_obs_n", "sum"),
        launch_angle_sum=("launch_angle_sum", "sum"),
        launch_angle_obs_n=("launch_angle_obs_n", "sum"),
        runs_proxy_source=("runs_proxy_source", _mode_or_na),
    )

    lg_hits_pa = float(all_pa["hit_n"].sum() / max(1.0, all_pa["pa_n"].sum()))
    lg_1b_pa = float(all_pa["1b_n"].sum() / max(1.0, all_pa["pa_n"].sum()))
    lg_2b_pa = float(all_pa["2b_n"].sum() / max(1.0, all_pa["pa_n"].sum()))
    lg_3b_pa = float(all_pa["3b_n"].sum() / max(1.0, all_pa["pa_n"].sum()))
    lg_hr_pa = float(all_pa["hr_n"].sum() / max(1.0, all_pa["pa_n"].sum()))
    lg_xbh_pa = float(all_pa["xbh_n"].sum() / max(1.0, all_pa["pa_n"].sum()))
    lg_runs_pa = float(all_pa["runs_proxy_n"].sum() / max(1.0, all_pa["pa_n"].sum()))

    has_rbi = bool(pd.to_numeric(all_pa["rbi_n"], errors="coerce").notna().any() and pd.to_numeric(all_pa["rbi_n"], errors="coerce").fillna(0.0).sum() > 0)
    lg_rbi_pa = float(pd.to_numeric(all_pa["rbi_n"], errors="coerce").fillna(0.0).sum() / max(1.0, all_pa["pa_n"].sum())) if has_rbi else np.nan

    lg_bb_pa = float(all_pa["bb_n"].sum() / max(1.0, all_pa["pa_n"].sum()))
    lg_k_pa = float(all_pa["k_n"].sum() / max(1.0, all_pa["pa_n"].sum()))
    lg_babip = float(all_pa["hit_on_bip_n"].sum() / max(1.0, all_pa["bip_n"].sum()))
    lg_ls = float(pd.to_numeric(all_pa["launch_speed_sum"], errors="coerce").sum() / max(1.0, pd.to_numeric(all_pa["launch_speed_obs_n"], errors="coerce").sum()))
    lg_la = float(pd.to_numeric(all_pa["launch_angle_sum"], errors="coerce").sum() / max(1.0, pd.to_numeric(all_pa["launch_angle_obs_n"], errors="coerce").sum()))

    park["park_factor_hits_hist"] = _safe_div(_safe_div(park["hit_n"], park["pa_n"]), lg_hits_pa)
    park["park_factor_1b_hist"] = _safe_div(_safe_div(park["one_b_n"], park["pa_n"]), lg_1b_pa)
    park["park_factor_2b_hist"] = _safe_div(_safe_div(park["two_b_n"], park["pa_n"]), lg_2b_pa)
    park["park_factor_3b_hist"] = _safe_div(_safe_div(park["three_b_n"], park["pa_n"]), lg_3b_pa)
    park["park_factor_hr_hist"] = _safe_div(_safe_div(park["hr_n"], park["pa_n"]), lg_hr_pa)
    park["park_factor_xbh_hist"] = _safe_div(_safe_div(park["xbh_n"], park["pa_n"]), lg_xbh_pa)
    park["park_factor_runs_hist"] = _safe_div(_safe_div(park["runs_proxy_n"], park["pa_n"]), lg_runs_pa)
    if has_rbi and np.isfinite(lg_rbi_pa) and lg_rbi_pa > 0:
        park["park_factor_rbi_hist"] = _safe_div(_safe_div(park["rbi_n"], park["pa_n"]), lg_rbi_pa)
    park["park_factor_bb_hist"] = _safe_div(_safe_div(park["bb_n"], park["pa_n"]), lg_bb_pa)
    park["park_factor_k_hist"] = _safe_div(_safe_div(park["k_n"], park["pa_n"]), lg_k_pa)
    park["park_factor_babip_hist"] = _safe_div(_safe_div(park["hit_on_bip_n"], park["bat_balls_in_play_n"]), lg_babip)
    park["park_factor_avg_launch_speed_hist"] = _safe_div(_safe_div(park["launch_speed_sum"], park["launch_speed_obs_n"]), lg_ls)
    park["park_factor_avg_launch_angle_hist"] = _safe_div(_safe_div(park["launch_angle_sum"], park["launch_angle_obs_n"]), lg_la)

    w_hits = park["pa_n"] / (park["pa_n"] + args.k_hits)
    w_hr = park["pa_n"] / (park["pa_n"] + args.k_hr)
    w_runs = park["pa_n"] / (park["pa_n"] + args.k_runs)
    w_xbh = park["pa_n"] / (park["pa_n"] + args.k_xbh)
    park["park_factor_hits_hist_shrunk"] = (w_hits * park["park_factor_hits_hist"]) + (1.0 - w_hits)
    park["park_factor_hr_hist_shrunk"] = (w_hr * park["park_factor_hr_hist"]) + (1.0 - w_hr)
    park["park_factor_runs_hist_shrunk"] = (w_runs * park["park_factor_runs_hist"]) + (1.0 - w_runs)
    park["park_factor_xbh_hist_shrunk"] = (w_xbh * park["park_factor_xbh_hist"]) + (1.0 - w_xbh)

    output_cols = [
        "canonical_park_key",
        "canonical_park_key_norm",
        "representative_park_id",
        "representative_venue_id",
        "representative_park_name",
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
        "park_factor_bb_hist",
        "park_factor_k_hist",
        "park_factor_babip_hist",
        "park_factor_avg_launch_speed_hist",
        "park_factor_avg_launch_angle_hist",
        "park_factor_hits_hist_shrunk",
        "park_factor_hr_hist_shrunk",
        "park_factor_runs_hist_shrunk",
        "park_factor_xbh_hist_shrunk",
    ]
    if "park_factor_rbi_hist" in park.columns:
        output_cols.insert(output_cols.index("park_factor_bb_hist"), "park_factor_rbi_hist")

    out = park[output_cols].copy()

    out_path = dirs["reference_dir"] / "parks.parquet"
    write_parquet(out, out_path)

    logging.info("park_factors seasons_included=%s", sorted(set(included_seasons)))
    logging.info("park_factors totals pa_rows=%s joined_rows=%s missing_game_pk=%s included_rows=%s", total_pa_rows, total_joined_rows, total_missing_game_pk, len(all_pa))
    logging.info("park_factors raw_to_norm_collapsed_count=%s", collapsed_n)
    logging.info("park_factors raw_to_norm_multi_raw_examples=%s", examples)
    logging.info("park_factors unique_normalized_parks=%s", out["canonical_park_key_norm"].nunique())

    logging.info("park_factors run_proxy_source_by_park=%s", park[["canonical_park_key_norm", "runs_proxy_source"]].head(20).to_dict(orient="records"))

    for metric in ["park_factor_hits_hist_shrunk", "park_factor_hr_hist_shrunk", "park_factor_runs_hist_shrunk"]:
        logging.info("park_factors top10_%s=%s", metric, out.nlargest(10, metric)[["canonical_park_key_norm", metric, "pa_n"]].to_dict(orient="records"))
        logging.info("park_factors bottom10_%s=%s", metric, out.nsmallest(10, metric)[["canonical_park_key_norm", metric, "pa_n"]].to_dict(orient="records"))

    null_rates = {c: float(out[c].isna().mean()) for c in out.columns}
    logging.info("park_factors null_rates=%s", null_rates)
    logging.info("park_factors sample_rows=%s", out.head(10).to_dict(orient="records"))
    logging.info("park_factors complete rows=%s path=%s", len(out), out_path)
    print(f"parks_out={out_path}")


if __name__ == "__main__":
    main()
