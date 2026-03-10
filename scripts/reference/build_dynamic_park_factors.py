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
XBH_EVENTS = {"double", "triple", "home_run"}
BIP_OUT_EVENTS = {
    "field_out", "force_out", "grounded_into_double_play", "double_play", "triple_play",
    "fielders_choice", "fielders_choice_out", "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play", "lineout", "flyout", "groundout", "pop_out",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build leakage-safe dynamic park factors for an in-season year.")
    p.add_argument("--season", type=int, default=2026)
    p.add_argument("--k-pa", type=float, default=4000.0)
    p.add_argument("--k-bip", type=float, default=2200.0)
    p.add_argument("--max-dynamic-weight", type=float, default=0.50)
    p.add_argument("--config", type=Path, default=Path("configs/project.yaml"))
    return p.parse_args()


def _pick(cols: list[str], cands: list[str]) -> str | None:
    colset = set(cols)
    for c in cands:
        if c in colset:
            return c
    return None


def _load_path(base: Path, season: int, stem: str) -> Path:
    by_season = base / "by_season" / f"{stem}_{season}.parquet"
    root = base / f"{stem}_{season}.parquet"
    if by_season.exists():
        return by_season
    if root.exists():
        return root
    raise FileNotFoundError(f"Missing {stem} file for season {season}: checked {by_season} and {root}")


def _build_norm_key(df: pd.DataFrame) -> pd.Series:
    venue = pd.to_numeric(df.get("venue_id"), errors="coerce")
    park = pd.to_numeric(df.get("park_id"), errors="coerce")
    raw = df.get("canonical_park_key", pd.Series(pd.NA, index=df.index)).astype("string")
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out = out.where(~venue.notna(), "venue:" + venue.astype("Int64").astype(str))
    out = out.fillna(pd.Series(np.where(park.notna(), "park:" + park.astype("Int64").astype(str), pd.NA), index=df.index))
    out = out.fillna(raw.where(raw.notna() & (raw.str.strip() != "") & (raw.str.lower() != "nan")))
    return out.astype("string")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return pd.to_numeric(num, errors="coerce") / pd.to_numeric(den, errors="coerce").replace(0, np.nan)


def _shifted_cumsum(group: pd.Series) -> pd.Series:
    return group.cumsum().shift(1)


def main() -> None:
    args = parse_args()

    repo_root = get_repo_root()
    config_path = (repo_root / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    config = load_config(config_path)
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    configure_logging(dirs["logs_dir"] / "build_dynamic_park_factors.log")
    log_header("scripts/reference/build_dynamic_park_factors.py", repo_root, config_path, dirs)

    pa_path = _load_path(dirs["processed_dir"], args.season, "pa")
    games_path = _load_path(dirs["processed_dir"], args.season, "games")
    logging.info("dynamic_park_factors resolved_inputs pa=%s games=%s", pa_path, games_path)

    pa = pd.read_parquet(pa_path)
    games = pd.read_parquet(games_path)

    if "game_pk" not in pa.columns or "game_pk" not in games.columns:
        raise ValueError("dynamic park factors require game_pk in both PA and games tables")

    event_col = _pick(list(pa.columns), ["events", "event_type", "event"])
    if event_col is None:
        raise ValueError("dynamic park factors require events/event_type/event column")

    game_date_col = _pick(list(games.columns), ["game_date", "date"])
    if game_date_col is None:
        raise ValueError("dynamic park factors require game_date/date in games table")

    keep_games = [c for c in ["game_pk", "game_date", "park_id", "venue_id", "canonical_park_key", "park_name"] if c in games.columns]

    pa["game_pk"] = pd.to_numeric(pa["game_pk"], errors="coerce").astype("Int64")
    games["game_pk"] = pd.to_numeric(games["game_pk"], errors="coerce").astype("Int64")
    games["game_date"] = pd.to_datetime(games[game_date_col], errors="coerce").dt.normalize()

    joined = pa.merge(games[keep_games].drop_duplicates(subset=["game_pk"], keep="last"), on="game_pk", how="left")
    joined = joined[joined["game_pk"].notna() & joined["game_date"].notna()].copy()
    joined["canonical_park_key_norm"] = _build_norm_key(joined)
    joined = joined[joined["canonical_park_key_norm"].notna()].copy()
    if joined.empty:
        raise FileNotFoundError("No joined PA rows with normalized park keys for dynamic park factors")

    ev = joined[event_col].astype(str).str.lower().str.strip()
    joined["hit_n"] = ev.isin(HIT_EVENTS).astype(float)
    joined["hr_n"] = (ev == "home_run").astype(float)
    joined["xbh_n"] = ev.isin(XBH_EVENTS).astype(float)
    joined["bip_n"] = ((joined["hit_n"] - joined["hr_n"]) > 0).astype(float) + ev.isin(BIP_OUT_EVENTS).astype(float)
    joined["hit_on_bip_n"] = (joined["hit_n"] - joined["hr_n"]).clip(lower=0.0)

    if "rbi" in joined.columns and pd.to_numeric(joined["rbi"], errors="coerce").notna().any():
        joined["runs_proxy_n"] = pd.to_numeric(joined["rbi"], errors="coerce").fillna(0.0)
    elif "delta_run_exp" in joined.columns:
        joined["runs_proxy_n"] = np.clip(pd.to_numeric(joined["delta_run_exp"], errors="coerce").fillna(0.0), 0.0, None)
    else:
        joined["runs_proxy_n"] = np.nan

    ls_col = _pick(list(joined.columns), ["launch_speed", "launch_speed_mean", "launch_speed_mph"])
    la_col = _pick(list(joined.columns), ["launch_angle", "launch_angle_mean"])
    joined["launch_speed_sum"] = pd.to_numeric(joined[ls_col], errors="coerce") if ls_col else np.nan
    joined["launch_speed_obs_n"] = pd.to_numeric(joined[ls_col], errors="coerce").notna().astype(float) if ls_col else 0.0
    joined["launch_angle_sum"] = pd.to_numeric(joined[la_col], errors="coerce") if la_col else np.nan
    joined["launch_angle_obs_n"] = pd.to_numeric(joined[la_col], errors="coerce").notna().astype(float) if la_col else 0.0

    agg_cols = [
        "hit_n", "hr_n", "xbh_n", "runs_proxy_n", "bip_n", "hit_on_bip_n",
        "launch_speed_sum", "launch_speed_obs_n", "launch_angle_sum", "launch_angle_obs_n",
    ]
    daily_park = joined.groupby(["game_date", "canonical_park_key_norm"], as_index=False).agg(
        pa_n=("game_pk", "size"),
        games_n=("game_pk", "nunique"),
        representative_park_id=("park_id", lambda s: pd.to_numeric(s, errors="coerce").dropna().mode().iloc[0] if pd.to_numeric(s, errors="coerce").dropna().size else pd.NA),
        representative_venue_id=("venue_id", lambda s: pd.to_numeric(s, errors="coerce").dropna().mode().iloc[0] if pd.to_numeric(s, errors="coerce").dropna().size else pd.NA),
        representative_park_name=("park_name", lambda s: s.dropna().astype(str).mode().iloc[0] if s.dropna().size else pd.NA),
        **{c: (c, "sum") for c in agg_cols},
    )

    daily_league = joined.groupby("game_date", as_index=False).agg(
        league_pa_n=("game_pk", "size"),
        **{f"league_{c}": (c, "sum") for c in agg_cols},
    )

    daily_park = daily_park.sort_values(["canonical_park_key_norm", "game_date"]).reset_index(drop=True)
    daily_league = daily_league.sort_values("game_date").reset_index(drop=True)

    for c in ["pa_n", "games_n"] + agg_cols:
        daily_park[f"prior_{c}"] = daily_park.groupby("canonical_park_key_norm", dropna=False)[c].transform(_shifted_cumsum)
    for c in ["league_pa_n"] + [f"league_{x}" for x in agg_cols]:
        daily_league[f"prior_{c}"] = daily_league[c].cumsum().shift(1)

    out = daily_park.merge(
        daily_league[["game_date"] + [c for c in daily_league.columns if c.startswith("prior_")]],
        on="game_date",
        how="left",
    )

    out["season"] = int(args.season)
    out["games_n_2026_to_date"] = pd.to_numeric(out["prior_games_n"], errors="coerce")
    out["pa_n_2026_to_date"] = pd.to_numeric(out["prior_pa_n"], errors="coerce")
    out["bat_balls_in_play_n_2026_to_date"] = pd.to_numeric(out["prior_bip_n"], errors="coerce")

    park_rates = {
        "hits": _safe_div(out["prior_hit_n"], out["prior_pa_n"]),
        "hr": _safe_div(out["prior_hr_n"], out["prior_pa_n"]),
        "xbh": _safe_div(out["prior_xbh_n"], out["prior_pa_n"]),
        "runs": _safe_div(out["prior_runs_proxy_n"], out["prior_pa_n"]),
        "babip": _safe_div(out["prior_hit_on_bip_n"], out["prior_bip_n"]),
        "avg_launch_speed": _safe_div(out["prior_launch_speed_sum"], out["prior_launch_speed_obs_n"]),
        "avg_launch_angle": _safe_div(out["prior_launch_angle_sum"], out["prior_launch_angle_obs_n"]),
    }
    league_rates = {
        "hits": _safe_div(out["prior_league_hit_n"], out["prior_league_pa_n"]),
        "hr": _safe_div(out["prior_league_hr_n"], out["prior_league_pa_n"]),
        "xbh": _safe_div(out["prior_league_xbh_n"], out["prior_league_pa_n"]),
        "runs": _safe_div(out["prior_league_runs_proxy_n"], out["prior_league_pa_n"]),
        "babip": _safe_div(out["prior_league_hit_on_bip_n"], out["prior_league_bip_n"]),
        "avg_launch_speed": _safe_div(out["prior_league_launch_speed_sum"], out["prior_league_launch_speed_obs_n"]),
        "avg_launch_angle": _safe_div(out["prior_league_launch_angle_sum"], out["prior_league_launch_angle_obs_n"]),
    }

    k_map = {
        "hits": args.k_pa,
        "hr": args.k_pa * 1.8,
        "xbh": args.k_pa * 1.4,
        "runs": args.k_pa * 1.3,
        "babip": args.k_bip,
        "avg_launch_speed": args.k_bip,
        "avg_launch_angle": args.k_bip,
    }

    for metric in ["hits", "hr", "xbh", "runs", "babip", "avg_launch_speed", "avg_launch_angle"]:
        raw = park_rates[metric] / league_rates[metric].replace(0, np.nan)
        raw = raw.replace([np.inf, -np.inf], np.nan).clip(0.70, 1.30)
        sample = out["pa_n_2026_to_date"] if metric in {"hits", "hr", "xbh", "runs"} else out["bat_balls_in_play_n_2026_to_date"]
        w = (pd.to_numeric(sample, errors="coerce") / (pd.to_numeric(sample, errors="coerce") + float(k_map[metric]))).clip(0.0, 1.0)
        out[f"park_factor_{metric}_2026_roll"] = 1.0 + w * (raw - 1.0)

    dyn_weight = (pd.to_numeric(out["pa_n_2026_to_date"], errors="coerce") / (pd.to_numeric(out["pa_n_2026_to_date"], errors="coerce") + args.k_pa)).clip(0.0, args.max_dynamic_weight)
    out["park_factor_dynamic_weight"] = dyn_weight
    out["park_factor_hist_weight"] = 1.0 - dyn_weight
    out["dynamic_sample_confidence"] = (pd.to_numeric(out["pa_n_2026_to_date"], errors="coerce") / (pd.to_numeric(out["pa_n_2026_to_date"], errors="coerce") + args.k_pa)).clip(0.0, 1.0)

    keep_cols = [
        "season", "game_date", "canonical_park_key_norm", "representative_park_id", "representative_venue_id", "representative_park_name",
        "games_n_2026_to_date", "pa_n_2026_to_date", "bat_balls_in_play_n_2026_to_date",
        "park_factor_hits_2026_roll", "park_factor_hr_2026_roll", "park_factor_xbh_2026_roll", "park_factor_runs_2026_roll",
        "park_factor_babip_2026_roll", "park_factor_avg_launch_speed_2026_roll", "park_factor_avg_launch_angle_2026_roll",
        "park_factor_dynamic_weight", "park_factor_hist_weight", "dynamic_sample_confidence",
    ]
    out = out[keep_cols].sort_values(["season", "game_date", "canonical_park_key_norm"]).reset_index(drop=True)

    dynamic_cols = [c for c in out.columns if c.startswith("park_factor_") or c in {"dynamic_sample_confidence"}]
    null_rates = {c: float(pd.to_numeric(out[c], errors="coerce").isna().mean()) for c in dynamic_cols}
    latest_date = out["game_date"].max()
    latest = out[out["game_date"] == latest_date].copy()

    logging.info(
        "dynamic_park_factors season=%s joined_rows=%s unique_parks=%s min_game_date=%s max_game_date=%s output_rows=%s",
        args.season,
        len(joined),
        int(out["canonical_park_key_norm"].nunique()),
        out["game_date"].min(),
        out["game_date"].max(),
        len(out),
    )
    logging.info("dynamic_park_factors null_rates=%s", null_rates)
    if not latest.empty:
        logging.info(
            "dynamic_park_factors latest_date=%s top10_hits=%s",
            latest_date,
            latest.nlargest(10, "park_factor_hits_2026_roll")[["canonical_park_key_norm", "park_factor_hits_2026_roll", "pa_n_2026_to_date"]].to_dict(orient="records"),
        )
        logging.info(
            "dynamic_park_factors latest_date=%s bottom10_hits=%s",
            latest_date,
            latest.nsmallest(10, "park_factor_hits_2026_roll")[["canonical_park_key_norm", "park_factor_hits_2026_roll", "pa_n_2026_to_date"]].to_dict(orient="records"),
        )
        logging.info(
            "dynamic_park_factors latest_date=%s top10_runs=%s",
            latest_date,
            latest.nlargest(10, "park_factor_runs_2026_roll")[["canonical_park_key_norm", "park_factor_runs_2026_roll", "pa_n_2026_to_date"]].to_dict(orient="records"),
        )
        logging.info(
            "dynamic_park_factors latest_date=%s bottom10_runs=%s",
            latest_date,
            latest.nsmallest(10, "park_factor_runs_2026_roll")[["canonical_park_key_norm", "park_factor_runs_2026_roll", "pa_n_2026_to_date"]].to_dict(orient="records"),
        )
    logging.info("dynamic_park_factors sample_rows=%s", out.head(10).to_dict(orient="records"))

    out_path = dirs["reference_dir"] / f"parks_dynamic_{args.season}.parquet"
    write_parquet(out, out_path)
    logging.info("dynamic_park_factors wrote path=%s", out_path)
    print(f"parks_dynamic_out={out_path}")


if __name__ == "__main__":
    main()
