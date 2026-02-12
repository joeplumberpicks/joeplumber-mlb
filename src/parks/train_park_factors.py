from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_cfg(path: Path) -> dict:
    if not path.exists():
        return {
            "shrink_k": 30,
            "min_games_prior": 10,
            "hr_mult_clip": [0.75, 1.30],
            "runs_delta_clip": [-0.8, 0.8],
            "yrfi_delta_clip": [-0.08, 0.08],
        }
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _derive_hr_count_by_game(targets_hitter_path: Path, events_path: Path) -> pd.DataFrame:
    if targets_hitter_path.exists():
        th = pd.read_parquet(targets_hitter_path, engine="pyarrow")
        if {"game_pk", "hr"}.issubset(th.columns):
            th["game_pk"] = pd.to_numeric(th["game_pk"], errors="coerce").astype("Int64")
            th["hr"] = pd.to_numeric(th["hr"], errors="coerce").fillna(0.0)
            return th.groupby("game_pk", observed=False)["hr"].sum().reset_index(name="hr_count")

    if events_path.exists():
        ev = pd.read_parquet(events_path, engine="pyarrow")
        if {"game_pk", "events"}.issubset(ev.columns):
            ev["game_pk"] = pd.to_numeric(ev["game_pk"], errors="coerce").astype("Int64")
            ev["hr_count"] = (ev["events"].astype("string").str.lower() == "home_run").astype(int)
            return ev.groupby("game_pk", observed=False)["hr_count"].sum().reset_index()

    raise FileNotFoundError(
        "Could not derive hr_count: expected targets_hitter_game parquet with hr or events_pa parquet with events."
    )


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    if n < 10:
        raise ValueError("Need at least 10 rows to train park factors")
    split_idx = min(max(int(np.floor(0.8 * n)), 1), n - 1)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def _compute_asof_factors(
    df: pd.DataFrame,
    *,
    mean_hr: float,
    mean_runs: float,
    mean_yrfi: float,
    shrink_k: float,
    min_games_prior: int,
    hr_clip: tuple[float, float],
    runs_clip: tuple[float, float],
    yrfi_clip: tuple[float, float],
) -> pd.DataFrame:
    out = df.sort_values(["park_id", "game_date", "game_pk"], kind="mergesort").copy()

    out["park_games_prior"] = out.groupby("park_id", observed=False).cumcount()
    out["hr_prior_sum"] = out.groupby("park_id", observed=False)["hr_count"].cumsum() - out["hr_count"]
    out["runs_prior_sum"] = out.groupby("park_id", observed=False)["total_runs"].cumsum() - out["total_runs"]
    out["yrfi_prior_sum"] = out.groupby("park_id", observed=False)["yrfi"].cumsum() - out["yrfi"]

    denom = out["park_games_prior"] + float(shrink_k)
    out["shrunk_hr_rate"] = (out["hr_prior_sum"] + float(shrink_k) * mean_hr) / denom
    out["shrunk_runs_mean"] = (out["runs_prior_sum"] + float(shrink_k) * mean_runs) / denom
    out["shrunk_yrfi_rate"] = (out["yrfi_prior_sum"] + float(shrink_k) * mean_yrfi) / denom

    out["park_hr_mult"] = (out["shrunk_hr_rate"] / max(mean_hr, 1e-9)).clip(*hr_clip)
    out["park_runs_delta"] = (out["shrunk_runs_mean"] - mean_runs).clip(*runs_clip)
    out["park_yrfi_delta"] = (out["shrunk_yrfi_rate"] - mean_yrfi).clip(*yrfi_clip)

    neutral_mask = out["park_games_prior"] < int(min_games_prior)
    out.loc[neutral_mask, "park_hr_mult"] = 1.0
    out.loc[neutral_mask, "park_runs_delta"] = 0.0
    out.loc[neutral_mask, "park_yrfi_delta"] = 0.0

    return out


def train_park_factors(
    season: int,
    park_game_path: Path,
    targets_game_path: Path,
    targets_hitter_path: Path,
    events_path: Path,
    output_path: Path,
    *,
    start: str | None = None,
    end: str | None = None,
    config_path: Path = Path("config/parks.yaml"),
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    log = logger or logging.getLogger(__name__)
    if not park_game_path.exists():
        raise FileNotFoundError(f"Missing park_game parquet: {park_game_path}")
    if not targets_game_path.exists():
        raise FileNotFoundError(f"Missing targets_game parquet: {targets_game_path}")

    cfg = _load_cfg(config_path)
    shrink_k = float(cfg.get("shrink_k", 30))
    min_games_prior = int(cfg.get("min_games_prior", 10))
    hr_clip = tuple(float(x) for x in cfg.get("hr_mult_clip", [0.75, 1.30]))
    runs_clip = tuple(float(x) for x in cfg.get("runs_delta_clip", [-0.8, 0.8]))
    yrfi_clip = tuple(float(x) for x in cfg.get("yrfi_delta_clip", [-0.08, 0.08]))

    park = pd.read_parquet(park_game_path, engine="pyarrow")
    tg = pd.read_parquet(targets_game_path, engine="pyarrow")
    hr = _derive_hr_count_by_game(targets_hitter_path=targets_hitter_path, events_path=events_path)

    park["game_date"] = pd.to_datetime(park["game_date"], errors="coerce").dt.normalize()
    tg["game_date"] = pd.to_datetime(tg["game_date"], errors="coerce").dt.normalize()
    for frame in [park, tg, hr]:
        frame["game_pk"] = pd.to_numeric(frame["game_pk"], errors="coerce").astype("Int64")

    req_tg = {"game_pk", "game_date", "total_runs", "yrfi"}
    missing_tg = req_tg - set(tg.columns)
    if missing_tg:
        raise ValueError(f"targets_game missing required columns: {sorted(missing_tg)}")

    tg["total_runs"] = pd.to_numeric(tg["total_runs"], errors="coerce")
    tg["yrfi"] = pd.to_numeric(tg["yrfi"], errors="coerce")
    hr["hr_count"] = pd.to_numeric(hr["hr_count"], errors="coerce")

    df = park[["game_date", "game_pk", "park_id"]].merge(
        tg[["game_date", "game_pk", "total_runs", "yrfi"]], on=["game_date", "game_pk"], how="inner"
    )
    df = df.merge(hr[["game_pk", "hr_count"]], on="game_pk", how="left")
    df = df.dropna(subset=["game_date", "game_pk", "park_id", "total_runs", "yrfi", "hr_count"]).copy()
    df = df.loc[df["game_date"].dt.year == int(season)].copy()

    if start:
        df = df.loc[df["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        df = df.loc[df["game_date"] <= pd.to_datetime(end).normalize()].copy()

    train_df, _ = _time_split(df)
    mean_hr = float(train_df["hr_count"].mean())
    mean_runs = float(train_df["total_runs"].mean())
    mean_yrfi = float(train_df["yrfi"].mean())

    log.info("league_baseline_hr=%.4f runs=%.4f yrfi=%.4f", mean_hr, mean_runs, mean_yrfi)

    asof = _compute_asof_factors(
        df,
        mean_hr=mean_hr,
        mean_runs=mean_runs,
        mean_yrfi=mean_yrfi,
        shrink_k=shrink_k,
        min_games_prior=min_games_prior,
        hr_clip=(float(hr_clip[0]), float(hr_clip[1])),
        runs_clip=(float(runs_clip[0]), float(runs_clip[1])),
        yrfi_clip=(float(yrfi_clip[0]), float(yrfi_clip[1])),
    )

    out = asof[
        [
            "game_date",
            "game_pk",
            "park_id",
            "park_games_prior",
            "park_hr_mult",
            "park_runs_delta",
            "park_yrfi_delta",
        ]
    ].copy()
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    neutral_pct = float((out["park_games_prior"] < min_games_prior).mean()) if len(out) else 0.0
    log.info("park_factors_neutral_pct=%.4f parks=%d", neutral_pct, out["park_id"].nunique())
    park_mean = out.groupby("park_id", observed=False)["park_hr_mult"].mean().sort_values()
    log.info("park_hr_mult_bottom5=%s", park_mean.head(5).to_dict())
    log.info("park_hr_mult_top5=%s", park_mean.tail(5).to_dict())
    for col in ["park_hr_mult", "park_runs_delta", "park_yrfi_delta"]:
        log.info("null_rate_%s=%.4f", col, float(out[col].isna().mean()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_park_factors season=%s rows=%d path=%s", season, len(out), output_path)
    return out


def run_smoke_test(logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger(__name__)
    season = 2099
    dates = pd.date_range(f"{season}-04-01", periods=20, freq="D")
    park_ids = ["PARK_A" if i % 2 == 0 else "PARK_B" for i in range(len(dates))]

    park = pd.DataFrame({"game_date": dates, "game_pk": range(1, len(dates) + 1), "park_id": park_ids})
    tg = pd.DataFrame(
        {
            "game_date": dates,
            "game_pk": range(1, len(dates) + 1),
            "total_runs": [6 + (i % 4) for i in range(len(dates))],
            "yrfi": [1 if i % 3 == 0 else 0 for i in range(len(dates))],
        }
    )
    th = pd.DataFrame(
        {
            "game_date": dates.repeat(2),
            "game_pk": np.repeat(range(1, len(dates) + 1), 2),
            "hr": [1 if i % 5 == 0 else 0 for i in range(len(dates) * 2)],
        }
    )

    tmp = Path("data/processed")
    tmp.mkdir(parents=True, exist_ok=True)
    park_p = tmp / "park_game_smoke.parquet"
    tg_p = Path("data/processed/targets/targets_game_smoke.parquet")
    th_p = Path("data/processed/targets/targets_hitter_game_smoke.parquet")
    out_p = tmp / "park_factors_game_smoke.parquet"
    tg_p.parent.mkdir(parents=True, exist_ok=True)

    park.to_parquet(park_p, index=False, engine="pyarrow")
    tg.to_parquet(tg_p, index=False, engine="pyarrow")
    th.to_parquet(th_p, index=False, engine="pyarrow")

    out = train_park_factors(
        season=season,
        park_game_path=park_p,
        targets_game_path=tg_p,
        targets_hitter_path=th_p,
        events_path=Path("data/processed/events_pa_missing.parquet"),
        output_path=out_p,
        logger=log,
    )

    assert out.equals(out.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True))
    assert out["park_hr_mult"].between(0.75, 1.30).all()
    assert out["park_runs_delta"].between(-0.8, 0.8).all()
    assert out["park_yrfi_delta"].between(-0.08, 0.08).all()
    assert out.iloc[0]["park_hr_mult"] == 1.0
    assert out.iloc[0]["park_runs_delta"] == 0.0
    assert out.iloc[0]["park_yrfi_delta"] == 0.0

    for p in [park_p, tg_p, th_p, out_p]:
        if p.exists():
            p.unlink()
    log.info("park_factors smoke test passed")
