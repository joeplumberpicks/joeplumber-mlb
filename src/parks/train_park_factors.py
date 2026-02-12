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
            "min_games_prior_split": 8,
            "hr_mult_clip": [0.75, 1.30],
            "runs_delta_clip": [-0.8, 0.8],
            "yrfi_delta_clip": [-0.08, 0.08],
        }
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _normalize_side(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.upper().str.strip()
    return s.where(s.isin(["R", "L"]))


def _find_first(columns: pd.Index, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


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


def _derive_hr_split_by_game(targets_hitter_path: Path, events_path: Path, logger: logging.Logger) -> tuple[pd.DataFrame, str]:
    # Preferred: hitter targets with batter hand.
    if targets_hitter_path.exists():
        th = pd.read_parquet(targets_hitter_path, engine="pyarrow")
        hand_col = _find_first(th.columns, ["batter_stand", "stand", "bat_side", "bats", "batter_hand"])  # preferred
        if hand_col and {"game_pk", "hr"}.issubset(th.columns):
            d = th[["game_pk", "hr", hand_col]].copy()
            d["game_pk"] = pd.to_numeric(d["game_pk"], errors="coerce").astype("Int64")
            d["hr"] = pd.to_numeric(d["hr"], errors="coerce").fillna(0.0)
            d["hand"] = _normalize_side(d[hand_col]).fillna("UNK")
            agg = d.groupby(["game_pk", "hand"], observed=False).agg(hr=("hr", "sum"), opp=("hr", "size")).reset_index()
            piv_hr = agg.pivot(index="game_pk", columns="hand", values="hr").fillna(0.0)
            piv_opp = agg.pivot(index="game_pk", columns="hand", values="opp").fillna(0.0)
            out = pd.DataFrame({"game_pk": piv_hr.index.astype("Int64")})
            out["hr_vsR"] = piv_hr.get("R", pd.Series(0.0, index=out.index)).to_numpy()
            out["hr_vsL"] = piv_hr.get("L", pd.Series(0.0, index=out.index)).to_numpy()
            out["opp_vsR"] = piv_opp.get("R", pd.Series(0.0, index=out.index)).to_numpy()
            out["opp_vsL"] = piv_opp.get("L", pd.Series(0.0, index=out.index)).to_numpy()
            return out.reset_index(drop=True), "batter_hand"

    # Fallback: events PA with batter/pitcher hand columns.
    if events_path.exists():
        ev = pd.read_parquet(events_path, engine="pyarrow")
        if {"game_pk", "events"}.issubset(ev.columns):
            hand_col = _find_first(ev.columns, ["batter_stand", "stand", "bat_side", "bats", "batter_hand"])
            basis = "batter_hand"
            if hand_col is None:
                hand_col = _find_first(ev.columns, ["pitcher_throws", "p_throws", "pitcher_hand"])
                basis = "pitcher_hand"
            if hand_col is not None:
                d = ev[["game_pk", "events", hand_col]].copy()
                d["game_pk"] = pd.to_numeric(d["game_pk"], errors="coerce").astype("Int64")
                d["hr"] = (d["events"].astype("string").str.lower() == "home_run").astype(float)
                d["hand"] = _normalize_side(d[hand_col]).fillna("UNK")
                agg = d.groupby(["game_pk", "hand"], observed=False).agg(hr=("hr", "sum"), opp=("hr", "size")).reset_index()
                piv_hr = agg.pivot(index="game_pk", columns="hand", values="hr").fillna(0.0)
                piv_opp = agg.pivot(index="game_pk", columns="hand", values="opp").fillna(0.0)
                out = pd.DataFrame({"game_pk": piv_hr.index.astype("Int64")})
                out["hr_vsR"] = piv_hr.get("R", pd.Series(0.0, index=out.index)).to_numpy()
                out["hr_vsL"] = piv_hr.get("L", pd.Series(0.0, index=out.index)).to_numpy()
                out["opp_vsR"] = piv_opp.get("R", pd.Series(0.0, index=out.index)).to_numpy()
                out["opp_vsL"] = piv_opp.get("L", pd.Series(0.0, index=out.index)).to_numpy()
                return out.reset_index(drop=True), basis

    logger.warning("No handedness source found for park split factors; using neutral split values.")
    return pd.DataFrame(columns=["game_pk", "hr_vsR", "hr_vsL", "opp_vsR", "opp_vsL"]), "none"


def _time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    if n < 10:
        raise ValueError("Need at least 10 rows to train park factors")
    split_idx = min(max(int(np.floor(0.8 * n)), 1), n - 1)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def _safe_div(num: pd.Series, den: pd.Series, default: float) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.fillna(default)


def _compute_asof_factors(
    df: pd.DataFrame,
    *,
    mean_hr: float,
    mean_runs: float,
    mean_yrfi: float,
    mean_hr_vsR: float,
    mean_hr_vsL: float,
    shrink_k: float,
    min_games_prior: int,
    min_games_prior_split: int,
    hr_clip: tuple[float, float],
    runs_clip: tuple[float, float],
    yrfi_clip: tuple[float, float],
) -> pd.DataFrame:
    out = df.sort_values(["park_id", "game_date", "game_pk"], kind="mergesort").copy()

    # Overall prior-only park and league context.
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out["league_games_prior"] = np.arange(len(out), dtype=float)
    out["league_hr_prior_sum"] = out["hr_count"].cumsum() - out["hr_count"]
    out["league_runs_prior_sum"] = out["total_runs"].cumsum() - out["total_runs"]
    out["league_yrfi_prior_sum"] = out["yrfi"].cumsum() - out["yrfi"]
    out["league_hr_rate_prior"] = _safe_div(out["league_hr_prior_sum"], out["league_games_prior"], mean_hr)
    out["league_runs_mean_prior"] = _safe_div(out["league_runs_prior_sum"], out["league_games_prior"], mean_runs)
    out["league_yrfi_rate_prior"] = _safe_div(out["league_yrfi_prior_sum"], out["league_games_prior"], mean_yrfi)

    out = out.sort_values(["park_id", "game_date", "game_pk"], kind="mergesort")
    out["park_games_prior"] = out.groupby("park_id", observed=False).cumcount()
    out["hr_prior_sum"] = out.groupby("park_id", observed=False)["hr_count"].cumsum() - out["hr_count"]
    out["runs_prior_sum"] = out.groupby("park_id", observed=False)["total_runs"].cumsum() - out["total_runs"]
    out["yrfi_prior_sum"] = out.groupby("park_id", observed=False)["yrfi"].cumsum() - out["yrfi"]

    denom = out["park_games_prior"] + float(shrink_k)
    out["shrunk_hr_rate"] = (out["hr_prior_sum"] + float(shrink_k) * out["league_hr_rate_prior"]) / denom
    out["shrunk_runs_mean"] = (out["runs_prior_sum"] + float(shrink_k) * out["league_runs_mean_prior"]) / denom
    out["shrunk_yrfi_rate"] = (out["yrfi_prior_sum"] + float(shrink_k) * out["league_yrfi_rate_prior"]) / denom

    out["park_hr_mult"] = (out["shrunk_hr_rate"] / out["league_hr_rate_prior"].replace(0, np.nan)).fillna(1.0).clip(*hr_clip)
    out["park_runs_delta"] = (out["shrunk_runs_mean"] - out["league_runs_mean_prior"]).clip(*runs_clip)
    out["park_yrfi_delta"] = (out["shrunk_yrfi_rate"] - out["league_yrfi_rate_prior"]).clip(*yrfi_clip)

    neutral_mask = out["park_games_prior"] < int(min_games_prior)
    out.loc[neutral_mask, "park_hr_mult"] = 1.0
    out.loc[neutral_mask, "park_runs_delta"] = 0.0
    out.loc[neutral_mask, "park_yrfi_delta"] = 0.0

    # Split by handedness, prior-only using opportunities.
    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out["league_hr_vsR_prior_sum"] = out["hr_vsR"].cumsum() - out["hr_vsR"]
    out["league_hr_vsL_prior_sum"] = out["hr_vsL"].cumsum() - out["hr_vsL"]
    out["league_opp_vsR_prior_sum"] = out["opp_vsR"].cumsum() - out["opp_vsR"]
    out["league_opp_vsL_prior_sum"] = out["opp_vsL"].cumsum() - out["opp_vsL"]
    out["league_hr_rate_vsR_prior"] = _safe_div(out["league_hr_vsR_prior_sum"], out["league_opp_vsR_prior_sum"], mean_hr_vsR)
    out["league_hr_rate_vsL_prior"] = _safe_div(out["league_hr_vsL_prior_sum"], out["league_opp_vsL_prior_sum"], mean_hr_vsL)

    out = out.sort_values(["park_id", "game_date", "game_pk"], kind="mergesort")
    out["park_games_prior_vsR"] = (
        out.assign(_has=(out["opp_vsR"] > 0).astype(int)).groupby("park_id", observed=False)["_has"].cumsum()
        - (out["opp_vsR"] > 0).astype(int)
    )
    out["park_games_prior_vsL"] = (
        out.assign(_has=(out["opp_vsL"] > 0).astype(int)).groupby("park_id", observed=False)["_has"].cumsum()
        - (out["opp_vsL"] > 0).astype(int)
    )

    out["park_hr_vsR_prior_sum"] = out.groupby("park_id", observed=False)["hr_vsR"].cumsum() - out["hr_vsR"]
    out["park_hr_vsL_prior_sum"] = out.groupby("park_id", observed=False)["hr_vsL"].cumsum() - out["hr_vsL"]
    out["park_opp_vsR_prior_sum"] = out.groupby("park_id", observed=False)["opp_vsR"].cumsum() - out["opp_vsR"]
    out["park_opp_vsL_prior_sum"] = out.groupby("park_id", observed=False)["opp_vsL"].cumsum() - out["opp_vsL"]

    out["park_hr_rate_vsR_prior"] = (
        out["park_hr_vsR_prior_sum"] + float(shrink_k) * out["league_hr_rate_vsR_prior"]
    ) / (out["park_opp_vsR_prior_sum"] + float(shrink_k))
    out["park_hr_rate_vsL_prior"] = (
        out["park_hr_vsL_prior_sum"] + float(shrink_k) * out["league_hr_rate_vsL_prior"]
    ) / (out["park_opp_vsL_prior_sum"] + float(shrink_k))

    out["park_hr_mult_vsR"] = (
        out["park_hr_rate_vsR_prior"] / out["league_hr_rate_vsR_prior"].replace(0, np.nan)
    ).fillna(1.0).clip(*hr_clip)
    out["park_hr_mult_vsL"] = (
        out["park_hr_rate_vsL_prior"] / out["league_hr_rate_vsL_prior"].replace(0, np.nan)
    ).fillna(1.0).clip(*hr_clip)

    neutral_r = out["park_games_prior_vsR"] < int(min_games_prior_split)
    neutral_l = out["park_games_prior_vsL"] < int(min_games_prior_split)
    out.loc[neutral_r, "park_hr_mult_vsR"] = 1.0
    out.loc[neutral_l, "park_hr_mult_vsL"] = 1.0

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
    min_games_prior_split = int(cfg.get("min_games_prior_split", min_games_prior))
    hr_clip = tuple(float(x) for x in cfg.get("hr_mult_clip", [0.75, 1.30]))
    runs_clip = tuple(float(x) for x in cfg.get("runs_delta_clip", [-0.8, 0.8]))
    yrfi_clip = tuple(float(x) for x in cfg.get("yrfi_delta_clip", [-0.08, 0.08]))

    park = pd.read_parquet(park_game_path, engine="pyarrow")
    tg = pd.read_parquet(targets_game_path, engine="pyarrow")
    hr = _derive_hr_count_by_game(targets_hitter_path=targets_hitter_path, events_path=events_path)
    split, split_basis = _derive_hr_split_by_game(targets_hitter_path=targets_hitter_path, events_path=events_path, logger=log)
    log.info("park_split_basis=%s", split_basis)

    park["game_date"] = pd.to_datetime(park["game_date"], errors="coerce").dt.normalize()
    tg["game_date"] = pd.to_datetime(tg["game_date"], errors="coerce").dt.normalize()
    for frame in [park, tg, hr, split]:
        if "game_pk" in frame.columns:
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
    if not split.empty:
        df = df.merge(split[["game_pk", "hr_vsR", "hr_vsL", "opp_vsR", "opp_vsL"]], on="game_pk", how="left")

    for col in ["hr_vsR", "hr_vsL", "opp_vsR", "opp_vsL"]:
        if col not in df.columns:
            df[col] = 0.0
    df[["hr_vsR", "hr_vsL", "opp_vsR", "opp_vsL"]] = df[["hr_vsR", "hr_vsL", "opp_vsR", "opp_vsL"]].fillna(0.0)

    df = df.dropna(subset=["game_date", "game_pk", "park_id", "total_runs", "yrfi", "hr_count"]).copy()
    df = df.loc[df["game_date"].dt.year == int(season)].copy()

    if start:
        df = df.loc[df["game_date"] >= pd.to_datetime(start).normalize()].copy()
    if end:
        df = df.loc[df["game_date"] <= pd.to_datetime(end).normalize()].copy()

    train_df, _ = _time_split(df)
    mean_hr = float(train_df["hr_count"].mean())
    mean_runs = float(train_df["total_runs"].mean())
    mean_yrfi = float(np.clip(train_df["yrfi"].mean(), 1e-9, 1 - 1e-9))
    mean_hr_vsR = float((train_df["hr_vsR"].sum() / max(train_df["opp_vsR"].sum(), 1.0)))
    mean_hr_vsL = float((train_df["hr_vsL"].sum() / max(train_df["opp_vsL"].sum(), 1.0)))
    mean_hr_vsR = max(mean_hr_vsR, 1e-9)
    mean_hr_vsL = max(mean_hr_vsL, 1e-9)

    log.info(
        "league_baseline_hr=%.4f runs=%.4f yrfi=%.4f hr_vsR=%.6f hr_vsL=%.6f",
        mean_hr,
        mean_runs,
        mean_yrfi,
        mean_hr_vsR,
        mean_hr_vsL,
    )

    asof = _compute_asof_factors(
        df,
        mean_hr=mean_hr,
        mean_runs=mean_runs,
        mean_yrfi=mean_yrfi,
        mean_hr_vsR=mean_hr_vsR,
        mean_hr_vsL=mean_hr_vsL,
        shrink_k=shrink_k,
        min_games_prior=min_games_prior,
        min_games_prior_split=min_games_prior_split,
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
            "park_hr_mult_vsR",
            "park_hr_mult_vsL",
            "park_games_prior_vsR",
            "park_games_prior_vsL",
            "park_hr_rate_vsR_prior",
            "park_hr_rate_vsL_prior",
        ]
    ].copy()

    out = out.sort_values(["game_date", "game_pk"], kind="mergesort")
    out = out.drop_duplicates(subset=["game_pk"], keep="first").reset_index(drop=True)

    neutral_pct = float((out["park_games_prior"] < min_games_prior).mean()) if len(out) else 0.0
    neutral_pct_r = float((out["park_games_prior_vsR"] < min_games_prior_split).mean()) if len(out) else 0.0
    neutral_pct_l = float((out["park_games_prior_vsL"] < min_games_prior_split).mean()) if len(out) else 0.0
    log.info(
        "park_factors_neutral_pct overall=%.4f vsR=%.4f vsL=%.4f parks=%d",
        neutral_pct,
        neutral_pct_r,
        neutral_pct_l,
        out["park_id"].nunique(),
    )

    park_mean = out.groupby("park_id", observed=False)["park_hr_mult"].mean().sort_values()
    log.info("park_hr_mult_bottom5=%s", park_mean.head(5).to_dict())
    log.info("park_hr_mult_top5=%s", park_mean.tail(5).to_dict())
    log.info(
        "park_hr_mult_vsR_mean=%.4f park_hr_mult_vsL_mean=%.4f",
        float(out["park_hr_mult_vsR"].mean()) if len(out) else 1.0,
        float(out["park_hr_mult_vsL"].mean()) if len(out) else 1.0,
    )
    for col in [
        "park_hr_mult",
        "park_runs_delta",
        "park_yrfi_delta",
        "park_hr_mult_vsR",
        "park_hr_mult_vsL",
        "park_games_prior_vsR",
        "park_games_prior_vsL",
    ]:
        log.info("null_rate_%s=%.4f", col, float(out[col].isna().mean()))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_park_factors season=%s rows=%d path=%s", season, len(out), output_path)
    return out


def run_smoke_test(logger: logging.Logger | None = None) -> None:
    log = logger or logging.getLogger(__name__)
    season = 2099
    dates = pd.date_range(f"{season}-04-01", periods=24, freq="D")

    park = pd.DataFrame(
        {
            "game_date": dates,
            "game_pk": range(1, len(dates) + 1),
            "park_id": ["PARK_A" if i < 12 else "PARK_B" for i in range(len(dates))],
        }
    )
    tg = pd.DataFrame(
        {
            "game_date": dates,
            "game_pk": range(1, len(dates) + 1),
            "total_runs": [7 + (i % 5) for i in range(len(dates))],
            "yrfi": [1 if i % 4 == 0 else 0 for i in range(len(dates))],
        }
    )

    rows = []
    for i, d in enumerate(dates, start=1):
        for b in range(10):
            side = "R" if b < 6 else "L"
            hr = 1 if ((i % 6 == 0 and side == "R") or (i % 7 == 0 and side == "L")) else 0
            rows.append({"game_date": d, "game_pk": i, "batter_stand": side, "hr": hr})
    th = pd.DataFrame(rows)

    tmp = Path("data/processed")
    tmp.mkdir(parents=True, exist_ok=True)
    park_p = tmp / "park_game_smoke.parquet"
    tg_p = Path("data/processed/targets/targets_game_smoke.parquet")
    th_p = Path("data/processed/targets/targets_hitter_game_smoke.parquet")
    cfg_p = Path("config/parks_smoke.yaml")
    out_p = tmp / "park_factors_game_smoke.parquet"
    tg_p.parent.mkdir(parents=True, exist_ok=True)

    park.to_parquet(park_p, index=False, engine="pyarrow")
    tg.to_parquet(tg_p, index=False, engine="pyarrow")
    th.to_parquet(th_p, index=False, engine="pyarrow")
    cfg_p.write_text(
        "\n".join(
            [
                "shrink_k: 10",
                "min_games_prior: 1",
                "min_games_prior_split: 1",
                "hr_mult_clip: [0.75, 1.30]",
                "runs_delta_clip: [-0.8, 0.8]",
                "yrfi_delta_clip: [-0.08, 0.08]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = train_park_factors(
        season=season,
        park_game_path=park_p,
        targets_game_path=tg_p,
        targets_hitter_path=th_p,
        events_path=Path("data/processed/events_pa_missing.parquet"),
        output_path=out_p,
        config_path=cfg_p,
        logger=log,
    )

    assert out.equals(out.sort_values(["game_date", "game_pk"], kind="mergesort").reset_index(drop=True))
    assert out["game_pk"].is_unique
    assert out["park_hr_mult"].between(0.75, 1.30).all()
    assert out["park_hr_mult_vsR"].between(0.75, 1.30).all()
    assert out["park_hr_mult_vsL"].between(0.75, 1.30).all()

    first = out.iloc[0]
    assert first["park_games_prior"] == 0
    assert first["park_games_prior_vsR"] == 0
    assert first["park_games_prior_vsL"] == 0
    assert first["park_hr_mult"] == 1.0
    assert first["park_hr_mult_vsR"] == 1.0
    assert first["park_hr_mult_vsL"] == 1.0

    park_a_second = out[out["park_id"] == "PARK_A"].iloc[1]
    assert park_a_second["park_games_prior"] >= 1

    for p in [park_p, tg_p, th_p, out_p, cfg_p]:
        if p.exists():
            p.unlink()
    log.info("park_factors smoke test passed")
