from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


PPMI_METRICS = [
    "xwoba_on_contact",
    "hard_hit_rate",
    "whiff_rate",
    "hr_per_bip",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.astype("float64").replace(0.0, np.nan)
    return numerator.astype("float64") / den


def _normalize_matchup_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for alias, canonical in [("batter", "batter_id"), ("pitcher", "pitcher_id")]:
        if alias in out.columns and canonical not in out.columns:
            out = out.rename(columns={alias: canonical})

    for col in ["batter_id", "pitcher_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    for col, default in [("batter_stand", "UNK"), ("pitcher_throws", "UNK")]:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip().str.upper().replace("", pd.NA).fillna(default)
        else:
            out[col] = default

    return out


def _load_matchups(season: int, processed_dir: Path, explicit_path: Path | None) -> pd.DataFrame:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Matchups file not found: {explicit_path}")
        return pd.read_parquet(explicit_path, engine="pyarrow")

    spine_path = processed_dir / "model_spine_game.parquet"
    if spine_path.exists():
        spine = pd.read_parquet(spine_path, engine="pyarrow")
        # model_spine_game may not yet have batter/pitcher matchup granularity.
        required = {"game_date", "batter_id", "pitcher_id"}
        if required.issubset(spine.columns):
            return spine

    fallback = processed_dir / f"matchups_{season}.parquet"
    if fallback.exists():
        return pd.read_parquet(fallback, engine="pyarrow")

    raise FileNotFoundError(
        "No matchup source found. Provide --matchups, or create "
        f"{fallback}, or ensure model_spine_game.parquet contains batter/pitcher matchup rows."
    )


def _load_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} parquet: {path}")
    return pd.read_parquet(path, engine="pyarrow")


def _prepare_pitcher_mix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["pitcher_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    out["pitch_type"] = out["pitch_type"].astype("string").str.strip().str.upper()
    if "batter_stand" in out.columns:
        out["batter_stand"] = out["batter_stand"].astype("string").str.strip().str.upper().replace("", pd.NA).fillna("UNK")
    else:
        out["batter_stand"] = "UNK"

    out["pitch_usage_pct"] = pd.to_numeric(out.get("pitch_usage_pct"), errors="coerce")
    out = out.dropna(subset=["pitcher_id", "pitch_type", "pitch_usage_pct"]).copy()
    return out


def _prepare_hitter_pitchtype(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["batter_id"] = pd.to_numeric(out["batter_id"], errors="coerce").astype("Int64")
    out["pitch_type"] = out["pitch_type"].astype("string").str.strip().str.upper()
    if "pitcher_throws" in out.columns:
        out["pitcher_throws"] = out["pitcher_throws"].astype("string").str.strip().str.upper().replace("", pd.NA).fillna("UNK")
    else:
        out["pitcher_throws"] = "UNK"

    for metric in PPMI_METRICS:
        if metric in out.columns:
            out[metric] = pd.to_numeric(out[metric], errors="coerce")
        else:
            out[metric] = np.nan

    out = out.dropna(subset=["batter_id", "pitch_type"]).copy()

    # UNK -> overall fallback by batter/pitch_type, weighted by pitches when present.
    if "pitches" in out.columns:
        weights = pd.to_numeric(out["pitches"], errors="coerce").fillna(0.0)
    else:
        weights = pd.Series(1.0, index=out.index)

    overall = out[["batter_id", "pitch_type"]].copy()
    overall["_w"] = weights
    for metric in PPMI_METRICS:
        overall[f"_num_{metric}"] = out[metric] * weights

    grouped = overall.groupby(["batter_id", "pitch_type"], dropna=False, observed=False).sum().reset_index()
    for metric in PPMI_METRICS:
        grouped[metric] = grouped[f"_num_{metric}"] / grouped["_w"].replace(0.0, np.nan)

    overall_fallback = grouped[["batter_id", "pitch_type", *PPMI_METRICS]].copy()
    return out, overall_fallback


def build_ppmi_matchup_features(
    season: int,
    pitcher_mix: pd.DataFrame,
    hitter_pitchtype: pd.DataFrame,
    matchups: pd.DataFrame,
    *,
    coverage_threshold: float = 0.60,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Build usage-weighted pitcher-pitchmix matchup indices for batter/pitcher pairs."""
    log = logger or logging.getLogger(__name__)

    pm = _prepare_pitcher_mix(pitcher_mix)
    hp, hp_overall = _prepare_hitter_pitchtype(hitter_pitchtype)
    mu = _normalize_matchup_columns(matchups)

    needed_matchups = ["game_date", "batter_id", "pitcher_id", "batter_stand", "pitcher_throws"]
    missing = [c for c in needed_matchups if c not in mu.columns]
    if missing:
        raise ValueError(f"matchups missing required columns: {missing}")

    mu = mu.dropna(subset=["batter_id", "pitcher_id"]).copy()
    mu["season"] = season

    out_rows: list[dict] = []
    low_coverage_count = 0

    pm_idx = {
        (pid, stand): g.sort_values("pitch_usage_pct", ascending=False, kind="mergesort")
        for (pid, stand), g in pm.groupby(["pitcher_id", "batter_stand"], dropna=False, observed=False)
    }

    hp_exact_idx = {
        (bid, pthrow, ptype): row
        for _, row in hp.iterrows()
        for bid, pthrow, ptype in [(row["batter_id"], row["pitcher_throws"], row["pitch_type"])]
    }
    hp_overall_idx = {
        (bid, ptype): row
        for _, row in hp_overall.iterrows()
        for bid, ptype in [(row["batter_id"], row["pitch_type"])]
    }

    for _, m in mu.iterrows():
        batter_id = m["batter_id"]
        pitcher_id = m["pitcher_id"]
        batter_stand = m["batter_stand"] if pd.notna(m["batter_stand"]) else "UNK"
        pitcher_throws = m["pitcher_throws"] if pd.notna(m["pitcher_throws"]) else "UNK"

        pm_slice = pm_idx.get((pitcher_id, batter_stand))
        if pm_slice is None:
            pm_slice = pm_idx.get((pitcher_id, "UNK"))
        if pm_slice is None:
            pm_slice = pm.loc[pm["pitcher_id"] == pitcher_id].sort_values(
                "pitch_usage_pct", ascending=False, kind="mergesort"
            )

        if pm_slice.empty:
            out_rows.append(
                {
                    "game_date": m["game_date"],
                    "batter_id": batter_id,
                    "pitcher_id": pitcher_id,
                    "ppmi_xwoba": np.nan,
                    "ppmi_hardhit": np.nan,
                    "ppmi_whiff": np.nan,
                    "ppmi_hr": np.nan,
                    "ppmi_pitch_coverage": 0.0,
                    "ppmi_missing_pitch_types": 0,
                    "sp_top_pitch_1": pd.NA,
                    "sp_top_pitch_1_usage": np.nan,
                    "sp_top_pitch_2": pd.NA,
                    "sp_top_pitch_2_usage": np.nan,
                }
            )
            continue

        top2 = pm_slice.head(2)
        top_pitch_1 = top2.iloc[0]["pitch_type"] if len(top2) >= 1 else pd.NA
        top_pitch_1_usage = float(top2.iloc[0]["pitch_usage_pct"]) if len(top2) >= 1 else np.nan
        top_pitch_2 = top2.iloc[1]["pitch_type"] if len(top2) >= 2 else pd.NA
        top_pitch_2_usage = float(top2.iloc[1]["pitch_usage_pct"]) if len(top2) >= 2 else np.nan

        coverage = 0.0
        missing_pitch_types = 0
        sums = {"xwoba": 0.0, "hardhit": 0.0, "whiff": 0.0, "hr": 0.0}

        for _, p in pm_slice.iterrows():
            usage = float(p["pitch_usage_pct"])
            pitch_type = p["pitch_type"]

            hrow = hp_exact_idx.get((batter_id, pitcher_throws, pitch_type))
            if hrow is None and pitcher_throws == "UNK":
                hrow = hp_overall_idx.get((batter_id, pitch_type))

            if hrow is None:
                missing_pitch_types += 1
                continue

            coverage += usage
            if pd.notna(hrow.get("xwoba_on_contact")):
                sums["xwoba"] += usage * float(hrow["xwoba_on_contact"])
            if pd.notna(hrow.get("hard_hit_rate")):
                sums["hardhit"] += usage * float(hrow["hard_hit_rate"])
            if pd.notna(hrow.get("whiff_rate")):
                sums["whiff"] += usage * float(hrow["whiff_rate"])
            if pd.notna(hrow.get("hr_per_bip")):
                sums["hr"] += usage * float(hrow["hr_per_bip"])

        row = {
            "game_date": m["game_date"],
            "batter_id": batter_id,
            "pitcher_id": pitcher_id,
            "ppmi_xwoba": sums["xwoba"],
            "ppmi_hardhit": sums["hardhit"],
            "ppmi_whiff": sums["whiff"],
            "ppmi_hr": sums["hr"],
            "ppmi_pitch_coverage": coverage,
            "ppmi_missing_pitch_types": int(missing_pitch_types),
            "sp_top_pitch_1": top_pitch_1,
            "sp_top_pitch_1_usage": top_pitch_1_usage,
            "sp_top_pitch_2": top_pitch_2,
            "sp_top_pitch_2_usage": top_pitch_2_usage,
        }

        if coverage < coverage_threshold:
            low_coverage_count += 1
            row["ppmi_xwoba"] = np.nan
            row["ppmi_hardhit"] = np.nan
            row["ppmi_whiff"] = np.nan
            row["ppmi_hr"] = np.nan

        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if not out.empty:
        out = out.sort_values(["game_date", "pitcher_id", "batter_id"], kind="mergesort").reset_index(drop=True)

    log.info("ppmi_rows=%d", len(out))
    log.info("ppmi_low_coverage_rows=%d threshold=%.2f", low_coverage_count, coverage_threshold)
    return out


def build_and_write_ppmi_matchup(
    season: int,
    pitcher_mix_path: Path,
    hitter_pitchtype_path: Path,
    processed_dir: Path,
    output_path: Path,
    *,
    matchups_path: Path | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Load inputs, build PPMI matchup features, and write parquet output."""
    log = logger or logging.getLogger(__name__)

    pitcher_mix = _load_required(pitcher_mix_path, "pitcher mix")
    hitter_pitchtype = _load_required(hitter_pitchtype_path, "hitter pitch-type")
    matchups = _load_matchups(season, processed_dir=processed_dir, explicit_path=matchups_path)

    out = build_ppmi_matchup_features(
        season,
        pitcher_mix=pitcher_mix,
        hitter_pitchtype=hitter_pitchtype,
        matchups=matchups,
        logger=log,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False, engine="pyarrow")
    log.info("wrote_ppmi_matchup season=%s rows=%d path=%s", season, len(out), output_path)
    return out
