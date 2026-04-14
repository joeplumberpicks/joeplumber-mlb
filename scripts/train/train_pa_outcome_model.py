#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.pa_outcome_model import (
    PA_OUTCOME_CLASSES,
    fit_pa_outcome_model,
    predict_pa_outcome_class,
    predict_pa_outcome_proba,
    save_pa_outcome_artifact,
)
from src.utils.config import load_config
from src.utils.drive import resolve_data_dirs


SAFE_EXACT_FEATURES = [
    "inning",
    "outs_before_pa",
    "base_state_before",
    "on_1b",
    "on_2b",
    "on_3b",
    "base_runner_count",
    "risp_flag",
    "bases_empty_flag",
    "is_top_inning",
    "is_bot_inning",
    "two_out_flag",
    "inning_bucket",
    "lineup_slot",
]

SAFE_BAT_PREFIXES = [
    "bat_pa_roll",
    "bat_ab_roll",
    "bat_hits_roll",
    "bat_hr_roll",
    "bat_tb_roll",
    "bat_bb_roll",
    "bat_so_roll",
    "bat_hit_rate_roll",
    "bat_hr_rate_roll",
    "bat_tb_pa_rate_roll",
    "bat_bb_rate_roll",
    "bat_so_rate_roll",
    "bat_contact_rate_roll",
    "bat_whiff_rate_roll",
    "bat_hard_hit_rate_roll",
    "bat_barrel_rate_roll",
    "bat_avg_ev_roll",
    "bat_avg_la_roll",
    "bat_gb_rate_roll",
    "bat_fb_rate_roll",
    "bat_pull_rate_roll",
    "bat_iso_roll",
    "bat_hit_rate_calc_roll",
    "bat_hr_rate_calc_roll",
    "bat_bb_rate_calc_roll",
    "bat_so_rate_calc_roll",
    "bat_tb_pa_rate_calc_roll",
    "bat_hit_rate_trend_7v30",
    "bat_hr_rate_trend_7v30",
    "bat_bb_rate_trend_7v30",
    "bat_so_rate_trend_7v30",
    "bat_contact_trend_7v30",
    "bat_hard_hit_trend_7v30",
    "bat_barrel_trend_7v30",
]

SAFE_PIT_PREFIXES = [
    "pit_batters_faced_roll",
    "pit_hits_allowed_roll",
    "pit_hr_allowed_roll",
    "pit_bb_allowed_roll",
    "pit_so_roll",
    "pit_runs_allowed_roll",
    "pit_tb_allowed_roll",
    "pit_k_rate_roll",
    "pit_bb_rate_roll",
    "pit_hr_rate_roll",
    "pit_hit_rate_roll",
    "pit_runs_rate_roll",
    "pit_contact_rate_roll",
    "pit_whiff_rate_roll",
    "pit_hard_hit_rate_roll",
    "pit_barrel_rate_roll",
    "pit_avg_ev_roll",
    "pit_avg_la_roll",
    "pit_gb_rate_roll",
    "pit_fb_rate_roll",
    "pit_hit_rate_calc_roll",
    "pit_hr_rate_calc_roll",
    "pit_bb_rate_calc_roll",
    "pit_runs_rate_calc_roll",
    "pit_hit_rate_trend_7v30",
    "pit_hr_rate_trend_7v30",
    "pit_bb_rate_trend_7v30",
    "pit_k_rate_trend_7v30",
    "pit_hard_hit_trend_7v30",
    "pit_barrel_trend_7v30",
]

SAFE_MATCHUP_PREFIXES = [
    "matchup_hit_rate_diff",
    "matchup_hr_rate_diff",
    "matchup_bb_rate_diff",
    "matchup_k_pressure_diff",
    "matchup_contact_diff",
    "matchup_whiff_diff",
    "matchup_hard_hit_diff",
    "matchup_barrel_diff",
    "matchup_power_diff",
    "matchup_hr_pressure_x",
    "matchup_hit_pressure_x",
    "matchup_walk_pressure_x",
    "matchup_k_pressure_x",
    "matchup_contact_x",
    "matchup_hard_hit_x",
    "matchup_barrel_x",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PA outcome model from mart with strict safe features.")
    parser.add_argument("--train-seasons", type=str, default="2019,2020,2021,2022,2023,2024")
    parser.add_argument("--test-seasons", type=str, default="2025")
    parser.add_argument("--config", type=str, default="configs/project.yaml")
    parser.add_argument("--mart-path", type=str, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _parse_seasons(text: str) -> list[int]:
    vals = [int(t.strip()) for t in str(text).split(",") if t.strip()]
    if not vals:
        raise ValueError(f"No seasons parsed from: {text}")
    return vals


def _load_mart(mart_path: Path) -> pd.DataFrame:
    if not mart_path.exists():
        raise FileNotFoundError(f"Missing mart: {mart_path}")
    df = pd.read_parquet(mart_path)
    print(f"Loaded mart: {mart_path} rows={len(df):,} cols={len(df.columns):,}")
    return df


def _prep_mart(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")

    if "season" not in out.columns:
        raise ValueError("Mart missing required column: season")
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")

    if "pa_outcome_target" not in out.columns:
        raise ValueError("Mart missing required column: pa_outcome_target")
    out["pa_outcome_target"] = out["pa_outcome_target"].astype("string")

    out = out[out["season"].notna()].copy()
    out = out[out["pa_outcome_target"].notna()].copy()
    out = out[out["pa_outcome_target"].isin(set(PA_OUTCOME_CLASSES))].copy()
    return out


def _is_safe_feature(col: str) -> bool:
    if col in SAFE_EXACT_FEATURES:
        return True
    for prefix in SAFE_BAT_PREFIXES + SAFE_PIT_PREFIXES + SAFE_MATCHUP_PREFIXES:
        if col.startswith(prefix):
            return True
    return False


def _build_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [c for c in df.columns if _is_safe_feature(c)]
    if not feature_columns:
        raise ValueError("No strict safe feature columns found in mart.")
    return sorted(feature_columns)


def _multiclass_metrics(df: pd.DataFrame, proba: pd.DataFrame, pred: pd.Series) -> dict:
    y_true = df["pa_outcome_target"].astype("string")
    y_true_id = y_true.map({name: i for i, name in enumerate(PA_OUTCOME_CLASSES)}).astype(int).to_numpy()

    proba_cols = [f"p_{c}" for c in PA_OUTCOME_CLASSES]
    proba_mat = proba[proba_cols].to_numpy()

    return {
        "n_test": int(len(df)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "top2_accuracy": float(
            top_k_accuracy_score(
                y_true_id,
                proba_mat,
                k=2,
                labels=np.arange(len(PA_OUTCOME_CLASSES)),
            )
        ),
        "logloss": float(
            log_loss(
                y_true_id,
                proba_mat,
                labels=np.arange(len(PA_OUTCOME_CLASSES)),
            )
        ),
        "class_rates_test": {c: float((y_true == c).mean()) for c in PA_OUTCOME_CLASSES},
        "pred_rates_test": {c: float((pred == c).mean()) for c in PA_OUTCOME_CLASSES},
        "mean_predicted_probability": {c: float(proba[f"p_{c}"].mean()) for c in PA_OUTCOME_CLASSES},
    }


def main() -> None:
    args = parse_args()

    train_seasons = _parse_seasons(args.train_seasons)
    test_seasons = _parse_seasons(args.test_seasons)

    config = load_config((REPO_ROOT / args.config).resolve())
    dirs = resolve_data_dirs(config=config, prefer_drive=True)

    marts_dir = Path(dirs["marts_dir"])
    models_dir = Path(dirs["models_dir"]) / "pa_outcome_xgb"
    backtests_dir = Path(dirs["backtests_dir"]) / "pa_outcome_xgb"

    models_dir.mkdir(parents=True, exist_ok=True)
    backtests_dir.mkdir(parents=True, exist_ok=True)

    mart_path = Path(args.mart_path) if args.mart_path else marts_dir / "pa_outcome" / "pa_outcome_features.parquet"

    print("========================================")
    print("JOE PLUMBER STRICT SAFE PA OUTCOME TRAIN")
    print("========================================")
    print(f"train_seasons={train_seasons}")
    print(f"test_seasons={test_seasons}")
    print(f"mart_path={mart_path}")

    df = _load_mart(mart_path)
    df = _prep_mart(df)

    feature_columns = _build_feature_columns(df)

    train_df = df[df["season"].isin(train_seasons)].copy()
    test_df = df[df["season"].isin(test_seasons)].copy()

    if train_df.empty:
        raise ValueError(f"No train rows for seasons={train_seasons}")
    if test_df.empty:
        raise ValueError(f"No test rows for seasons={test_seasons}")

    artifact = fit_pa_outcome_model(
        train_df=train_df,
        feature_columns=feature_columns,
        random_state=args.random_state,
    )

    test_proba = predict_pa_outcome_proba(artifact, test_df)
    test_pred = predict_pa_outcome_class(artifact, test_df)

    metrics = _multiclass_metrics(test_df, test_proba, test_pred)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_name = f"pa_outcome_xgb_strictsafe_TEST{'-'.join(map(str, test_seasons))}_{stamp}"

    saved = save_pa_outcome_artifact(
        artifact=artifact,
        out_dir=models_dir,
        artifact_name=artifact_name,
    )

    preview_cols = [c for c in ["game_date", "season", "game_pk", "pa_index", "batter_id", "pitcher_id", "inning", "outs_before_pa", "base_state_before", "pa_outcome_target"] if c in test_df.columns]
    preview = test_df[preview_cols].copy()
    preview["pred_pa_outcome"] = test_pred
    preview = pd.concat([preview.reset_index(drop=True), test_proba.reset_index(drop=True)], axis=1)

    preview_path = backtests_dir / f"{artifact_name}_preview.parquet"
    metrics_path = backtests_dir / f"{artifact_name}_metrics.json"
    feature_list_path = backtests_dir / f"{artifact_name}_features.txt"

    preview.to_parquet(preview_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "artifact_name": artifact_name,
                "train_seasons": train_seasons,
                "test_seasons": test_seasons,
                "feature_count": len(feature_columns),
                "feature_columns": feature_columns,
                "model_path": saved["model_path"],
                "meta_path": saved["meta_path"],
                "preview_path": str(preview_path),
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    with feature_list_path.open("w", encoding="utf-8") as f:
        for col in feature_columns:
            f.write(col + "\n")

    print("")
    print("=== TRAIN SUMMARY ===")
    print(f"train_rows={len(train_df):,}")
    print(f"test_rows={len(test_df):,}")
    print(f"feature_count={len(feature_columns):,}")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"top2_accuracy={metrics['top2_accuracy']:.4f}")
    print(f"logloss={metrics['logloss']:.4f}")

    print("")
    print("=== TEST CLASS RATES ===")
    for c in PA_OUTCOME_CLASSES:
        print(
            f"{c:>10s} | actual={metrics['class_rates_test'][c]:.4f} "
            f"| pred={metrics['pred_rates_test'][c]:.4f} "
            f"| mean_p={metrics['mean_predicted_probability'][c]:.4f}"
        )

    print("")
    print("model_out:", saved["model_path"])
    print("meta_out:", saved["meta_path"])
    print("metrics_out:", metrics_path)
    print("preview_out:", preview_path)
    print("feature_list_out:", feature_list_path)


if __name__ == "__main__":
    main()
